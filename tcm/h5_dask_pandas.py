#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: load from/save to hdf5 using pandas/dask libraries
"""

from glob import escape as glob_escape
import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from tables.exceptions import HDF5ExtError, ClosedFileError
try:
    import dask.array as da
    from dask import compute, dataframe as dd
    from dask.diagnostics import ProgressBar  # or distributed.progress when using the distributed scheduler
except ImportError:
    # not all functions/their options will work
    dd = pd
    da = np
# my
from .h5toh5 import h5remove, ReplaceTableKeepingChilds, df_data_append_fun, df_log_append_fun
from .utils2init import Ex_nothing_done, set_field_if_no, standard_error_info, dir_create_if_need, LoggingStyleAdapter
from .utils_time import timzone_view, multiindex_timeindex, multiindex_replace, minInterval

pd.set_option('io.hdf.default_format', 'table')

lf = LoggingStyleAdapter(logging.getLogger(__name__))

qstr_range_pattern = "index>='{}' & index<='{}'"


def h5q_interval2coord(
        db_path,
        table,
        t_interval: Optional[Sequence[Union[str, pd.Timestamp]]] = None,
        time_range: Optional[Sequence[Union[str, pd.Timestamp]]] = None) -> pd.Index:
    """
    Edge coordinates of index range query
    As it is nealy part of h5toh5.h5select() may be depreshiated? See Note
    :param: db_path, str
    :param: table, str
    :param: t_interval: array or list with strings convertable to pandas.Timestamp
    :param: time_range: same as t_interval (but must be flat numpy array)
    :return: ``qstr_range_pattern`` edge coordinates
    Note: can use instead:
    >>> from to_pandas_hdf5.h5toh5 import h5load_points
    ... with pd.HDFStore(db_path, mode='r') as store:
    ...     df, bbad = h5load_points(store,table,columns=None,query_range_lims=time_range)

    """

    if not t_interval:
        t_interval = time_range
    if not (isinstance(t_interval, list) and isinstance(t_interval[0], str)):
        t_interval = np.array(t_interval).ravel()

    qstr = qstr_range_pattern.format(*t_interval)
    with pd.HDFStore(db_path, mode='r') as store:
        lf.debug("loading range from {:s}/{:s}: {:s} ", db_path, table, qstr)
        try:
            ind_all = store.select_as_coordinates(table, qstr)
        except Exception as e:
            lf.debug("- not loaded: {:s}", e)
            raise
        if len(ind_all):
            ind = ind_all[[0, -1]]  # .values
        else:
            ind = []
        lf.debug('- gets {}', ind)
    return ind


def h5q_intervals_indexes_gen(
        db_path,
        table: str,
        t_prev_interval_start: pd.Timestamp,
        t_intervals_start: Iterable[pd.Timestamp],
        i_range: Optional[Sequence[Union[str, pd.Timestamp]]] = None) -> Iterator[pd.Index]:
    """
    Yields start and end coordinates (0 based indexes) of hdf5 store table index which values are next nearest to intervals start input
    :param db_path
    :param table, str (see h5q_interval2coord)
    :param t_prev_interval_start: first index value
    :param t_intervals_start:
    :param i_range: Sequence, 1st and last element will limit the range of returned result
    :return: Iterator[pd.Index] of lower and upper int limits (adjacent intervals)
    """

    for t_interval_start in t_intervals_start:
        # load_interval
        start_end = h5q_interval2coord(db_path, table, [t_prev_interval_start.isoformat(), t_interval_start.isoformat()])
        if len(start_end):
            if i_range is not None:  # skip intervals that not in index range
                start_end = minInterval([start_end], [i_range], start_end[-1])[0]
                if not len(start_end):
                    if 0 < i_range[-1] < start_end[0]:
                        raise Ex_nothing_done
                    continue
            yield start_end
        else:  # no data
            print('-', end='')
        t_prev_interval_start = t_interval_start


def h5q_ranges_gen(cfg_in: Mapping[str, Any], df_intervals: pd.DataFrame):
    """
    Loading intervals using ranges dataframe (defined by Index and DateEnd column - like in h5toGrid hdf5 log tables)
    :param df_intervals: dataframe, with:
        index - pd.DatetimeIndex for starts of intervals
        DateEnd - pd.Datetime col for ends of intervals
    :param cfg_in: dict, with fields:
        db_path, str
        table, str
    Exsmple:
    >>> df_intervals = pd.DataFrame({'DateEnd': pd.DatetimeIndex([2,4,6])}, index=pd.DatetimeIndex([1,3,5]))
    ... a = h5q_ranges_gen(df_intervals, cfg['out'])
    """
    with pd.HDFStore(cfg_in['db_path'], mode='r') as store:
        print("loading from {db_path}: ".format_map(cfg_in), end='')
        # Query table tblD by intervals from table tblL
        # dfL = store[tblL]
        # dfL.index= dfL.index + dtAdd
        df = pd.DataFrame()
        for n, r in enumerate(df_intervals.itertuples()):  # if n == 3][0]  # dfL.iloc[3], r['Index']= dfL.index[3]
            qstr = qstr_range_pattern.format(r.Index, r.DateEnd)  #
            df = store.select(cfg_in['table'], qstr)  # or dd.query?
            print(qstr)
            yield df


def h5_load_range_by_coord(
        db_path,
        table,
        range_coordinates: Optional[Sequence] = None,
        columns=None,
        chunksize=1000000,
        sorted_index=True,
        **kwargs) -> dd.DataFrame:
    """
    Load (range by integer indexes of) hdf5 data to dask dataframe
    :param range_coordinates: control/limit range of data loading:
        tuple of int, start and end indexes - limit returned dask dataframe by this range
        empty tuple - raise Ex_nothing_done
        None, to load all data
    :param db_path, str/Path
    :param table, str
        dask.read_hdf() parameters:
    :param chunksize,
    :param sorted_index: bool (optional), default True
    :param columns: passed without change to dask.read_hdf()
    """

    db_path_esc = glob_escape(db_path) if isinstance(db_path, Path) else db_path  # need for dask, not compatible with pandas if path contains "["
    if isinstance(columns, pd.Index):
        columns = columns.to_list()  # need for dask\dataframe\io\hdf.py (else ValueError: The truth value of a Index is ambiguous...)

    if range_coordinates is None:  # not specify start and stop.
        print("h5_load_range_by_coord(all)")
        # ?! This is the only option in dask to load sorted index
        ddpart = dd.read_hdf(db_path_esc, table,
                             chunksize=chunksize,
                             lock=True,
                             mode='r',
                             columns=columns,
                             sorted_index=sorted_index)
    elif not len(range_coordinates):
        raise Ex_nothing_done('no data')
    else:
        ddpart_size = -np.subtract(*range_coordinates)
        if not ddpart_size:
            return dd.from_array(
                np.zeros(0, dtype=[('name', 'O'), ('index', 'M8')]))  # DataFrame({},'NoData', {}, [])  # None
        # if ddpart_size < chunksize:
        #     chunksize = ddpart_size  # !? needed to not load more data than need
        # else:
        chunksize = ddpart_size  # !? else loads more data than needs. Do I need to adjust chunksize to divide ddpart_on equal parts?
        # sorted_index=cfg_in['sorted_index'] not works with start/stop so loading without

        for c in [False, True]:  # try with specified columns first
            try:
                # todo: find out why not works any more with distributed scheduler
                ddpart = dd.read_hdf(
                    db_path_esc, table,
                    chunksize=chunksize,
                    # lock=True,  default already
                    # mode='r',  default already
                    columns=columns,
                    start=range_coordinates[0],
                    stop=range_coordinates[-1]
                )
                break
            except ValueError:
                # An error occurred while calling the read_hdf method registered to the pandas backend.
                # Original Message: Stop keyword exceeds dataset number of rows (15000)
                with pd.HDFStore(db_path, mode='r') as store:
                    range_coordinates[-1] = store.get_storer(table).group.table.shape[0]
            except KeyError:  # some of specified columns not exist
                # use only existed columns
                with pd.HDFStore(db_path, mode='r') as store:
                    columns = store[table].columns.join(columns, how='inner').to_list()
                print('found columns:', ', '.join(columns))


        # because of no 'sorted_index' we need:
        ddpart = ddpart.reset_index().set_index(ddpart.index.name or 'index', sorted=sorted_index)  # 'Time'
    return ddpart


def i_bursts_starts_dd(tim, dt_between_blocks: Optional[np.timedelta64] = None):
    raise NotImplementedError
    """ Determine starts of burst in datafreame's index and mean burst size
    :param: tim, dask array or dask index: "Dask Index Structure"
    :param: dt_between_blocks, pd.Timedelta or None - minimum time between blocks.
            Must be greater than delta time within block
            If None then auto find: greater than min of two first intervals + 1s       
    return: (i_burst, mean_burst_size) where:
    - i_burst - indexes of starts of bursts,
    - mean_burst_size - mean burst size.

    >>> tim = pd.date_range('2018-04-17T19:00', '2018-04-17T20:10', freq='2ms').to_series()
    ... di_burst = 200000  # start of burst in tim i.e. burst period = period between samples in tim * period (period is a freq argument) 
    ... burst_len = 100
    ... ix = np.arange(1, len(tim) - di_burst, di_burst) + np.int32([[0], [burst_len]])
    ... tim = pd.concat((tim[st:en] for st,en in ix.T)).index
    ... i_bursts_starts(tim)
    (array([  0, 100, 200, 300, 400, 500, 600, 700, 800, 900]), 100.0)
    # same from i_bursts_starts(tim, dt_between_blocks=pd.Timedelta(minutes=2))
    """

    if isinstance(tim, pd.DatetimeIndex):
        tim = tim.values
    dtime = np.diff(tim.base)
    if dt_between_blocks is None:
        # auto find it: greater interval than min of two first + constant.
        # Some intervals may be zero (in case of bad time resolution) so adding constant enshures that intervals between blocks we'll find is bigger than constant)
        dt_between_blocks = (dtime[0] if dtime[0] < dtime[1] else dtime[1]) + np.timedelta64(1,
                                                                                             's')  # pd.Timedelta(seconds=1)

    # indexes of burst starts
    i_burst = np.append(0, np.flatnonzero(dtime > dt_between_blocks) + 1)

    # calculate mean_block_size
    if len(i_burst) > 1:
        if len(i_burst) > 2:  # amount of data is sufficient to not include edge (likely part of burst) in statistics
            mean_burst_size = np.mean(np.diff(i_burst[1:]))
        if len(i_burst) == 2:  # select biggest of two burst parts we only have
            mean_burst_size = max(i_burst[1], len(tim) - i_burst[1])
    else:
        mean_burst_size = len(tim)

    # dtime_between_bursts = dtime[i_burst-1]     # time of hole  '00:39:59.771684'
    return i_burst, mean_burst_size


# @+node:korzh.20180520212556.1: *4* i_bursts_starts
def i_bursts_starts(tim, dt_between_blocks: Optional[np.timedelta64] = None) -> Tuple[np.array, int, np.timedelta64]:
    """
    Starts of bursts in datafreame's index and mean burst size by calculating difference between each index value
    :param: tim, pd.datetimeIndex
    :param: dt_between_blocks, pd.Timedelta or None or np.inf - minimum time between blocks.
            Must be greater than delta time within block
            If None then auto find: greater than min of two first intervals + 1s
            If np.inf returns (array(0), len(tim))
    return (i_burst, mean_burst_size, max_hole) where:
    - i_burst: indexes of starts of bursts, with first element is 0 (points to start of data)
    - mean_burst_size: mean burst size
    - max_hole: max time distance between bursts found

    >>> tim = pd.date_range('2018-04-17T19:00', '2018-04-17T20:10', freq='2ms').to_series()
    ... di_burst = 200000  # start of burst in tim i.e. burst period = period between samples in tim * period (period is a freq argument)
    ... burst_len = 100
    ... ix = np.arange(1, len(tim) - di_burst, di_burst) + np.int32([[0], [burst_len]])
    ... tim = pd.concat((tim[st:en] for st,en in ix.T)).index
    ... i_bursts_starts(tim)
    (array([  0, 100, 200, 300, 400, 500, 600, 700, 800, 900]), 100.0)
    # same from i_bursts_starts(tim, dt_between_blocks=pd.Timedelta(minutes=2))
    """
    dt_zero = np.timedelta64(0)
    max_hole = dt_zero

    if isinstance(tim, pd.DatetimeIndex):
        tim = tim.values
    if not len(tim):
        return np.int32([]), 0, max_hole

    dtime = np.diff(tim)

    # Checking time is increasing

    if np.any(dtime <= dt_zero):
        lf.warning('Not increased time detected ({:d}+{:d}, first at {:d})!',
                  np.sum(dtime < dt_zero), np.sum(dtime == dt_zero), np.flatnonzero(dtime <= dt_zero)[0])
    # Checking dt_between_blocks
    if dt_between_blocks is None:
        # Auto find it: greater interval than min of two first + constant. Constant = 1s i.e. possible worst time
        # resolution. If bad resolution then 1st or 2nd interval can be zero and without constant we will search everything
        dt_between_blocks = dtime[:2].min() + np.timedelta64(1, 's')
    elif isinstance(dt_between_blocks, pd.Timedelta):
        dt_between_blocks = dt_between_blocks.to_timedelta64()
    elif dt_between_blocks is np.inf:
        return np.int32([0]), len(tim), max_hole

    # Indexes of burst starts
    i_burst = np.flatnonzero(dtime > dt_between_blocks)

    # Calculate mean_block_size
    if i_burst.size:
        if i_burst.size > 1:  # amount of data is sufficient to not include edge (likely part of burst) in statistics
            mean_burst_size = np.mean(np.diff(i_burst))
        elif len(i_burst) == 1:  # select biggest of two burst parts we only have
            i_burst_st = i_burst[0] + 1
            mean_burst_size = max(i_burst_st, len(tim) - i_burst_st)

        max_hole = dtime[i_burst].max()
    else:
        mean_burst_size = len(tim)

    return np.append(0, i_burst + 1), mean_burst_size, max_hole


# ----------------------------------------------------------------------


def add_tz_if_need(v, tim: Union[pd.Index, dd.Index]) -> pd.Timestamp:
    """
    If tim has tz then ensure v has too
    :param v: time value that need be comparable with tim
    :param tim: series/index from which tz will be copied. If dask.dataframe.Index mast have known divisions
    :return: v with added timezone if needed
    """
    try:
        tz = getattr(tim.dtype if isinstance(tim, dd.Index) else tim, 'tz')
        try:
            if v.tzname() is None:
                v = v.tz_localize(tz=tz)
        except AttributeError:
            v = pd.Timestamp(v, tz='UTC')
    except AttributeError:
        try:  # if v had time zone then pd.Timestamp(v, tz=None) not works
            if v.tzname() is not None:
                v = v.astimezone(None)  # so need this
        except AttributeError:
            v = pd.Timestamp(v)
            if v.tz:  # not None?
                v = v.astimezone(None)
    return v


def filterGlobal_minmax(a, tim=None, cfg_filter=None, b_ok=True, not_warn_if_no_col=[]) -> pd.Series:
    """
    Filter min/max limits
    :param a:           numpy record array or Dataframe
    :param tim:         time array (convrtable to pandas Datimeinex) or None then use a.index instead
    :param cfg_filter:  dict with keys max_'field', min_'field', where 'field' must be
     in _a_ or 'date' (case insensitive)
    :param b_ok: initial mask - True means not filtered yet <=> da.ones(len(tim), dtype=bool, chunks = tim.values.chunks) if isinstance(a, dd.DataFrame) else np.ones_like(tim, dtype=bool)  # True #
    :return: boolean pandas.Series
    """

    def filt_max_or_min(array, lim, v):
        """
        Emplicitly logical adds new check to b_ok
        :param array: numpy array or pandas series to filter
        :param lim:
        :param v:
        :return: array of good rows or None
        """
        nonlocal b_ok  # :param b_ok: logical array
        if isinstance(array, da.Array):
            if lim == 'min':
                b_ok &= (array > v).compute()  # da.logical_and(b_ok, )
            elif lim == 'max':
                b_ok &= (array < v).compute()  # da.logical_and(b_ok, )
            elif lim == 'fun':
                b_ok &= getattr(da, v)(array).compute()
        else:
            if lim == 'min':           # matplotlib error when stop in PyCharm debugger!
                b_ok &= (array > v)  # np.logical_and(b_ok, )
            elif lim == 'max':
                b_ok &= (array < v)  # np.logical_and(b_ok, )
            elif lim == 'fun':
                b_ok &= getattr(np, v)(array)

    if tim is None:
        tim = a.index

    for key, v in cfg_filter.items():  # between(left, right, inclusive=True)
        if v is None:
            continue
        try:
            key, lim = key.rsplit('_', 1)
        except ValueError:  # not enough values to unpack
            continue  # not filter field

        # swap if need (depreciate?):
        if key in ('min', 'max', 'fun'):
            key, lim = lim, key
        # else:
        #     continue        # not filter field
        # key may be lowercase(field) when parsed from *.ini so need find field yet:
        field = [field for field in (a.dtype.names if isinstance(a, np.ndarray
                                                                 ) else a.columns.to_numpy()) if
                 field.lower() == key.lower()]
        if field:
            field = field[0]
            if field == 'date':
                # v= pd.to_datetime(v, utc=True)
                v = pd.Timestamp(v, tz='UTC')
                filt_max_or_min(tim, lim, v)
            else:
                filt_max_or_min(a[field], lim, v)

        elif key == 'date':  # 'index':
            # v= pd.to_datetime(v, utc=True)
            filt_max_or_min(tim, lim, add_tz_if_need(v, tim))
        elif key in ('dict', 'b_bad_cols_in_file', 'corr_time') or key in not_warn_if_no_col:
            # config fields not used here
            pass
        else:
            lf.warning('filter warning: no field "{}"!'.format(key))
    return pd.Series(b_ok, index=tim)


def filter_global_minmax(a: Union[pd.DataFrame, dd.DataFrame],
                         cfg_filter: Optional[Mapping[str, Any]] = None
                         ) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Executes query that filters rows where some values outside min/max limits
    :param a: dask or pandas Dataframe. If need to filter datetime columns then their name must start with 'date'
    :param cfg_filter: dict with
    - keys: max_`col`, min_`col`, where `col` must be in ``a`` (case-insensitive) to filter lower/upper values of `col`.
      To filter by index the `col` part must be equal "date".
    - values: are float or ifs str repr - to compare with col/index values
    :return: dask bool array of good rows (or array if tim is not dask and only tim is filtered)
    """
    if cfg_filter is None:
        return a

    qstrings = []

    # key may be lowercase(field) when parsed from *.ini so need find field yet:
    cols = {col.lower(): col for col in (a.dtype.names if isinstance(a, np.ndarray) else a.columns.values)}

    for lim_key, val in cfg_filter.items():  # between(left, right, inclusive=True)
        try:
            lim, key = lim_key.rsplit('_', 1)
        except ValueError:  # not enough values to unpack
            continue  # not filter field

        if lim not in ('min', 'max') or val is None or isinstance(val, dict):
            continue  # if val is None then filtering would get AttributeError: 'NaTType' object has no attribute 'tz'

        if key == 'date':  # 'index':
            # val= pd.to_datetime(val, utc=True)
            # cf[lim + '_' + key] = pd.Timestamp(val, tz='UTC') not works for queries (?!)
            col = 'index'
            # tim = a.index
            # tim = a.Time.apply(lambda x: x.tz_localize(None), meta=[('ts', 'datetime64[ns]')])
            val = f"'{add_tz_if_need(val, a.index)}'"
        else:
            try:
                col = cols[key.lower()]
                if col.startswith('date'):  # have datetime column
                    # cf[lim_key] = pd.Timestamp(val, tz='UTC')
                    val = f"'{add_tz_if_need(val, a[col])}'"
            except KeyError:
                lf.warning('filter warning: no column "{}"!'.format(key))
                continue

        # Add expression to query string
        qstrings.append(f"{col}{'>' if lim == 'min' else '<'}{val}")
    # numexpr.set_num_threads(1)
    try:
        return a.query(' & '.join(qstrings)) if any(qstrings) else a
    except (TypeError, ValueError):  # Cannot compare tz-naive and tz-aware datetime-like objects
        lf.exception('filter_global_minmax filtering "{:s}" error! Continuing...', ' & '.join(qstrings))
        return a

    # @cf['{}_{}] not works in dask


def filter_local(d: Union[pd.DataFrame, dd.DataFrame, Mapping[str, pd.Series], Mapping[str, dd.Series]],
                 cfg_filter: Mapping[str, Any], ignore_absent: Optional[set] = None
                 ) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Filtering values without changing output size: set to NaN if exceed limits
    :param d: DataFrame
    :param cfg_filter: This is a dict with dicts "min" and "max" having fields with:
     - keys equal to column names to filter or regex strings to selelect columns: "*" or "[" must be present to detect
    it as regex.
     - values are min and max limits consequently.
    :param ignore_absent: list of cfg_filter['min'] or cfg_filter['max'] fields to not warning if they are absent in d.
    :return: filtered d with bad values replaced by NaN

    """
    for limit, f_compare in [('min', lambda x, lim: x > lim), ('max', lambda x, lim: x < lim)]:
        # todo: check if is better to use between(left, right, inclusive=True)
        if not cfg_filter.get(limit):
            continue
        for key, lim in cfg_filter[limit].items():
            if ('*' in key) or ('[' in key):  # get multiple keys by regex
                keys = [c for c in d.columns if re.fullmatch(key, c)]
                d[keys] = d.loc[:, keys].where(f_compare(d.loc[:, keys], lim))
                key = ', '.join(keys)  # for logging only
            else:
                try:
                    d[key] = d.loc[:, key].where(f_compare(d.loc[:, key], lim))
                except (KeyError, TypeError) as e:  # allow redundant parameters in config
                    # It is strange, but we have "TypeError: cannot do slice indexing on DatetimeIndex with these indexers [key] of type str" if just no "key" column
                    if not (ignore_absent and key in ignore_absent):
                        lf.warning('Can not filter this parameter {:s}', standard_error_info(e))
                    continue
                except AssertionError:  # handle one strange dask error if d is empty
                    if not d.count().compute().any():
                        return d.persist()  # very cheap dask dataframe
                    else:
                        raise  # not have guessed - other error
            lf.debug('filtering {:s}({:s}) = {:g}', limit, key, lim)
    return d


def filter_local_arr(d: Mapping[str, Sequence],
                 cfg_filter: Mapping[str, Any]
                 ) -> Mapping[str, np.ndarray]:
    """
    Same as filter_local but for dict of arrays
    Filtering values without changing output size: set to NaN if exceed limits
    :param d: dict of arrays
    :param cfg: must have field 'filter'. This is a dict with dicts "min" and "max" having fields with:
     - keys equal to column names to filter or regex strings to selelect columns: "*" or "[" must be present to detect
    it as regex.
     - values are min and max limits consequently.
    :return: filtered input where filtered values are np.ndarrays having filtered values replaced by NaN

    """
    for limit, f_compare in [('min', lambda x, v: x < v), ('max', lambda x, v: x > v)]:
        # todo: check if is better to use between(left, right, inclusive=True)
        if not cfg_filter.get(limit):
            continue
        for key, v in cfg_filter[limit].items():
            if ('*' in key) or ('[' in key):  # get multiple keys by regex
                # Filter multiple columns at once
                keys = [c for c in d.columns if re.fullmatch(key, c)]
                d[keys][f_compare(d[keys], v)] = np.NaN
                key = ', '.join(keys)  # for logging only
            else:
                try:
                    d[key][f_compare(d[key], v)] = np.NaN
                except KeyError as e:  # allow redundant parameters in config
                    lf.warning('Can not filter this parameer {:s}', standard_error_info(e))
            lf.debug('filtering {:s}({:s}) = {:g}', limit, key, v)
    return d


#   Veusz inline version of this (viv):
# dstime = np.diff(stime)
# burst_i = nonzero(dstime>(dstime[0] if dstime[1]>dstime[0] else dstime[1])*2)[0]+1
# mean_burst_size = burst_i[1]-burst_i[0] if len(burst_i)>0 else np.diff(USEi[0,:])
# @+node:korzh.20180520185242.1: *4* filt_blocks_array
def filt_blocks_array(x, i_starts, func=None):
    """
    Filter each block of numpy array separate using provided function.
    :param x: numpy array, to filter
    :param i_starts: numpy array, indexes of starts of bocks
    :param func: filters_scipy.despike() used if None
    returns: numpy array of same size as x with bad values replased with NaNs

    """
    if func is None:
        from filters_scipy import despike

        func = lambda x: despike(x, offsets=(20, 5), blocks=len(x), ax=None, label=None)[0]

    y = da.from_array(x, chunks=(tuple(np.diff(np.append(i_starts, len(x))).tolist()),), name='filt')
    with ProgressBar():
        y_out = y.map_blocks(func, dtype=np.float64, name='blocks_arr').compute()
    return y_out

    # for ist_en in np.c_[i_starts[:-1], i_starts[1:]]:
    # sl = slice(*ist_en)
    # y[sl], _ = despike(x[sl], offsets=(200, 50), block=block, ax=None, label=None)
    # return y


# @+node:korzh.20180604062900.1: *4* filt_blocks_da
def filt_blocks_da(dask_array, i_starts, i_end=None, func=None, *args):
    """
    Apply function to each block of numpy array separately (function is , can be provided other to, for example, filter array)
    :param dask_array: dask array, to filter, may be with unknown chunks as for dask series.values
    :param i_starts: numpy array, indexes of starts of bocks
    :param i_end: len(dask_array) if None then last element of i_starts must be equal to it else i_end should not be in i_starts
    # specifing this removes warning 'invalid value encountered in less'
    :param func: numpy.interp by default interp(NaNs) used if None
    returns: dask array of same size as x with func upplied

    >>> Pfilt = filt_blocks_da(a['P'].values, i_burst, i_end=len(a))
    ... sum(~isfinite(a['P'].values.compute())), sum(~isfinite(Pfilt))  # some nans was removed
    : (6, 0)
    # other values unchanged
    >>> allclose(Pfilt[isfinite(a['P'].values.compute())], a['P'].values[isfinite(a['P'].values)].compute())
    : True
    """
    if func is None:
        func = np.interp
    if i_end:
        i_starts = np.append(i_starts, i_end)
    else:
        i_end = i_starts[-1]

    if np.isnan(dask_array.size):  # unknown chunks delayed transformation
        dask_array = da.from_delayed(dask_array.to_delayed()[0], shape=(i_end,), dtype=np.float64, name='filt')

    y = da.rechunk(dask_array, chunks=(tuple(np.diff(i_starts).tolist()),))
    y_out = y.map_blocks(func, dtype=np.float64, name='blocks_da')
    return y_out

    # for ist_en in np.c_[i_starts[:-1], i_starts[1:]]:
    # sl = slice(*ist_en)
    # y[sl], _ = despike(x[sl], offsets=(200, 50), block=block, ax=None, label=None)
    # return y


def cull_empty_partitions(ddf: dd.DataFrame, lengths=None):
    """
    Remove empty partitions
    :param: ddf
    :return: ddf, lengths: dask dataframe without zero length partitions and its lengths
    """
    if lengths is None:
        lengths = tuple(ddf.map_partitions(len).compute())
    if all(lengths):
        return ddf, lengths

    delayed = ddf.to_delayed()
    delayed_f = []
    lengths_f = []
    for df, len_df in zip(delayed, lengths):
        if len_df:
            delayed_f.append(df)
            lengths_f.append(len_df)
    ddf = dd.from_delayed(delayed_f, meta=delayed[lengths.index(0)].compute())
    return ddf, lengths_f


def df_to_csv(df, cfg_out, add_subdir='', add_suffix=''):
    """
    Exports df to Path(cfg_out['db_path']).parent / add_subdir / pattern_date.format(df.index[0]) + cfg_out['table'] + '.txt'
    where 'pattern_date' = '{:%y%m%d_%H%M}' without lower significant parts than cfg_out['period']
    :param df: pandas.Dataframe
    :param cfg_out: dict with fields:
        db_path
        table
        dir_export (optional) will save here
        period (optional), any from (y,m,d,H,M), case-insensitive - to round time pattern that constructs file name
    modifies: creates if not exist:
        cfg_out['dir_export'] if 'dir_export' is not in cfg_out
        directory 'V,P_txt'

    >>> df_to_csv(df, cfg['out'])
    """
    if 'dir_export' not in cfg_out:
        cfg_out['dir_export'] = Path(cfg_out['db_path']).parent / add_subdir
        if not cfg_out['dir_export'].exists():
            cfg_out['dir_export'].mkdir()

    if 'period' in cfg_out:
        i_period_letter = '%y%m%d_%H%M'.lower().find(cfg_out['period'].lower())
        # if index have not found (-1) then keep all else include all from start to index:
    else:
        i_period_letter = 100  # big enough
    pattern_date = '{:%' + 'y%m%d_%H%M'[:i_period_letter] + '}'

    fileN_time_st = pattern_date.format(df.index[0])
    path_export = cfg_out['dir_export'] / (fileN_time_st + cfg_out['table'] + add_suffix + '.txt')
    print('df_to_csv "{}" is going...'.format(path_export.name), end='')
    df.to_csv(path_export, index_label='DateTime_UTC', date_format='%Y-%m-%dT%H:%M:%S.%f')
    print('Ok')


def dd_to_csv(
        d: dd.DataFrame,
        text_path=None,
        text_date_format: Optional[str] = None,
        text_columns=None,
        aggregate_period=None,
        suffix='',
        single_file_name=True,
        progress=None, client=None,
        b_continue=False
        ):
    """
    Save to ascii if _text_path_ is not None
    :param d: dask dataframe
    :param text_path: None or directory path. If not a dir tries to create and if this fails (like if more than one level) then adds this as prefix to nemes
    :param b_continue: append to the end of existed text file
    :param text_date_format: If callable then create "Date" column by calling it (dd.index), retain index only if "Time" in text_columns. If string use it as format for index (Time) column
    :param text_columns: optional
    :param aggregate_period: [seconds] str or class with repr() to add "bin{}" suffix to files names
    :param suffix: str, will be added to filenamme with forbidden characters removed/replaced
    :param single_file_name:
    - True or str: save all to one file of this name if str or to autogenerated name if True
    - False: generate name for each partition individually and save to multiple files according to d divisions
    :param progress: progress bar object, used with client
    :param client: dask.client, used with progress
    """
    if text_path is None:
        return

    tab = '\t'
    sep = tab
    ext = '.tsv' if sep == tab else '.csv'
    lf.info(
        '{} *{:s}: {:s}',
        'Saving (continue)' if b_continue else 'Saving',
        ext,
        single_file_name if b_continue else '1 file' if single_file_name else f'{d.npartitions} files'
    )
    try:
        dir_create_if_need(text_path)

        def combpath(dir_or_prefix, s):
            return str(dir_or_prefix / s)
    except:
        lf.exception('Dir not created!')

        def combpath(dir_or_prefix, s):
            return f'{dir_or_prefix}{s}'

    def name_that_replaces_asterisk(i_partition):
        return f'{d.divisions[i_partition]:%y%m%d_%H%M}'
        # too long variant: '{:%y%m%d_%H%M}-{:%H%M}'.format(*d.partitions[i_partition].index.compute()[[0,-1]])

    suffix_mod = re.sub(r'[\\/*?:"<>\.]', '', suffix.replace('|', ',').replace('{}', 'all'))
    filename = single_file_name if isinstance(single_file_name, str) else (
        combpath(
            text_path,
            f"{name_that_replaces_asterisk(0) if single_file_name else '*'}{{}}{suffix_mod}{ext}".format(
                f'bin{aggregate_period.lower()}' if aggregate_period else ''  # lower seconds: S -> s
            ))
    )

    d_out = d.round({'Vdir': 4, 'inclination': 4, 'Pressure': 3})
    # if not cfg_out.get('b_all_to_one_col'):
    #     d_out.rename(columns=map_to_suffixed(d.columns, suffix))
    if callable(text_date_format):
        arg_out = {'index': bool(text_columns) and 'Time' in text_columns,
                   'columns': bool(text_columns) or d_out.columns.insert(0, 'Date')
                   }
        d_out['Date'] = d_out.map_partitions(lambda df: text_date_format(df.index))
    else:
        if text_date_format in ('s', '%Y-%m-%d %H:%M:%S'):                   # speedup
            d_out.index = d_out.index.dt.tz_convert(None).dt.ceil(freq='s')  # very speedups!
            arg_out = {'columns': text_columns or None  # for write all columns if empty (replaces to None)
                       }
        else:
            arg_out = {'date_format': text_date_format,  # lead to very long saving (tenths howers) for 2s and smaller resolution data!
                       'columns': text_columns or None  # for write all columns if empty (replaces to None)
                       }

    # Common pandas and dask to_csv() args
    args_to_csv = {
        'float_format': '%.5g',
        'sep': sep,
        'encoding': 'ascii',
        # 'compression': 'zip',
        **arg_out
    }
    if b_continue and single_file_name:  # using pandas because can not append using dask
        d_out = d_out.compute()
        with Path(filename).open(mode='a', newline='') as hfile:
            d_out.to_csv(
                path_or_buf=hfile,
                header=False,
                **args_to_csv  # auto distribute between workers: prevent write by one row (very long)
            )
    else:
        if progress is None:
            pbar = ProgressBar(dt=10)
            pbar.register()
            
        # with dask.config.set(scheduler='processes'):  # need because saving to csv mainly under GIL
        to_csv = d_out.to_csv(
            filename=filename,
            single_file=bool(single_file_name),
            name_function=None if single_file_name else name_that_replaces_asterisk,  # 'epoch' not works
            **args_to_csv,
            compute=False,
            compute_kwargs={'scheduler': 'processes'},
            chunksize=200000  # -1 not works: auto distribute between workers: prevent write by one row (very long)
        )
        # disabling the chain assignment pandas option made my ETL job go from running out of memory after
        # 90 minutes to taking 17 minutes! I think we can close this issue since its related to pandas - not helps:
        # pd.set_option('chained_assignment', None)  #  'warn' (the default), 'raise' (raises an exception),
        # or None (no checks are made).
    
        if progress is None:
            compute(to_csv)
            pbar.unregister()
        else:
            futures = client.compute(to_csv)
            progress(futures)
            # to_csv.result()
            client.gather(futures)
    return filename

def h5_append_dummy_row(df: Union[pd.DataFrame, dd.DataFrame],
                        freq=None,
                        tim: Optional[Sequence[Any]] = None) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Add row of NaN with index value that will between one of last data and one of next data start
    :param df: pandas dataframe, dask.dataframe supported only tim is not None
    :param freq: frequency to calc index. If logically equal to False, then will be calculated using tim
    :param tim: sequence having in last elements time of 2 last rows
    :return: appended dataframe
    """
    if tim is not None:
        try:
            dindex = pd.Timedelta(seconds=0.5 / freq) if freq else np.abs(tim[-1] - tim[-2]) / 2
        except IndexError:  # only one element => we think they are seldom so use 1s
            dindex = pd.Timedelta(seconds=1)
        ind_new = [tim[-1] + dindex]
    else:
        df_index, itm = multiindex_timeindex(df.index)
        try:
            dindex = pd.Timedelta(seconds=0.5 / freq) if freq else np.abs(df_index[-1] - df_index[-2]) / 2
        except (IndexError, NotImplementedError):
            # only one element => we think they are seldom so use 1s or NotImplemented in Dask
            dindex = pd.Timedelta(seconds=1)
        ind_new = multiindex_replace(df.index[-1:], df_index[-1:] + dindex, itm)

    dict_dummy = {}
    tip0 = None
    same_types = True  # tries prevent fall down to object type (which is bad handled by pandas.pytables) if possible
    for name, field in df.dtypes.items():
        typ = field.type
        dict_dummy[name] = typ(0) if np.issubdtype(typ, np.integer) else np.NaN if np.issubdtype(typ,
                                                                                                 np.floating) else ''

        if same_types:
            if typ != tip0:
                if tip0 is None:
                    tip0 = typ
                else:
                    same_types = False

    df_dummy = pd.DataFrame(
        dict_dummy, columns=df.columns.values, index=ind_new, dtype=tip0 if same_types else None
    ).rename_axis('Time')

    if isinstance(df, dd.DataFrame):
        return dd.concat([df, df_dummy], axis=0, interleave_partitions=True)  # buggish dask not always can append
    else:
        return pd.concat([df, df_dummy])  # df.append(df_dummy)

    # np.array([np.int32(0) if np.issubdtype(field.type, int) else
    #           np.NaN if np.issubdtype(field.type, float) else
    #           [] for field in df.dtypes.values]).view(
    #     dtype=np.dtype({'names': df.columns.values, 'formats': df.dtypes.values})))

    # insert separator # 0 (can not use np.nan in int) [tim[-1].to_pydatetime() + pd.Timedelta(seconds = 0.5/cfg['in']['fs'])]
    #   df_dummy= pd.DataFrame(0, columns=cfg_out['names'], index= (pd.NaT,))
    #   df_dummy= pd.DataFrame(np.full(1, np.NaN, dtype= df.dtype), index= (pd.NaT,))
    # used for insert separator lines


def h5append_on_inconsistent_index(cfg_out, tbl_parent, df, df_append_fun, e, msg_func):
    """
    Align types: Make index to be UTC
    :param cfg_out:
    :param tbl_parent:
    :param df:
    :param df_append_fun:
    :param e:
    :param msg_func:
    :return:
    """

    if tbl_parent is None:
        tbl_parent = cfg_out['table']

    error_info_list = [s for s in e.args if isinstance(s, str)]
    msg = msg_func + ' Error:'.format(e.__class__) + '\n==> '.join(error_info_list)
    if not error_info_list:
        lf.error(msg)
        raise e
    b_correct_time = False
    b_correct_str = False
    b_correct_cols = False
    str_check = 'invalid info for [index] for [tz]'
    if error_info_list[0].startswith(str_check) or error_info_list[0] == 'Not consistent index':
        if error_info_list[0] == 'Not consistent index':
            msg += 'Not consistent index detected'
        lf.error(msg + 'Not consistent index time zone? Changing index to standard UTC')
        b_correct_time = True
    elif error_info_list[0].startswith('Trying to store a string with len'):
        b_correct_str = True
        lf.error(msg + error_info_list[0])  # ?
    elif error_info_list[0].startswith('cannot match existing table structure'):
        b_correct_cols = True
        lf.error(f'{msg} => Adding columns...')
        # raise e #?
    elif error_info_list[0].startswith('invalid combination of [values_axes] on appending data') or \
            error_info_list[0].startswith('invalid combination of [non_index_axes] on appending data'):
        # old pandas version has word "combinate" insted of "combination"!
        b_correct_cols = True
        lf.error(f'{msg} => Adding columns/convering type...')
    else:  # Can only append to Tables - need resave?
        lf.error(f'{msg} => Can not handle this error!')
        raise e




    df_cor = cfg_out['db'][tbl_parent]
    b_df_cor_changed = False

    def align_columns(df, df_ref, columns=None):
        """

        :param df: changing dataframe. Will update implicitly!
        :param df_ref: reference dataframe
        :param columns:
        :return: updated df
        """
        if columns is None:
            columns = df.columns
        df = df.reindex(df_ref.columns, axis="columns", copy=False)
        for col, typ in df_ref[columns].dtypes.items():
            fill_value = np.array(
                0 if np.issubdtype(typ, np.integer) else np.NaN if np.issubdtype(typ, np.floating) else '',
                dtype=typ)
            df[col] = fill_value
        return df

    if b_correct_time:
        # change stored to UTC
        df_cor.index = pd.DatetimeIndex(df_cor.index.tz_convert(tz='UTC'))
        b_df_cor_changed = True

    elif b_correct_cols:
        new_cols = list(set(df.columns).difference(df_cor.columns))
        if new_cols:
            df_cor = align_columns(df_cor, df, columns=new_cols)
            b_df_cor_changed = True
            # df_cor = df_cor.reindex(columns=df.columns, copy=False)
        # add columns to df same as in store
        new_cols = list(set(df_cor.columns).difference(df.columns))
        if new_cols:
            if isinstance(df, dd.DataFrame):
                df = df.compute()
            df = align_columns(df, df_cor, columns=new_cols)

    elif b_correct_str:
        # error because our string longer => we need to increase store's limit
        b_df_cor_changed = True

    for col, dtype in zip(df_cor.columns, df_cor.dtypes):
        d = df_cor[col]
        if dtype != df[col].dtype:
            if b_correct_time and isinstance(d[0], pd.Timestamp):  # is it possible that time types are different?
                try:
                    df_cor[col] = d.dt.tz_convert(tz=df[col].dt.tz)
                    b_df_cor_changed = True
                except {AttributeError, ValueError}:  # AttributeError: Can only use .dt accessor with datetimelike values
                    pass
            elif b_correct_str:
                # todo:
                pass
            else:
                try:
                    dtype_max = np.result_type(df_cor[col].dtype, df[col].dtype)
                    if df[col].dtype != dtype_max:
                        df[col] = df[col].astype(dtype_max)
                    if df_cor[col].dtype != dtype_max:
                        df_cor[col] = df_cor[col].astype(dtype_max)
                        b_df_cor_changed = True
                except e:
                    lf.exception('Col "{:s}" have not numpy dtype?', col)
                    df_cor[col] = df_cor[col].astype(df[col].dtype)
                    b_df_cor_changed = True
                # pd.api.types.infer_dtype(df_cor.loc[df_cor.index[0], col], df.loc[df.index[0], col])
        elif b_correct_time and isinstance(d[0], pd.Timestamp):
            try:
                if d.dt.tz != df[col].dt.tz:
                    df_cor[col] = d.dt.tz_convert(tz=df[col].dt.tz)
                    b_df_cor_changed = True
            except (AttributeError, ValueError):  # AttributeError: Can only use .dt accessor with datetimelike values
                pass  # TypeError: Cannot convert tz-naive timestamps, use tz_localize to localize
            
    if b_df_cor_changed:
        # Update all cfg_out['db'] store data
        try:
            with ReplaceTableKeepingChilds([df_cor, df], tbl_parent, cfg_out, df_append_fun):
                pass
            return tbl_parent
        except Exception as e:
            lf.error('{:s} Can not write to store. May be data corrupted. {:s}', msg_func, standard_error_info(e))
            raise e
        except HDF5ExtError as e:
            lf.exception(e)
            raise e
    else:
        # Append corrected data to cfg_out['db'] store
        try:
            return df_append_fun(df, tbl_parent, cfg_out)
        except Exception as e:
            lf.error('{:s} Can not write to store. May be data corrupted. {:s}', msg_func, standard_error_info(e))
            raise e
        except HDF5ExtError as e:
            lf.exception(e)
            raise e


"""       store.get_storer(tbl_parent).group.__members__
           if tblD == cfg_out['table_log']:
                try:
                    df.to_hdf(store, tbl_parent, append=True,
                              data_columns=True)  # , compute=False
                    # store.append(tbl_parent, df, data_columns=True, index=False,
                    #              chunksize=cfg_out['chunksize'])

                except ValueError as e:
                    
            store.append(tbl_parent,
                         df_cor.append(df, verify_integrity=True),
                         data_columns=True, index=False,
                         chunksize=cfg_out['chunksize'])

            childs[tblD] = store[cfg_out['table_log']]

        dfLog = store[cfg_out['table_log']] if cfg_out['table_log'] in store  else None# copy before delete

        # Make index to be UTC
        df_cor.index = pd.to_datetime(store[tbl_parent].index, utc=True)
        store.remove(tbl_parent)
        store.append(tbl_parent,
                     df_cor.append(df, verify_integrity=True),
                     data_columns=True, index=False,
                     chunksize=cfg_out['chunksize'])
        if dfLog: # have removed only if it is a child
            store.remove(cfg_out['table_log'])
            store.append(cfg_out['table_log'], dfLog, data_columns=True, expectedrows=cfg_out['nfiles'], index=False, min_itemsize={'values': cfg_out['logfield_fileName_len']})  # append

"""


def h5add_log(log: Union[pd.DataFrame, MutableMapping, None], cfg_out: Dict[str, Any], tim, df, log_dt_from_utc):
    """
    Updates (or creates if need) metadata/log table in store
    :param cfg_out: dict with fields:
     - b_log_ready: if False or '' then updates log['Date0'], log['DateEnd'] using df
     - db: handle of opened hdf5 store
     - some of following fields (next will be tried if previous not defined):
         - table_log: str, path of log table
         - tables_log: List[str], path of log table in first element
         - table: str, path of log table will be constructed by adding '/log'
         - tables: List[str], path of log table will be constructed by adding '/log' to first element
     - logfield_fileName_len: optional, fixed length of string format of 'fileName' hdf5 column
    :param df: used to get log['Date0'] and log['DateEnd'] as start and end if cfg_out['b_log_ready']
    :param log: Mapping records or dataframe. updates 'Date0' and 'DateEnd' if no 'Date0' or it is {} or None
    :param tim: used to get 'Date0' and 'DateEnd' if they are none or not cfg_out.get('b_log_ready')
    :param log_dt_from_utc:
    :return:
    :updates: log's fields 'Date0', 'DateEnd' if not cfg_out.get('b_log_ready')
    """
    if cfg_out.get('b_log_ready') and (isinstance(log, Mapping) and not log):
        return

    # synchro ``tables_log`` and ``table_log`` (last is more user-friendly but not so universal)
    if cfg_out.get('table_log'):
        table_log = cfg_out['table_log']
    else:
        table_log = cfg_out.get('tables_log')
        if table_log:
           table_log = t0.format(cfg_out['table']) if '{}' in (t0 := table_log[0]) else t0
        else:  # set default for (1st) data table
            try:
                table_log = f"{cfg_out['table']}/log"
            except KeyError:
                table_log = f"{cfg_out['tables'][0]}/log"

    set_field_if_no(cfg_out, 'logfield_fileName_len', 255)

    if (not cfg_out.get('b_log_ready')) or (log.get('DateEnd') is None):
        try:
            t_lims = (
                tim if tim is not None else
                df.index.compute() if isinstance(df, dd.DataFrame) else
                df.index
            )[[0, -1]]
        except IndexError:
            lf.debug('no data')
            return
        log['Date0'], log['DateEnd'] = timzone_view(t_lims, log_dt_from_utc)
    # dfLog = pd.DataFrame.from_dict(log, np.dtype(np.unicode_, cfg_out['logfield_fileName_len']))
    if not isinstance(log, pd.DataFrame):
        try:
            log = pd.DataFrame(log).set_index('Date0')
        except ValueError as e:  # , Exception
            log = pd.DataFrame.from_records(
                log, exclude=['Date0'],
                index=log['Date0'] if isinstance(log['Date0'], pd.DatetimeIndex) else [log['Date0']]
                )  # index='Date0' not work for dict

    try:
        return df_log_append_fun(log, table_log, cfg_out)
    except ValueError as e:
        return h5append_on_inconsistent_index(cfg_out, table_log, log, df_log_append_fun, e, 'append log')
    except ClosedFileError as e:
        lf.warning('Check code: On reopen store update store variable')


def h5_append(cfg_out: Mapping[str, Any],
              df: Union[pd.DataFrame, dd.DataFrame],
              log: MutableMapping[str, Any],
              log_dt_from_utc=pd.Timedelta(0),
              tim: Optional[pd.DatetimeIndex] = None):
    """
    Append dataframe to Store:
     - df to cfg_out['table'] ``table`` node of opened cfg_out['db'] store and
     - child table with 1 row - metadata including 'index' and 'DateEnd' (which is calculated as first and last elements
     of df.index)

    :param df: pandas or dask dataframe to append. If dask then log_dt_from_utc must be None (not assign log metadata here)
    :param log: dict which will be appended to child tables (having name of cfg_out['tables_log'] value)
    :param cfg_out: dict with fields:
        db: opened hdf5 store in write mode
        table: name of table to update (or tables: list, then used only 1st element). if not none tables[0] is ignored
        table_log: name of child table (or tables_log: list, then used only 1st element). if not none tables_log[0] is ignored
        tables: None - to return with done nothing!
                list of str - to assign cfg_out['table'] = cfg_out['tables'][0]
        tables_log: list of str - to assign cfg_out['table_log'] = cfg_out['tables_log'][0]
        b_insert_separator: (optional), freq (optional)
        data_columns: optional, list of column names to write.
        chunksize: may be None but then must be chunksize_percent to calcW ake Up:
            chunksize = len(df) * chunksize_percent / 100
    :param log_dt_from_utc: 0 or pd.Timedelta - to correct start and end time: index and DateEnd.
        Note: if log_dt_from_utc is None then start and end time: 'Date0' and 'DateEnd' fields of log must be filled right already
    :param tim: df time index
    :return: None
    :updates:
        log:
            'Date0' and 'DateEnd'
        cfg_out: only if not defined already:
            cfg_out['table_log'] = cfg_out['tables_log'][0]
            table_log
            tables_written set addition (or creation) with tuple `(table, table_log)`
    """
    table = None
    df_len = len(df.index) if tim is None else len(tim)  # use computed values if possible for faster dask
    if df_len:  # dask.dataframe.empty is not implemented
        if cfg_out.get('b_insert_separator'):
            # Add separation row of NaN
            msg_func = f'{df_len}rows+1dummy'
            cfg_out.setdefault('fs')
            df = h5_append_dummy_row(df, cfg_out['fs'], tim)
            df_len += 1
        else:
            msg_func = f'{df_len}rows'

        # Save to store
        # check/set tables names
        if 'tables' in cfg_out:
            if cfg_out['tables'] is None:
                lf.info('selected({:s})... ', msg_func)
                return
            set_field_if_no(cfg_out, 'table', cfg_out['tables'][0])

        lf.info('h5_append({:s})... ', msg_func)
        set_field_if_no(cfg_out, 'nfiles', 1)

        if 'chunksize' in cfg_out and cfg_out['chunksize'] is None:
            if 'chunksize_percent' in cfg_out:  # based on first file
                cfg_out['chunksize'] = int(df_len * cfg_out['chunksize_percent'] / 1000) * 10
                if cfg_out['chunksize'] < 10000:
                    cfg_out['chunksize'] = 10000
            else:
                cfg_out['chunksize'] = 10000

                if df_len <= 10000 and isinstance(df, dd.DataFrame):
                    df = df.compute()  # dask not writes "all NaN" rows
        # Append data
        try:
            table = df_data_append_fun(df, cfg_out['table'], cfg_out)
        except ValueError as e:
            table = h5append_on_inconsistent_index(cfg_out, cfg_out['table'], df, df_data_append_fun, e, msg_func)
        except TypeError as e:  # (, AttributeError)?
            if isinstance(df, dd.DataFrame):
                last_nan_row = df.loc[df.index.compute()[-1]].compute()
                # df.compute().query("index >= Timestamp('{}')".format(df.index.compute()[-1].tz_convert(None))) ??? works
                # df.query("index > Timestamp('{}')".format(t_end.tz_convert(None)), meta) #df.query(f"index > {t_end}").compute()
                if all(last_nan_row.isna()):
                    lf.exception(f'{msg_func}: dask not writes separator? Repeating using pandas')
                    table = df_data_append_fun(last_nan_row, cfg_out['table'], cfg_out, min_itemsize={c: 1 for c in (
                        cfg_out['data_columns'] if cfg_out.get('data_columns', True) is not True else df.columns)})
                    # sometimes pandas/dask get bug (thinks int is a str?): When I add row of NaNs it tries to find ``min_itemsize`` and obtain NaN (for float too, why?) this lead to error
                else:
                    lf.exception(msg_func)
            else:
                lf.error('{:s}: Can not write to store. {:s}', msg_func, standard_error_info(e))
                raise e
        except Exception as e:
            lf.error('{:s}: Can not write to store. {:s}', msg_func, standard_error_info(e))
            raise e
    # Append log rows
    # run even if df is empty because may be writing the log is needed only
    table_log = h5add_log(log, cfg_out, tim, df, log_dt_from_utc)
    if table_log:
        _t = (table, table_log) if table else (table_log,)
    else:
        if table:
            _t = (table,)
        return
    if 'tables_written' in cfg_out:
        cfg_out['tables_written'].add(_t)
    else:
        cfg_out['tables_written'] = {_t}


def h5_append_to(dfs: Union[pd.DataFrame, dd.DataFrame],
                 tbl: str,
                 cfg_out: Mapping[str, Any],
                 log: Optional[Mapping[str, Any]] = None,
                 msg: Optional[str] = None
                 ):
    """
    Append data to opened cfg_out['db'] by h5_append() without modifying cfg_out['tables_written'] instead returning it
    """
    if cfg_out['db'] is None:
        return set()
    if dfs is not None:
        if msg:
            lf.info(msg)
        # try:  # tbl was removed by h5temp_open() if b_overwrite is True:
        #     if h5remove(cfg_out['db'], tbl):
        #         lf.info('Writing to new table {}/{}', Path(cfg_out['db'].filename).name, tbl)
        # except Exception as e:  # no such table?
        #     pass
        cfg_out_mod = {**cfg_out, 'table': tbl, 'table_log': f'{tbl}/logFiles', 'tables_written': set()}
        try:
            del cfg_out_mod['tables']
        except KeyError:
            pass
        h5_append(cfg_out_mod, dfs, {} if log is None else log)
        # dfs_all.to_hdf(cfg_out['db_path'], tbl, append=True, format='table', compute=True)
        return cfg_out_mod['tables_written']
    else:
        print('No data.', end=' ')
        return set()
