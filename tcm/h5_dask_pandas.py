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
from .utils2init import Ex_nothing_done, standard_error_info, dir_create_if_need, LoggingStyleAdapter

pd.set_option('io.hdf.default_format', 'table')

lf = LoggingStyleAdapter(logging.getLogger(__name__))

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
        # ?! This is the only option in dask to load sorted index (can not use opened db handle)
        ddpart = dd.read_hdf(
            db_path_esc,
            table,
            chunksize=chunksize,
            lock=True,
            mode="r",
            columns=columns,
            sorted_index=sorted_index,
        )
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
     - keys equal to column names to filter or regex strings to select columns: "*" or "[" must be present to detect
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
                d[keys][f_compare(d[keys], v)] = np.nan
                key = ', '.join(keys)  # for logging only
            else:
                try:
                    d[key][f_compare(d[key], v)] = np.nan
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
        suffix='',
        single_file_name=True,
        progress=None, client=None,
        b_continue=False
        ):
    """
    Save to ascii if _text_path_ is not None
    :param d: dask dataframe
    :param text_path: None or directory path. If not a dir tries to create and if this fails (such as when if
    more than one level) then add this as prefix to names
    :param b_continue: append to the end of existed text file
    :param text_date_format: If callable then create "Date" column by calling it (dd.index), retain index only
    if "Time" in text_columns. If string use it as format for index (Time) column
    :param text_columns: optional
    :param suffix: str, will be added to filename with forbidden characters removed/replaced
    :param single_file_name:
    - True or str: save all to one file of this name if str or to autogenerated name if True
    - False: generate name for each `d` partition individually and save each to separate file
    :param progress: progress bar object, used with client
    :param client: dask.client, used with progress
    """
    if text_path is None:
        return

    tab = '\t'
    sep = tab
    ext = '.tsv' if sep == tab else '.csv'

    try:
        dir_create_if_need(text_path)

        def comb_path(dir_or_prefix, s):
            return str(dir_or_prefix / s)
    except:
        lf.exception('Dir not created!')

        def comb_path(dir_or_prefix, s):
            return f'{dir_or_prefix}{s}'

    def name_that_replaces_asterisk(i_partition):
        return f'{d.divisions[i_partition]:%y%m%d_%H%M}'
    # too long variant: '{:%y%m%d_%H%M}-{:%H%M}'.format(*d.partitions[i_partition].index.compute()[[0,-1]])

    suffix_mod = re.sub(r'[\\/*?:"<>\.]', '', suffix.replace('|', ',').replace('{}', 'all'))
    filename = Path(single_file_name if isinstance(single_file_name, str) else (
        comb_path(
            text_path,
            f"{name_that_replaces_asterisk(0) if single_file_name else '*'}{suffix_mod}{ext}")
    ))
    lf.info(
        "Saving{} *{:s}: {}",
        " (continue)" if b_continue else "",
        ext,
        Path(*filename.parts[-2:]) if b_continue else
        "1 file" if single_file_name else
        f"{d.npartitions} files",
    )

    d_out = d.round({'Vdir': 4, 'inclination': 4, 'Pressure': 3})
    # if not cfg_out.get('b_all_to_one_col'):
    #     d_out.rename(columns=map_to_suffixed(d.columns, suffix))
    if callable(text_date_format):
        arg_out = {
            "index": bool(text_columns) and "Time" in text_columns,
            "columns": bool(text_columns) or d_out.columns.insert(0, "Date"),
        }
        d_out['Date'] = d_out.map_partitions(lambda df: text_date_format(df.index))
    else:
        if text_date_format in ('s', '%Y-%m-%d %H:%M:%S'):                   # speedup
            d_out.index = d_out.index.dt.tz_convert(None).dt.ceil(freq='s')  # very speedups!
            arg_out = {'columns': text_columns or None}  # for write all columns if empty (replaces to None)
        else:
            arg_out = {
                "date_format": text_date_format,  # lead to very long saving (tenths howers) for 2s and smaller resolution data!
                "columns": text_columns or None,  # for write all columns if empty (replaces to None)
            }

    # Common pandas and dask to_csv() args
    args_to_csv = {
        'float_format': '%.5g',
        'sep': sep,
        'encoding': 'ascii',
        # 'compression': 'zip',
        **arg_out
    }
    if b_continue and single_file_name:
        # Append
        d_out = d_out.compute()  # using pandas because can not append using dask
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
