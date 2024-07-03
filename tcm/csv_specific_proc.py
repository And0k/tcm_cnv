"""
different format converters mainly for csv2h5
"""

import re
from datetime import datetime, timezone
from typing import (Any, AnyStr, Callable, Dict, Iterable, Mapping, Match, Optional, Union, Sequence, TypeVar,
                    BinaryIO, TextIO, Tuple)
import io
import numpy as np
import pandas as pd

# if __debug__:  'commented because functions in this file used by dask in separate threads so start plot here is problematic
#     from matplotlib import pyplot as plt
from pathlib import Path, PurePath
from .utils2init import set_field_if_no, FakeContextIfOpen, standard_error_info, dir_create_if_need, my_logging
from functools import partial, wraps

from .utils_time import matlab2datetime64ns, date_from_filename
from .utils_time_corr import plot_bad_time_in_thread

lf = my_logging(__name__)
century = b'20'


def chars_array_to_datetimeindex(date: Union[np.ndarray, pd.Series],
                                 dtype: np.dtype,
                                 format: str = '%Y-%m-%dT%H:%M:%S') -> pd.DatetimeIndex:
    """
    Error corrected conversion: replace bad data with previous (next)
    :param date: Numpy array of strings
    :param dtype: destination Numpy type
    :return: pandas DatetimeIndex of destination type

    Bug in supervisor convertor corrupts dates: gets in local time zone.
    Ad hoc: for daily records we use 1st date. But can not use if 1st date is wrong,
    so we use full correct + wrong time at start (day jump down)
    + up and down must be alternating

    """
    try:
        is_series = isinstance(date, pd.Series)
        if isinstance(date.iat[0] if is_series else date[0], bytes):
            # convert 'bytes' to 'strings' needed for pd.to_datetime()
            date = (
                date.str.decode if is_series else (
                    lambda **kwargs: np.char.decode(date, **kwargs)
                    ))(encoding='utf-8', errors='replace')
    except Exception as e:
        pass

    try:
        date = date.astype(dtype)
    except TypeError as e:
        print(f'Date strings converting to {dtype} error: ', standard_error_info(e))
    except ValueError as e:
        print('Bad date: ', standard_error_info(e))

    try:
        date = pd.to_datetime(date, format=format, errors='coerce')  # new method
        t_is_bad = date.isna()
        t_is_bad_sum = t_is_bad.sum()
        if t_is_bad_sum:
            lf.warning('replacing {:d} bad strings with previous', t_is_bad_sum)
            date.ffill(inplace=True)
            # 'nearest' interpolating date is not works, so
            # s2 = date.astype('i8').astype('f8')
            # s2[t_is_bad] = np.nan
            # s2.interpolate(method='nearest', inplace=True)
            # date[t_is_bad] = pd.to_datetime(s2[t_is_bad])
        if date.dtype != dtype:
            date = date.astype(dtype)
    except Exception as e:
        b_new_method = False
        print('to_datetime not works', standard_error_info(e))
        raise e
    return pd.DatetimeIndex(date)

# def old_method():
#     # old method of bad date handling
#     ...
#     try:
#         ibad = np.flatnonzero(date == bytes(re.search(r'(?:")(.*)(?:")', e.args[0]).group(1), 'ascii'))
#         print("- replacing element{} #{} by its neighbor".format(
#             *('s', ibad) if ibad.size > 1 else ('', ibad[0])), end=' ')
#
#         # use pd.replace to correct bad strings?
#     except ValueError as e:
#         print('Can not find bad date caused error: ', standard_error_info(e))
#         raise
#     except TypeError as e:
#         print('Can not find bad date caused error: ', standard_error_info(e))
#         raise (e)
#     date.iloc[ibad] = date.iloc[np.where(ibad > 0, ibad - 1, ibad + 1)].asobject
#     print("({})".format(date.iloc[ibad]))


def fill0(arr: np.ndarray, width: int) -> np.ndarray:
    """ Fills right with byte char b'0' """
    return np.char.rjust(arr.astype('|S%d' % width), width, fillchar=b'0')  # dd.from_array()
    # return arr.str.decode('utf-8').str.pad(width=2,fillchar='0')


RT = TypeVar('RT')  # return type


def meta_out(attribute: Any) -> Callable[[Callable[..., RT]], Callable[..., RT]]:
    """
    Decorator that adds attribute ``meta_out`` to decorated function.

    In this file it is used to return partial of out_fields(),
    It is used in this module to get ``metadata`` parameter for dask functions that returns dask dataframe.
    :param attribute: any, will be assigned to ``meta_out`` attribute of decorated function.
    :return:
    """
    @wraps(attribute)
    def meta_out_dec(fun: Callable[..., RT]) -> Callable[..., RT]:
        """
        Decorator that depending on ``attribute(fun())`` returns needed columns from decorated function.
        :param fun:
        :return:
        """
        @wraps(fun)
        def fun_with_meta_out(*args, **kwargs):

            out = fun(*args, **kwargs)

            # output columns = attribute((cfg_in or out) dtype)
            try:  # If decorated function has 2nd or named parameter ``cfg_in`` dict, and it has ``dtype``:
                _ = (args[1] if len(args) > 1 else kwargs['cfg_in'])['dtype']
                # then get list of columns to return only them from dataframe:
                cols_out = attribute([(k, v[0]) for k, v in _.fields.items()]).keys()
            except KeyError:
                cols_out = attribute(out.dtypes.iteritems()).keys()
                if list(cols_out) == out.columns.to_list():
                    return out
            return out.loc[:, cols_out]

        fun_with_meta_out.meta_out = attribute
        return fun_with_meta_out

    return meta_out_dec


def out_fields(
        types: Mapping[str, type], keys_del: Optional[Iterable[str]] = (),
        add_before: Optional[Mapping[str, type]] = None,
        add_after: Optional[Mapping[str, type]] = None) -> Dict[str, type]:
    """
    Removes from ``types`` fields with keys that in ``keys_del`` and adds fields ``add*``
    :param types:
    :param keys_del:
    :param add_before:
    :param add_after:
    :return: dict of {field: dtype} metadata
    """
    if add_before is None:
        add_before = {}
    if add_after is None:
        add_after = {}
    return {**add_before, **{k: v for k, v in types if k not in keys_del}, **add_after}


def meta_out_fields(**known):
    # meta_out(partial(out_fields, **known)))
    def out_fields_not_known(*args, **kwargs):
        return out_fields(*args, **kwargs, **known)

    return meta_out(out_fields_not_known)


# ----------------------------------------------------------------------
def log_csv_specific_param_operation(
        key_logged: str,
        functions_str: Sequence[str],
        cfg_in) -> None:
    """
    Shows info message of csv_specific_param operations.
    Sets cfg_in['csv_specific_param_logged' + key_logged] to not repeat message on repeating call
    :param key_logged: key of csv_specific_param triggering specific calculations when func will be called and this
    message about
    :param functions_str: string representing operations applied
    :param cfg_in: dict to set/check field
    :return:
    """

    key_logged_full = f'csv_specific_param_logged{"-" if key_logged else ""}{key_logged}'
    if not cfg_in.get(key_logged_full):  # log 1 time i.e. in only one 1 dask partition
        cfg_in[key_logged_full] = True
        lf.info(f'csv_specific_param {list(functions_str)} modifications applied')  # todo: add time or block


def param_funs_closure(
        csv_specific_param: Mapping[str, Union[Callable[[str], Any], float]],
        cfg_in: Mapping[str, Any]) -> Mapping[str, Callable[[str], Any]]:
    """
    Used by loaded_corr(). Converts dict `csv_specific_param` to new dict by removing key suffixes and
    replace each value with function of one Mapping like variable
    :param csv_specific_param:
    :param cfg_in: used to keep temporary variables between multiple calls in log_csv_specific_param_operation
    :return: dict of functions, having arguments of one Mapping like variable
    """
    params_funs = {}
    # def fun(param, fun_id, v):
    #     param_closure = param
    #     if fun_id == 'fun':
    #         def fun_closure(x):
    #             return v(x[param_closure])
    #     elif fun_id == 'add':
    #         def fun_closure(x):
    #             return x[param_closure] + v
    #         # or a.eval(f"{param} = {param} + {v}", inplace=True)
    #     else:
    #         raise KeyError(f'Error in csv_specific_param: {k}: {v}')
    #     return fun_closure

    for k, fun_or_const in csv_specific_param.items():
        param, fun_id = k.rsplit('_', 1)
        if fun_id == 'fun':

            def fun(prm, fun):
                param_closure = prm
                v_closure = fun

                params_closure = fun.__code__.co_varnames
                if len(params_closure) <= 1:

                    def fun_closure(x):
                        return v_closure(x[param_closure])
                else:
                    params_closure = list(params_closure)

                    def fun_closure(x):
                        return v_closure(*x[params_closure].T.values)

                return fun_closure

        elif fun_id == 'add':

            def fun(prm, const):
                param_closure = prm
                v_closure = const

                def fun_closure(x):
                    return x[param_closure] + v_closure

                # or a.eval(f"{prm} = {prm} + {const}", inplace=True)
                return fun_closure

        else:
            # raise KeyError(f'Error in csv_specific_param: {k}: {fun_or_const}')
            continue
        params_funs[param] = fun(param, fun_or_const)  # f"{param} = {fun_or_const}({param})
    if params_funs:
        log_csv_specific_param_operation('', params_funs.keys(), cfg_in)
    return params_funs


def loaded_corr(
        a: Union[pd.DataFrame, np.ndarray],
        cfg_in: Mapping[str, Any],
        csv_specific_param: Optional[Mapping[str, Any]] = None
        ) -> pd.DataFrame:
    """
    Specified prep&proc of data:

    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :param csv_specific_param: {param_suffix: fun_expr} where ``suffix`` in ``param_suffix`` key can be 'fun' or 'add':
    - 'fun': fun_expr specifies function assign to ``param``
    - 'add': fun_expr specifies value to add to ``param`` to modify it

    Note: fun code must have args and outputs that are column names of loaded DataFrame

    :return: pandas.DataFrame
    """
    if csv_specific_param is not None:
        params_funs = param_funs_closure(csv_specific_param, cfg_in)
        if params_funs:
            return a.assign(**params_funs)
    return a  # .copy() need?


# Specific format loaders ---------------------------------------------

def loaded_Idronaut(a: Union[pd.DataFrame, np.ndarray],
                         cfg_in: Optional[Mapping[str, Any]] = None) -> pd.DatetimeIndex:
    """
    # Idronaut specified proc: convert Time column from data like "14-07-2019 07:57:32.28" to pandas DateTime
    :param a:
    :param cfg_in:
    :return: ISO time string in current zone
    """
    # convert to ISO8601 date strings
    date = a['date'].to_numpy(dtype='|S10').view([('DD', 'S2'), ('-MM-', 'S4'), ('YYYY', 'S4')])
    date = np.column_stack((date['YYYY'], date['-MM-'], date['DD'])).view([('date', 'S10'), ('junk', 'S2')])['date']
    date = np.append(date, np.broadcast_to(b'T', date.shape), axis=1).view([('date', 'S11'), ('junk', 'S9')])['date']
    date = np.append(date, a['txtT'].to_numpy(dtype='|S11')[:, np.newaxis], axis=1).view('|S22').ravel()
    # np.array(date, 'datetime64[ms]')
    return chars_array_to_datetimeindex(date, 'datetime64[ns]', format='%Y-%m-%dT%H:%M:%S.%f')


@meta_out_fields(keys_del={'Date', 'Time'}, add_before={'Time': 'M8[ns]'})
def loaded_Idronaut_terminal(
        a: Union[pd.DataFrame, np.ndarray],
        cfg_in: Optional[Mapping[str, Any]] = None,
        # cols_out: Union[slice, Sequence[Any]] = slice(None)
        ) -> pd.DatetimeIndex:
    """
    Not implemented
    :param a:
    :param cfg_in:
    :return:
    """
    # date['YY'], date['MM'], date['DD'] = [cfg_in['file_stem'][(slice(k, k + 2))] for k in range(0, 5, 2)]
    # date = np.array(century + date['YY'].astype(np.object) + b'-' + date['MM'].astype(np.object) + b'-' +
    #                 date['DD'].astype(np.object) + b'T' + txtT['hh'].astype(np.object) + b':' +
    #                 txtT['mm'].astype(np.object) + b':' + txtT['ss'].astype(np.object), '|S19', ndmin=1)
    # date = chars_array_to_datetimeindex(date, 'datetime64[ns]')  # convert ISO8601 date strings
    # date = day_jumps_correction(cfg_in, date)

    # extract date from file name
    if not cfg_in.get('fun_date_from_filename'):
        cfg_in['fun_date_from_filename'] = date_from_filename
    str_date = cfg_in['fun_date_from_filename'](cfg_in['file_stem'], century.decode())
    date = pd.to_datetime(str_date, format='%Y-%m-%d')
    tim = pd.to_datetime(a['Time'].str.decode('utf-8', errors='replace'), format='%H:%M:%S.%f') - pd.Timestamp('1900-01-01')  # default date '1900-01-01' found experimentally!
    date += tim
    a['Time'] = pd.NaT  # also changes column's type
    a.loc[:, 'Time'] = day_jumps_correction(cfg_in, date.to_numpy())


    #
    # t = pd.to_datetime(str_date) + \
    #     pd.to_timedelta(a['Time'].str.decode('utf-8', errors='replace'))
    # # t = day_jumps_correction(cfg_in, t.values)
    return a


@meta_out_fields(keys_del={'Date', 'Time'}, add_before={'Time': 'M8[ns]'})
def loaded_sbe(
        a: Union[pd.DataFrame, np.ndarray], cfg_in: Mapping[str, Any],
        csv_specific_param: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """
    Specified prep&proc of data :
    - Time calc: gets string for time (mm/dd/yyyy	hh:mm:ss) in current zone

    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :param csv_specific_param: {param_suffix: fun_expr} where ``suffix`` in param_suffix string can be 'fun' or 'add':
    - 'fun': fun_expr specifies function assign to ``param``
    - 'add': fun_expr specifies value to add to ``param`` to modify it
    :return: pandas.DataFrame

    """
    date = pd.to_datetime(a['Date'], format='%m/%d/%Y') + pd.to_timedelta(a['Time'])  #, unit='ms'
    return a.assign(Time=date)


@meta_out_fields(keys_del={'Date', 'Time'}, add_before={'Time': 'M8[ns]'})
def loaded_sst(
        a: Union[pd.DataFrame, np.ndarray], cfg_in: Mapping[str, Any],
        csv_specific_param: Optional[Mapping[str, Any]] = None,
        ) -> pd.DataFrame:
    """
    Specified prep&proc of data :
    - Time calc: gets string for time in current zone

    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :param csv_specific_param: {param_suffix: fun_expr} where ``suffix`` in param_suffix string can be 'fun' or 'add':
    - 'fun': fun_expr specifies function assign to ``param``
    - 'add': fun_expr specifies value to add to ``param`` to modify it
    :return: pandas.DataFrame
    """

    date = (  # .str.decode('utf-8', errors='replace')
        pd.to_datetime(a['Date'], format='%d.%m.%Y') + pd.to_timedelta(a['Time'])  #, unit='ms'
    )
    # check that used
    return a.assign(Time=date)

@meta_out_fields(keys_del={'Time', 'Lat', 'Lon'}, add_before={'Time': 'M8[ns]'}, add_after={'Lat': 'f8', 'Lon': 'f8'})
def loaded_sst_mws_with_coord(
        a: Union[pd.DataFrame, np.ndarray], cfg_in: Mapping[str, Any],
        csv_specific_param: Optional[Mapping[str, Any]] = None,
        # cols_out: Union[slice, Sequence[Any]] = slice(None)
) -> pd.DataFrame:
    """

    :param a: Lat: 60°02'52.11'' N	Lon: 26°46'59.67'' E
    :param cfg_in:
    :param csv_specific_param:
    :return:
    """

    def deg_min_sec_text_to_deg(dms):
        """
        :param degmin: 26°46'59.67'' E
        """
        dms = dms.str.split("['° ]", n=3, expand=True, regex=True).iloc[:, :3].astype(float)
        return dms.iloc[:, 0] + dms.iloc[:, 1]/60 + dms.iloc[:, 2]/3600

    a.Lat = deg_min_sec_text_to_deg(a.Lat)
    a.Lon = deg_min_sec_text_to_deg(a.Lon)
    a.Time = chars_array_to_datetimeindex(a.Time, 'datetime64[ns]')
    return a


@meta_out_fields(keys_del={'Time'}, add_before={'Time': 'M8[ns]'})
def loaded_sst_mws(
        a: Union[pd.DataFrame, np.ndarray], cfg_in: Mapping[str, Any],
        csv_specific_param: Optional[Mapping[str, Any]] = None,
        # cols_out: Union[slice, Sequence[Any]] = slice(None)
        ) -> pd.DataFrame:
    """
    Specified prep&proc of data :
    - Time calc: gets string for time in current zone


    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :param csv_specific_param: {param_suffix: fun_expr} where ``suffix`` in param_suffix string can be 'fun' or 'add':
    - 'fun': fun_expr specifies function assign to ``param``
    - 'add': fun_expr specifies value to add to ``param`` to modify it
    :return: pandas.DataFrame
    """

    # extract date from file name
    if not cfg_in.get('fun_date_from_filename'):
        with cfg_in['file_cur'].open(encoding='utf-8', errors='replace') as f:
            line = f.readline()
        str_date = line[len('Data Start: '):len('Data Start: 2022-11-06')]
    else:
        str_date = cfg_in['fun_date_from_filename'](cfg_in['file_cur'], century.decode())

    try:
        date = pd.to_datetime(str_date, format='%Y-%m-%d')
    except ValueError as e:
        lf.warning('Not found correct "%Y-%m-%d" date in line: {:s}', line)
        date = pd.to_datetime(input('correct date in "%Y-%m-%d" format'))

    if not isinstance(a['Time'][0], bytes):
        lf.warning('Second pass?')
        return a
    else:
        tim = pd.to_datetime(a['Time'].str.decode('utf-8', errors='replace'), format='%H:%M:%S') - pd.Timestamp('1900-01-01')  # default date '1900-01-01' found experimentally!

    date += tim
    a['Time'] = pd.NaT  # also changes column's type
    a.loc[:, 'Time'] = day_jumps_correction(cfg_in, date.to_numpy())
    return a

    # # check that used
    # return a.assign(Time=date).loc[:, list(
    #     attribute(cfg_in['dtype_out']).keys())]


def loaded_sst_old(a: pd.DataFrame, cfg_in: Optional[Mapping[str, Any]] = None) -> pd.Series:
    """
    SST specified proc
    :param a:
    :param cfg_in:
    :return: Time
    """

    if False:  # shifts not works now!
        # Time calc: gets string for time in current zone
        date = np.array(a['Date'].values if isinstance(a, pd.DataFrame) else (a['Date']).astype(object),
                        dtype={'DD': ('a2', 0), 'MM': ('a2', 3), 'YYYY': ('a4', 6)})
        # dtype={'DD': ('a2', 1), 'MM': ('a2', 4), 'YYYY': ('a4', 7)})
        date = (date['YYYY'].astype(np.object) + b'-' +
                date['MM'].astype(np.object) + b'-' +
                date['DD'].astype(np.object) + b'T' +
                fill0(a['Time'], 12)).ravel()
    else:
        date = pd.to_datetime(a['Date'].str.decode('utf-8', errors='replace'), format='%d.%m.%Y') + \
               pd.to_timedelta(a['Time'].str.decode('utf-8', errors='replace'), unit='ms')
    return date


# cfg['in']['converters']= lambda cfg_in:\
#     {cfg_in['coldate']: lambda D_M_Y:   #very slow
#         np.datetime64('{2}-{1}-{0}'.format(*D_M_Y.decode().strip().split('.')), 'D'),
#      cfg_in['coltime']: lambda hh_mm_s: #very slow
#         np.sum(np.float64(hh_mm_s.split(b':')) * np.float64([3600, 60, 1]))
#     }

@meta_out_fields(keys_del={'Date', 'Time'}, add_before={'Time': 'M8[ns]'})
def loaded_rock(
        a: Union[pd.DataFrame, np.ndarray], cfg_in: Mapping[str, Any],
        csv_specific_param: Optional[Mapping[str, Any]] = None
    ) -> pd.DataFrame:
    """
    Specified prep&proc of ROCK CTD data :
    ROCK_CTD, S/N 2301
    Pres Temp Cond Date Time
    -0.0013 3.8218 0.1531 2023-11-21 11:53:22

    - Time calc: gets string for time in current zone

    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :param csv_specific_param: {param_suffix: fun_expr} where ``suffix`` in param_suffix string can be 'fun' or 'add':
    - 'fun': fun_expr specifies function assign to ``param``
    - 'add': fun_expr specifies value to add to ``param`` to modify it
    :return: pandas.DataFrame
    """

    date = (
        pd.to_datetime(a['Date'].str.decode('utf-8', errors='replace'), format='%Y-%m-%d') +
        pd.to_timedelta(a['Time'].str.decode('utf-8', errors='replace'))  # , unit='ms'
    )
    # check that used
    return a.assign(Time=date)



# ----------------------------------------------------------------------
def loaded_csv_log(a: Union[pd.DataFrame, np.ndarray], cfg_in: Optional[Mapping[str, Any]] = None) -> np.ndarray:
    # Log format specific proc
    # Time calc: gets string for time in current zone
    # "07.04.2017 11:27:00" -> YYYY-MM-DDThh:mm:ss

    date = np.array(a['Time'].astype(object), dtype={'DD': ('a2', 0),
                                                     'MM': ('a2', 3),
                                                     'YYYY': ('a4', 6),
                                                     'Time': ('a8', 11)
                                                     })
    date = (date['YYYY'].astype(np.object) + b'-'
            + date['MM'].astype(np.object) + b'-'
            + date['DD'].astype(np.object)
            + b'T' + date['Time'].astype(np.object)).ravel()

    return date


# ----------------------------------------------------------------------
def view_fields(a: Union[pd.DataFrame, np.ndarray], fields) -> Union[pd.DataFrame, np.ndarray]:
    """
    `a` must be a numpy structured array.
    `names` is the collection of field names to keep.

    Returns a view of the array `a` (not a copy).
    """
    dt = a.dtype
    formats = [dt.fields[name][0] for name in fields]
    offsets = [dt.fields[name][1] for name in fields]
    itemsize = a.dtype.itemsize
    newdt = np.dtype(dict(names=fields,
                          formats=formats,
                          offsets=offsets,
                          itemsize=itemsize))
    b = a.view(newdt)
    return b


@meta_out_fields(keys_del={'Time'}, add_before={'Time': 'M8[ns]'}, add_after={'N^2': 'f8'})
def loaded_Baklan(
        a: Union[pd.DataFrame, np.ndarray], cfg_in: Optional[Mapping[str, Any]] = None,
        # cols_out: Union[slice, Sequence[Any]] = slice(None)
        ) -> pd.DataFrame:
    # Baklan Matlab processed data specified proc
    # Time calc: gets string for time in current zone
    # "7/25/2015"	"2:14:36" -> YYYY-MM-DDThh:mm:ss

    set_field_if_no(cfg_in, 'decimate_rate', 1)
    set_field_if_no(cfg_in, 'lat', 55.3)  # ~ Slupsk furrow
    set_field_if_no(cfg_in, 'lon', 16.3)  # ~ Slupsk furrow

    out_len = np.ceil(a.shape[0] / cfg_in['decimate_rate']).astype(
        np.int64)  # , dtype=np.uint64, casting='unsafe' - not works. np.floor_divide(a.shape[0], cfg_in['decimate_rate']) + 1 is too small if no reminder

    # # adding field cannot affect a on exit function
    # a_last_field = a.dtype.fields[a.dtype.names[-1]]
    # out_dtype = dict(a.dtype.fields)
    # out_dtype['N^2']= (np.float64, a.itemsize)

    date = matlab2datetime64ns(a['Time'][::cfg_in['decimate_rate']])
    out = np.ndarray(out_len, cfg_in['dtype_out'])

    from scipy import signal
    from gsw.conversions import t90_from_t68, CT_from_t
    from gsw import SA_from_SP, Nsquared

    out_names_need_decimate = set(cfg_in['dtype_out'].names).difference(['Time', 'N^2'])
    # w: set(a.dtype.names); names_not_time.remove('Time')

    names_in = set(a.dtype.names if isinstance(a, np.ndarray) else a.columns)

    for name_out in out_names_need_decimate:
        if name_out in names_in:
            series = a[name_out]
        elif name_out == 'GsumMinus1':
            # calculate GsumMinus1
            dtype_Gx = cfg_in['dtype']['Gx_m\s^2']
            dtype_Gxyz = np.dtype({'names': ['Gx_m\s^2', 'Gy_m\s^2', 'Gz_m\s^2'], 'formats': 3 * (dtype_Gx,)})

            series = np.ndarray((len(a), 3), cfg_in['dtype']['Gx_m\s^2'], a,
                                offset=cfg_in['dtype'].fields['Gx_m\s^2'][1],
                                strides=(a.strides[0], dtype_Gx.itemsize)
                                ) if isinstance(a, np.ndarray) else a.loc[:, dtype_Gxyz.names].to_numpy(dtype=dtype_Gx)
            # a.astype(dtype_Gxyz).view(dtype_Gx).reshape(-1, 3) #[['Gx_m\s^2', 'Gy_m\s^2', 'Gz_m\s^2']]

            series = np.sqrt(
                np.sum(np.square(series), 1)) / 9.8 - 1  # a[['Gx_m\s^2', 'Gy_m\s^2', 'Gz_m\s^2']]/9.8) is not supported
        else:
            continue

        out[name_out] = signal.decimate(series, cfg_in['decimate_rate'], ftype='fir',
                                        zero_phase=True)  # 'fir' becouse of "When using IIR
        # downsampling, it is recommended to call decimate multiple times for downsampling factors higher than 13"

    Temp90 = t90_from_t68(out['Temp'])
    # out['Sal'] = SP_from_C(out['Cond'], out['Temp'], out['Pres'])  # recalc here?
    SA = SA_from_SP(out['Sal'], out['Pres'], cfg_in['lon'], cfg_in['lat'])  # Absolute Salinity  [g/kg]

    conservative_temperature = CT_from_t(SA, Temp90, out['Pres'])
    N2 = np.append(Nsquared(SA, conservative_temperature, out['Pres'], cfg_in['lat'])[0], np.nan)

    # np.append(Nsquared(SA, conservative_temperature, out['Pres'], cfg_in['lat'])[1], np.nan) - out['Pres'] is small
    return out.assign(**{'Time': date, 'N^2': N2})

    # works before change shape of a in caller only:
    # a.resize(out.shape, refcheck=False)
    # a[:] = out  # np.ndarray(out.shape, a.dtype, out, 0, out.strides)  # cfg_in['a'] = out  #

    # import numpy.lib.recfunctions as rfn
    # data = rfn.append_fields(data, ['x', 'y'], [x, y])
    #


def loaded_ctd_Schuka(a: Union[pd.DataFrame, np.ndarray],
                           cfg_in: Optional[Mapping[str, Any]] = None) -> np.ndarray:
    # Schuka data specified proc
    # Time calc: gets string for time in current zone
    # "7/25/2015"	"2:14:36" -> YYYY-MM-DDThh:mm:ss

    def for_remEnds(ls):
        return ([s[1:-1] for s in ls])

    def for_align_right_by_add_0_to_left(ls, min_len_where):
        return ([(b'0' + s if len(s) <= min_len_where else s) for s in ls])
        # if len(s)==10 else '0'+s[1:-1]

    txtT = np.array(for_align_right_by_add_0_to_left(for_remEnds(a['Time']), 7))
    date = np.array(for_align_right_by_add_0_to_left(for_remEnds(a['Date']), 9))
    date = np.array(date.astype(object), dtype={'MM': ('a2', 0),
                                                'DD': ('a2', 3),
                                                'YYYY': ('a4', 6)})
    date = (date['YYYY'].astype(np.object) + b'-'
            + date['MM'].astype(np.object) + b'-'
            + date['DD'].astype(np.object)
            + b'T' + txtT.astype(np.object)).ravel()

    return date
    # np.array(a['Time'].astype(object), dtype={'hh': ('a2', 0), 'mm': ('a2', 3), 'ss': ('a2', 6)})
    # date= (np.char.replace(np.array(date),b'/',b'-').astype(np.object)


# ----------------------------------------------------------------------
def loaded_ctd_Schuka_HHMM(a: Union[pd.DataFrame, np.ndarray],
                                cfg_in: Optional[Mapping[str, Any]] = None) -> np.ndarray:
    # Schuka data #2 specified proc
    # Time calc: gets string for time in current zone
    # "08.06.2015"	"5:50" -> YYYY-MM-DDThh:mm:ss

    def for_remEnds(ls):
        return ([s[1:-1] for s in ls])

    def for_align_right_by_add_0_to_left(ls, min_len_where):
        return ([(b'0' + s if len(s) <= min_len_where else s) for s in ls])
        # if len(s)==10 else '0'+s[1:-1]

    txtT = np.array(for_align_right_by_add_0_to_left(for_remEnds(a['Time']), 4))
    date = np.array(for_remEnds(a['Date']))
    date = np.array(date.astype(object), dtype={'DD': ('a2', 0),
                                                'MM': ('a2', 3),
                                                'YYYY': ('a4', 6)})
    date = (date['YYYY'].astype(np.object) + b'-'
            + date['MM'].astype(np.object) + b'-'
            + date['DD'].astype(np.object)
            + b'T' + txtT.astype(np.object) + b':00').ravel()

    return date


def day_jumps_correction(cfg_in: Mapping[str, Any], t: Union[np.ndarray, pd.DatetimeIndex]):
    """
    Correction of day jumps in time. Source invalid time might be result as sum of day and time part obtained from different not synchronised sourses
    :param cfg_in: dict with field 'time_last' (optional) to check if first time values are need to be corrected
    :param t: time
    :return: corrected time
    creates field 'time_last' equal to t[0] if was not
    """
    dT_day_jump = np.timedelta64(1, 'D')

    set_field_if_no(cfg_in, 'time_last', t[0])
    dT = np.diff(np.insert(t if isinstance(t, np.ndarray) else t.to_numpy(), 0,
                           0 if cfg_in['time_last'] is pd.NaT else cfg_in[
                               'time_last']))  # .copy() ???, not np.ediff1d because it not allows change type from time to timedelta
    dT_resolution = max(np.timedelta64(1, 's'), np.median(dT)) * 10  # with permissible tolerance
    # correct day jump up
    fun_b_day_jump = lambda dT, dT_day_jump=dT_day_jump, dT_resolution=dT_resolution: \
        np.logical_and(dT_day_jump - dT_resolution < dT, dT <= dT_day_jump + dT_resolution)
    jumpU = np.flatnonzero(fun_b_day_jump(dT))
    # correct day jump down
    fun_b_day_jump = lambda dT, dT_day_jump=dT_day_jump, dT_resolution=dT_resolution: \
        np.logical_and(-dT_day_jump + dT_resolution > dT, dT >= -dT_day_jump - dT_resolution)
    jumpD = np.flatnonzero(fun_b_day_jump(dT))

    # if dT[0] < 0: #b_corU_from0 #softer crieria for detect jump at start?
    # jumpU.insert(txtDlast,0)  # need up time data at start

    lU = len(jumpU)
    lD = len(jumpD)
    if lU or lD:
        jumps = np.hstack((jumpU, jumpD))
        ijumps = np.argsort(jumps)
        jumps = np.append(jumps[ijumps], len(t))
        bjumpU = np.append(np.ones(lU, np.bool_), np.zeros(lD, np.bool_))[ijumps]
        t_orig = t
        # if __debug__:  # if run  under debugger
        #     plt.plot(t, color='r', alpha=0.5)  # ; plt.show()
        for bjU, jSt, jEn in zip(bjumpU[::2], jumps[:-1:2], jumps[1::2]):  # apply_day_shifting
            t_datetime = (
                datetime.fromtimestamp(t[jSt].astype(datetime) * 1e-9, timezone.utc) if isinstance(t, np.ndarray) else
                t[jSt]
            )
            if bjU:
                t[jSt:jEn] -= dT_day_jump
                print('Date correction to {:%d.%m.%y}UTC: day jumps up was '
                      'detected in [{}:{}] rows'.format(t_datetime, jSt, jEn))
            else:
                t[jSt:jEn] += dT_day_jump
                print('Date correction to {:%d.%m.%y}UTC: day jumps down was '
                      'detected in [{}:{}] rows'.format(t_datetime, jSt, jEn))
        # if __debug__:
        #     plt.plot(t)  # ; plt.show()
        plot_bad_time_in_thread(cfg_in, t, np.ones_like(t, np.bool_), None, t_orig,
                                None, 'day_jumps_corr', 'day jumps correction')

    return t

# ----------------------------------------------------------------------
#@meta_out_fields(keys_del={'yyyy', 'mm', 'dd', 'HH', 'MM', 'SS'}, add_before={'Time': 'M8[ns]'})) - not need if only date returns
def loaded_chain_Baranov(a: Union[pd.DataFrame, np.ndarray],
                              cfg_in: Optional[Mapping[str, Any]] = None,
                              csv_specific_param: Optional[Mapping[str, Any]] = None) -> pd.DatetimeIndex:
    """
    Specified prep&proc of data from program "Supervisor":
    - Time calc: gets time in current zone

    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :param csv_specific_param: {depth_add: str_val}
    :return: numpy 'datetime64[ns]' array

    Example input:
    a = {
    'yyyy': b"2017", 'mm': b"10", 'dd': b"14",
    'HH': b'59','MM': b'59', 'SS': b'59'}
    """

    # todo: use dd.to_datime(a[['year', 'month', 'day', 'hower', 'minute', 'second']], ) instead

    if csv_specific_param:
        lf.info('unknown key(s) in csv_specific_param')

    # Baranov format specified proc
    # Time calc: gets string for time in current zone
    date = np.add.reduce([a['yyyy'], b'-', a['mm'], b'-', a['dd'], b'T',
                          a['HH'], b':', a['MM'], b':', a['SS']])

    # np.array(a['yyyy'].astype(np.object) + b'-' + a['mm'].astype(np.object) + b'-' +
    #                 a['dd'].astype(np.object) + b'T' + a['HH'].astype(np.object) + b':' +
    #                 a['MM'].astype(np.object) + b':' + a['SS'].astype(np.object), '|S19', ndmin=1)
    return chars_array_to_datetimeindex(date, 'datetime64[ns]')  # convert ISO8601 date strings


loaded_inclin_Baranov = loaded_chain_Baranov  # to find this function by csv_inclin_Baranov.ini config file name


def concat_to_iso8601(a: pd.DataFrame) -> pd.Series:
    """
    Concatenate date/time parts to get iso8601 strings
    :param a: pd.DataFrame with columns 'yyyy', 'mm', 'dd', 'HH', 'MM', 'SS' having byte strings
    :return:  series of byte strings of time in iso8601 format
    """

    d = a['yyyy'].str.decode("utf-8")
    d = d.str.cat([a[c].str.decode("utf-8").str.zfill(2) for c in ['mm', 'dd']], sep='-')
    t = a['HH'].str.decode("utf-8").str.zfill(2)
    t = t.str.cat([a[c].str.decode("utf-8").str.zfill(2) for c in ['MM', 'SS']], sep=':')
    return d.str.cat(t, sep='T')


@meta_out_fields(keys_del={'yyyy', 'mm', 'dd', 'HH', 'MM', 'SS'}, add_before={'Time': 'M8[ns]'})
def loaded_tcm(
        a: pd.DataFrame, cfg_in: Mapping[str, Any] = None,
        csv_specific_param: Optional[Mapping[str, Any]] = None,
        # cols_out: Union[slice, Sequence[Any]] = slice(None)
        ) -> pd.DataFrame:
    """
    Specified prep&proc of raw data from Kondrashov inclinometers:
    - Time calc: gets time in current zone

    :param a:
    :param cfg_in: dict
    :param csv_specific_param:
    if it is {'invert_magnetometer': True} then multiply magnetometer channels on -1
    :return: numpy 'datetime64[ns]' array

    Example input:
    a = {
    'yyyy': b"2017", 'mm': b"1", 'dd': b"4",
    'HH': b'9','MM': b'5', 'SS': b''}
    """

    try:
        date = concat_to_iso8601(a)  # .compute() #da.from_delayed(, (a.shape[0],), '|S19') #, ndmin=1)
    except Exception as e:
        lf.exception('Can not convert date in {}', a)
        raise e

        # Bad date? Try set manually like this:
        time_st = '2023-06-16T09:13:59'  # '2023-05-08T23:24:00'
        time_en = '2023-07-24T13:10:00'  # '24.07.2023 в 13:10'
        n_rows = date.size  # 6322918
        date = pd.date_range(start=time_st, end=time_en, periods=n_rows)

    tim_index = chars_array_to_datetimeindex(date, 'datetime64[ns]')  # a['Time']

    key = 'time_shift'
    try:
        cfg_dt = csv_specific_param[key]
        if cfg_dt:
            dt0 = pd.Timedelta(cfg_dt.get('dt0', '0s'))

            time_raw_st = cfg_in.get('time_raw_st')
            if not time_raw_st:
                time_raw_st = tim_index[0]
                # Saving if 1st chunk to use in next chunks.
                # Must be cleared when start loading new data file (as in csv_load.csv_read_gen())
                cfg_in['time_raw_st'] = time_raw_st

            time_st = time_raw_st + dt0  # real time start
            time_en = pd.Timestamp(cfg_dt['time_en'])
            dt_full = (time_en - time_st).as_unit('ns')  # full good time range

            linear_len = cfg_dt.get('linear_len')
            if linear_len:
                dt = np.arange(cfg_in['n_rows'], cfg_in['n_rows'] + len(tim_index)) * (dt_full / linear_len)
                tim_index = pd.DatetimeIndex(time_st + dt)
                lf.info(
                    '{} applied: replacing time by series of {:g} Hz from {:%y-%m-%d %H:%M:%S} + {}',
                    key, linear_len/dt_full.total_seconds(), time_st, dt0 or '0s'
                )
            else:
                # Difference between good and bad end
                dt_end = pd.Timedelta(cfg_dt.get('dt_end', pd.NaT))  # dt_end = time_en - time_en_bad
                if pd.isna(dt_end):
                    time_raw_en = pd.Timestamp(cfg_dt['time_raw_en'])
                    try:
                        dt_end = time_en - time_raw_en
                    except KeyError:
                        lf.info(
                            'Can not apply {} to data: (time_en and time_raw_en started) or dt_end must '
                            'be specified. Other parameters specified: time_raw_st={:%y-%m-%d %H:%M:%S}, '
                            'dt0={}, linear_len={}',
                            key, time_raw_st, dt0 or '0s', linear_len
                        )
                else:
                    time_raw_en = time_en - dt_end
                    # Full raw time range
                    dt_raw_full = (time_raw_en - time_raw_st).as_unit('ns')

                dt_end_ns = dt_end.as_unit('ns')  # with this, division before multiplication is needed to not overflow
                # for small dt_end_ns there may be loss in precision, so select max to divide first?:
                # div_max_1st = lambda x, y, z: np.where(y.to_numpy().astype(np.int64) > x, x*(y/z), (x/z)*y)

                dt = (tim_index - time_raw_st) * (dt_end_ns / dt_raw_full)
                tim_index += (dt0 + dt)

                lf.info(
                    '{} applied. Parameters: time_st={:%y-%m-%d %H:%M:%S} + {}, dt_end={}',
                    key, time_raw_st, dt0 or '0s', dt_end or '0s'
                )
    except KeyError:
        pass

    try:
        lf.info('Time {:%y-%m-%d %H:%M} – {:%d %H:%M} converted', *tim_index[[0, -1]].to_pydatetime())
    except KeyError:
        lf.warning('Time of block size {} converted', tim_index.size)

    try:
        key = 'invert_magnetometer'
        if csv_specific_param[key]:
            magnetometer_channels = ['Mx', 'My', 'Mz']
            # Log csv_specific_param operation with message to indicate how meny redundant calls of this have been made.
            # (possible side effect of this is canceled by copy() below, except a long calculation time)
            lf.info('{key} applied',  # almost now
                    {'key': key, 'add_same_counter': 1, 'id': id(a.loc[:, magnetometer_channels[0]].values)  # Is id() always detects same input data?
                     }
                    )
            a.loc[:, magnetometer_channels] = -a.loc[:, magnetometer_channels].values           # Is "*= -1" better?
            a = a.copy()  # make function idempotent for use with dask without side effects     # Comment and check above message changes?
        elif csv_specific_param:
            lf.info(f'{key} is not specified in csv_specific_param (skipped)')
    except KeyError:
        if csv_specific_param is not None:
            lf.warning('Unknown key(s) in csv_specific_param!')

    return a.assign(Time=tim_index)


# same but do not use csv_specific_param (key = 'invert_magnetometer') here because no input magnetometer data
loaded_wavegauge = loaded_tcm


def f_repl_by_dict(replist: Iterable[AnyStr], binary_str=True) -> Callable[[Match[AnyStr]], AnyStr]:  # , repldictvalues={None: b''}
    """
    Returns function ``fsub()`` that returns replacement using ``regex.sub()``. ``fsub(line)`` uses multiple alternative
    search groups, specified by ``replist``, it uses last found or returns source ``line`` if no match.
    :param replist: what to search: list of regexes for all possible bad matches
    :return: function bytes = fsub(bytes), to replace
    """
    # ?, last group can have name (default name is None) which must be in repldictvalues.keys to use it for rep last group
    #:param repldictvalues: dict: key name of group, value: by what to replace. Default {None: b''} that is delete group from source

    # This is closure to close scope of definitions for replfunc() replacing function.
    # "or" list:
    regex = re.compile((b'|' if binary_str else '|').join(x for x in replist))  # re.escape()

    def replfunc(match):
        """
        :param match: re.match object
        :return: if last found group has name then matched string else empty string
        """

        if match.lastgroup:
            return match.group(match.lastgroup)  # regex.sub(replfunc, match.string)
        else:
            return b'' if binary_str else ''
        # return repldictvalues[match.lastgroup]

        # re.sub('\g<>'.(match.lastgroup), repldictvalues[match.lastgroup])
        # match.group(match.lastgroup)
        # if re.sub(repldictvalues[match.lastgroup])

    def fsub(line):
        """
        Use last found group.
        :param line:
        :return: None If no npatern found fsub
        """
        # try:
        return regex.sub(replfunc, line)
        # Note: replfunc even not called if regex not get any matches so this error catching useless
        # except KeyError:  # nothing to replace
        #     return None # line
        # except AttributeError:  # nothing to replace
        #     return None # line

    return fsub


def correct_old_zip_name_ru(name):
    """
    Correct invalid zip encoding of Russian file names
    code try to work with old style zips (having encoding CP437 and just has it broken), and if fails, it seems that zip archive is new style (UTF-8)
    :param name:
    :return:
    """
    from chardet import detect

    try:
        name_bytes = name.encode('cp437')
    except UnicodeEncodeError:
        name_bytes = name.encode('utf8')
    encoding = detect(name_bytes)['encoding']
    return name_bytes.decode(encoding)


def rep_in_file(file_in: Union[str, PurePath, BinaryIO, TextIO], file_out,
                f_replace: Union[Callable[[bytes], bytes], Callable[[str], str]],
                header_rows=0, block_size=None, min_out_length=2,
                f_replace_in_header: Optional[Callable[[bytes], bytes]] = None,
                binary_mode=True) -> int:
    """
    Replacing text in a file, applying f_replace
    :param file_in: str/Path or opened file handle, file to read from
    :param file_out: str/Path, file name to write here
    :param f_replace: bytes = function(bytes) or str = function(str) if binary_mode=False
    :param header_rows: int/range copy rows, if range and range[0]>0 remove rows till range[0] then copy till range[-1]
    :param block_size: int or None, if None - reads line by line else size in bytes
    :param min_out_length: int, do not write lines which length is lesser
    :param f_replace_in_header: bytes = function(bytes) or str = function(str) if binary_mode=False. function to make replacements only in header  todo: Not Implemented
    :return: None
    """
    try:
        file_in_path = Path(file_in)
        if not file_in_path.is_file():
            print(f'{file_in_path} not found')
            return None

    except TypeError:  # have opened file handle
        file_in_path = Path(file_in.name)  # losts abs path for archives so
        # next check will be filed, but we not writing to archives so it's safe.

    lf.warning('preliminary correcting csv file {:s} by removing irregular rows, writing to {}.',
              file_in_path.name, file_out)

    # Check if we modyfing input file
    file_out = Path(file_out)
    if file_in_path == file_out:
        # Modyfing input file -> temporary out file and final output files:
        file_out, file_out_original = file_out.with_suffix(f'{file_out.suffix}.bak'), file_out
    else:
        file_out_original = ''

    sum_deleted = 0
    with FakeContextIfOpen(
            lambda x: open(x, **{'mode': 'rb'} if binary_mode else {'mode': 'r', 'errors': 'ignore'}),
            file_in
    ) as fin, open(file_out, 'wb' if binary_mode else 'w') as fout:
        if isinstance(header_rows, range):
            for row in range(header_rows[0]):
                # removing rows
                newline = fin.readline()
            for row in header_rows:
                # copying rows
                newline = fin.readline()
                fout.write(newline)
        else:  # copying rows
            for row in range(header_rows):
                newline = fin.readline()
                fout.write(newline)

        # # commented replacing time part to uniformly increasing with period dt
        # t_start = pd.Timestamp('2021-07-26T08:06:17')  # '2021-08-27T18:49:30'
        # dt = pd.Timedelta('00:00:00.200803832')  # 200253794 203379827
        if block_size is None:
            # replacing rows
            for line in fin:
                # for i, line in enumerate(fin):
                #     t_cur = t_start + i*dt
                #     line = f"{t_cur:%Y,%m,%d,%H,%M,%S},{line.split(',', maxsplit=6)[-1]}"

                newline = f_replace(line)
                if len(newline) > min_out_length:  # newline != rb'\n':
                    fout.write(newline)
                else:  # after correcting line is bad
                    sum_deleted += 1
        else:
            # replacing blocks
            for block in iter(lambda: fin.read(block_size), (the_end := b'' if binary_mode else '')):
                block = f_replace(block)
                if block != the_end:
                    fout.write(block)

    if file_out_original:
        file_out.replace(file_out_original)

    return sum_deleted


def parse_name(name: str):
    """
    Extract logical parts of inclinometer / wave gauge name/glob from source raw csv file name

    :param name: name/glob of source raw csv file
    :return: dict with fields:
    - type:
    - model: abbreviation of one letter among (i, b, d, p, w) or '' if not recognized
    - number:
    - chars1 and chars2: before and after ``number``
    """
    name = name.lower()
    m = re.match(  # (?P<chars0>\*?) search
        r'@?(?P<type>[iw])(?P<chars1>nkl_|nkl|ncl_|ncl|\*_|_\*?|\*|)'
        r'(?P<model>[bdp]|\[[0bdp]{1,4}]|)(?P<number>\*?\d{1,4})(?P<chars2>\D*)',
        name
    )

    if m:
        m = m.groupdict()
    else:  # unusual i / w
        m = re.match(r'@?(?P<type>voln_v)(?P<chars1>\D*)(?P<number>\d\d)(?P<chars2>\D*)', name)
        if m:
            m = m.groupdict()
            m['type'] = 'w'
            m['model'] = ''
    return m


def mod_name(file_in: Union[str, PurePath], add_prefix=None, parse=True) -> Tuple[str, Path]:
    """
    Extract 1. `model` of inclinometer / wave gauge and construct (as we use inclinometer with pressure sensor
    now, and it includes wave gauge functionality, type is not needed as now it always "inclinometer")
    2. Path/name or glob pattern (keeps asterixes if in prefixes) of corrected raw data file (of inclinometer
    / wave gauge) from name/glob of source raw csv file

    :param parse: if False then return `model` = None and `file_in` prefixed with one `add_prefix`
    :param file_in: full path name/glob of source raw csv file
    :param add_prefix:
    :return:
        - model abbreviation: one letter among (i, b, d, p, w) or '' if not recognized
        - modified file name (of corrected not here raw inclinometer / wave gauge data).
    """
    file_in = PurePath(file_in)
    name = file_in.stem
    if parse:
        m = parse_name(name.lower().replace('inkl', 'incl'))  # convert Translit to English)
        # Return "i" for inclinometer if model is not specified else matched regex model; "w" for wave gauge
        if m:
            model = m['model']
            if not model:
                model = m['type']
            name = '{type}{chars1}{model}{number}{chars2}'.format_map(m)
            if not ('*' in name or '?' in name) and not m['number']:
                print(f'Bad probe name {file_in}: probe number not detected')
        else:
            model = None
            print(f'Not known probe name: {file_in}')

        # Paste `add_prefix` before extension, keeping asterixes if exist
        if add_prefix:
            def rep(matchobj):
                """ if add_prefix (excluding @) in ``name`` replace it else append"""
                substr = matchobj.group(0)
                # if (add_str1 := add_prefix.replace('@', '')) in substr:
                #     return substr.replace(add_str1, add_prefix)
                # else:
                return f'{add_prefix}{substr}'
                # '{0}{2}.{1}'.format(*name.rsplit('.', 1), add_prefix) f'{substr}{add_prefix}'

            name = re.sub(r'^\*?([^*.]*)', rep, name)
    else:
        model = None
        name = f'@{name.lstrip(add_prefix)}'

    file_out = file_in.with_name(name).with_suffix(file_in.suffix)
    return model, file_out

# Note: this is replaced by correct_baranov_txt() that now uses universal replacer correct_txt()
# def correct_baranov_txt(file_in: Union[str, PurePath], file_out: Optional[PurePath] = None,
#                         dir_out: Optional[PurePath] = None) -> Path:
#     """
#     Replaces bad strings in csv file and writes corrected file which named by replacing 'W_0' by 'w' in file_in
#     :param file_in:
#     :return: name of file to write.
#     """
#     # 2019	11	18	18	00	00	02316	22206	16128	31744	32640	33261	32564	32529
#     #
#     # 2019	11	18	18	00	00	02342	22204	16128	31744	32576	33149	32608	32582
#     fsub = f_repl_by_dict([
#         b'^\r?(?P<use>20\d{2}(\t\d{1,2}){5}(\t\d{5}){8}).*',
#         b'^.+'])
#     # last element in list is used to delete non captured by previous: it has no group so sub() returns empty line,
#     # that will be checked and deleted
#
#     file_in = Path(file_in)
#     if file_out:
#         pass
#     elif dir_out:
#         file_out = dir_out / mod_incl_name(file_in.name)
#     else: # autofind out path
#         file_out = file_in.with_name(mod_incl_name(file_in.name))  # re.sub(r'W_?0*(\d{2})', r'w\1', file_in.name)
#
#     if file_out.is_file():
#         lf.warning(f'skipping of pre-correcting csv file {file_in.name} to {file_out}: destination exist')
#         return file_out
#     elif not file_in.is_file():
#         print(f'{file_in} not found')
#         return None
#     else:
#         lf.warning('preliminary correcting csv file {} by removing irregular rows, writing to {}.'.format(
#             file_in.name, str(file_out)))
#     sum_deleted = rep_in_file(file_in, file_out, fsub)
#
#     if sum_deleted:
#         lf.warning('{} bad line deleted'.format(sum_deleted))
#
#     return file_out


def correct_txt(
        file_in: Union[str, Path, BinaryIO, TextIO],
        file_out: Optional[Path] = None,
        dir_out: Optional[PurePath] = None,
        mod_file_name: Callable[[PurePath], PurePath] = lambda n: n.parent.with_name(n.name.replace('.', '_clean.')),
        sub_str_list: Sequence[bytes] = None,
        **kwargs
        ) -> Path:
    """
    Replaces bad strings in csv file and writes corrected file (saves regular table format file)
    :param file_in: file name or file-like object of csv file or csv file opened in archive that RarFile.open() returns
    :param file_out: full output file name. If None combine dir_out and name of output file which is generated by mod_file_name(file_in.name)
    :param dir_out: output dir, if None then dir of file_in, but
        Note: in opened archives it may not contain original path info (they may be temporary archives).
    :param mod_file_name: function to get out file from input file (usually relative)
    :param sub_str_list: f_repl_by_dict() argument that will be decoded here to str if needed. If Falsy then do not
     replace anything in content
    :param kwargs: rep_in_file() keyword arguments except first 3
    :return: name of file to write.
    """

    is_opened = isinstance(file_in, (io.TextIOBase, io.RawIOBase))
    msg_file_in = file_in

    # Set _file_out_ name if not yet
    if file_out:
        pass
    elif dir_out:
        msg_file_in = (Path(file_in) if isinstance(file_in, str) else file_in).name
        name_maybe_with_sub_dir = mod_file_name(msg_file_in)
        file_out = dir_out / Path(name_maybe_with_sub_dir).name  # flattens archive subdirs
    else:  # autofind out path
        # handle opened files too

        # Set out file dir based on file_in dir
        if is_opened:
            # handle opened files from archive too
            inf = getattr(file_in, '_inf', None)
            if inf:
                file_in_path = Path(inf.volume_file)
                file_out = mod_file_name(file_in_path.parent / file_in.name)  # exclude archive name in out path
                file_in_path /= file_in.name  # archive name+name / file_in.name for logging
            else:
                # cmd_contained_path_of_archive = getattr(file_in, '_cmd', None)  # depreciated
                # if cmd_contained_path_of_archive:
                #     # archive path+name
                #     file_in_path = Path([word for word in cmd_contained_path_of_archive if (len(word)>3 and word[-4]=='.')][0])
                #     # here '.' used to find path word (of arhive with extension .rar)
                #     file_out = mod_file_name(file_in_path.parent / file_in.name)     # exclude archive name in out path
                #     file_in_path /= file_in.name                       # archive name+name / file_in.name for logging
                # else:
                file_in_path = Path(file_in.name)
                file_out = mod_file_name(file_in_path)
            msg_file_in = file_in_path.name
        elif not file_out:
            file_in_path = Path(file_in)
            file_out = file_in_path.with_name(str(mod_file_name(Path(file_in_path.name))))

    out_dir = file_out.parent
    out_dir = out_dir.with_name(out_dir.name.replace('.', '-'))  # Windows can not create dir named same as sibling file
    dir_create_if_need(out_dir)
    file_out = out_dir / file_out.name

    if file_out.is_file() and file_out.stat().st_size > 100:  # If file size is small it may be damaged.
        # Try to reprocess (not hard to reprocess small files)
        if is_opened:
            msg_file_in = correct_old_zip_name_ru(msg_file_in)
        lf.warning(f'skipping of pre-correcting csv file {msg_file_in} to {file_out.name}: destination exist')
        return file_out

    binary_mode = isinstance(file_in, io.RawIOBase)
    if sub_str_list:
        fsub = f_repl_by_dict([x if binary_mode else bytes.decode(x) for x in sub_str_list], binary_str=binary_mode)

        # {'use': b'\g<use>'})  # $ not works without \r\n so it is useless
        # b'((?!2018)^.+)': '', b'( RS)(?=\r\n)': ''
        # '^Inklinometr, S/N 008, ABIORAS, Kondrashov A.A.': '',
        # '^Start datalog': '',
        # '^Year,Month,Day,Hour,Minute,Second,Ax,Ay,Az,Mx,My,Mz,Battery,Temp':
        sum_deleted = rep_in_file(file_in, file_out, fsub, **{'binary_mode': binary_mode, **kwargs})
        if sum_deleted:
            lf.warning('{} bad lines deleted'.format(sum_deleted))
    else:
        lf.warning(
            f'skipping of pre-correcting csv file {msg_file_in} to {file_out.name}: just extracting to output dir')
        block_size = 1000000
        the_end = b'' if binary_mode else ''
        with open(file_out, 'w') as fout:
            for block in iter(lambda: file_in.read(block_size), the_end):
                if block != the_end:
                    fout.write(block)

    return file_out


# def correct_kondrashov_txt(
#         file_in: Union[str, Path, BinaryIO, TextIO],
#         file_out: Optional[Path] = None,
#         dir_out: Optional[PurePath] = None, **kwargs) -> Path:
#     """
#     Replaces bad strings in csv file and writes corrected file which named by replacing 'W_0' by 'w' in file_in
#     :param file_in:
#     :return: name of file to write.
#
#     Supported _file_in_ format examples:
#     # 2018,4,30,23,59,53,-1088,-640,-15648,-14,74,556,7.82,5.50
#     # 2018,12,1,0,0,0,-544,-1136,-15568,-44,90,550,7.82,5.50
#     """
#     return correct_txt(
#         file_in, file_out, dir_out,
#         mod_file_name=mod_incl_name,
#         sub_str_list=[
#             b'^(?P<use>20\d{2}(,\d{1,2}){5}(,\-?\d{1,6}){6},\d{1,2}(\.\d{1,2})?,\-?\d{1,3}(\.\d{1,2})?).*',
#             b'^.+'],
#         **kwargs
#     )
#
#
# def correct_baranov_txt(
#         file_in: Union[str, Path, BinaryIO, TextIO],
#         file_out: Optional[Path] = None,
#         dir_out: Optional[PurePath] = None, **kwargs) -> Path:
#     """
#     Replaces bad strings in csv file and writes corrected file which named by replacing 'W_0' by 'w' in file_in
#     :param file_in:
#     :return: name of file to write.
#     """
#     # 2019	11	18	18	00	00	02316	22206	16128	31744	32640	33261	32564	32529
#     #
#     # 2019	11	18	18	00	00	02342	22204	16128	31744	32576	33149	32608	32582
#     return correct_txt(
#         file_in, file_out, dir_out,
#         mod_file_name=mod_incl_name,
#         sub_str_list=[
#             b'^\r?(?P<use>20\d{2}(\t\d{1,2}){5}(\t\d{5}){8}).*',
#             b'^.+'],
#         **kwargs
#     )


def correct_idronaut_terminal_txt(
        file_in: Union[str, Path, BinaryIO, TextIO],
        file_out: Optional[Path] = None,
        dir_out: Optional[PurePath] = None, **kwargs) -> Path:
    """
    Replaces bad strings of Idronaut OS310 recorded in terminal and writes corrected file
     261.23  8.812 23.917 21.561    2.53    0.26  7.858 -180.8 09:26:03.96
     0.28 25.430  0.017  0.017-1148.98  -21.71  9.026  107.4 09:53:02.56
    """
    return correct_txt(
        file_in, file_out, dir_out,
        mod_file_name = lambda n: n.replace('.', '_clean.'),
        sub_str_list = [
            b'^(?P<use> *\d{1,4}.\d{1,4}( *[ -]\d{1,4}.\d{1,5}){5,7}( +\d{2}:\d{2}:\d{2}\.\d{2})).*',
            #b'^(?P<use> *\d{1,4}.\d{1,2}( *[ -]\d{1,4}.\d{1,3}){7}( +\d{2}:\d{2}:\d{2}\.\d{2})).*',
            b'^.+'
            ],
        **kwargs
    )


# navigation loaders -------------------------------------------------------------

# ----------------------------------------------------------------------
def deg_min_float_as_text2deg(degmin):
    """
    :param degmin: 5539.33 where 55 are degrees, 39.33 are minutes
    """
    minut, deg = np.modf(degmin / 100)
    deg += minut / 0.6
    return deg


# 'yyyy', 'mm', 'dd', 'HH', 'MM', 'SS'
@meta_out_fields(keys_del={'date', 'Lat_NS', 'Lon_WE'}, add_before={'Time': 'M8[ns]'})
def loaded_nav_supervisor(a: Union[pd.DataFrame, np.ndarray], cfg_in: Mapping[str, Any],
        csv_specific_param: Optional[Mapping[str, Any]] = None,
        # cols_out: Union[slice, Sequence[Any]] = slice(None)
        ) -> pd.DataFrame:
    """
    Specified prep&proc of navigation data from program "Supervisor":
    - Time calc: gets string for time in current zone
    - Lat, Lon to degrees conversion


    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :param csv_specific_param: {depth_add: str_val}
    :return: numpy 'datetime64[ns]' array

    Example input:
    a = {
    'date': "17 10 14", 'Time': "   00 00 01",
    'Lat': 5522.949219, 'Lat_NS': 'N',
    'Lon': 1817.303223, 'Lon_WE': 'E',
    'DepEcho': 10
    }
    todo: replace bad strings. Now it is need to manual delete multiple headers
    """

    if not isinstance(a, np.recarray):
        txtT = pd.to_datetime(a['Time'].str.decode('utf-8', errors='replace'), format='%H %M %S', errors='coerce') - \
               pd.to_datetime('00:00:00', format='%H:%M:%S')
        date = pd.to_datetime(a['date'].str.decode('utf-8', errors='replace'), format='%y %m %d',
                              errors='coerce') + txtT
        names_in = a.columns
    else:
        # old (array input)
        # whitespaces before time
        c = 3 if Path(cfg_in['cfgFile']).stem.endswith('nav_supervisor') else 1
        date = np.array(a['date'].astype(object), dtype={'YY': ('a2', 0), 'MM': ('a2', 3), 'DD': ('a2', 6)})
        txtT = np.array(a['Time'].astype(object), dtype={'hh': ('a2', 0 + c), 'mm': ('a2', 3 + c), 'ss': ('a2', 6 + c)})
        bBad = np.int8(date['YY']) > np.int8(datetime.now().strftime('%y'))
        if np.any(bBad):
            print('Bad date => extract from file name')
            date['YY'], date['MM'], date['DD'] = [cfg_in['file_stem'][(slice(k, k + 2))] for k in range(0, 5, 2)]
            # date = np.where(bBad,
        date = np.array(century + date['YY'].astype(np.object) + b'-' + date['MM'].astype(np.object) + b'-' +
                        date['DD'].astype(np.object) + b'T' + txtT['hh'].astype(np.object) + b':' +
                        txtT['mm'].astype(np.object) + b':' + txtT['ss'].astype(np.object), '|S19', ndmin=1)
        # Bug in supervisor convertor corrupts dates: gets in local time zone.
        # Ad hoc: for daily records we use 1st date - can not use if 1st date is wrong
        # so use full correct + wrong time at start (day jump down)
        # + up and down must be alternating
        date = chars_array_to_datetimeindex(date, 'datetime64[ns]')  # convert ISO8601 date strings
        names_in = a.dtype.names
    date = day_jumps_correction(cfg_in, date)

    if 'Lat' in names_in:
        # Lat, Lon to degrees. Not used for meteo data
        a['Lat'] = deg_min_float_as_text2deg(a['Lat'])
        a['Lon'] = deg_min_float_as_text2deg(a['Lon'])

        bMinus = a['Lat_NS'] == b'S'
        if np.any(bMinus):
            a['Lat'] = np.where(bMinus, -1 * a['Lat'], a['Lat'])
        bMinus = a['Lon_WE'] == b'W'
        if np.any(bMinus):
            a['Lon'] = np.where(bMinus, -1 * a['Lon'], a['Lon'])

    if csv_specific_param is not None:
        if 'DepEcho_add' in csv_specific_param:
            a.eval(f"DepEcho = DepEcho + {csv_specific_param['DepEcho_add']}",
                   inplace=True)  # @corrections['DepEcho_add'] if float
            log_csv_specific_param_operation('loaded_nav_supervisor', csv_specific_param.keys(), cfg_in)

    return a.assign(**{'Time': date})


def loaded_nav_HYPACK(a: Union[pd.DataFrame, np.ndarray], cfg_in: Mapping[str, Any]) -> pd.DatetimeIndex:
    """
    Specified prep&proc of navigation data from program "HYPACK":
    - Time calc: gets string for time in current zone

    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :return: numpy 'datetime64[ns]' array

    Example input:
    a = {
    'date': "02:13:12.30", #'Time'
    'Lat': 55.94522129,
    'Lon': 18.70426069,
    'Depth': 43.01}
    """

    # extract date from file name
    if not cfg_in.get('fun_date_from_filename'):
        cfg_in['fun_date_from_filename'] = date_from_filename
    str_date = cfg_in['fun_date_from_filename'](cfg_in['file_stem'], century.decode())
    t = pd.to_datetime(str_date) + \
        pd.to_timedelta(a['Time'].str.decode('utf-8', errors='replace'))
    # t = day_jumps_correction(cfg_in, t.values)
    return t


@meta_out_fields(keys_del={'Time', 'LatNS', 'LonEW', 'DatePC', 'TimePC'},
                 add_before={'Time': 'M8[ns]', 'Lat': 'f8', 'Lon': 'f8'})
def loaded_nav_HYPACK_SES2000(
        a: Union[pd.DataFrame, np.ndarray], cfg_in: Mapping[str, Any],
        # cols_out: Union[slice, Sequence[Any]] = slice(None)
        ) -> pd.DatetimeIndex:
    """
    Specified prep&proc of SES2000 data from program "HYPACK":
    - Time calc: gets string for time in current zone
    - Lat, Lon to degrees conversion

    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :return: numpy 'datetime64[ns]' array


1 01855.748,E 5558.892,N 191533 370793.50 6205960.01 -120.65 24.08.2020 19:08:23
    Example input:
    a = {
    'date': "02:13:12.30", #'Time'
    'Lat': 55.94522129,
    'Lon': 18.70426069,
    'Depth': 43.01}
    """
    # sources to out columns loaded_nav_HYPACK_SES2000.meta_out(cfg_in['dtype']).keys()
    uniq = ~a.duplicated(subset=['Time'])  #'LatNS', 'LonEW', , 'DepEcho']

    date = pd.to_datetime(a.loc[uniq, 'DatePC'].str.decode('utf-8', errors='replace'), format='%d.%m.%Y')
    tim = pd.to_datetime(a.loc[uniq, 'Time'].str.decode('utf-8', errors='replace'), format='%H%M%S') - pd.Timestamp('1900-01-01')  # default date '1900-01-01' found experimentally!
    date += tim
    a['Time'] = pd.NaT  # also changes column's type
    a.loc[uniq, 'Time'] = day_jumps_correction(cfg_in, date.to_numpy())

    for in_f, out_f in (('LatNS', 'Lat'),('LonEW', 'Lon')):
        a[out_f] = np.nan
        a.loc[uniq, out_f] = deg_min_float_as_text2deg(
            pd.to_numeric(a.loc[uniq, in_f].str.decode('utf-8', errors='replace').str.split(',').str.get(0))
            )
    a.loc[uniq, 'DepEcho'] = -a.loc[uniq, 'DepEcho']  # make positive below sea top
    return a


@meta_out_fields(keys_del={'Date', 'Time'}, add_before={'Time': 'M8[ns]'})
def loaded_nav_HYPACK_EA600(
        a: Union[pd.DataFrame, np.ndarray], cfg_in: Mapping[str, Any],
        # cols_out: Union[slice, Sequence[Any]] = slice(None)
        ) -> pd.DatetimeIndex:
    """
    Specified prep&proc of EA600 data (from program "HYPACK"?):
    - Time calc: gets string for time in current zone
    - Lat, Lon to degrees conversion

    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :return: numpy 'datetime64[ns]' array


    54.7396999 19.5789835 86.37 30062021 222129.06 0.00
    Example input:
    a = {
    'date': "2021-06-30T22:21:29.06", #'Time'
    'Lat': 54.7396999,
    'Lon': 19.5789835,
    'Depth': 86.37}
    """
    # sources to out columns loaded_nav_EA600.meta_out(cfg_in['dtype']).keys()
    uniq = ~a.duplicated(subset=['Time'])  #'LatNS', 'LonEW', , 'DepEcho']

    date = pd.to_datetime(a.loc[uniq, 'Date'].str.decode('utf-8', errors='replace'), format='%d%m%Y')
    tim = pd.to_datetime(a.loc[uniq, 'Time'].str.decode('utf-8', errors='replace'), format='%H%M%S.%f') - pd.Timestamp('1900-01-01')  # default date '1900-01-01' found experimentally!
    date += tim
    a['Time'] = pd.NaT  # also changes column's type
    a.loc[uniq, 'Time'] = day_jumps_correction(cfg_in, date.to_numpy())

    return a


#@meta_out_fields(keys_del={'Time'}, add_before={'Time': 'M8[ns]'})
def loaded_nav_ADCP_WinRiver2_at(a: Union[pd.DataFrame, np.ndarray], cfg_in: Mapping[str, Any]) -> pd.DatetimeIndex:
    """
    Specified prep&proc of Depth and navigation from ADCP data, exported by WinRiver II with at.ttf settings:
    - Time calc: gets string for time in UTC from b'20,9,9,13,36,18,78' (yy,mm,dd,HH,MM,SS )

    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :return: numpy 'datetime64[ns]' array
    """

    tim = pd.to_datetime(a['Time'].str.decode('utf-8', errors='replace'), format='%y,%m,%d,%H,%M,%S,%f')
    return tim


@meta_out_fields(keys_del={'Time', 'X', 'Y'}, add_before={'Time': 'M8[ns]', 'Lat': 'f8', 'Lon': 'f8'})
def loaded_nav_Dudkov_HydroProfiles(
        a: Union[pd.DataFrame, np.ndarray], cfg_in: Mapping[str, Any],
        # cols_out: Union[slice, Sequence[Any]] = slice(None)
        ) -> pd.DatetimeIndex:
    """
    Specified prep&proc of Dudkov\tr_ABP049_000-078_1sec_processed_HydroProfiles:
    - Time calc: time is already UTC in ISO format
    - X, Y WGS84_UTM34N to Lat, Lon degrees conversion

    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :return: numpy 'datetime64[ns]' array


    Example input:
    a = {
    'date': "02:13:12.30", #'Time'
    'Lat': 55.94522129,
    'Lon': 18.70426069,
    'Depth': 43.01}
    """
    from pyproj import Proj
    p = Proj(proj='utm', zone=34, ellps='WGS84', preserve_units=False)
    a['Lon'], a['Lat'] = p(*a[['X', 'Y']].values.T, inverse=True)  # inverse transform
    a['Time'] = np.array(a['Time'].values, 'M8[ns]')
    # date = pd.to_datetime(a.loc[uniq, 'Time'].str.decode('utf-8', errors='replace'), format='%d.%m.%YT%H:%M:%S.%F')
    # 'DepEcho'
    return a
    # may use a.loc[:, list(cfg_in['fun_proc_loaded']... instead?
