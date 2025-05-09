#!/usr/bin/env python
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: pandas time convert utils
  Created: 26.02.2016
"""
import logging
import re
from typing import Optional, Union, Tuple, Sequence
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from numba import jit
from pandas.tseries.frequencies import to_offset
# from  pandas.tseries.offsets import DateOffset
from dateutil.tz import tzoffset

# from future.moves.itertools import zip_longest
# from builtins import input
# from debug import __debug___print

# my:
from .filters import l
from .utils2init import LoggingStyleAdapter

lf = LoggingStyleAdapter(logging.getLogger(__name__))
if __debug__:
    # datetime converter for a matplotlib plotting method
    try:
        import matplotlib
        from pandas.plotting import register_matplotlib_converters
    except ImportError:
        lf.warning("matplotlib not installed, but may be needed for display/save data images")
    else:
        register_matplotlib_converters()

dt64_1s = np.int64(1e9)

# def numpy_to_datetime(arr):
#     return np.apply_along_axis(np.ndarray.item, 0, np.array([arr], 'datetime64[s]'))


def datetime_fun(fun, *args, type_of_operation='<M8[s]', type_of_result='<M8[s]'):
    """
    :param fun, args: function and its arguments array to apply on
    :param type_of_operation: type to convert args before apply fun to not overflow
    :param type_of_result: type to convert result after
    :return: fun result of type type_of_operation

    >>> import pandas as pd; df_index = pd.DatetimeIndex(['2017-10-20 12:36:32', '2017-10-20 12:41:32'], dtype='datetime64[ns]', name='time', freq=None)
    >>> datetime_fun(lambda x: -np.subtract(*x)/2 .view('<m8[s]'), df_index[-2:].values)
    # 150s
    """
    return np.int64(fun(*[x.astype(type_of_operation).view('i8') for x in args])).view(type_of_result)


def datetime_mean(x: np.ndarray, y: np.ndarray, type_of_operation='<M8[s]'):
    """
    Compute mean vector of two time vectors
    :param x: numpy datetime64 vector
    :param y: numpy datetime64 vector
    :param type_of_operation: numpy type to convert x and y before average to not overflow
    :return: numpy datetime64 vector of type_of_operation
    """
    result = datetime_fun(lambda x2d: np.mean(x2d, 1), np.column_stack((x, y)), type_of_operation=type_of_operation)
    return result


# def datetime_mean(x, y):
#     return np.int64((x.astype('<M8[s]').view('i8') + y.astype('<M8[s]').view('i8') / 2)).view('<M8[s]')


def multiindex_timeindex(df_index):
    """
    Extract DatetimeIndex even it is in MultiIndex
    :param df_index: pandas index
    :return: df_t_index, DatetimeIndex,
        itm - next MultiIndex level if exist else None
    """
    b_MultiIndex = isinstance(df_index, pd.MultiIndex)
    if b_MultiIndex:
        itm = [isinstance(L, pd.DatetimeIndex) for L in df_index.levels].index(True)
        df_t_index = df_index.get_level_values(itm)  # not use df_index.levels[itm] which returns sorted
    else:
        df_t_index = df_index
        itm = None
    return df_t_index, itm


def multiindex_replace(pd_index, new1index, itm):
    """
    replace timeindex_even if_multiindex
    :param pd_index:
    :param new1index: replacement for pandas 1D index in pd_index
    :param itm: index of dimention in pandas MultiIndex which is need to replace by new1index. Use None if not MultiIndex
    :return: modified MultiIndex if itm is not None else new1index
    """
    if not itm is None:
        pd_index.set_levels(new1index, level=itm, verify_integrity=False)
        # pd_index = pd.MultiIndex.from_arrays([[new1index if i == itm else L] for i, L in enumerate(pd_index.values)], names= pd_index.names)
        # pd.MultiIndex([new1index if i == itm else L for i, L in enumerate(pd_index.levels)], labels=pd_index.labels, verify_integrity=False)
        # pd.MultiIndex.from_tuples([ind_new.values])
    else:
        pd_index = new1index
    return pd_index


def timezone_view(t, dt_from_utc=0):
    """
    Convert a given time 't' to a specific timezone offset from UTC.
    If the time 't' is timezone-naive, it's assumed to be in UTC.

    :param t: Pandas Timestamp or DatetimeIndex. The time to be converted.
    :param dt_from_utc: int or any pd.to_timedelta() compatible argument.
        The offset from UTC in seconds. Defaults to 0.
    :return: The time 't' converted to the timezone offset by 'dt_from_utc' from UTC.
    """
    # If dt_from_utc is 0 or equivalent to pd.Timedelta(0), set timezone info to 'UTC'
    tzinfo = (
        "UTC"
        if dt_from_utc in (0, pd.Timedelta(0))
        else tzoffset(None, pd.to_timedelta(dt_from_utc).total_seconds())
    )

    # Check if 't' is either a Pandas DatetimeIndex or a Timestamp
    if isinstance(t, (pd.DatetimeIndex, pd.Timestamp)):
        # If 't' is timezone-naive, localize it to 'UTC'
        if t.tz is None:
            t = t.tz_localize("UTC")
        # Convert 't' to the desired timezone
        return t.tz_convert(tzinfo)
    else:
        # If 't' is not a subclass of pd.Timestamp/DatetimeIndex, convert it
        lf.error(
            "Bad time format {}: {} - it is not subclass of pd.Timestamp/DatetimeIndex => Converting...",
            type(t),
            t,
        )
        t = pd.to_datetime(t).tz_localize(tzinfo)
        return t


# ----------------------------------------------------------------------
def pd_period_to_timedelta(period: str) -> pd.Timedelta:
    """
    Converts str to pd.Timedelta. May be better to use pd.Timedelta(*to_offset(period))
    :param period: str, in format of pandas offset string 'D' (Y, D, 5D, H, ...)
    :return:
    """
    number_and_units = re.search(r'(^\d*)(.*)', period).groups()
    if not number_and_units[0]:
        number_and_units = (1, number_and_units[1])
    else:
        number_and_units = (int(number_and_units[0]), number_and_units[1])
    try:
        return pd.Timedelta(*number_and_units)
    except Exception as e:  # pandas._libs.tslibs.np_datetime.OutOfBoundsTimedelta: Cannot cast 365100 from D to 'ns' without overflow.
        out_frac = pd.Timedelta(number_and_units[0]/1000, number_and_units[1], resolution='ms')
        return out_frac * 1000

def intervals_from_period(
    datetime_range: Optional[np.ndarray] = None,
    min_date: Optional[pd.Timestamp] = None,
    max_date: Optional[pd.Timestamp] = None,
    period: Optional[str] = '999D',
    **kwargs
) -> Tuple[pd.Timestamp, pd.DatetimeIndex]:
    """
    Divide datetime_range on intervals of period, normalizes starts[1:] if period>1D and returns them in tuple's 2nd element
    :param period: pandas offset string 'D' (D, 5D, h, ...) if None such field must be in cfg_in
    :param datetime_range: list of 2 elements, use something like np.array(['0', '9999'], 'datetime64[s]') for all data.
    If not provided 'min_date' and 'max_date' will be used
    :param min_date, max_date: used if datetime_range is None. If neither provided then use range from 2000/01/01 to now
    :return (start, ends): (Timestamp, fixed frequency DatetimeIndex)
    """

    # Set _datetime_range_ if need and its _start_
    if datetime_range is not None:
        start = pd.Timestamp(datetime_range[0])  # (temporarely) end of previous interval
    else:
        start = pd.to_datetime(min_date) if min_date else pd.Timestamp(year=2000, month=1, day=1)
        if max_date is not None:
            t_interval_last = pd.to_datetime(max_date)  # last
        else:
            t_interval_last = datetime.now()  # i.e. big value
        datetime_range = [start, t_interval_last]

    if period:
        period_timedelta = to_offset(period)  # pd_period_to_timedelta(

        # Set next start on the end of day if interval is bigger than day
        if period_timedelta >= pd.Timedelta(1, 'D'):
            start_next = start.normalize()
            if start_next <= start:
                start_next += period_timedelta
        else:
            start_next = start

        if start_next > datetime_range[-1]:
            ends = pd.DatetimeIndex(datetime_range[-1:])
        else:
            ends = pd.date_range(
                start=start_next,
                end=max(datetime_range[-1], start_next + period_timedelta),
                freq=period)
            # make last start bigger than datetime_range[-1]
            if ends[-1] < datetime_range[-1]:
                ends = ends.append(pd.DatetimeIndex(datetime_range[-1:]))
    else:
        ends = pd.DatetimeIndex(datetime_range[1:])

    return start, ends


def positiveInd(i: Sequence, l: int) -> int:
    """
    Positive index
    :param i: index
    :param l: length of indexing array
    :return: index i if i>0 else if i is negative then its positive python equivalent
    """
    ia = np.int64(i)
    return np.where(ia < 0, l - ia, ia)


def minInterval(range1: Sequence, range2: Sequence, length: int):
    """
    Intersect of two ranges
    :param range1: 1st and last elements are [min, max] of 1st range
    :param range2: 1st and last elements are [min, max] of 2nd range
    :param length: length of indexing array
    :return: tuple: (max of first iLims elements, min of last iLims elements)
    """
    # todo: update to: [
    # (lambda st, en: transpose([max(st), min(en)]))(*positiveInd([take(range1, [0, -1]), take(range2, [0, -1])], l).T)
    # ]
    def maxmin(lims1, lims2):
        return np.transpose([max(lims1[:, 0], lims2[:, 0]), min(lims1[:, -1], lims2[:, -1])])

    return maxmin(positiveInd(range1, length), positiveInd(range2, length))


def check_time_diff(
        t_queried: Union[pd.Series, np.ndarray], t_found: Union[pd.Series, np.ndarray],
        dt_warn: Union[pd.Timedelta, np.timedelta64, timedelta],
        msg: str = ('Bad nav. data time coverage at {} points where time difference [{units}] to nearest data exceeds '
                    '{dt_warn}:'),
        return_diffs: bool = False, max_msg_rows=1000
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Check time difference between found and requested time points and prints info if big difference is found

    :param t_queried: pandas TimeSeries or numpy array of 'datetime64[ns]'
    :param t_found:   pandas TimeSeries or numpy array of 'datetime64[ns]'
    :param dt_warn: pd.Timedelta - prints info about bigger differences found only
    :param return_diffs: if True also returns time differences (t_found - t_queried_values, 'timedelta64[ns]')
    :param msg: message header before bad rows found. May use placeholders:
    :param max_msg_rows:
    - {n} to include info about number of rows,
    - {dt_warn},
    - {units}
    :return: mask where time difference is bigger than ``dt_warn`` and time differences if return_diffs=True
    """
    try:
        if not np.issubdtype(t_queried.dtype, np.dtype('datetime64[ns]')):  # isinstance(, ) pd.Ti
            t_queried_values = t_queried.values
        else:
            t_queried_values = t_queried
    except TypeError:  # not numpy 'datetime64[ns]'
        t_queried_values = t_queried.values

    dt_arr = np.array(t_found - t_queried_values, 'timedelta64[ns]')
    bbad = abs(dt_arr) > np.timedelta64(dt_warn)
    if msg:
        n = bbad.sum()
        if n:
            if dt_warn > timedelta(minutes=1):
                units = 'min'
                dt_div = 60
            else:
                units = 's'
                dt_div = 1
            msg = '\n'.join([msg.format(n=n, dt_warn=dt_warn, units=units)] + [
                '{}. {}:\t{}{:.1f}'.format(i, tdat, m, dt / dt_div) for i, tdat, m, dt in zip(
                    np.flatnonzero(bbad)[:max_msg_rows], t_queried[bbad], np.where(dt_arr[bbad].astype(np.int64) < 0, '-', '+'),
                    np.abs(dt_arr[bbad]) / np.timedelta64(1, 's')
                )] + ['...' if n > max_msg_rows else ''])
            lf.warning(msg)
    return (bbad, dt_arr) if return_diffs else bbad


# str_time_short= '{:%d %H:%M}'.format(r.Index.to_datetime())
# timeUTC= r.Index.tz_convert(None).to_datetime()
# @jit(nopython=True)  # astype(datetime64[ns]) not supported on array(float64, 1d, C)

def matlab2datetime64ns(matlab_datenum: np.ndarray) -> np.ndarray:
    """
    Matlab serial day to numpy datetime64[ns] conversion
    :param matlab_datenum: serial day from 0000-00-00
    :return: numpy datetime64[ns] array
    """

    origin_day = -719529  # np.int64(np.datetime64('0000-01-01', 'D') - np.timedelta64(1, 'D'))
    day_ns = 24 * 60 * 60 * 1e9

    # LOCAL_ZONE_m8= np.timedelta64(tzlocal().utcoffset(datetime.now()))
    return ((matlab_datenum + origin_day) * day_ns).astype('datetime64[ns]')  # - LOCAL_ZONE_m8


# assert matlab2datetime64ns.nopython_signatures  # this was compiled in nopython mode

def date_from_filename(file_stem: str, century: str = '20'):
    """
    Reformat string from "%y%d%m" (format usually used to record dates in file names) to "%y-%d-%m" ISO 8601 format
    :param file_stem: str of length >= 6, should start from date in "%y%d%m" format
    :param century: str of length 2
    :return:
    """
    return f"{century}{'-'.join(file_stem[(slice(k, k + 2))] for k in (6, 3, 0))}"
