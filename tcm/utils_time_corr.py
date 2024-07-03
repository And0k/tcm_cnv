from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from time import sleep

from .filters import find_sampling_frequency, longest_increasing_subsequence_i, rep2mean, repeated2increased, \
    make_linear, rep2mean_with_const_freq_ends
from .utils2init import dir_create_if_need
from .utils_time import lf, datetime_fun, check_time_diff
#from to_pandas_hdf5.h5_dask_pandas import filter_global_minmax


tim_min_save: pd.Timestamp     # can only decrease in time_corr(), set to pd.Timestamp('now', tz='UTC') before call
tim_max_save: pd.Timestamp     # can only increase in time_corr(), set to pd.Timestamp(0, tz='UTC') before call
def time_corr(
        date: Union[pd.Series, pd.Index, np.ndarray],
        cfg_in: Mapping[str, Any],
        process: Union[str, bool, None] = None,
        path_save_image='corr_time_mode') -> Tuple[pd.Series, np.bool_]:
    """
    Correct time values:
    increases resolution (i.e. adds trend to repeated values) and
    deletes values leading to inversions (considered as outlines). But not changes its positions (i.e. no sorting) here.
    :param date: numpy np.ndarray elements may be datetime64 or text in ISO 8601 format
    :param cfg_in: dict with fields:
    - dt_from_utc: correct time by adding this constant
    - fs: sampling frequency
    - corr_time_mode: same as :param process, used only if :param process is None. Default: None.
    - b_keep_not_a_time: NaNs in :param date will be unchanged
    - path: where save images of bad time corrected
    - min_date, min_date: optional limits - to set out time beyond limits to constants slitly beyond limits
    :param process:
    - True ,'True' or 'increase': increase duplicated time values (increase time resolution by custom interpolation),
    - False ,'False', None: do not check time inversions (default),
    - 'delete_inversions': mask inversions (i.e. set them in returning b_ok) and not interpolate them
    :return: (tim, b_ok) where
    - tim: pandas time series, same size as date input
    - b_ok: mask of not decreasing elements
    Note: converts to UTC time if ``date`` in text format, properly formatted for conv.
    todo: use Kalman filter?
    """
    if not date.size:
        return pd.DatetimeIndex([], tz='UTC'), np.bool_([])
    if process is None:
        process = cfg_in.get('corr_time_mode')
    if process == 'False':
        process = False
    elif process in ('True', 'increase'):
        process = True
    if __debug__:
        lf.debug('time_corr (time correction) started')
    if dt_from_utc := cfg_in.get('dt_from_utc'):
        if isinstance(date.iat[0] if isinstance(date, pd.Series) else date[0], str):
            # add zone that compensate time shift
            hours_from_utc_f = dt_from_utc.total_seconds() / 3600
            Hours_from_UTC = int(hours_from_utc_f)
            hours_from_utc_f -= Hours_from_UTC
            if abs(hours_from_utc_f) > 0.0001:
                print('For string data can add only fixed number of hours! Adding', Hours_from_UTC / 3600, 'Hours')
            tim = pd.to_datetime((date.astype(np.object) + '{:+03d}'.format(Hours_from_UTC)).astype('datetime64[ns]'),
                                 utc=True)
        elif isinstance(date, pd.Index):
            tim = date
            tim -= dt_from_utc
            try:
                tim = tim.tz_localize('UTC')
            except TypeError:  # "Already tz-aware, use tz_convert to convert." - not need localize
                lf.warning('subtracted {} from input (already) UTC data!', dt_from_utc)
                pass

            # if Hours_from_UTC != 0:
            # tim.tz= tzoffset(None, -Hours_from_UTC*3600)   #invert localize
            # tim= tim.tz_localize(None).tz_localize('UTC')  #correct
        else:
            try:
                if isinstance(date, pd.Series):
                    tim = pd.to_datetime(date - np.timedelta64(dt_from_utc), utc=True)

                else:
                    tim = pd.to_datetime(date.astype('datetime64[ns]') - np.timedelta64(
                        pd.Timedelta(dt_from_utc)), utc=True)  # hours=Hours_from_UTC
            except OverflowError:  # still need??
                tim = pd.to_datetime(datetime_fun(
                    np.subtract, tim.values, np.timedelta64(dt_from_utc), type_of_operation='<M8[ms]'
                    ), utc=True)
            # tim += np.timedelta64(pd.Timedelta(hours=hours_from_utc_f)) #?
        lf.info('Time constant ({}) {:s}', abs(dt_from_utc),
                'subtracted' if dt_from_utc > timedelta(0) else 'added')
    else:
        if not isinstance(date.iat[0] if isinstance(date, pd.Series) else date[0], pd.Timestamp):  # isinstance(date, (pd.Series, np.datetime64))
            date = date.astype('datetime64[ns]')
        tim = pd.to_datetime(date, utc=True)  # .tz_localize('UTC')tz_convert(None)

    if cfg_min_date := cfg_in.get('min_date'):
        cfg_min_date = pd.Timestamp(cfg_min_date, tz=None if cfg_min_date.tzinfo else 'UTC')

        # Skip processing if data is out of filtering range
        global tim_min_save, tim_max_save
        tim_min = tim.min(skipna=True)
        tim_max = tim.max(skipna=True)
        # also collect statistics of min&max for messages:
        tim_min_save = min(tim_min_save, tim_min)
        tim_max_save = max(tim_max_save, tim_max)

        # set time beyond limits to special values keeping it sorted for dask and mark out of range as good values
        if tim_max < cfg_min_date:
            tim[:] = cfg_min_date - np.timedelta64(1, 'ns')  # pd.NaT                      # ns-resolution maximum year
            return tim, np.ones_like(tim, dtype=bool)

        is_time_filt = False
        if cfg_max_date := cfg_in.get('max_date'):
            cfg_max_date = pd.Timestamp(cfg_max_date, tz=None if cfg_max_date.tzinfo else 'UTC')
            if tim_min > cfg_max_date:
                tim[:] = cfg_max_date + np.timedelta64(1, 'ns')  # pd.Timestamp('2262-01-01')  # ns-resolution maximum year
                return tim, np.ones_like(tim, dtype=bool)
            if tim_max > cfg_max_date:
                b_ok_in = (tim.values <= cfg_max_date.to_numpy())
                is_time_filt = True
        if tim_min < cfg_min_date:
            if is_time_filt:
                b_ok_in &= (tim.values >= cfg_min_date.to_numpy())
            else:
                b_ok_in = (tim.values >= cfg_min_date.to_numpy())
                is_time_filt = True

        if is_time_filt:
            it_se = np.flatnonzero(b_ok_in)[[0,-1]]
            it_se[1] += 1
            tim = tim[slice(*it_se)]
    else:
        is_time_filt = False

    b_ok_in = tim.notna()
    n_bad_in = b_ok_in.size - b_ok_in.sum()
    if n_bad_in:
        if cfg_in.get('b_keep_not_a_time'):
            tim = tim[b_ok_in]
    try:
        b_ok_in = b_ok_in.to_numpy()
    except AttributeError:
        pass  # we already have numpy array


    t = tim.to_numpy(np.int64)
    if process and tim.size > 1:
        # Check time resolution and increase if needed to avoid duplicates
        if n_bad_in and not cfg_in.get('b_keep_not_a_time'):
            t = np.int64(rep2mean(t, bOk=b_ok_in))
            b_ok_in[:] = True
        freq, n_same, n_decrease, i_different = find_sampling_frequency(t, precision=6, b_show=False)
        if freq:
            cfg_in['fs_last'] = freq  # fallback freq to get value for next files on fail
        elif cfg_in['fs_last']:
            lf.warning('Using fallback (last) sampling frequency fs = {:s}', cfg_in['fs_last'])
            freq = cfg_in['fs_last']
        elif cfg_in.get('fs'):
            lf.warning('Using specified sampling frequency fs = {:s}', cfg_in['fs'])
            freq = cfg_in['fs']
        elif cfg_in.get('fs_old_method'):
            lf.warning('Using specified sampling frequency fs_old_method = {:s}', cfg_in['fs_old_method'])
            freq = cfg_in['fs_old_method']
        else:
            lf.warning('Ready to set sampling frequency to default value: fs = 1Hz')
            freq = 1

        # # show linearity of time # plt.plot(date)
        # fig, axes = plt.subplots(1, 1, figsize=(18, 12))
        # t = date.values.view(np.int64)
        # t_lin = (t - np.linspace(t[0], t[-1], len(t)))
        # axes.plot(date, / dt64_1s)
        # fig.savefig(os_path.join(cfg_in['dir'], cfg_in['file_stem'] + 'time-time_linear,s' + '.png'))
        # plt.close(fig)
        b_ok = None
        idel = None
        msg = ''
        if n_decrease > 0:
            # Excude elements

            # if True:
            #     # try fast method
            #     b_bad_new = True
            #     k = 10
            #     while np.any(b_bad_new):
            #         k -= 1
            #         if k > 0:
            #             b_bad_new = b1spike(t[b_ok], max_spike=2 * np.int64(dt64_1s / freq))
            #             b_ok[np.flatnonzero(b_ok)[b_bad_new]] = False
            #             print('step {}: {} spikes found, deleted {}'.format(k, np.sum(b_bad_new),
            #                                                                 np.sum(np.logical_not(b_ok))))
            #             pass
            #         else:
            #             break
            # if k > 0:  # success?
            #     t = rep2mean(t, bOk=b_ok)
            #     freq, n_same, n_decrease, b_same_prev = find_sampling_frequency(t, precision=6, b_show=False)
            #     # print(np.flatnonzero(b_bad))
            # else:
            #     t = tim.values.view(np.int64)
            # if n_decrease > 0:  # fast method is not success
            # take time:i
            # lf.warning(Fast method is not success)

            # Excluding inversions
            # find increased elements (i_different is i_inc only if single spikes):
            i_inc = i_different[longest_increasing_subsequence_i(t[i_different])]
            # try trusting to repeating values, keeping them to not interp near holes (else use np.zeros):
            dt = np.ediff1d(t, to_end=True)
            b_ok = dt == 0
            b_ok[i_inc] = True
            # b_ok= nondecreasing_b(t, )
            # t = t[b_ok]

            t_ok = t[b_ok]
            i_dec = np.flatnonzero(np.ediff1d(t_ok, to_end=True) < 0)
            n_decrease_remains = len(i_dec)
            if n_decrease_remains:
                lf.warning('Decreased time among duplicates ({:d} times). Not trusting repeated values...',
                          n_decrease_remains)
                b_ok = np.zeros_like(t, dtype=np.bool_)
                b_ok[i_inc] = True

                if process == 'delete_inversions':
                    # selecting one of the two bad time values that lead to the bad diff element and mask these elements
                    for s, e in i_dec + np.int32([0, 1]):
                        b_ok[t == (t_ok[e if b_ok[s] else s])] = False
                    if cfg_in.get('b_keep_not_a_time'):
                        (b_ok_in[b_ok_in])[~b_ok] = False
                    else:
                        b_ok_in[~b_ok] = False
            else:  # Decreased time not in duplicates
                i_dec = np.delete(i_different, np.searchsorted(i_different, i_inc))
                assert np.alltrue(i_dec == i_different[~np.in1d(i_different, i_inc)])  # same results
                # assert np.alltrue(i_dec == np.setdiff1d(i_different, i_inc[:-1]))  # same results
                if process == 'delete_inversions':
                    b_ok_in[np.flatnonzero(b_ok_in)[i_dec] if cfg_in.get('b_keep_not_a_time') else i_dec] = False

            b_ok[b_ok] = np.ediff1d(t[b_ok], to_end=True) > 0  # adaption for next step

            idel = np.flatnonzero(~b_ok)
            n_del = len(idel)
            msg = f"Filtered time: {n_del}/{t.size} values " \
                  f"{'masked' if process == 'delete_inversions' else 'will be interpolated'} (1st and last: " \
                  f"{pd.to_datetime(t[idel[[0, -1]]], utc=True)})"
            if n_decrease:
                lf.warning('decreased time ({}) was detected! {}', n_decrease, msg)
            else:
                lf.warning(msg)


        if n_same > 0 and cfg_in.get('fs') and not cfg_in.get('fs_old_method'):
            # The most simple operation that should be done usually for CTD, but requires fs:
            t = repeated2increased(t, cfg_in['fs'], b_ok if n_decrease else None)  # if n_decrease then b_ok is calculated before
            tim = pd.to_datetime(t, utc=True)
        elif n_same > 0 or n_decrease > 0:
            # message with original t

            # Replace t by linear increasing values using constant frequency excluding big holes
            if cfg_in.get('fs_old_method'):
                lf.info('Flatten time interval using provided freq = {:f}Hz (determined: {:f})',
                        cfg_in.get('fs_old_method'), freq)
                freq = cfg_in.get('fs_old_method')
            else:  # constant freq = filtered mean
                lf.info('Flatten time interval using median* freq = {:f}Hz determined', freq)
            b_show = n_decrease > 0
            if freq <= 1:
                # Skip: typically data resolution is sufficient for this frequency
                lf.warning('Not flattening for frequency < 1')
            else:
                # Increase time resolution by recalculating all values

                dt_interp_between = cfg_in.get('dt_interp_between', timedelta(seconds=1.5))
                # interp to can use as pandas index even if any bad. Note: if ~b_ok at i_st_hole then interp. is bad
                tim_before = pd.to_datetime(np.int64(rep2mean(t, bOk=b_ok)))
                i_st_hole = make_linear(t, freq, dt_big_hole=dt_interp_between)  # changes t (and tim?)
                # Check if we can use them
                bbad = np.ones_like(t, dtype=np.bool_)
                bbad[i_st_hole] = False
                bbad[i_st_hole[1:] - 1] = False  # last data point before hole
                bbad[bbad] = check_time_diff(
                    t[bbad].view('M8[ns]'), tim_before[bbad],
                    dt_warn=cfg_in.get('dt_max_interp_err', timedelta(seconds=1.5)),  # pd.Timedelta(minutes=2)
                    msg='Time difference [{units}] in {n} points exceeds {dt_warn} after flattening:',
                    max_msg_rows=20
                )
                # replace regions of big error with original data ignoring single values with big error
                _ = bbad[:-1] & bbad[1:]  # repeated error mask
                if _.any():
                    bbad[:-1] = _
                    bbad[1:] = _
                    n_repeated = bbad.sum()
                    b_ok = ~bbad
                    b_show = True
                    t[bbad] = tim_before.to_numpy(dtype=np.int64)[bbad]
                    dt = np.ediff1d(t, to_begin=1)
                    if (dt < 0).any():
                        t = tim_before.to_numpy(dtype=np.int64)
                        lf.info("flattening frequency failed => keeping variable frequency")
                    else:
                        lf.info(f'There are {n_repeated} adjacent among them => '
                                'revert to original not constant frequency there')
                else:
                    lf.info("Discarded isolated gaps")
            # Show what is done
            if b_show:
                if b_ok is None:
                    dt = np.ediff1d(t, to_begin=1)
                    b_ok = dt > 0
                plot_bad_time_in_thread(
                    cfg_in, t, b_ok, idel, tim,
                    (tim_min, tim_max) if cfg_in.get('min_date') else None,
                    path_save_image,
                    msg
                )
        # Checking all is ok

        dt = np.ediff1d(t, to_begin=1)
        b_ok = dt > 0
        # tim.is_unique , len(np.flatnonzero(tim.duplicated()))
        b_decrease = dt < 0  # with set of first element as increasing
        n_decrease = b_decrease.sum()
        if n_decrease > 0:
            lf.warning(
                'Decreased time ({:d}) remains:{:s}{:s}{:s}! => rows masked',
                n_decrease,
                '\n' if n_decrease > 1 else ' ',
                '\n'.join(' follows '.join(
                    '{:%y.%m.%d %H:%M:%S.%f}'.format(_) for _ in tim.iloc[se].to_numpy()
                    ) for se in np.flatnonzero(b_decrease)[:3, None] + np.int32([0, 1])),  # [-1, 0]
                '...' if n_decrease > 3 else ''
            )
            b_ok &= ~b_decrease

        b_same_prev = np.ediff1d(t, to_begin=1) == 0  # with set of first element as changing
        n_same = b_same_prev.sum()

        if cfg_in.get('b_keep_not_a_time'):
            if n_same > 0:
                lf.warning('non-increasing time ({:d} times) is detected! => interp ', n_same)
        else:
            # prepare to interp all non-increased (including NaNs)
            if n_bad_in:
                b_same_prev &= ~b_ok_in

            msg = ', '.join(
                f'{fault} time ({n} times)' for (n, fault) in ((n_same, 'non-increased'), (n_bad_in, 'NaN')) if n > 0
                )
            if msg:
                lf.warning('{:s} is detected! => interp ', msg)

        if n_same > 0 or n_decrease > 0:
            # rep2mean(t, bOk=np.logical_not(b_same_prev if n_decrease==0 else (b_same_prev | b_decrease)))
            b_bad = b_same_prev if n_decrease == 0 else (b_same_prev | b_decrease)
            t = rep2mean_with_const_freq_ends(t, ~b_bad, freq)
            dt = np.ediff1d(t, to_begin=1)
            b_ok = dt > 0
    else:  # not need to check / correct time
        b_ok = np.ones(tim.size, np.bool_)
    # make initial shape: paste NaNs back
    if n_bad_in and cfg_in.get('b_keep_not_a_time'):
        # place initially bad elements back
        t, t_in = (np.nan + np.empty_like(b_ok_in)), t
        t[b_ok_in] = t_in
        b_ok_in[b_ok_in] = b_ok
        b_ok = b_ok_in
    elif process == 'delete_inversions':
        b_ok &= b_ok_in
    # make initial shape: pad with constants of config. limits where data was removed because input is beyond this limits
    if is_time_filt:   # cfg_min_date and np.any(it_se != np.int64([0, date.size])):
        pad_width = (it_se[0], date.size - it_se[1])
        t = np.pad(
            t, pad_width,
            constant_values=np.array((cfg_in['min_date'], cfg_in['max_date']), 'M8[ns]').astype(np.int64)
        )
        b_ok = np.pad(b_ok, pad_width, constant_values=False)
    assert t.size == b_ok.size

    return pd.to_datetime(t, utc=True), b_ok


def get_version_via_com(filename):
    from win32com.client import Dispatch

    parser = Dispatch("Scripting.FileSystemObject")
    try:
        version = parser.GetFileVersion(filename)
    except Exception:
        return None
    return version


def plot_bad_time_in_thread(cfg_in, t: np.ndarray, b_ok=None, idel=None,
                            tim: Union[pd.Series, pd.Index, np.ndarray, None] = None,
                            tim_range: Optional[Tuple[Any, Any]] = None, path_save_image=None, msg='') -> None:
    """
    # Used instead of plt that can hang (have UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.)
    To can export png the chromedriver must be placed to c:\Programs\_net\Selenium\chromedriver{chrome_version[:2]}.exe
    On fail tries export html
    :param cfg_in:
    :param t: array of values to show their idel and b_ok parts, for example currently filtered part of tim, converted to array
    :param idel: indexes of t
    :param b_ok:
    :param tim: original input
    :param tim_range:
    :param path_save_image:
    :param msg:
    :return:
    """
    from bokeh.plotting import figure, output_file, show
    from bokeh.io import export_png, export_svgs
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    save_format_suffix = '.png'
    # output figure name
    fig_name = '{:%y%m%d_%H%M}-{:%H%M}'.format(*(
        tim_range if (tim_range is not None and tim_range[0]) else
        tim[[0, -1]] if isinstance(tim, pd.DatetimeIndex) else
        tim.iloc[[0, -1]] if isinstance(tim, pd.Series) else
        (x for x in tim[[0, -1]].astype('M8[m]').astype(datetime))))
    if path_save_image:
        path_save_image = Path(path_save_image)
        if not path_save_image.is_absolute():
            for p in ['path', 'text_path', 'file_cur']:
                p_use = cfg_in.get(p)
                if p_use:
                    break
            path_save_image = dir_create_if_need(Path(p_use).with_name(str(path_save_image)))
        fig_name = (path_save_image / fig_name).with_suffix(save_format_suffix)
        if fig_name.is_file():
            # work have done before
            return

    # prepare saving/exporting method
    if isinstance(fig_name, Path):
        lf.info('saving figure to {!s}', fig_name)
        if save_format_suffix != '.html':
            from selenium.common.exceptions import SessionNotCreatedException, WebDriverException, NoSuchDriverException

            try:
                # To png/svg
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                from selenium import webdriver

                chrome_options.binary_location = r'C:\Program Files (x86)\Slimjet\Slimjet.exe'
                # version = [get_version_via_com(p) for p in paths if p is not None][0]
                # chrome_version = '104'  # get_version_via_com(chrome_options.binary_location)  - not Chrome version

                # see chromium chrome://settings/help
                # (downdloaded from https://chromedriver.chromium.org/downloads)
                # os.environ["webdriver.chrome.driver"] = chrome_driver_path  # seems not works
                n_tries = range(2)
                e = None
                for k in n_tries:
                    chrome_driver_path = 'C:/Users/and0k/.wdm/drivers/chromedriver/win32/111.0.5563.41/chromedriver.exe'
                    *_, chrome_driver_path = list(Path(chrome_driver_path).parent.parent.glob('*'))
                    # rf'c:\Programs\_net\Selenium\chromedriver{chrome_version}.exe'
                    try:
                        # try:
                        #     # web_driver = webdriver.Edge()  # webdriver.Chrome()
                        #     web_driver = webdriver.Firefox()
                        # except NoSuchDriverException:
                        #     from selenium.webdriver.firefox.service import Service as FirefoxService
                        #     from webdriver_manager.firefox import GeckoDriverManager
                        #     web_driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()))


                        # Not works without getting ver
                        #
                        # from selenium.webdriver.edge.service import Service as EdgeService
                        # from webdriver_manager.microsoft import EdgeChromiumDriverManager
                        # web_driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()))

                        from webdriver_manager.core.os_manager import ChromeType  # old version: webdriver_manager.core.utils
                        from selenium.webdriver.chrome.service import Service as ChromeService

                        options = webdriver.ChromeOptions()
                        options.binary_location = r'C:\Program Files (x86)\Slimjet\slimjet.exe'
                        service = ChromeService(executable_path=chrome_driver_path)
                        try:
                            web_driver = webdriver.Chrome(service=service, options=options)
                        except SessionNotCreatedException as e:
                            lf.exception('')
                            import re
                            from webdriver_manager.chrome import ChromeDriverManager
                            from webdriver_manager.core.os_manager import PATTERN
                            version = re.search(rf'(?<=version is )[{PATTERN[ChromeType.CHROMIUM]}]*', e.msg).group()

                            chrome_driver_path = ChromeDriverManager(version=version).install()
                            service = ChromeService(executable_path=chrome_driver_path) # chrome_type=ChromeType.CHROMIUM
                            web_driver = webdriver.Chrome(service=service, options=options)

                    except WebDriverException as e:
                        lf.exception('')
                        # try old code
                        # service = Service(executable_path='C:\Program Files\Chrome Driver\chromedriver.exe')
                        # web_driver = webdriver.Chrome(service=service)
                        try:
                            web_driver = webdriver.Chrome(executable_path=chrome_driver_path, options=chrome_options)
                        except WebDriverException as e:  # "chrome not reachable" - may be hung or a while
                            try:
                                # retrieve right current browser version from error message
                                chrome_version = e.msg.split('Current browser version is ')[1].split('.')[0]
                                continue
                            except Exception as e:
                                pass
                        except TypeError as e:
                            print('old web_driver not works too')
                            return
                            # selenium.common.exceptions.NoSuchDriverException: Message: Unable to obtain C:\Users\and0k\.wdm\drivers\chromedriver\win32\111.0.5563.41 using Selenium Manager; Message: Unable to obtain working Selenium Manager binary; C:\Users\and0k\conda\envs\py3.11h5togrid\Lib\site-packages\selenium\webdriver\common\windows\selenium-manager.exe
                            # ; For documentation on this error, please visit: https://www.selenium.dev/documentation/webdriver/troubleshooting/errors/driver_location
                        msg_trace = '\n==> '.join((s for s in e.args if isinstance(s, str)))
                        sleep(1)
                        lf.error(f'{k}/{len(range(n_tries))}. {e.__class__}: {msg_trace}')
                        web_driver = None
                    except ImportError:
                        lf.exception('Skipping of figure creating because there is no needed package...')

            except SessionNotCreatedException:
                lf.exception('Can not save png so will save html instead')
                save_format_suffix = '.html'
                # todo: parse message to get Current browser version (or better method):
                # selenium.common.exceptions.SessionNotCreatedException: Message: session not created: This version of ChromeDriver only supports Chrome version 94
                # Current browser version is 104.0.5112.39 with binary path ...


        if save_format_suffix == '.html':
            # To static HTML file: big size but with interactive zoom, datashader?
            output_file(fig_name.with_suffix('.html'), title=msg)
            # web_driver.get("http://www.python.org")  # for testing


    # Create a new plot with a datetime axis type
    p = figure(width=1400, height=700, y_axis_type="datetime")   # old: plot_width=1400, plot_height=700  # plt.figure('Decreasing time corr')
    p.title.text = 'Decreasing time corr'

    # add renderers
    if idel is not None:
        p.circle(idel, pd.to_datetime(t[idel], utc=True), legend_label='deleting', size=4, color='magenta', alpha=0.8)
    p.line(np.arange(t.size), tim, legend_label='all', color='red', alpha=0.2)
    if b_ok is not None:
        p.line(np.flatnonzero(b_ok), pd.to_datetime(t[b_ok], utc=True), legend_label='good', color='green', alpha=0.8)


    # NEW: customize by setting attributes
    p.legend.location = "top_left"
    p.grid.grid_line_alpha = 0
    p.xaxis.axis_label = 'Counts'
    p.yaxis.axis_label = 'Date'
    p.ygrid.band_fill_color = "olive"
    p.ygrid.band_fill_alpha = 0.1

    # show(p)  # show the results

    # export figure
    if isinstance(fig_name, Path) and save_format_suffix != '.html':
        try:
            (export_png if save_format_suffix == '.png' else export_svgs)(
                p, filename=fig_name.with_suffix(save_format_suffix), webdriver=web_driver)
        except Exception as e:
            lf.exception('Can not save figure of bad source time detected')
