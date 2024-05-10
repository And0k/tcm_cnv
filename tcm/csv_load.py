import logging
import re
import sys
from collections import defaultdict
from functools import partial
from itertools import islice
from io import StringIO
from pathlib import Path, PurePath
import glob
from typing import (
    Any, AnyStr, Callable, Dict, Iterable, List, Mapping, Match, Optional, Union, Sequence, Tuple, TypeVar,
    BinaryIO, TextIO, Iterator
)
from operator import gt, lt

import numpy as np
import pandas as pd
# from pandas.tseries.frequencies import to_offset
# from yaml import safe_dump as yaml_safe_dump
# import cfg_dataclasses as cfg_d
# import incl_h5clc_hy
from . import (csv_specific_proc, utils_time_corr)
# import my scripts
# from to_pandas_hdf5.csv2h5 import main as csv2h5
from .utils2init import (
    Ex_nothing_done, ExitStatus, LoggingStyleAdapter, set_field_if_no, dir_create_if_need,
    this_prog_basename, init_logging, open_csv_or_archive_of_them, standard_error_info,  # open_csv_or_archive_of_them ...
    FakeContextIfOpen
)
lf = LoggingStyleAdapter(logging.getLogger(__name__))


def init_input_cols(cfg_in=None):
    """
        Append/modify dictionary cfg_in for parameters of dask/pandas load_csv() function and of save to hdf5.
    :param cfg_in: dictionary, may has fields:
        header (required if no 'cols') - comma/space separated string. Column names in source file data header
        (as in Veusz standard input dialog), used to find cfg_in['cols'] if last is not cpecified
        dtype - type of data in column (as in Numpy loadtxt)
        converters - dict (see "converters" in Numpy loadtxt) or function(cfg_in) to make dict here
        cols_load - list of used column names

    :return: modified cfg_in dictionary. Will have fields:
        cols - list constructed from header by spit and remove format cpecifiers: '(text)', '(float)', '(time)'
        cols_load - list[int], indexes of ``cols`` in needed to save order
        coltime/coldate - assigned to index of 'Time'/'Date' column
        dtype: numpy.dtype of data after using loading function but before filtering/calculating fields
            numpy.float64 - default and for '(float)' format specifier
            numpy string with length cfg_in['max_text_width'] - for '(text)'
            datetime64[ns] - for coldate column (or coltime if no coldate) and for '(time)'
        col_index_name - index name for saving Pandas frame. Will be set to name of cfg_in['coltime'] column if not exist already
        used in main() default time postload proc only (if no specific loader which calculates and returns time column for index)
        cols_loaded_save_b - columns mask of cols_load to save (some columns needed only before save to calulate
        of others). Default: excluded (text) columns and index and coldate
        (because index saved in other variable and coldate may only used to create it)

    Example
    -------
    header= u'`Ensemble #`,txtYY_M_D_h_m_s_f(text),,,Top,`Average Heading (degrees)`,`Average Pitch (degrees)`,stdPitch,`Average Roll (degrees)`,stdRoll,`Average Temp (degrees C)`,txtu_none(text) txtv_none(text) txtVup(text) txtErrVhor(text) txtInt1(text) txtInt2(text) txtInt3(text) txtInt4(text) txtCor1(text) txtCor2(text) txtCor3(text) txtCor4(text),,,SpeedE_BT SpeedN_BT SpeedUp ErrSpeed DepthReading `Bin Size (m)` `Bin 1 Distance(m;>0=up;<0=down)` absorption IntScale'.strip()
    """

    if cfg_in is None: cfg_in = dict()
    set_field_if_no(cfg_in, 'max_text_width', 2000)

    dtype_text_max = '|S{:.0f}'.format(cfg_in['max_text_width'])  # '2000 #np.str

    if cfg_in.get('header'):  # if header specified
        re_sep = ' *(?:(?:,\n)|[\n, ]) *'  # process ",," right
        cfg_in['cols'] = re.split(re_sep, cfg_in['header'])
        # re_fast = re.compile(u"(?:[ \n,]+[ \n]*|^)(`[^`]+`|[^`,\n ]*)", re.VERBOSE)
        # cfg_in['cols']= re_fast.findall(cfg_in['header'])
    elif 'cols' not in cfg_in:  # cols is from header, is specified or is default
        KeyError('no "cols" no "header" in config.in')
        cfg_in['cols'] = ('stime', 'latitude', 'longitude')

    # Default parameters dependent from ['cols']
    cols_load_b = np.ones(len(cfg_in['cols']), np.bool_)
    set_field_if_no(cfg_in, 'comments', '"')

    # assign data type of input columns
    b_was_no_dtype = 'dtype' not in cfg_in
    if b_was_no_dtype:
        cfg_in['dtype'] = np.array([np.float64] * len(cfg_in['cols']))
        # 32 gets trunkation errors after 6th sign (=> shows long numbers after dot)
    elif isinstance(cfg_in['dtype'], str):
        cfg_in['dtype'] = np.array([np.dtype(cfg_in['dtype'])] * len(cfg_in['cols']))
    elif isinstance(cfg_in['dtype'], list):
        # prevent numpy array(list) guess minimal dtype because dtype of dtype_text_max may be greater
        numpy_cur_dtype = np.min_scalar_type(cfg_in['dtype'])
        numpy_cur_dtype_len = numpy_cur_dtype.itemsize / np.dtype((numpy_cur_dtype.kind, 1)).itemsize
        cfg_in['dtype'] = np.array(cfg_in['dtype'], '|S{:.0f}'.format(
            max(len(dtype_text_max), numpy_cur_dtype_len)))

    for col, col_name in (['coltime', 'Time'], ['coldate', 'Date']):
        if col not in cfg_in:
            # if cfg['col(time/date)'] is not provided try find 'Time'/'Date' column name
            if not (col_name in cfg_in['cols']):
                col_name = col_name + '(text)'
            if not (col_name in cfg_in['cols']):
                continue
            cfg_in[col] = cfg_in['cols'].index(col_name)  # assign 'Time'/'Date' column index to cfg['col(time/date)']
        elif isinstance(cfg_in[col], str):
            cfg_in[col] = cfg_in['cols'].index(cfg_in[col])

    if 'converters' not in cfg_in:
        cfg_in['converters'] = None
    elif cfg_in['converters']:
        if not isinstance(cfg_in['converters'], dict):
            # suspended evaluation required
            cfg_in['converters'] = cfg_in['converters'](cfg_in)
        if b_was_no_dtype:
            # converters produce datetime64[ns] for coldate column (or coltime if no coldate):
            cfg_in['dtype'][cfg_in['coldate' if 'coldate' in cfg_in else 'coltime']] = 'datetime64[ns]'

    # process format cpecifiers: '(text)','(float)','(time)' and remove it from ['cols'],
    # also find not used cols cpecified by skipping name between commas like in 'col1,,,col4'
    for i, s in enumerate(cfg_in['cols']):
        if len(s) == 0:
            cols_load_b[i] = 0
            cfg_in['cols'][i] = f'NotUsed{i}'
        else:
            b_i_not_in_converters = (not (i in cfg_in['converters'].keys())) \
                if cfg_in['converters'] else True
            i_suffix = s.rfind('(text)')
            if i_suffix > 0:  # text
                cfg_in['cols'][i] = s[:i_suffix]
                if (cfg_in['dtype'][
                        i] == np.float64) and b_i_not_in_converters:  # reassign from default float64 to text
                    cfg_in['dtype'][i] = dtype_text_max
            else:
                i_suffix = s.rfind('(float)')
                if i_suffix > 0:  # float
                    cfg_in['cols'][i] = s[:i_suffix]
                    if b_i_not_in_converters:
                        # assign to default. Already done?
                        assert cfg_in['dtype'][i] == np.float64
                else:
                    i_suffix = s.rfind('(time)')
                    if i_suffix > 0:
                        cfg_in['cols'][i] = s[:i_suffix]
                        if (cfg_in['dtype'][i] == np.float64) and b_i_not_in_converters:
                            cfg_in['dtype'][i] = 'datetime64[ns]'  # np.str

    if any(cfg_in.get('cols_load', [])):
        cols_load_b &= np.isin(cfg_in['cols'], cfg_in['cols_load'])
    else:
        cfg_in['cols_load'] = np.array(cfg_in['cols'])[cols_load_b]

    col_names_out = cfg_in['cols_load'].copy()
    # Convert ``cols_load`` to index (to be compatible with numpy loadtxt()), names will be in cfg_in['dtype'].names
    cfg_in['cols_load'] = np.int32([
        cfg_in['cols'].index(c) for c in cfg_in['cols_load'] if c in cfg_in['cols']
        ])
    # not_cols_load = np.array([n in cfg_in['cols_not_save'] for n in cfg_in['cols']], np.bool_)
    # cfg_in['cols_load']= np.logical_and(~not_cols_load, cfg_in['cols_load'])
    # cfg_in['cols']= np.array(cfg_in['cols'])[cfg_in['cols_load']]
    # cfg_in['dtype']=  cfg_in['dtype'][cfg_in['cols_load']]
    # cfg_in['cols_load']= np.flatnonzero(cfg_in['cols_load'])
    # cfg_in['dtype']= np.dtype({'names': cfg_in['cols'].tolist(), 'formats': cfg_in['dtype'].tolist()})

    cfg_in['cols'] = np.array(cfg_in['cols'])
    cfg_in['dtype_raw'] = np.dtype({'names': cfg_in['cols'],
                                    'formats': cfg_in['dtype'].tolist()})
    cfg_in['dtype'] = np.dtype({
        'names': cfg_in['cols'][cfg_in['cols_load']],
        'formats': cfg_in['dtype'][cfg_in['cols_load']].tolist()
        })

    # Get index name for saving Pandas frame
    b_index_exist = cfg_in.get('coltime') is not None
    if b_index_exist:
        set_field_if_no(cfg_in, 'col_index_name', cfg_in['cols'][cfg_in['coltime']])

    # Mask of only needed output columns

    # Output columns mask
    if 'cols_loaded_save_b' in cfg_in:  # list to array
        cfg_in['cols_loaded_save_b'] = np.bool_(cfg_in['cols_loaded_save_b'])
    else:
        cfg_in['cols_loaded_save_b'] = np.logical_not(np.array(
            [cfg_in['dtype'].fields[n][0].char == 'S' for n in
             cfg_in['dtype'].names]))  # a.dtype will = cfg_in['dtype']

        if 'coldate' in cfg_in:
            cfg_in['cols_loaded_save_b'][
                cfg_in['dtype'].names.index(
                    cfg_in['cols'][cfg_in['coldate']])] = False

    # Exclude index from cols_loaded_save_b
    if b_index_exist and cfg_in['col_index_name']:
        cfg_in['cols_loaded_save_b'][cfg_in['dtype'].names.index(
            cfg_in['col_index_name'])] = False  # (must index be used separately?)

    if 'cols_not_save' in cfg_in:
        b_cols_load_in_used = np.isin(
            cfg_in['dtype'].names, cfg_in['cols_not_save'], invert=True)
        if not np.all(b_cols_load_in_used):
            cfg_in['cols_loaded_save_b'] &= b_cols_load_in_used

    # Output columns dtype
    col_names_out = np.array(col_names_out)[cfg_in['cols_loaded_save_b']].tolist() + cfg_in.get('cols_save', [])
    cfg_in['dtype_out'] = np.dtype({
        'formats': [cfg_in['dtype'].fields[n][0] if n in cfg_in['dtype'].names else
                    np.dtype(np.float64) for n in col_names_out],
        'names': col_names_out})

    return cfg_in


def csv_process(
    df: pd.DataFrame, cfg_in: Mapping[str, Any], t_prev=None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """

    :param df:
    :param cfg_in:
    :param t_prev: will be prepended to df.Time before time filtering and removed after
    :return:
    """
    utils_time_corr.tim_min_save = pd.Timestamp('now', tz='UTC')  # initialisation for time_corr_df()
    utils_time_corr.tim_max_save = pd.Timestamp(0, tz='UTC')
    meta_time_and_mask = {'Time': 'datetime64[ns, utc]', 'b_ok': np.bool_}
    n_overlap = 2 * int(np.ceil(cfg_in['fs'])) if cfg_in.get('fs') else 50
    b_overlap = t_prev is not None
    # Process df and get date in ISO string or numpy standard format
    try:
        try:
            get_out_types = cfg_in['fun_proc_loaded'].meta_out
        except AttributeError:
            lf.debug('time correction...')                           # only Time column
            date = cfg_in['fun_proc_loaded'](df, cfg_in)

        else:  # get_out_types is not None => fun_proc_loaded() will return full data DataFrame
            lf.debug('processing csv data with time correction...')  # not only Time column
            if callable(get_out_types):
                meta_out = get_out_types([(k, v[0]) for k, v in cfg_in['dtype'].fields.items()])
            df = cfg_in['fun_proc_loaded'](df, cfg_in)
            date = df.Time

        if b_overlap:
            date = pd.concat([t_prev, date])
        time_cor, b_ok = utils_time_corr.time_corr(date, cfg_in)  # may be long
        if b_overlap:
            _ = t_prev.shape[0]
            time_cor = time_cor[_:]
            b_ok = b_ok[_:]

        # functions defined under csv_specific_param dict keys 'fun' and 'add' was added to cfg_in['fun_proc_loaded'], so this not needed:
        # if cfg_in.get('csv_specific_param'):
        #     df = csv_specific_proc.loaded_corr(df, cfg_in, cfg_in['csv_specific_param'])
    except IndexError:
        lf.warning('No data?')
        return None

    # # Show message # commented as it shows many non-increased time rows set as bad
    # nbad_time = len(b_ok) - b_ok.sum()
    # if nbad_time:
    #     lf.info(
    #         'Bad time values ({:d}): {}{:s}',
    #         nbad_time, time_cor[~(b_ok|time_cor.isna())].to_numpy()[:20],
    #         ' (shows first 20)' if nbad_time > 20 else ''
    #     )

    # Filter data that is out of config range (after range_message()' args defined as both uses df_time_ok['b_ok'])
    for date_lim, op in [('min_date', lt), ('max_date', gt)]:  # todo: time_range also
        date_lim = cfg_in.get(date_lim)
        if not date_lim:
            continue
        b_ok[op(time_cor, pd.Timestamp(date_lim, tz=None if date_lim.tzinfo else 'UTC'))] = False

    t_prev = df.loc[df.index[-n_overlap:], 'Time']
    # Removing rows with bad time
    df_filt = df.loc[b_ok, list(cfg_in['dtype_out'].names)]
    time_filt = time_cor[b_ok]
    df_filt = df_filt.set_index(time_filt).rename_axis('Time')

    if False and __debug__:
        len_out = len(time_filt)
        print('out data length:', len_out)
        # print('index_limits:', df_time_ok.divisions[0::df_time_ok.npartitions])
        sum_na_out, df_time_ok_time_min, df_time_ok_time_max = (
            time_filt.notnull().sum(), time_filt.min(), time_filt.max())
        print('out data len, nontna, min, max:', sum_na_out, df_time_ok_time_min, df_time_ok_time_max)

    return df_filt, t_prev


def range_message(range_source, n_ok_time, n_all_rows):
    t = range_source() if isinstance(range_source, Callable) else range_source
    lf.info('loaded source range: {:s}',
            f'{t["min"]:%Y-%m-%d %H:%M:%S} - {t["max"]:%Y-%m-%d %H:%M:%S %Z}, {n_ok_time:d}/{n_all_rows:d} rows')


def csv_read_gen(paths: Sequence[Union[str, Path]],
                 edge_rows_only: bool = False,
                 **cfg_in: Mapping[str, Any]
    ) -> Iterator[Tuple[int, Path, Optional[pd.DataFrame]]]:
    """
    Reads csv to pandas DataFrame in chunks
    Calls `cfg_in['fun_proc_loaded']()` (if specified)
    Calls `time_corr()`: corrects/checks Time (with arguments defined in cfg_in fields)
    Sets Time as output dataframe index
    :param paths: list of file names
    :param edge_rows_only: returned `df` will have only edge csv rows and other rows will not be read from csv
    :param cfg_in: contains fields for arguments of `pandas.read_csv()` correspondence:
    - names=cfg_in['cols'][cfg_in['cols_load']]
    - usecols=cfg_in['cols_load']
    - on_bad_lines=cfg_in['on_bad_lines']
    - comment=cfg_in['comment']
    - chunksize=cfg_in['blocksize']
    Other arguments corresponds to fields with same name:
    - dtype=cfg_in['dtype']
    - delimiter=cfg_in['delimiter']
    - converters=cfg_in['converters']
    - skiprows=cfg_in['skiprows']

    Also cfg_in has fields:
    - dtype_out: numpy.dtype, which "names" field used to detrmine output columns
    - fun_proc_loaded: None or Callable[
    [Union[pd.DataFrame, np.array], Mapping[str, Any], Optional[Mapping[str, Any]]],

        Union[pd.DataFrame, pd.DatetimeIndex]]
    If it returns pd.DataFrame then it also must have attribute meta_out:
        Callable[[np.dtype, Iterable[str], Mapping[str, dtype]], Dict[str, np.dtype]]

    See also `time_corr()` for used fields


    :yield: tuple (i1_path, i_chunk, path, df_filt) where
    - i1_path: 1-based counter of yielded data,
    - i_chunk: 0-based counter of chunks,
    - path: file name,
    - df_filt: dataframe with time index and only columns listed in `cfg_in['dtype_out']`.names
    """

    read_csv_args_to_cfg_in = {
        'dtype': 'dtype_raw',
        'names': 'cols',
        'on_bad_lines': 'on_bad_lines',
        'comment': 'None',
        'delimiter': 'delimiter',
        'converters': 'converters',
        'skiprows': 'skiprows',
        'chunksize': 'blocksize',  # load in chunks
        'encoding': 'encoding'
        }
    read_csv_args = {arg: cfg_in[key] for arg, key in read_csv_args_to_cfg_in.items() if key in cfg_in}
    read_csv_args.update({
        'skipinitialspace': True,
        'usecols': cfg_in['dtype'].names,
        'header': None})
    # removing "ParserWarning: Both a converter and dtype were specified for column k - only the converter will be used"
    if read_csv_args['converters']:
        read_csv_args['dtype'] = {k: v[0] for i, (k, v) in enumerate(read_csv_args['dtype'].fields.items()) if i not in read_csv_args['converters']}

    t_prev = None
    for i1_path, path in enumerate(paths, start=1):
        # Save params that may be need in `csv_process()` (to extract date)
        cfg_in['file_cur'] = Path(path)
        cfg_in['file_stem'] = cfg_in['file_cur'].stem
        cfg_in['n_rows'] = 0  # number of file rows loaded
        cfg_in['time_raw_st'] = None  # can be set after time is calculated (in csv_specific_proc)
        for i_retry in [False, True]:  # try again on ParserError with tuned loading parameters
            try:
                if edge_rows_only:
                    # Yield only 1st and last rows (metadata)

                    # Remove `skiprows` `pandas.read_csv()` parameter as we'll skip rows other way
                    read_csv_args_ = read_csv_args.copy()
                    skip_rows, read_csv_args_['skiprows'] = \
                        read_csv_args.get('skiprows', 0), 0
                    # Remove `chunksize` as we will load only several rows
                    try:
                        del read_csv_args_['chunksize']
                    except KeyError:
                        pass

                    file_size = path.stat().st_size
                    with open(path, 'r') as f:
                        # Read the first row and determine its length skipping (less than 5) empty lines
                        for line in islice(f, skip_rows, skip_rows + 5):
                            len_line = len(line)
                            if len_line:  # this is not needed: bad lines have been deleted in precorrection
                                break
                        df_1st_row = pd.read_csv(
                            StringIO(line), **read_csv_args_, index_col=False)
                        f.seek(file_size - len_line*5)  # seek to the last rows
                        _ = next(f)  # skip row that most probably red not from start

                        # Read the last rows
                        df_last_rows = pd.read_csv(f, **read_csv_args_, index_col=False)

                    df = pd.concat([df_1st_row, df_last_rows.iloc[[-1]]])
                    df_filt, t_prev = csv_process(df, cfg_in, t_prev)
                    cfg_in['n_rows'] = float('nan')  # we don't know number of rows available
                    yield i1_path, 0, path, df_filt  # i_chunk = 0 (we not use chunks)
                else:
                    # Normal loading

                    for i_chunk, df in enumerate(pd.read_csv(path, **read_csv_args, index_col=False)):
                        df_filt, t_prev = csv_process(df, cfg_in, t_prev)
                        cfg_in['n_rows'] += len(df)
                        if df_filt is None:
                            continue
                        yield i1_path, i_chunk, path, df_filt
                break
            except pd.errors.ParserError:  # for example NotImplementedError if bad file
                if (read_csv_args['on_bad_lines'] not in ('warn', 'skip')
                    or read_csv_args.get('engine') != 'python'
                    ):
                    lf.exception('Trying set [in].on_bad_lines = "warn" and retry\n')
                    read_csv_args['on_bad_lines'] = 'warn'
                    read_csv_args['engine'] = 'python'
                else:
                    lf.exception('Failed loading file')

            # lf.exception('If file "{}" have no data try to delete it?', paths)

# #############################################################################################################
# def text_type(file_stem):
#     """
#     Type of file (probe).
#     :param file_stem: file name without extension
#     :return: 'p' if file_stem is like "INKL_P01"
#     """
#     return 'p' if 'P' in file_stem[5] else 'i'


def config_model(text_type, get_regex_only=False) -> Union[bytes, dict[str, Any]]:
    """
    Parameters to correct and (optionally) load text file (csv) data
    :param text_type: one letter among (i, p) to get corresponding parameters (for any other returns None).
    todo: Implement for b, d, w.
    :param get_regex_only: get only values of 'text_line_regex' field of normally returned dict
    :return: dict with fields:
    - text_line_regex: regex to check / correct lines of text data (see csv_specific_proc.correct_txt())
    - header:
    - dtype:
    or only its 'text_line_regex' field value if ``get_regex_only`` is True
    """
    if text_type is None:
        return {}
    if text_type not in ('i', 'p', 'b', 'd', 'w'):
        raise TypeError('Probe model not recognised!')
    b_default_type = text_type in ('i', '', 'b')
    text_line_regex = b'^(?P<use>' + csv_specific_proc.century + br'\d{2}(,\d{1,2}){5}' + (  # date & time
        br'(,\-?\d{1,6}){6},\d{1,2}(\.\d{1,2})?,\-?\d{1,3}(\.\d{1,2})?).*' if b_default_type else
        br'(,\-?\d{1,6}){6}(,\-?\d{1,8}),\d{1,2}(\.\d{1,2})?,\-?\d{1,3}(\.\d{1,2})?).*'  # if text_type == 'p'
        # Ax,y,z,Mx,y,z,ADC,T,Bat
    )                                   # ,160,-2576,-13808,-111,62,547
        # br'(,\-?\d{1,2}\.\d{2}){2}).*' if text_type == 'w'           # ,1849841,21.63,5.42

    if get_regex_only:
        return text_line_regex

    cfg_for_type = {
        'text_line_regex': text_line_regex,
        'header': 'yyyy(text),mm(text),dd(text),HH(text),MM(text),SS(text),Ax,Ay,Az,Mx,My,Mz' + (
            ',Battery,Temp' if b_default_type else ',P_counts,Temp,Battery'),
        'dtype': '|S4 |S2 |S2 |S2 |S2 |S2 i2 i2 i2 i2 i2 i2 f8 f8'.split() + ([] if b_default_type else ['f8']),
    }
    return cfg_for_type


raw_pattern_default = '*{prefix:}{number:0>2}*.[tT][xX][tT]'


def correct_files(
    raw_parent, raw_subdir='',
    raw_pattern=raw_pattern_default,
    prefix='I*',  # 'INKL_'
    probes: Optional[Iterable[Any]] = None,
    mod_name=partial(csv_specific_proc.mod_name, add_prefix='@'),
    fun_correct=None,
    sub_str_list: Optional[Union[Callable[[str], Union[str, bytes]], Sequence[Union[str, bytes]]]] = None,
    **kwargs
) -> Dict[Tuple[str, str], list[Path]]:
    """
    Find and correct raw files for specified probes. If file for probe is not a path/mask then search corrected file.
    :param raw_parent:
    :param raw_subdir: Optional subdir or zip/rar archive name (data will be unpacked) in "path_cruise/_raw".
    Symbols * and ? are not supported for directories (to load from many directories/archives) but `prefix` and `number`
    format strings are available instead.
    :param raw_pattern: mask to find raw files: Python "format" command pattern to format with {prefix} and {number}
        placeholders (where prefix is a ``prefix`` argument and number will be one of specified in ``probes`` arg).
        To match files in archive the subdirectories can be prepended with "/" but files will be unpacked with flattened
        path in created dir with name constructed from path in archive
    :param prefix: word, that should be in raw input file
    :param probes: iterable of probes identifiers. Can be
    - digit: if only one device type for each prober number (include device type in prefix).
    - {model}{number}: {model} should not contain digits, do not repeat model in ``prefix``.
    - None: we use 2-digit numbers from '01' to '99'
    :param fun_correct: callable returning Path of corrected file with arguments (as in csv_specific_proc.correct_txt()
     except ``sub_str_list`` argument):
        file_in,
        dir_out: ,
        binary_mode=False,
        mod_file_name
    :param sub_str_list: ``sub_str_list`` argument of fun_correct() or function to get it from file name (
    see correct_txt())
    :param mod_name: function to get probe models and names of corrected raw files from source raw file names. Default:
        csv_specific_proc.mod_name(x, add_prefix='@')
    :param kwargs: not used
    :return: probes_found: dict with:
    - keys: tuples (model, pid) where:
      - model: model abbreviation (one letter among i, b, d, p, w) or '' if not recognized
      - pid: probe identifier = item from input probes: model and numder
    - values: list of corrected files paths for probes
    """

    # counter of processed files
    n_files_total = 0
    # collect files for each pid here
    probes_found = defaultdict(list)

    if '{number' not in raw_pattern and '{id}' not in raw_pattern:
        if not probes:
            pattern_name_parts = csv_specific_proc.parse_name(raw_pattern.split('.', maxsplit=1)[0])
            if pattern_name_parts is None:
                probes = [None]
            else:
                probes = [int(pattern_name_parts['number'])]
        else:
            lf.warning(' '.join([
                'No {number} or {id} part in in.raw_pattern configuration, so all matched files will be assigned to ',
                f'each of specified {len(probes)} probes number' if len(probes) > 1
                else f'specified probe "{probes[0]}"'
            ]))
    # Try determine if we use arcive (for now without account for specific probe subdir we deteimine laner)
    arc_supported_suffixes = ('.zip', '.rar')
    arc_suffix_ = raw_parent.suffix.lower()
    arc_suffix_ = [sfx for sfx in arc_supported_suffixes if sfx == arc_suffix_]
    arc_suffix_ = arc_suffix_[0] if arc_suffix_ else None
    for pid in (range(1, 100) if probes is None else probes):
        # Search files matched `subdir`/`pattern` matched current probe `pid` or its parts:
        if isinstance(pid, int):
            model = '[bdp]'
            number = f'{pid:0>2}'
            pid = f'[0bdp]{number}'
        elif pid:
            model = pid.rstrip('0123456789')
            number = pid[len(model):]
        else:
            model = None
            number = None
        # Fill patten's placeholders `prefix`, `model`, `number`, `id`:
        # prefix_model = f"{prefix}{model}*"  #  if not prefix or prefix[-1] not in 'ibdpw' else '*'
        subdir = raw_subdir.format(prefix=prefix, model=model, number=number, id=pid)
        pattern = raw_pattern.format(prefix=prefix, model=model, number=number, id=pid)

        # Search files for each probe sequentially (in dir specific to probe)
        for cur_dir in [raw_parent] if arc_suffix_ else raw_parent.glob(subdir):
            if arc_suffix_:
                arc_suffix = arc_suffix_
            else:
                arc_suffix = cur_dir.suffix.lower()
                arc_suffix = [sfx for sfx in arc_supported_suffixes if sfx == arc_suffix]
                arc_suffix = arc_suffix[0] if arc_suffix else None

            paths_not_corr = []
            if arc_suffix:
                # We will open archive and search inside later if no corrected files will be found in dir

                # Corrected raw files output dir name if input is archive (for fun_correct())
                _ = f"{arc_suffix[1:]}^{str(cur_dir.stem).replace('/', '_')}"
                dir_cor_path = raw_parent / _

                glob_processed_in_arc, pattern_to_get_corr_if_arc = \
                    pattern, str(Path(glob.escape(dir_cor_path)) / pattern)
                if not glob_processed_in_arc[0] == '@':
                    glob_processed_in_arc = f'@{glob_processed_in_arc}'
            else:
                # Search inside current probe raw subdir or use provided path to raw file
                if cur_dir.is_dir():
                    paths_not_corr = list(
                        file_in for file_in in cur_dir.glob(pattern) if not file_in.stem.startswith('@')
                    )
                    dir_cor_path = cur_dir
                else:
                    dir_cor_path = raw_parent
                    # else:
                #     parent_path = ''  # no more search raw files here for current pid
                glob_processed_in_arc = '@*'  # not compare name from archive

            # If already have corrected files for pid with glob name = fun_correct_name(pattern) then use them
            model, glob_corrected = mod_name(
                pattern_to_get_corr_if_arc if arc_suffix else pattern, parce=bool(pid))
            paths = list((dir_cor_path if arc_suffix else cur_dir).glob(glob_corrected.name))
            # or (raw_parent / cur_dir).glob(f"@{prefix_model}{number:0>2}.txt"  # f"{tbl}.txt"
            if paths:
                lf.info('Corrected csv files found:\n{:s}', ',\n'.join(r.name for r in paths))
                if paths_not_corr:
                    # keep only that paths that has no corresponding corrected path
                    _, paths_not_corr = paths_not_corr, []
                    for p in _:
                        __, p_cor = mod_name(p, parce=False)
                        if p_cor not in paths:
                            paths_not_corr.append(p)
                    if paths_not_corr:
                        lf.info('Not corrected yet:\n{:s}', ',\n'.join(r.name for r in paths_not_corr))
            n_files = 0  # files for pid found
            for file_in in (
                (paths + paths_not_corr) or open_csv_or_archive_of_them(raw_parent / subdir, pattern=pattern)
            ):
                model, file = mod_name(file_in.name, parce=bool(pid))
                # (".name" is required for mod_name() if file_in is TextIOWrapper)
                file = fun_correct(
                    file_in,
                    dir_out=dir_cor_path, binary_mode=False,
                    mod_file_name=lambda _: file,
                    sub_str_list=(
                        sub_str_list(model) if callable(sub_str_list) else
                        # If file_in is already converted file in archive just extracts to output dir:
                        None if PurePath(file_in.name).match(glob_processed_in_arc) else
                        sub_str_list
                    )
                )
                n_files += 1
                for model_number, files in probes_found.items():
                    if file in files:
                        if model_number == (model, number):
                            break
                        raise FileExistsError(''.join([
                            'Going to process same file again as another probe/data part! ',
                            'Maximum one file should be matched for each probe number: correct input file '
                            f'name pattern (current raw_pattern={raw_pattern}) or rename files. ',
                            'Note: if You set path of input text file as dir, then default raw_pattern = '
                            f'{raw_pattern_default} will be used which is usually Ok'
                            if raw_pattern != raw_pattern_default else ''
                        ]))
                else:
                    probes_found[(model, number)].append(file)
            else:
                if n_files == 0:
                    (lf.debug if probes is None else lf.warning)('No {:s} files found', pattern)
                    continue
            n_files_total += n_files

    lf.info(
        'Raw files {:s}for {:d} probes found:\n{:s}',
        f'({n_files_total}) ' if n_files_total != len(probes_found) else '', len(probes_found),
        '\n'.join('{}{}: {}'.format(m, n, ', '.join(f'{f!s}' for f in ff)) for (m, n), ff in probes_found.items())
    )
    return probes_found


#############################################################################################################
# Inclinometer file format and other parameters
cfg = {
    'in':      {
        'fun_proc_loaded':    csv_specific_proc.loaded_tcm,  # function to convert csv to dask dataframe
        'delimiter':          ',',  # \t not specify if need "None" useful for fixed length format
        'skiprows':           3,  # ignore this number of top rows both in preliminary correction and read_csv
        'on_bad_lines':       'warn',  #'error',
        # '--min_date', '07.10.2017 11:04:00',  # not output data < min_date
        # '--max_date', '29.12.2018 00:00:00',  # UTC, not output data > max_date
        'blocksize':          1_000_000,  # 5_000_000  # 15_000_000 hangs my comp
        'b_interact':         '0',
        'csv_specific_param': {
            'invert_magnetometer': True,
        # Bad time correction
        #     'time_shift': {
        #         'dt0': '0s',
        #         'time_st': None,                    # needed start time or it will be taken from existed bad time
        #         'time_en': '2023-07-24T13:10:00',   # required
        #         'time_raw_en': '2023-07-19T23:31:10',    # required if not linear_len and not time_raw_en
        #         'dt_end': None,  # can specify this (time_en - time_raw_en) interval instead time_en & bad
        #
        #         'linear_len': 16404000  # replace time using linear increased values of this length
        #         # (instead of linear transformation of existed values)
        #     }

        },
        'dt_max_interp_err': pd.Timedelta('15s'),   # 11s = (1.5s)*(time_en - time_st)/(time_end_bad - time_st)

        'dt_interp_between': pd.Timedelta('1.5s'),  # default
        'encoding':           'CP1251',  # read_csv() encoding parameter
        'max_text_width':     1000,  # used here to get sample for dask that should be > max possible row length
        # '--dt_from_utc_seconds', str(cfg['in']['dt_from_utc'][probe].total_seconds()),
        # '--fs_float', str(p_type[cfg['in']['probes_prefix']]['fs']),  # f'{fs(probe, file_in.stem)}',
        'corr_time_mode': True,  # to make sorted index: required to can process loaded data as timeseries by dask
        'text_type': None,
        'text_line_regex': None,
        'dt_from_utc': 0
    },
    'out':     {},
    'filter':  {},
    'program': {
        'log':     'tcm_csv.log',
        'verbose': 'INFO',
        # 'dask_scheduler': 'synchronous'
    }
    # Warning! If True and b_incremental_update= True then not replace temporary file with result before proc.
    # '--log', 'log/csv2h5_inclin_Kondrashov.log'  # log operations
}


def load_from_csv_gen(
        cfg_in: Mapping[str, Any],
        cfg_in_probe: Optional[Mapping[str, Any]] = None,
        return_: str = ''
) -> Iterator[Tuple[int, Any, Path, pd.DataFrame]]:
    """
    Yields metadata and data for each discovered paths ordered by probe.
    filter, correct, and group data files based on probe identifiers. The order of processing is determined by the order of probes specified in the probes parameter or by default numeric sequencing when probes is not provided



       as represented by the paths_csv list in the output tuple.


    pandas dataframes loaded from dir/mask csv-files
    :param cfg_in: subdict of global `cfg`: cfg['in']. Required field:
      - path: input directory/file mask
    Other fields are optional (see global cfg). For example, overwrite this field:
      - fun_proc_loaded: function to convert csv to dask dataframe
      - text_type: to specify cfg_in fields 'text_line_regex', 'header', 'dtype' for known probes
        (specified in config_model()). If probe not known then set these fields keeping
        default text_type = None
      ...
    :param cfg_in_probe: optional dict with fields named equal to probe(s) {pid}(s) to update cfg_in
     in the cycle of loading probe data with settings specified to current probe
    :param return_:
    - None: iterator will yield all data
    - edge_rows_only: yielded `df` will have only edge csv rows and other rows will not be read from csv
    - files_list: yielded `df` will be None
    :return: iterator of (i1_probe, probe, paths_csv, df):
    - i1_probe: int, 1-based index;
    - probe: f'{text_type}{probe}';
    - paths_csv: list of all Paths found for probe;
    - loaded probe data (may be part)
    """

    global cfg, lf
    lf = LoggingStyleAdapter(init_logging(__name__, cfg['program']['log'], cfg['program']['verbose']))

    dir_csv_cor = None  # dir for cleaned (intermediate) raw files, None => same as cfg['in']['dir']

    # other fields init
    cfg['in'].update(cfg_in)
    # Prepare loading and processing by rad_csv() specific to raw data format

    # - extract date from file name if needed
    if cfg['in'].get('fun_date_from_filename') and isinstance(cfg['in']['fun_date_from_filename'], str):
        cfg['in']['fun_date_from_filename'] = eval(
            compile("lambda file_stem, century=None: {}".format(cfg['in']['fun_date_from_filename']), '', 'eval'))

    # - additional calculation in read_csv() if needed
    if cfg['in'].get('fun_proc_loaded') is None:
        # Default time processing after loading by dask/pandas.read_csv()
        if 'coldate' not in cfg['in']:  # if Time includes Date then we will just return it
            cfg['in']['fun_proc_loaded'] = lambda a, cfg_in, dummy=None: a[cfg_in['col_index_name']]
        else:                           # else will return Time + Date
            cfg['in']['fun_proc_loaded'] = lambda a, cfg_in, dummy=None: a['Date'] + np.array(
                np.int32(1000 * a[cfg_in['col_index_name']]), dtype='m8[ms]')

    if cfg['in']['csv_specific_param']:
        # Split 'csv_specific_param' fields into two parts for :
        # 1. loaded_corr() - oneliner operations ('fun', 'add'), and rest embed into
        # 2. cfg['in']['fun_proc_loaded']()
        arg_loaded_corr = {}
        arg_fun_proc_loaded = {}
        for k, v in cfg['in']['csv_specific_param'].items():
            (arg_loaded_corr if k.rsplit('_', 1)[-1] in ('fun', 'add') else arg_fun_proc_loaded)[k] = v
        arg_fun_proc_loaded = {'csv_specific_param': arg_fun_proc_loaded} if arg_fun_proc_loaded else {}
        arg_loaded_corr = {'csv_specific_param': arg_loaded_corr} if arg_loaded_corr else {}

        # Incorporate these two types of custom operations in cfg['in']['fun_proc_loaded'] function
        # to be executed in our read_csv() without extra arguments
        fun_proc_loaded = cfg['in']['fun_proc_loaded']

        def fun_loaded_and_loaded_corr(a, cfg_in):
            _ = fun_proc_loaded(a, cfg_in=cfg_in, **arg_fun_proc_loaded)
            b = csv_specific_proc.loaded_corr(_, cfg_in, **arg_loaded_corr)
            return b

        try:  # Save 'meta_out' attribute before wrapping (defines output parameters needed for dask)
            meta_out = fun_proc_loaded.meta_out
        except AttributeError:
            pass
            # cfg['in']['fun_proc_loaded'] = partial(fun_proc_loaded, **arg_fun_proc_loaded)
        else:  # Wrapping
            # 'meta_out' attribute means we will modify parameters in same step as Time loading by apply function:
            fun_loaded_and_loaded_corr.meta_out = meta_out  # add lost parameter back

        cfg['in']['fun_proc_loaded'] = fun_loaded_and_loaded_corr

    # Find raw files and preliminary format correction: non-regularity in source (primary) csv
    # todo: move correction into the load cycle
    raw_parent = cfg['in']['path'].parent
    raw_pattern = cfg['in']['path'].name
    if cfg['in']['path'].is_dir() or (not cfg['in']['path'].suffix or
        sum(p.suffix.lower() in ('.zip', '.rar') or p.is_dir() for p in raw_parent.glob(raw_pattern))):
        # raw_parent was found
        raw_subdir = raw_pattern
        raw_pattern = '{prefix}{id}*.[tT][xX][tT]'  # number:0>2
    else:
        raw_subdir = raw_parent.name
        raw_parent = raw_parent.parent
        # raw_pattern was found

    # type may be specified to get ``raw_corrected`` files
    cfg_text_type = cfg['in']['text_type']
    if cfg_text_type:
        _ = cfg['in']['text_line_regex']        # save custom regex
        cfg['in'].update(config_model(cfg_text_type))
        if _ is not None:
            cfg['in']['text_line_regex'] = _    # recover custom regex
        cfg['in'] = init_input_cols(cfg['in'])
    else:  # wait getting ``raw_corrected`` files first, we run init_input_cols() later in loop
        cfg_text_type = 'run init_input_cols() later in loop'  # not text_type we'll get in loop allows to update cfg_in there
    raw_corrected = correct_files(
        raw_parent=raw_parent, raw_subdir=raw_subdir, raw_pattern=raw_pattern,
        fun_correct=partial(
            csv_specific_proc.correct_txt, dir_out=dir_csv_cor, binary_mode=False,
            header_rows=cfg['in']['skiprows']
        ),
        sub_str_list=(
            (lambda txt_type: [config_model(txt_type, get_regex_only=True), b'^.+']
             ) if cfg['in']['text_line_regex'] is None else [cfg['in']['text_line_regex'], b'^.+']
        ),
        **cfg['in']
        )
    n_raw_cor_found = len(raw_corrected)

    if n_raw_cor_found == 0:
        lf.warning('No {:s} raw files found!', raw_pattern)
        sys.exit(ExitStatus.failure)

    if return_:
        if return_.startswith('<cfg_input_files_list'):
            lf.info('{:d} raw files...', n_raw_cor_found)
            for i1_probe, ((text_type, probe), paths_csv) in enumerate(raw_corrected.items(), start=1):
                for path_csv in paths_csv:
                    yield i1_probe, f'{text_type}{probe}', path_csv, None
            return
        edge_rows_only = 'meta' in return_
    else:
        edge_rows_only = False

    # Loading files of corrected format and processing their data
    #############################################################

    lf.info('Loading {:d} raw files...', n_raw_cor_found)
    cfg_in = cfg['in'].copy()
    for i1_probe, ((text_type, probe), paths_csv) in enumerate(raw_corrected.items(), start=1):
        if text_type != cfg_text_type:
            cfg_text_type = text_type
            cfg_in = cfg['in'].copy()
            cfg_in.update(config_model(text_type))
            pid = f'{text_type}{probe}'
            if cfg_in_probe and pid in cfg_in_probe:
                cfg_in.update(cfg_in_probe[pid])
            cfg_in = init_input_cols(cfg_in)
            if cfg_in.get('date_to_from'):
                cfg_in['dt_from_utc'] = (cfg_in['date_to_from'][1] - cfg_in['date_to_from'][0])
                lf.warning(
                    'Time shift to {} from {} will be performed ({} hours)',
                    *cfg_in['date_to_from'], -cfg_in['dt_from_utc']
                )

        # Loading csv in parts
        ######################
        n_paths = len(paths_csv)
        for i1_path, i_chunk, path_csv, df in csv_read_gen(
            **{**cfg_in, "paths": paths_csv}, edge_rows_only=edge_rows_only
        ):
            if not edge_rows_only:
                lf.warning(
                    '{: >2}. csv {:s}{:s}{:s} loading...',
                    i1_probe, path_csv.stem,
                    f' {i1_path}/{n_paths}' if n_paths > 1 else '',
                    f'.{i_chunk}' if i_chunk else ''
                )
            if df is None:
                if i_chunk == 0:
                    lf.warning('Not processed (empty) {}', path_csv.name)
                continue
            yield i1_probe, f'{text_type}{probe}', path_csv, df


if __name__ == '__main__':

    def main():
        global cfg
        filenames_default = '*.txt'
        if len(sys.argv) > 1:
            dir_in, raw_pattern_file = sys.argv[1].split('*', 1)
            dir_in = Path(dir_in)
            raw_pattern_file = f'*{raw_pattern_file}' if raw_pattern_file else filenames_default
            lf.info('Searching config file and input files in {:s} (default mask: {:s})', dir_in, raw_pattern_file)
        else:
            dir_in = Path.cwd().resolve()
            raw_pattern_file = filenames_default
            lf.info(
                'No input arguments => using current dir to search config file and input files (default mask: {:s})',
                raw_pattern_file
            )

        cfg_in = {'path': Path(dir_in) / raw_pattern_file}

        # file_ini = cfg['in']['dir'] / 'tcm.ini'
        # file_out = cfg['in']['dir'] / 'test.h5'
        # tbl_out = 'test'

        for i1_probe, probe, paths_csv, d in load_from_csv_gen(cfg_in):
            print(d.compute())
            # todo


    main()
