# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import enum
import io
import logging
import re
import sys
from os import path as os_path
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
import itertools
from pathlib import Path, PurePath
import glob
from typing import Any, AnyStr, Callable, Dict, Iterable, Mapping, Match, Optional, Union, Sequence, Tuple, TypeVar, BinaryIO, TextIO

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask import delayed, compute
from pandas.tseries.frequencies import to_offset

from yaml import safe_dump as yaml_safe_dump

from to_vaex_hdf5 import cfg_dataclasses as cfg_d
from inclinometer import incl_h5clc_hy

# import my scripts
# from to_pandas_hdf5.csv2h5 import main as csv2h5
# from to_pandas_hdf5.csv_specific_proc import rep_in_file, correct_txt, loaded_tcm

lf = None
# code from utils2init import open_csv_or_archive_of_them, ExitStatus ...
def standard_error_info(e):
    msg_trace = '\n==> '.join((s for s in e.args if isinstance(s, str)))
    return f'{e.__class__}: {msg_trace}'


class Ex_nothing_done(Exception):
    def __init__(self, msg=''):
        self.message = f'{msg} => nothing done. For help use "-h" option'


@enum.unique
class ExitStatus(enum.IntEnum):
    """Portable definitions for the standard POSIX exit codes.
    https://github.com/johnthagen/exitstatus/blob/master/exitstatus.py
    """
    success = 0  # Indicates successful program completion
    failure = 1  # Indicates unsuccessful program completion in a general sense


class FakeContextIfOpen:
    """
    Context manager that does nothing if file is not str/PurePath or custom open function is None/False
    useful if instead file want use already opened file object
    """

    def __init__(self,
                 fn_open_file: Optional[Callable[[Any], Any]] = None,
                 file: Optional[Any] = None,
                 opened_file_object = None):
        """
        :param fn_open_file: if not bool(fn_open_file) is True then context manager will do nothing on exit
        :param file:         if not str or PurePath then context manager will do nothing on exit
        """
        if opened_file_object:  # will return opened_file_object and do nothing
            self.file = opened_file_object
            self._do_open_close = False
        else:
            self.file = file
            self.fn_open_file = fn_open_file
            self._do_open_close = (
                isinstance(self.file, (str, PurePath))
                and self.fn_open_file
            )

    def __enter__(self):
        """
        :return: opened handle or :param file: from __init__ if not need open
        """
        self.handle = self.fn_open_file(self.file) if self._do_open_close else self.file
        return self.handle

    def __exit__(self, exc_type, ex_value, ex_traceback):
        """
        Closes handle returned by fn_open_file() if need
        """
        if exc_type is None and self._do_open_close:
            # self.handle is not None and
            self.handle.close()
        return False


class Message:
    def __init__(self, fmt, args):
        self.fmt = fmt
        self.args = args

    def __str__(self):
        return self.fmt.format(*self.args)


class LoggingStyleAdapter(logging.LoggerAdapter):
    """
    Uses str.format() style in logging messages. Usage:
    logger = LoggingStyleAdapter(logging.getLogger(__name__))
    also prepends message with [self.extra['id']]
    """
    def __init__(self, logger, extra=None):
        if isinstance(logger, str):
            logger = logging.getLogger(logger)
        super(LoggingStyleAdapter, self).__init__(logger, extra or {})

    def process(self, msg, kwargs):
        return (f'[{self.extra["id"]}] {msg}' if 'id' in self.extra else msg,
                kwargs)

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger._log(level, Message(msg, args), (), **kwargs)


def set_field_if_no(dictlike, dictfield, value=None):
    """
    Modifies dict: sets field to value only if it not exists
    :param dictlike: dict
    :param dictfield: field
    :param value: value
    :return: None
    """
    try:
        if dictlike[dictfield] is not None:
            return
    except KeyError:  # may be need check also MissingMandatoryValue or omegaconf.errors.ValidationError or set Optional if None
        pass
    dictlike[dictfield] = value


def dir_create_if_need(dir_like: Union[str, PurePath, Path], b_interact: bool = True) -> Path:
    """
    :param dir_like:
    :param b_interact:
    :return: Path(dir_like)
    """
    if dir_like:
        dir_like = Path(dir_like)
        if not dir_like.is_dir():
            print(f' ...making dir "{dir_like}"... ')
            try:
                dir_like.mkdir(exist_ok=True)  # exist_ok=True is need because dir may be just created in other thread
            except FileNotFoundError as e:
                ans = input(
                    f'There are several directories levels to create is needed. Are you sure to make: "{dir_like}"? Y/n: '
                    ) if b_interact else 'n'
                if 'n' in ans or 'N' in ans:
                    print('answered No')
                    raise FileNotFoundError(
                        f'Can make only 1 level of dir. without interact. Can not make: "{dir_like}"'
                        )
                else:
                    print('creating dir...', end='')
                dir_like.mkdir(parents=True, exist_ok=True)
    return dir_like


def this_prog_basename(path=sys.argv[0]):
    return os_path.splitext(os_path.split(path)[1])[0]


def init_logging(logger_name=__name__, log_file=None, level_file='INFO', level_console=None):
    """
    Logging to file flogD/flogN.log and console with piorities level_file and levelConsole
    :param logging: logging class from logging library
    :param logger_name:
    :param log_file: name of log file. Default: & + "this program file name"
    :param level_file: 'INFO'
    :param level_console: 'WARN'
    :return: logging Logger

    Call example:
    lf= init_logging(__name__, None, args.verbose)
    lf.warning(msgFile)
    """
    if log_file:
        if not os_path.isabs(log_file):
            # if flogD is None:
            flogD = os_path.dirname(sys.argv[0])
            log_file = os_path.join(flogD, log_file)
    else:
        # if flogD is None:
        flogD = os_path.join(os_path.dirname(sys.argv[0]), 'log')
        dir_create_if_need(flogD)
        log_file = os_path.join(flogD, f'&{this_prog_basename()}.log')  # '&' is for autoname indication

    if lf:
        try:  # a bit more check that we already have logger
            l = logging.getLogger(logger_name)
        except Exception as e:
            pass
        if l and l.hasHandlers():
            l.handlers.clear()  # or if have good handlers return l
    else:
        l = logging.getLogger(logger_name)
    try:
        filename = Path(log_file)
        b_default_path = not filename.parent.exists()
    except FileNotFoundError:
        b_default_path = True
    if b_default_path:
        filename = Path(__file__).parent / 'logs' / filename.name
    logging.basicConfig(filename=filename, format='%(asctime)s %(message)s', level=level_file)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(level_console if level_console else 'INFO' if level_file != 'DEBUG' else 'DEBUG')  # logging.WARN
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')  # %(name)-12s: %(levelname)-8s ...
    console.setFormatter(formatter)
    l.addHandler(console)
    l.propagate = True  # to default

    if b_default_path:
        l.warning('Bad log path: %s! Using new path with default dir: %s', log_file, filename)

    return LoggingStyleAdapter(l)

#######################################################################################################################
lf = LoggingStyleAdapter(logging.getLogger(__name__))

# code from to_pandas_hdf5.csv_specific_proc import mod_incl_name ...
def mod_incl_name(file_in: Union[str, PurePath], add_prefix=None):
    """
    Change name of raw inclinometer/wavegauge data file to name of corrected (regular table format) csv file
    """
    file_in = PurePath(file_in)
    name = file_in.name.lower().replace('inkl', 'incl')  # correct to English
    name, b_known_names = re.subn(
        r'\*?(?P<prefix>incl|i|w)\*?_0*(?P<number>\d\d)',
        lambda m: f"{m.group('prefix')}{m.group('number')}",
        name
        )
    if not (b_known_names or 'incl_b' in name):
        name, b_known_names = re.subn(r'voln_v*(?P<number>\d\d)',
                                      lambda m: f"w{m.group('number')}",
                                      name
                                      )
        if not (b_known_names or add_prefix):
            print('Not known probe name:', file_in)
    # Paste add_prefix before extension
    if add_prefix:
        def rep(matchobj):
            """ if add_prefix (excluding @) in ``name`` then replace it else append"""
            substr = matchobj.group(0)
            # if (add_str1 := add_prefix.replace('@', '')) in substr:
            #     return substr.replace(add_str1, add_prefix)
            # else:
            return f'{add_prefix}{substr}'
            # '{0}{2}.{1}'.format(*name.rsplit('.', 1), add_prefix) f'{substr}{add_prefix}'

        name = re.sub(r'^\*?([^*.]*)', rep, name)
        if not b_known_names and '*' in name or '?' in name:
            print('Files will have names:', name)
    file_out = file_in.with_name(name)  # r'inkl_?0*(\d{2})', r'incl\1'
    return file_out


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

    lf.warning('preliminary correcting csv file {:s} by removing irregular rows, writing to {:s}.',
              file_in_path.name, file_out)

    # Check if we modyfing input file
    file_out = Path(file_out)
    if file_in_path == file_out:
        # Modyfing input file -> temporary out file and final output files:
        file_out, file_out_original = file_out.with_suffix(f'{file_out.suffix}.bak'), file_out
    else:
        file_out_original = ''

    sum_deleted = 0
    with FakeContextIfOpen(lambda x: open(x, 'rb' if binary_mode else 'r'), file_in) as fin,\
         open(file_out, 'wb' if binary_mode else 'w') as fout:
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
            for block in iter(lambda: fin.read(block_size), b''):
                block = f_replace(block)
                if block != (b'' if binary_mode else ''):
                    fout.write(block)

    if file_out_original:
        file_out.replace(file_out_original)

    return sum_deleted



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


def correct_txt(
        file_in: Union[str, Path, BinaryIO, TextIO],
        file_out: Optional[Path] = None,
        dir_out: Optional[PurePath] = None,
        mod_file_name: Callable[[PurePath], PurePath] = lambda n: n.parent.with_name(n.name.replace('.', '_clean.')),
        sub_str_list: Sequence[bytes] = None,
        **kwargs
        ) -> Path:
    """
    Modified correct_kondrashov_txt() to be universal replacer
    Replaces bad strings in csv file and writes corrected file
    :param file_in: file name or file-like object of csv file or csv file opened in archive that RarFile.open() returns
    :param file_out: full output file name. If None combine dir_out and name of output file which is generated by mod_file_name(file_in.name)
    :param dir_out: output dir, if None then dir of file_in, but
        Note: in opened archives it may not contain original path info (they may be temporary archives).
    :param mod_file_name: function to get out file from input file (usually relative)
    :param sub_str_list: f_repl_by_dict() argument that will be decoded here to str if needed
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
                # cmd_contained_path_of_archive = getattr(file_in, '_cmd', None)  # deprechate
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
            file_out = (file_in_path := Path(file_in)).with_name(str(mod_file_name(Path(file_in_path.name))))


    if file_out.is_file() and file_out.stat().st_size > 100:  # If file size is small may be it is damaged. Try to
        # reprocess (not hard to reprocess small files)
        if is_opened:
            msg_file_in = correct_old_zip_name_ru(msg_file_in)
        lf.warning('skipping of pre-correcting csv file {} to {}: destination exist', msg_file_in, file_out.name)
        return file_out

    dir_create_if_need(file_out.parent)
    binary_mode = isinstance(file_in, io.RawIOBase)

    fsub = f_repl_by_dict([x if binary_mode else bytes.decode(x) for x in sub_str_list], binary_str=binary_mode)

    # {'use': b'\g<use>'})  # $ not works without \r\n so it is useless
    # b'((?!2018)^.+)': '', b'( RS)(?=\r\n)': ''
    # '^Inklinometr, S/N 008, ABIORAS, Kondrashov A.A.': '',
    # '^Start datalog': '',
    # '^Year,Month,Day,Hour,Minute,Second,Ax,Ay,Az,Mx,My,Mz,Battery,Temp':
    sum_deleted = rep_in_file(file_in, file_out, fsub, **{'binary_mode': binary_mode, **kwargs})

    if sum_deleted:
        lf.warning('{} bad lines deleted', sum_deleted)

    return file_out




# ---
RT = TypeVar('RT')  # return type


def meta_out(attribute: Any) -> Callable[[Callable[..., RT]], Callable[..., RT]]:
    """
    Decorator that adds attribute ``meta_out`` to decorated function
    In this file it is used to return partial of out_fields(),
    and that in this module used to get metadata parameter for dask functions.
    :param attribute: any data, will be assigned to ``meta_out`` attribute of decorated function
    :return:
    """
    def meta_out_dec(fun: Callable[..., RT]) -> Callable[..., RT]:
        fun.meta_out = attribute
        return fun

    return meta_out_dec


def out_fields(dtyp: np.dtype, keys_del: Optional[Iterable[str]] = (),
                add_before: Optional[Mapping[str, np.dtype]] = None,
                add_after: Optional[Mapping[str, np.dtype]] = None) -> Dict[str, np.dtype]:
    """
    Removes fields with keys that in ``keys_del`` and adding fields ``add*``
    :param dtyp:
    :param keys_del:
    :param add_before:
    :param add_after:
    :return: dict of {field: dtype} metadata
    """
    if add_before is None:
        add_before = {}
    if add_after is None:
        add_after = {}
    return {**add_before, **{k: v[0] for k, v in dtyp.fields.items() if k not in keys_del}, **add_after}


def log_csv_specific_param_operation(
        key_logged: str,
        functions_str: Optional[Sequence[str]],
        cfg_in) -> None:
    """
    Shows info message of csv_specific_param operations.
    Sets cfg_in['csv_specific_param_logged' + key_logged] to not repeat message on repeating call
    :param log_param: key of csv_specific_param triggering specific calculations when func will be called and this message about
    :param functions_str:
    :param msg_format_param: message format pattern with one parameter (%s) wich will be replaced by csv_specific_param.keys()
    :return:
    """

    key_logged_full = f'csv_specific_param_logged{"-" if key_logged else ""}{key_logged}'
    if not cfg_in.get(key_logged_full):  # log 1 time i.e. in only one 1 dask partition
        cfg_in[key_logged_full] = True
        lf.info('csv_specific_param {} modifications applied', list(functions_str.keys()))  # todo: add time or block


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
    applied_funs = []
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
        params_funs[param] = fun(param, fun_or_const)    # f"{param} = {fun_or_const}({param})
        applied_funs.append(k)
    if applied_funs:
        log_csv_specific_param_operation('', applied_funs, cfg_in)
    return params_funs


@meta_out(out_fields)
def loaded_corr(a: Union[pd.DataFrame, np.ndarray],
                     cfg_in: Mapping[str, Any],
                     csv_specific_param: Optional[Mapping[str, Any]] = None
                     ) -> pd.DataFrame:
    """
    Specified prep&proc of data:

    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :param csv_specific_param: {param_suffix: fun_expr} where ``suffix`` in param_suffix string that can be 'fun' or 'add':
    - 'fun': fun_expr specifies function assign to ``param``
    - 'add': fun_expr specifies value to add to ``param`` to modify it
    :return: pandas.DataFrame
    """
    if csv_specific_param is not None:
        params_funs = param_funs_closure(csv_specific_param, cfg_in)
        if params_funs:
            return a.assign(**params_funs)
    return a


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


century = b'20'


def convertNumpyArrayOfStrings(date: Union[np.ndarray, pd.Series],
                               dtype: np.dtype,
                               format: str = '%Y-%m-%dT%H:%M:%S') -> pd.DatetimeIndex:
    """
    Error corrected conversion: replace bad data with previous (next)
    :param date: Numpy array of strings
    :param dtype: destination Numpy type
    :return: pandas Series of destination type

    # Bug in supervisor convertor corrupts dates: gets in local time zone.
    #Ad hoc: for daily records we use 1st date - can not use if 1st date is wrong
    #so use full correct + wrong time at start (day jump down)
    # + up and down must be alternating

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
    while True:
        try:
            return pd.DatetimeIndex(date.astype(dtype))
        except TypeError as e:
            print('date strings converting to %s error: ' % dtype, standard_error_info(e))
        except ValueError as e:
            print('bad date: ', standard_error_info(e))

        try:
            date = pd.to_datetime(date, format=format, errors='coerce')  # new method
            t_is_bad = date.isna()
            t_is_bad_sum = t_is_bad.sum()
            if t_is_bad_sum:
                lf.warning('replacing {} bad strings with previous', t_is_bad_sum)
                date.ffill(inplace=True)
                # 'nearest' interpoating datime is not works, so
                # s2 = date.astype('i8').astype('f8')
                # s2[t_is_bad] = np.NaN
                # s2.interpolate(method='nearest', inplace=True)
                # date[t_is_bad] = pd.to_datetime(s2[t_is_bad])
            return date
        except Exception as e:
            b_new_method = False
            print('to_datetime not works', standard_error_info(e))
            raise (e)


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


@meta_out(partial(out_fields, keys_del={'yyyy', 'mm', 'dd', 'HH', 'MM', 'SS'}, add_before={'Time': 'M8[ns]'}))
def loaded_tcm(a: Union[pd.DataFrame, np.ndarray], cfg_in: Mapping[str, Any] = None,
                                  csv_specific_param: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """
    Specified prep&proc of navigation data from Kondrashov inclinometers:
    - Time calc: gets time in current zone

    :param a: numpy record array. Will be modified inplace.
    :param cfg_in: dict
    :param : {invert_magnetometer: True}
    :return: numpy 'datetime64[ns]' array

    Example input:
    a = {
    'yyyy': b"2017", 'mm': b"1", 'dd': b"4",
    'HH': b'9','MM': b'5', 'SS': b''}
    """

    try:
        date = concat_to_iso8601(a)  # .compute() #da.from_delayed(, (a.shape[0],), '|S19') #, ndmin=1)
    except Exception as e:
        lf.exception('Can not convert date: ')
        raise e
    tim_index = convertNumpyArrayOfStrings(date, 'datetime64[ns]')  # a['Time']

    if csv_specific_param is not None:
        key = 'invert_magnetometer'
        if csv_specific_param.get(key):
            magnetometer_channels = ['Mx', 'My', 'Mz']
            a.loc[:, magnetometer_channels] = -a.loc[:, magnetometer_channels]
            a = a.copy()
            # log_csv_specific_param_operation('loaded_tcm', csv_specific_param.keys(), cfg_in) mod:
            key_logged = 'csv_specific_param_logged-loaded_tcm'
            if not cfg_in.get(key_logged):  # log 1 time i.e. in only one 1 dask partition
                cfg_in[key_logged] = True
                lf.info('{} applied', key)
        elif csv_specific_param:
            lf.info('unknown key(s) in csv_specific_param')

    return a.assign(Time=tim_index).loc[:, list(loaded_tcm.meta_out(cfg_in['dtype_out']).keys())]

# ---

tim_min_save: pd.Timestamp = pd.Timestamp('now', tz='UTC')     # can only decrease in time_corr(), set to pd.Timestamp('now', tz='UTC') before call
tim_max_save: pd.Timestamp = pd.Timestamp(0, tz='UTC')         # can only increase in time_corr(), set to pd.Timestamp(0, tz='UTC') before call

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
        if isinstance(date[0], str):
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
        lf.info('Time constant: {} {:s}', abs(dt_from_utc),
                'subtracted' if dt_from_utc > timedelta(0) else 'added')
    else:
        if not isinstance(date[0], pd.Timestamp):  # isinstance(date, (pd.Series, np.datetime64))
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
            lf.warning('Ready to use specified sampling frequency fs = {:s}', cfg_in['fs'])
            freq = cfg_in['fs']
        elif cfg_in.get('fs_old_method'):
            lf.warning('Ready to use specified sampling frequency fs_old_method = {:s}', cfg_in['fs_old_method'])
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
                  f"{'masked' if process == 'delete_inversions' else 'interpolated'} (1st and last: " \
                  f"{pd.to_datetime(t[idel[[0, -1]]], utc=True)})"
            if n_decrease:
                lf.warning('decreased time ({}) was detected! {}', n_decrease, msg)
            else:
                lf.warning(msg)


        if n_same > 0 and cfg_in.get('fs') and not cfg_in.get('fs_old_method'):
            # This is most simple operation that should be done usually for CTD
            t = repeated2increased(t, cfg_in['fs'], b_ok if n_decrease else None)  # if n_decrease then b_ok is calculated before
            tim = pd.to_datetime(t, utc=True)
        elif n_same > 0 or n_decrease > 0:
            # message with original t

            # Replace t by linear increasing values using constant frequency excluding big holes
            if cfg_in.get('fs_old_method'):
                lf.warning('Linearize time interval using provided freq = {:f}Hz (determined: {:f})',
                          cfg_in.get('fs_old_method'), freq)
                freq = cfg_in.get('fs_old_method')
            else:  # constant freq = filtered mean
                lf.warning('Linearize time interval using median* freq = {:f}Hz determined', freq)
            t = np.int64(rep2mean(t, bOk=b_ok))  # interp to can use as pandas index even if any bad
            b_show = n_decrease > 0
            if freq <= 1:
                # Skip: typically data resolution is sufficient for this frequency
                lf.warning('Not linearizing for frequency < 1')
            else:
                # Increase time resolution by recalculating all values
                tim_before = pd.to_datetime(t, utc=True)
                make_linear(t, freq)  # changes t (and tim?)
                # Check if we can use them
                bbad = check_time_diff(tim_before, t.view('M8[ns]'), dt_warn=pd.Timedelta(minutes=2),
                                       msg='Big time diff after corr: difference [min]:')
                if np.any(bbad):
                    b_ok = ~bbad
                    b_show = True

            # Show what is done
            if b_show:
                if b_ok is None:
                    dt = np.ediff1d(t, to_begin=1)
                    b_ok = dt > 0
                plot_bad_time_in_thread(cfg_in, t, b_ok, idel, tim,
                                        (tim_min, tim_max) if cfg_in.get('min_date') else None, path_save_image, msg)

        # Checking all is ok

        dt = np.ediff1d(t, to_begin=1)
        b_ok = dt > 0
        # tim.is_unique , len(np.flatnonzero(tim.duplicated()))
        b_decrease = dt < 0  # with set of first element as increasing
        n_decrease = b_decrease.sum()
        if n_decrease > 0:
            lf.warning(
                'Decreased remaining time ({:d}) are masked!{:s}{:s}',
                n_decrease,
                '\n'.join(' < '.join('{:%y.%m.%d %H:%M:%S.%f%z}'.format(_) for _ in tim[se].to_numpy()) for se in
                         np.flatnonzero(b_decrease)[:3, None] + np.int32([-1, 0])),
                '...' if n_decrease > 3 else ''
                )

            b_ok &= ~b_decrease

        b_same_prev = np.ediff1d(t, to_begin=1) == 0  # with set of first element as changing
        n_same = b_same_prev.sum()

        if cfg_in.get('b_keep_not_a_time'):
            if n_same > 0:
                lf.warning('nonincreased time ({:d} times) is detected! ↦ interp ', n_same)
        else:
            # prepare to interp all nonincreased (including NaNs)
            if n_bad_in:
                b_same_prev &= ~b_ok_in

            msg = ', '.join(
                f'{fault} time ({n} times)' for (n, fault) in ((n_same, 'non-increased'), (n_bad_in, 'NaN')) if n > 0
                )
            if msg:
                lf.warning('{:s} is detected! ↦ interp ', msg)

        if n_same > 0 or n_decrease > 0:
            # rep2mean(t, bOk=np.logical_not(b_same_prev if n_decrease==0 else (b_same_prev | b_decrease)))
            b_bad = b_same_prev if n_decrease == 0 else (b_same_prev | b_decrease)
            t = rep2mean_with_const_freq_ends(t, ~b_bad, freq)

    else:  # not need to check / correct time
        b_ok = np.ones(tim.size, np.bool_)
    # make initial shape: paste NaNs back
    if n_bad_in and cfg_in.get('b_keep_not_a_time'):
        # place initially bad elements back
        t, t_in = (np.NaN + np.empty_like(b_ok_in)), t
        t[b_ok_in] = t_in
        b_ok_in[b_ok_in] = b_ok
        b_ok = b_ok_in
    elif process == 'delete_inversions':
        b_ok &= b_ok_in
    # make initial shape: pad with constants of config. limits where data was removed because input is beyond this limits
    if is_time_filt:   # cfg_min_date and np.any(it_se != np.int64([0, date.size])):
        pad_width = (it_se[0], date.size - it_se[1])
        t = np.pad(t, pad_width, constant_values=np.array((cfg_in['min_date'], cfg_in['max_date']), 'M8[ns]'))
        b_ok = np.pad(b_ok, pad_width, constant_values=False)
    assert t.size == b_ok.size

    return pd.to_datetime(t, utc=True), b_ok


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
    set_field_if_no(cfg_in, 'dt_from_utc', 0)
    dtype_text_max = '|S{:.0f}'.format(cfg_in['max_text_width'])  # '2000 #np.str

    if cfg_in.get('header'):  # if header specified
        re_sep = ' *(?:(?:,\n)|[\n,]) *'  # not isolate "`" but process ",," right
        cfg_in['cols'] = re.split(re_sep, cfg_in['header'])
        # re_fast = re.compile(u"(?:[ \n,]+[ \n]*|^)(`[^`]+`|[^`,\n ]*)", re.VERBOSE)
        # cfg_in['cols']= re_fast.findall(cfg_in['header'])
    elif 'cols' not in cfg_in:  # cols is from header, is specified or is default
        KeyError('no "cols" no "header" in config.in')
        cfg_in['cols'] = ('stime', 'latitude', 'longitude')

    # default parameters dependent from ['cols']
    cols_load_b = np.ones(len(cfg_in['cols']), np.bool_)
    set_field_if_no(cfg_in, 'comments', '"')

    # assign data type of input columns
    b_was_no_dtype = not 'dtype' in cfg_in
    if b_was_no_dtype:
        cfg_in['dtype'] = np.array([np.float64] * len(cfg_in['cols']))
        # 32 gets trunkation errors after 6th sign (=> shows long numbers after dot)
    elif isinstance(cfg_in['dtype'], str):
        cfg_in['dtype'] = np.array([np.dtype(cfg_in['dtype'])] * len(cfg_in['cols']))
    elif isinstance(cfg_in['dtype'], list):
        # prevent numpy array(list) guess minimal dtype because dtype for represent dtype of dtype_text_max may be greater
        numpy_cur_dtype = np.min_scalar_type(cfg_in['dtype'])
        numpy_cur_dtype_len = numpy_cur_dtype.itemsize / np.dtype((numpy_cur_dtype.kind, 1)).itemsize
        cfg_in['dtype'] = np.array(cfg_in['dtype'], '|S{:.0f}'.format(
            max(len(dtype_text_max), numpy_cur_dtype_len)))

    for sCol, sDefault in (['coltime', 'Time'], ['coldate', 'Date']):
        if (sCol not in cfg_in):
            # if cfg['col(time/date)'] is not provided try find 'Time'/'Date' column name
            if not (sDefault in cfg_in['cols']):
                sDefault = sDefault + '(text)'
            if not (sDefault in cfg_in['cols']):
                continue
            cfg_in[sCol] = cfg_in['cols'].index(sDefault)  # assign 'Time'/'Date' column index to cfg['col(time/date)']
        elif isinstance(cfg_in[sCol], str):
            cfg_in[sCol] = cfg_in['cols'].index(cfg_in[sCol])

    if not 'converters' in cfg_in:
        cfg_in['converters'] = None
    else:
        if not isinstance(cfg_in['converters'], dict):
            # suspended evaluation required
            cfg_in['converters'] = cfg_in['converters'](cfg_in)
        if b_was_no_dtype:
            # converters produce datetime64[ns] for coldate column (or coltime if no coldate):
            cfg_in['dtype'][cfg_in['coldate' if 'coldate' in cfg_in
            else 'coltime']] = 'datetime64[ns]'

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

    if cfg_in.get('cols_load'):
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


def read_csv(paths: Sequence[Union[str, Path]],
             **cfg_in: Mapping[str, Any]
             ) -> Tuple[Union[pd.DataFrame, dd.DataFrame, None], Mapping[str, Any]]:
    """
    Reads csv in dask DataFrame
    Calls cfg_in['fun_proc_loaded'] (if specified)
    Calls time_corr: corrects/checks Time (with arguments defined in cfg_in fields)
    Sets Time as index
    :param paths: list of file names
    :param cfg_in: contains fields for arguments of dask.read_csv correspondence:

        names=cfg_in['cols'][cfg_in['cols_load']]
        usecols=cfg_in['cols_load']
        on_bad_lines=cfg_in['on_bad_lines']
        comment=cfg_in['comment']

        Other arguments corresponds to fields with same name:
        dtype=cfg_in['dtype']
        delimiter=cfg_in['delimiter']
        converters=cfg_in['converters']
        skiprows=cfg_in['skiprows']
        blocksize=cfg_in['blocksize']

        Also cfg_in has filds:
            dtype_out: numpy.dtype, which "names" field used to detrmine output columns
            fun_proc_loaded: None or Callable[
            [Union[pd.DataFrame, np.array], Mapping[str, Any], Optional[Mapping[str, Any]]],
             Union[pd.DataFrame, pd.DatetimeIndex]]
            If it returns pd.DataFrame then it also must has attribute:
                meta_out: Callable[[np.dtype, Iterable[str], Mapping[str, dtype]], Dict[str, np.dtype]]

            See also time_corr() for used fields



    :return: tuple (a, b_ok) where
        a:      dask dataframe with time index and only columns listed in cfg_in['dtype_out'].names
        b_ok:   time correction rezult boolean array
    """
    read_csv_args_to_cfg_in = {
        'dtype': 'dtype_raw',
        'names': 'cols',
        'on_bad_lines': 'on_bad_lines',
        'comment': 'None',
        'delimiter': 'delimiter',
        'converters': 'converters',
        'skiprows': 'skiprows',
        'blocksize': 'blocksize',
        'encoding': 'encoding'
        }
    read_csv_args = {arg: cfg_in[key] for arg, key in read_csv_args_to_cfg_in.items() if key in cfg_in}
    read_csv_args.update({
        'skipinitialspace': True,
        'usecols': cfg_in['dtype'].names,
        'header': None})
    # removing "ParserWarning: Both a converter and dtype were specified for column k - only the converter will be used"
    
    chunksize = 10 ** 6
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        process(chunk)
    
    
    if read_csv_args['converters']:
        read_csv_args['dtype'] = {k: v[0] for i, (k, v) in enumerate(read_csv_args['dtype'].fields.items()) if i not in read_csv_args['converters']}
    try:
        try:
            # raise ValueError('Temporary')
            ddf = dd.read_csv(paths, **read_csv_args)
            # , engine='python' - may help load bad file

            # index_col=False  # force pandas to _not_ use the first column as the index (row names) - no in dask
            # names=None, squeeze=False, prefix=None, mangle_dupe_cols=True,
            # engine=None, true_values=None, false_values=None, skipinitialspace=False,
            #     nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False,
            #     skip_blank_lines=True, parse_dates=False, infer_datetime_format=False, keep_date_col=False,
            #     date_parser=None, dayfirst=False, iterator=False, chunksize=1000000, compression='infer',
            #     thousands=None, decimal=b'.', lineterminator=None, quotechar='"', quoting=0,
            #     escapechar=None, encoding=None, dialect=None, tupleize_cols=None,
            #      on_bad_lines='warn', skipfooter=0, skip_footer=0, doublequote=True,
            #     delim_whitespace=False, as_recarray=None, compact_ints=None, use_unsigned=None,
            #     low_memory=True, buffer_lines=None, memory_map=False, float_precision=None)
        except ValueError as e:
            lf.exception('dask lib can not load data. Trying pandas lib...')
            del read_csv_args['blocksize']  # because pandas.read_csv has no such arg
            for i, nf in enumerate(paths):
                df = pd.read_csv(nf, **read_csv_args, index_col=False)  # chunksize=cfg_in['blocksize']
                if i > 0:
                    raise NotImplementedError('list of files => need concatenate data')
            ddf = dd.from_pandas(df, chunksize=cfg_in['blocksize'])  #
        except NotImplementedError as e:
            lf.exception('If file "{}" have no data try to delete it?', paths)
            return None, None
    except Exception as e:  # for example NotImplementedError if bad file
        msg = 'Bad file. skip!'
        ddf = None
        if cfg_in['on_bad_lines'] == 'error':
            lf.exception('{:s}\n Try set [in].on_bad_lines = skip\n', msg)
            raise (e)
        else:
            lf.exception(msg)
    if __debug__:
        lf.debug('read_csv initialised')
    if ddf is None:
        return None, None

    meta_time = pd.Series([], name='Time', dtype='datetime64[ns, UTC]')  # np.dtype('datetime64[ns]')
    # meta_time_index = pd.DatetimeIndex([], dtype='datetime64[ns, UTC]', name='Time')
    # meta_df_with_time_col = cfg_in['cols_load']
    meta_time_and_mask = {'Time': 'datetime64[ns, utc]', 'b_ok': np.bool_}
    # meta_time_and_mask.time = meta_time_and_mask.time.astype('M8[ns]')
    # meta_time_and_mask.b_ok = meta_time_and_mask.b_ok.astype(np.bool_)


    utils_time_corr.tim_min_save = pd.Timestamp('now', tz='UTC')  # initialisation for time_corr_df()
    utils_time_corr.tim_max_save = pd.Timestamp(0, tz='UTC')

    n_overlap = 2 * int(np.ceil(cfg_in['fs'])) if cfg_in.get('fs') else 50

    # Process ddf and get date in ISO string or numpy standard format

    # may be need in func below to extract date:
    cfg_in['file_cur'] = Path(paths[0])
    cfg_in['file_stem'] = cfg_in['file_cur'].stem
    meta_out = None
    try:
        try:
            get_out_types = cfg_in['fun_proc_loaded'].meta_out
        except AttributeError:
            date = ddf.map_partitions(lambda *args, **kwargs: pd.Series(
                cfg_in['fun_proc_loaded'](*args, **kwargs)), cfg_in, meta=meta_time)  # meta_time_index
            # date = date.to_series()

            lf.info(*('time correction in {} blocks...', date.npartitions) if date.npartitions > 1 else
                    ('time correction...',))

            def time_corr_df(t, cfg_in):
                """ Convert tuple returned by time_corr() to dataframe
                """
                return pd.DataFrame.from_dict(dict(zip(
                    meta_time_and_mask.keys(), utils_time_corr.time_corr(t, cfg_in))))
                # return pd.DataFrame.from_items(zip(meta_time_and_mask.keys(), time_corr(t, cfg_in)))
                # pd.Series()

            df_time_ok = date.map_overlap(
                time_corr_df, before=n_overlap, after=n_overlap, cfg_in=cfg_in, meta=meta_time_and_mask)
            # try:
            #     df_time_ok = df_time_ok.persist()  # triggers all csv_specific_proc computations
            # except Exception as e:
            #     lf.exception(
            #         'Can not speed up by persist, doing something that can trigger error to help it identificate...')
            #     df_time_ok = time_corr_df(date.compute(), cfg_in=cfg_in)

            if cfg_in.get('csv_specific_param'):
                # need run this:
                # ddf = to_pandas_hdf5.csv_specific_proc.loaded_corr(
                #     ddf, cfg_in, cfg_in['csv_specific_param'])
                ddf = ddf.map_partitions(to_pandas_hdf5.csv_specific_proc.loaded_corr,
                                         cfg_in, cfg_in['csv_specific_param']
                                         )
            # except (TypeError, Exception) as e:

        else:  # get_out_types is not None => fun_proc_loaded() will return full data DataFrame (not only Time column)
            if callable(get_out_types):
                meta_out = get_out_types([(k, v[0]) for k, v in cfg_in['dtype'].fields.items()])
            lf.info('processing csv data with time correction{:s}...', f' in {ddf.npartitions} blocks' if
                    ddf.npartitions > 1 else '')

            # initialisation for utils_time_corr.time_corr():
            def fun_loaded_and_time_corr_df(df_):
                """fun_proc_loaded() then time_corr()
                """
                df_out = cfg_in['fun_proc_loaded'](df_, cfg_in)   # stop here to check that we calc once
                t = utils_time_corr.time_corr(df_out.Time, cfg_in)
                return df_out.assign(**dict(zip(meta_time_and_mask.keys(), t)))

            # ddf = ddf.map_partitions(cfg_in['fun_proc_loaded'], cfg_in, meta=meta_out)
            ddf = ddf.map_overlap(fun_loaded_and_time_corr_df, before=n_overlap, after=n_overlap,
                                  meta={**meta_out, **meta_time_and_mask})
            df_time_ok = ddf[['Time', 'b_ok']]

    except IndexError:
        print('no data?')
        return None, None

    if 'dtype_out' in cfg_in:
        meta_out = [(k, v[0]) for k, v in cfg_in['dtype_out'].fields.items()]
    if meta_out:
        if cfg_in.get('meta_out_df') is None:
            # construct meta (in format of dataframe to anable set index name)
            dict_dummy = {k: np.zeros(1, dtype=v) for k, v in meta_out}
            meta_out_df = pd.DataFrame(dict_dummy, index=pd.DatetimeIndex([], name='Time', tz='UTC'))
        else:
            meta_out_df = cfg_in['meta_out_df']
    else:
        meta_out_df = None

    if not isinstance(df_time_ok, dd.DataFrame):
        # show message
        nbad_time = len(df_time_ok['b_ok']) - df_time_ok['b_ok'].sum()
        lf.info(
            'Bad time values ({:d}): {}{:s}', nbad_time,
               df_time_ok['b_ok'].fillna(False).ne(True).to_numpy().nonzero()[0][:20],
            ' (shows first 20)' if nbad_time > 20 else ''
        )
    # Define range_message()' delayed args
    n_ok_time = df_time_ok['b_ok'].sum()
    n_all_rows = df_time_ok.shape[0]
    range_source = None

    for date_lim, op in [('min_date', lt), ('max_date', gt)]:
        date_lim = cfg_in.get(date_lim)
        if not date_lim:
            # condition for calc. tim_min_save/tim_max_save in utils_time_corr() and to set index=cfg_in['min_date'] or index=cfg_in['max_date'] where it is out of config range
            continue

        @delayed(pure=True)
        def range_source():
            # Only works as delayed (and in debug inspecting mode also) because utils_time_corr.time_corr(Time) must be processed to set 'tim_min_save', 'tim_max_save'
            return {k: getattr(utils_time_corr, attr) for k, attr in
                    (('min', 'tim_min_save'), ('max', 'tim_max_save'))}

        # Filter data that is out of config range (after range_message()' args defined as both uses df_time_ok['b_ok'])
        df_time_ok['b_ok'] = df_time_ok['b_ok'].mask(
            op(
                df_time_ok['Time'],
                pd.Timestamp(date_lim, tz=None if date_lim.tzinfo else 'UTC')
               ),
            False
            )

    if range_source is None:
        # to be computed before filtering
        range_source = df_time_ok['Time'].reduction(
            chunk=lambda x: pd.Series([x.min(), x.max()]),
            # combine=chunk_fun, not works - gets None in args
            aggregate=lambda x: pd.Series([x.iloc[0].min(), x.iloc[-1].max()], index=['min', 'max']),  # skipna=True is default
            meta=meta_time)
        # df_time_ok['Time'].apply([min, max], meta=meta_time) - not works for dask (works for pandas)
        # = df_time_ok.divisions[0::df_time_ok.npartitions] if (isinstance(df_time_ok, dd.DataFrame) and df_time_ok.known_divisions)

    range_message_args = [range_source, n_ok_time, n_all_rows]
    def range_message(range_source, n_ok_time, n_all_rows):
        t = range_source() if isinstance(range_source, Callable) else range_source
        lf.info('loaded source range: {:s}',
                f'{t["min"]:%Y-%m-%d %H:%M:%S} - {t["max"]:%Y-%m-%d %H:%M:%S %Z}, {n_ok_time:d}/{n_all_rows:d} rows')

    out_cols = list(cfg_in['dtype_out'].names)

    if not ddf.known_divisions:  # always

        def time_index_ok(new_df, new_time_ok):
            # need "values" below (i.e. drop index) because due to map_overlap() the time.index is shifted relative to df.index
            new_df_filt = new_df.loc[new_time_ok['b_ok'].values, out_cols]
            new_time_filt = new_time_ok.loc[new_time_ok['b_ok'].values, 'Time']
            return new_df_filt.set_index(new_time_filt)

        ddf_out = ddf.map_partitions(time_index_ok, df_time_ok, align_dataframes=False, meta=meta_out_df)
        try:
            # ddf_out, range_source = persist(ddf_out, range_source)
            ddf_out, *range_message_args = persist(ddf_out, *range_message_args)
        except MemoryError:
            lf.info('Persisting failed (not enough memory). Continue.... I give You 10sec, You better close some programs...')
            import gc
            gc.collect()  # frees many memory. Helps to not crash
            sleep(2)
            try:
                # ddf_out, range_source = persist(ddf_out, range_source)
                ddf_out, *range_message_args = persist(ddf_out, *range_message_args)
                lf.info('Persisted Ok now. Continue...')
            except MemoryError:
                lf.warning('Persisting failed (not enough memory). Continue, but this way is not tested...')
            # ddf_out.to_parquet('path/to/my-results/')
            # ddf_out = dd.read_parquet('path/to/my-results/')

            # Note: if not persist then compute() should be only once else
            # log['rows'] = out.shape[0].compute() cause to compute all from beginning 2nd time!

        # Two to_delayed() leads to call fun_loaded_and_time_corr_df() twice!
        # ddf_time_delayed = (ddf.to_delayed(), df_time_ok.to_delayed())
        # ddf_out_list = [
        #     delayed(time_index_ok, pure=True)(dl_f, dl_time_ok) for dl_f, dl_time_ok in zip(*ddf_time_delayed)
        #     ]
        # ddf_out = dd.from_delayed(ddf_out_list, divisions='sorted', meta=meta_out_df)
    else:
        # ddf, df_time_ok = compute(ddf, df_time_ok)  # for testing
        # Removing rows with bad time
        ddf_out = ddf.loc[df_time_ok['b_ok'], out_cols]

        if False and __debug__:
            len_out = len(df_time_ok)
            print('out data length before del unused blocks:', len_out)

        # Removing rows with bad time (continue)
        df_time_ok = df_time_ok[df_time_ok['b_ok']]

        if False and __debug__:
            len_out = len(df_time_ok)
            print('out data length:', len_out)
            print('index_limits:', df_time_ok.divisions[0::df_time_ok.npartitions])
            sum_na_out, df_time_ok_time_min, df_time_ok_time_max = compute(
                df_time_ok['Time'].notnull().sum(), df_time_ok['Time'].min(), df_time_ok['Time'].max())
            print('out data len, nontna, min, max:', sum_na_out, df_time_ok_time_min, df_time_ok_time_max)

        ddf_out = ddf_out.set_index(df_time_ok['Time'], sorted=True)  #

    # try:
    #     ddf_out = ddf_out.persist()  # triggers all csv_specific_proc computations
    # except Exception as e:
    #     lf.exception('Can not speed up by persist')


    # print('data loaded shape: {}'.format(ddf.compute(scheduler='single-threaded').shape))  # debug only
    # if nbad_time: #and cfg_in.get('b_keep_not_a_time'):
    #     df_time_ok = df_time_ok.set_index('Time', sorted=True)
    #     # ??? after I set index: ValueError: Not all divisions are known, can't align partitions. Please use `set_index` to set the index.
    #     ddf_out = ddf_out.loc[df_time_ok['b_ok'], :].repartition(freq='1D')

    # if isinstance(df_time_ok, dd.DataFrame) else df_time_ok['Time'].compute()
    # **({'sorted': True} if a_is_dask_df else {}
    # [cfg_in['cols_load']]
    # else:
    #     col_temp = ddf.columns[0]
    #     b = ddf[col_temp]
    #     b[col_temp] = b[col_temp].map_partitions(lambda s, t: t[s.index], tim, meta=meta)
    #     ddf = ddf.reset_index().set_index('index').set_index(b[col_temp], sorted=True).loc[:, list(cfg_in['dtype_out'].names)]

    # date = pd.Series(tim, index=ddf.index.compute())  # dd.from_dask_array(da.from_array(tim.values(),chunks=ddf.divisions), 'Time', index=ddf.index)
    # date = dd.from_pandas(date, npartitions=npartitions)
    # ddf = ddf.loc[:, list(cfg_in['dtype_out'].names)].set_index(date, sorted=True)

    # ddf = ddf.loc[:, list(cfg_in['dtype_out'].names)].compute()
    # ddf.set_index(tim, inplace=True)
    # ddf = dd.from_pandas(ddf, npartitions=npartitions)

    logger = logging.getLogger("dask")
    logger.addFilter(lambda s: s.getMessage() != "Partition indices have overlap.")
    # b_ok = df_time_ok['b_ok'].to_dask_array().compute() if isinstance(
    #     df_time_ok, dd.DataFrame) else df_time_ok['b_ok'].to_numpy()
    # ddf_out.index.name = 'Time' not works
    # b_ok_ds= df_time_ok.set_index('Time')['b_ok']
    return ddf_out, {'func': range_message, 'args': range_message_args}  # , b_ok_ds


# #####################################################################################################################

# inclinometer file format and other parameters
cfg = {
    'in': {
        'fun_proc_loaded': loaded_tcm,   # function to convert csv to dask dataframe
        'delimiter': ',',  # \t not specify if need "None" useful for fixed length format
        'skiprows': 0,  # use 0 because header will be removed by preliminary correction
        'header': 'yyyy(text),mm(text),dd(text),HH(text),MM(text),SS(text),Ax,Ay,Az,Mx,My,Mz,Battery,Temp',
        'dtype': '|S4 |S2 |S2 |S2 |S2 |S2 i2 i2 i2 i2 i2 i2 f8 f8'.split(),
        'on_bad_lines': 'error',
        # '--min_date', '07.10.2017 11:04:00',  # not output data < min_date
        # '--max_date', '29.12.2018 00:00:00',  # UTC, not output data > max_date
        'blocksize': 50_000_000,  # 50Mbt
        'b_interact': '0',
        'csv_specific_param': {'invert_magnetometer': True},
        'encoding': 'CP1251',  # read_csv() encoding parameter
        'max_text_width': 1000   # used here to get sample for dask that should be > max possible row length
        # '--dt_from_utc_seconds', str(cfg['in']['dt_from_utc'][probe].total_seconds()),
        # '--fs_float', str(p_type[cfg['in']['probes_prefix']]['fs']),  # f'{fs(probe, file_in.stem)}',
        },
    'out': {},
    'filter': {},
    'program': {
        'log': 'tcm_csv.log',
        'verbose': 'INFO',
        # 'dask_scheduler': 'synchronous'
        }
    # Warning! If True and b_incremental_update= True then not replace temporary file with result before proc.
    # '--log', 'log/csv2h5_inclin_Kondrashov.log'  # log operations
    }


def main():
    global lf
    lf = LoggingStyleAdapter(init_logging(__name__, cfg['program']['log'], cfg['program']['verbose']))

    filenames_default = '*.txt'
    if len(sys.argv) > 1:
        dir_in, raw_pattern_file = sys.argv[1].split('*', 1)
        dir_in = Path(dir_in)
        raw_pattern_file = f'*{raw_pattern_file}' if raw_pattern_file else filenames_default
        print(f'Searching config file and input files in {dir_in} (default mask: {raw_pattern_file})')
    else:
        dir_in = Path.cwd().resolve()
        raw_pattern_file = filenames_default
        print(
            f'No input arguments => using current dir to search config file and input files (default mask: {raw_pattern_file})')

    file_ini = dir_in / 'tcm.ini'
    file_out = dir_in / 'test.h5'
    tbl_out = 'test'

    dir_csv_cor = None  # dir for cleaned (intermediate) raw files, None => same as dir_in

    # Prepare loading and writing specific to format by rad_csv()

    cfg['in'] = init_input_cols(cfg['in'])

    # - if need extract date from file name
    if cfg['in'].get('fun_date_from_filename') and isinstance(cfg['in']['fun_date_from_filename'], str):
        cfg['in']['fun_date_from_filename'] = eval(
            compile("lambda file_stem, century=None: {}".format(cfg['in']['fun_date_from_filename']), '', 'eval'))

    # - if need additional calculation in read_csv()
    if cfg['in'].get('fun_proc_loaded') is None:
        # Default time processing after loading by dask/pandas.read_csv()
        if 'coldate' not in cfg['in']:  # if Time includes Date then we will just return it
            cfg['in']['fun_proc_loaded'] = lambda a, cfg_in, dummy=None: a[cfg_in['col_index_name']]
        else:  # else will return Time + Date
            cfg['in']['fun_proc_loaded'] = lambda a, cfg_in, dummy=None: a['Date'] + np.array(
                np.int32(1000 * a[cfg_in['col_index_name']]), dtype='m8[ms]')

    if cfg['in']['csv_specific_param']:
        # Move 'csv_specific_param' argument value into definition of to fun_proc_loaded() (because read_csv() not uses
        # additional args) and append it with loaded_corr() if fun_proc_loaded() has 'meta_out' attribute (else
        # fun_proc_loaded() returns only time, and we will run loaded_corr() separately (see read_csv()))

        # If there is 'meta_out' attribute then it will be lost during wrapping by "partial" so saving before
        meta_out = getattr(cfg['in']['fun_proc_loaded'], 'meta_out', None)
        fun_proc_loaded = cfg['in']['fun_proc_loaded']
        if 'csv_specific_param' in cfg['in']['fun_proc_loaded'].__code__.co_varnames:
            fun_proc_loaded = partial(fun_proc_loaded, csv_specific_param=cfg['in']['csv_specific_param'])

        if meta_out is not None:  # only functions with 'meta_out' allowed to modify parameters in same step as loading

            def fun_loaded_folowed_loaded_corr(a, cfg_in):
                a = fun_proc_loaded(a, cfg_in)
                a = loaded_corr(a, cfg_in, cfg['in']['csv_specific_param'])  # also applies functions defined by strings in csv_specific_param
                return a

            cfg['in']['fun_proc_loaded'] = fun_loaded_folowed_loaded_corr
            cfg['in']['fun_proc_loaded'].meta_out = meta_out  # add lost parameter back


    if cfg['program'].get('dask_scheduler'):
        if cfg['program']['dask_scheduler'] == 'distributed':
            from dask.distributed import Client
            # cluster = dask.distributed.LocalCluster(n_workers=2, threads_per_worker=1, memory_limit="5.5Gb")
            client = Client(processes=False)
            # navigate to http://localhost:8787/status to see the diagnostic dashboard if you have Bokeh installed
            # processes=False: avoid inter-worker communication for computations releases the GIL (numpy, da.array)  # without is error
        else:
            if cfg['program']['dask_scheduler'] == 'synchronous':
                lf.warning('using "synchronous" scheduler for debugging')
            import dask
            dask.config.set(scheduler=cfg['program']['dask_scheduler'])


    ######################################################################################################


    # Preliminary format correction: non-regularity in source (primary) csv raw files

    fun_correct_name = partial(mod_incl_name, add_prefix='@')
    raw_corrected = set(
        dir_in.glob(str(fun_correct_name(raw_pattern_file))))  # returns, may be not all, but only corrected file names
    raw_corrected = {fun_correct_name(file) for file in (set(dir_in.glob(
        raw_pattern_file)) - raw_corrected)}  # returns all corrected files + may be some other (cor of cor) file names
    raw_found = set(dir_in.glob(raw_pattern_file)) - raw_corrected  # excluding corrected and other not needed files
    correct_fun = partial(correct_txt,
                          dir_out=dir_csv_cor, binary_mode=False, mod_file_name=fun_correct_name,
                          sub_str_list=[
                              b'^(?P<use>20\d{2}(,\d{1,2}){5}(,\-?\d{1,6}){6},\d{1,2}(\.\d{1,2})?,\-?\d{1,3}(\.\d{1,2})?).*',
                              b'^.+'
                              ])
    # Correct
    if n_raw_found := len(raw_found):
        print('Cleaning', n_raw_found, f'{dir_in / raw_pattern_file}', 'found files...')
    n_raw_cor_found = 0
    for file_in in raw_found:
        file_in = correct_fun(file_in)
        raw_corrected.add(file_in)

    if (n_raw_cor_found := len(raw_corrected)) == 0:
        print('No', raw_pattern_file, end=' raw files found')
        sys.exit(ExitStatus.failure)
    else:
        print(f"Loading {n_raw_cor_found}{'' if n_raw_found else ' previously'} corrected files")
        # prints ' previously' if all source (primary) row files where deleted

    # Loading (files of corrected format) and processing its data

    for path_csv in raw_corrected:
        # Loading
        d, cfg['filter']['delayedfunc'] = read_csv(
            **{**cfg['in'], 'paths': [path_csv]},
            **{k: cfg['filter'].get(k) for k in ['min_date', 'max_date']}
            )

        if d is None:
            lf.warning('not processing')
            continue

        print(d.compute())






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
