#! /usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
  Purpose:  helper functions for input/output handling
  Author:   Andrey Korzh <ao.korzh@gmail.com>
  Created:  2016 - 2023

  used:
  call_with_valid_kwargs, set_field_if_no
  FakeContextIfOpen, standard_error_info, my_logging
  this_prog_basename, ini2dict, Ex_nothing_done, init_file_names, LoggingStyleAdapter
"""

import sys
from os import path as os_path, listdir as os_listdir, access as os_access, R_OK as os_R_OK, W_OK as os_W_OK
from ast import literal_eval
import enum
from fnmatch import fnmatch
from datetime import timedelta, datetime
from codecs import open
import configparser
import logging
import re
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Iterable, Iterator, BinaryIO, Sequence, TextIO, TypeVar, Tuple, Union
from inspect import currentframe
import io
from functools import wraps  # reduce
from dataclasses import dataclass

if sys.platform == "win32":
    from win32event import CreateMutex
    from win32api import CloseHandle, GetLastError
    from winerror import ERROR_ALREADY_EXISTS


def constant_factory(val):
    def default_val():
        return val
    return default_val


def dicts_values_addition(accumulator, element):
    for key, value in element.items():
        if key in accumulator:
            accumulator[key] += value
        else:
            accumulator[key] = value

    return accumulator


A = TypeVar('A')


def fallible(*exceptions, logger=None) \
        -> Callable[[Callable[..., A]], Callable[..., Optional[A]]]:
    """
    Decorator (very loosely inspired by the Maybe monad and lifting) to return None if specified errors are caught
    :param exceptions: a list of exceptions to catch
    :param logger: pass a custom logger; None means the default logger,
                   False disables logging altogether.

    >>> @fallible(ArithmeticError)
    ... def div(a, b):
    ...     return a / b
    ... div(1, 2)
    0.5


    >>> res = div(1, 0)
    ERROR:root:called <function div at 0x10d3c6ae8> with *args=(1, 0) and **kwargs={}
    Traceback (most recent call last):
        ...
    File "...", line 3, in div
        return a / b

    >>> repr(res)
    'None'
    """

    def fwrap(f: Callable[..., A]) -> Callable[..., Optional[A]]:

        @wraps(f)
        def wrapped(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                (logger or logging).exception('called %s with *args=%s and **kwargs=%s', f, args, kwargs)
                return None

        return wrapped

    return fwrap


def standard_error_info(e):
    msg_trace = '\n==> '.join((s for s in e.args if isinstance(s, str)))
    return f'{e.__class__}: {msg_trace}'


class Ex_nothing_done(Exception):
    def __init__(self, msg=''):
        self.message = f'{msg} => nothing done. For help use "-h" option'

def rerase(msg_before, e: Exception):
    try:
        e.message = msg_before + e.message
    except AttributeError:  # no message attribute
        e.args = (msg_before,) + e.args
    raise


def is_simple_sequence(arg):
    """not map not str, but may be set"""
    return not (isinstance(arg, Mapping) or hasattr(arg, "strip")) and (
        hasattr(arg, "__getitem__") or hasattr(arg, "__iter__"))


readable = lambda f: os_access(f, os_R_OK)
writeable = lambda f: os_access(f, os_W_OK)
l = {}


def dir_walker(root, fileMask='*', bGoodFile=lambda fname, mask: fnmatch(fname, mask),
               bGoodDir=lambda fname: True):
    """

    :param root: upper dir to start search files
    :param fileMask: mask for files to find
    :param bGoodFile: filter for files
    :param bGoodDir:  filter for dirs. If set False will search only in root dir
    :return: list of full names of files found
    """
    if root.startswith('.'):
        root = os_path.abspath(root)
    root = os_path.expanduser(os_path.expandvars(root))
    if readable(root):
        if not os_path.isdir(root):
            yield root
            return
        for fname in os_listdir(root):
            pth = os_path.join(root, fname)
            if os_path.isdir(pth):
                if bGoodDir(fname):
                    for entry in dir_walker(pth, fileMask, bGoodFile, bGoodDir):
                        yield entry
            elif readable(pth) and bGoodFile(fname, fileMask):
                yield pth


# Used in next two functions
bGood_NameEdge = lambda name, namesBadAtEdge: \
    all([name[-len(notUse):] != notUse and name[:len(notUse)] != notUse \
         for notUse in namesBadAtEdge])


def bGood_dir(dirName, namesBadAtEdge):
    if bGood_NameEdge(dirName, namesBadAtEdge):
        return True
    return False


def bGood_file(fname, mask, namesBadAtEdge, bPrintGood=True):
    # any([fname[i] == strProbe for i in range(min(len(fname), len(strProbe) + 1))])
    # in fnmatch.filter(os_listdir(root)
    if fnmatch(fname, mask) and bGood_NameEdge(fname, namesBadAtEdge):
        if bPrintGood: print(fname, end=' ')
        return True
    return False


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


def dir_from_cfg(path_parent: Path, path_child: Union[str, Path]) -> Path:
    """
    Create path_child if needed, if path_child is not absolute path then path_parent will be prepended before
    :param path_parent: parent dir path
    :param path_child: absolute path or relative
    :return: absolute Path(path_child)
    """
    path_child = Path(path_child)
    if not path_child.is_absolute():
        path_child = path_parent / path_child
    return dir_create_if_need(path_child)


# def path2rootAndMask(pathF):
# if pathF[-1] == '\\':
# root, fname = os_path.split(pathF)
# fname= '*'
# else:
# root, fname = os_path.split(pathF)

# return(root, fname)
# fileInF_All, strProbe):
# for fileInF in fileInF_All:
# DataDirName, fname = os_path.split(fileInF)
# if all([DataDirName[-len(noDir):] != noDir for noDir in (r'\bad', r'\test')]) and \
# any([fname[i] == strProbe for i in range(min(len(fname), len(strProbe) + 1))]) \
# and fname[-4:] == '.txt' and fname[:4] != 'coef':
##fileInF = os_path.join(root, fname)
# print(fname, end=' ')
# yield (DataDirName, fname)

def first_of_paths_text(paths):
    # Get only first path from paths text
    iSt = min(paths.find(r':', 3) - 1, paths.find(r'\\', 3)) + 2
    iEn = min(paths.find(r':', iSt) - 1, paths.find(r'\\', iSt))
    return paths[iSt - 2:iEn].rstrip('\\\n\r ')


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


def getDirBaseOut(mask_in_path, raw_dir_words: Optional[Sequence[str]]=None, replaceDir:str=None):
    """
    Finds 'Cruise' and 'Device' dirs. Also returns full path to 'Cruise'.
    If 'keyDir' in fileMaskIn and after 2 levels of dirs then treat next subsequence as:
    ...\\'keyDir'\\'Sea'\\'Cruise'\\'Device'\\... i.e. finds subdirs after 'keyDir'
    Else use subsequence before 'keyDir' (or from end of fileMaskIn if no ``keyDir``):
    ...\\'Sea'\\'Cruise'
    :param mask_in_path: path to analyse
    :param raw_dir_words: list of str - list of variants to find "keyDir" in priority order
    :param replaceDir: str, "dir" to replace "keyDir" in out_path
        + used instead "Device" dir if "keyDir" not in fileMaskIn
    :return: returns tuple, which contains:
    #1. out_path: full path to "Cruise" (see #2.)
        - if "replaceDir" is not None: with "keyDir" is replaced by "replaceDir"
        i.e. modify dir before "Sea"
    #2. "Cruise" dir: subdir of subdir of keyDir
        - if "keyDir" dir not found: parent dir of "Device" dir
        - if "Cruise" dir not found: parent of last subdir in fileMaskIn
    #3. "Device" dir: subdir of subdir of subdir of "keyDir"
        - if "keyDir" dir not found: "replaceDir" (or "" if "replaceDir" is None)
        - if "Cruise" dir not found: last subdir
    """
    # if isinstance(mask_in_path, PurePath):
    mask_in_str = str(mask_in_path)

    st = -1
    for source_dir_word in raw_dir_words:
        # Start of source_dir_word in 1st detected variant
        st = mask_in_str.find(source_dir_word, 3)  # .lower()
        if st >= 0: break

    if st < 0:
        print(
            "Directory structure should be ...*{}{}'Sea'{}'Cruise'{}'Device'{}!".format(
                source_dir_word, os_path.sep, os_path.sep, os_path.sep, os_path.sep
            )
        )
        out_path, cruise = os_path.split(mask_in_str)
        return out_path, cruise, ("" if replaceDir is None else replaceDir)

    else:
        parts_of_path = Path(mask_in_str[st:]).parts
        if len(parts_of_path) <= 2:
            # use last dirs for "\\'Sea'\\'Cruise'\\'Device'\\'{}'"
            path_device = Path(mask_in_str[:st])
            parts_of_path = path_device.parts
            cruise, device = parts_of_path[-2:]
            out_path = path_device.parent / replaceDir if replaceDir else path_device.parent
        else:
            cruise = parts_of_path[2]  # after keyDir and Sea
            try:
                device = parts_of_path[3]
            except IndexError:
                device = ''
            if replaceDir:
                out_path = Path(mask_in_str[:st]) / replaceDir / Path.joinpath(*parts_of_path[1:3])
            else:
                out_path = Path(mask_in_str[:st]).joinpath(*parts_of_path[:3])  # cruise path

        return str(out_path), cruise, device


def cfgfile2dict(arg_source: Union[Mapping[str, Any], str, PurePath, None] = None
                 ) -> Tuple[Union[Dict, configparser.RawConfigParser], Union[str, PurePath], str]:
    """
    Loads config to dict or passes dict though
    :param arg_source: one of:
        - path of *.ini or yaml file.
        - None - same effect as path name of called program with ini extension.
        - dict
    :return (config, arg_source):
        - config:
            dict loaded from yaml file if arg_source is name of file with yaml o yml extension
            configparser.RawConfigParser initialised with arg_source (has dict interface too)
        - file_path, arg_ext:
            arg_source splitted to base and ext if arg_source is not dict
            else '<dict>' and ''
    """

    def set_config():
        cfg = configparser.RawConfigParser(inline_comment_prefixes=(';',))  # , allow_no_value = True
        cfg.optionxform = lambda option: option  # do not lowercase options
        return cfg

    if not arg_source:
        return set_config(), '<None>', ''

    b_path = isinstance(arg_source, PurePath)
    if isinstance(arg_source, str) or b_path:
        # Load data from config file
        # Set default name of config file if it is not specified.

        if not b_path:
            arg_source = PurePath(arg_source)
        if not arg_source.is_absolute():
            arg_source = (Path(sys.argv[0]).parent / arg_source).resolve()
        arg_ext = arg_source.suffix
        try:
            dir_create_if_need(arg_source.parent)
        except FileNotFoundError:  # path is not consist of less than 1 level of new subdirs
            print('Dir of config file "{}" not found, continue without...'.format(arg_source))
            config = {}
        else:
            if arg_ext.lower() in ['.yml', '.yaml']:
                """ lazy-import PyYAML so that we doesn't have to dependend
                    on it unless this parser is used
                """
                try:
                    from ruamel.yaml import safe_load as yaml_safe_load
                except ImportError:
                    try:
                        from yaml import safe_load as yaml_safe_load
                    except ImportError:
                        raise ImportError(
                            "Could not import yaml or ruamel.yaml. It can be installed by running any of combinations:"
                            "pip/conda install PyYAML/ruamel.yaml")
                try:
                    with open(arg_source, encoding='utf-8') as f:
                        config = yaml_safe_load(f.read())
                except FileNotFoundError:  # path is not constist of less than 1 level of new subdirs
                    print('Ini file "{}" dir not found, continue...'.format(arg_source))
                    config = {}
            else:
                cfg_file = arg_source.with_suffix('.ini')
                # if not os_path.isfile(cfg_file):
                config = set_config()
                try:
                    with open(cfg_file, 'r', encoding='cp1251') as f:
                        config.read(cfg_file)
                except FileNotFoundError:
                    print('Ini file "{}" not found, continue...'.format(cfg_file))  # todo: l.warning
                    config = {}

    elif isinstance(arg_source, Mapping):
        # config = set_config()
        # config.read_dict(arg_source)  # todo: check if it is need
        config = arg_source
        arg_source = '<mapping>'
        arg_ext = ''
    return config, arg_source, arg_ext


def type_fix(name: str, opt: Any) -> Tuple[str, Any]:
    """
    Checking special words in parts of option's name splitted by '_'

    :param name: option's name. If special prefix/suffix provided then opt's type will be converted accordingly
    :param opt: option's value, usually str that need to convert to the type specified by oname's prefix/suffix.
    For different oname's prefix/suffixes use this formatting rules:
    - 'dict': do not use curles, field separator: ',' if no '\n' in it else '\n,', key-value separator: ': ' (or ':' if no ': ')
    - 'list', 'names': do not use brackets, item separator: "'," if fist is ``'`` else '",' if fist is ``"`` else ','

    ...
    :return: (new_name, new_opt)
    """

    key_splitted = name.split('_')
    key_splitted_len = len(key_splitted)
    if key_splitted_len < 1:
        return name, opt
    else:
        prefix = key_splitted[0]
        suffix = key_splitted[-1] if key_splitted_len > 1 else ''
    name_out = None


    def val_type_fix(parent_name, field_name, field_value):
        """
        Modify type of field_value basing on parent_name (dicts should have keys that defines types of values)
        """
        nonlocal name_out
        name_out, val = type_fix(parent_name, field_value)
        return field_name, val


    try:
        if is_simple_sequence(opt):
            opt = [type_fix('_'.join(key_splitted[0:-1]) if suffix in {'list', 'names'} else name, v)[1] for v in opt]
        if suffix in {'list', 'names'}:  # , '_endswith_list' -> '_endswith'
            # parse list. Todo: support square brackets
            name_out = '_'.join(key_splitted[0:-1])
            if not opt:
                opt_list_in = [None]
            elif opt[0] == "'":  # split to strings separated by "'," stripping " ',\n"
                opt_list_in = [n.strip(" ',\n") for n in opt.split("',")]
            elif opt[0] == '"':  # split to strings separated by '",' stripping ' ",\n'
                opt_list_in = [n.strip(' ",\n') for n in opt.split('",')]
            else:  # split to strings separated by ','
                opt_list_in = [n.strip() for n in opt.split(',')]

            opt_list = []
            for opt_in in opt_list_in:
                name_out_in, val_in = type_fix(name_out, opt_in)  # process next suffix
                opt_list.append(val_in)
            return name_out_in, ([] if opt_list_in == [None] else opt_list)


            # suffix = key_splitted[-2]  # check next suffix:
            # if suffix in {'int', 'integer', 'index'}:
            #     # type of list values is specified
            #     name_out = '_'.join(key_splitted[0:-2])
            #     return name_out, [int(n) for n in opt.split(',')] if opt else []
            # elif suffix in {'b', 'bool'}:
            #     name_out = '_'.join(key_splitted[0:-2])
            #     return name_out, list(literal_eval(opt))
            # else:
            #     name_out = '_'.join(key_splitted[0:-1])
            #     if not opt:
            #         return name_out, []
            #     elif opt[0] == "'":  # split to strings separated by "'," stripping " ',\n"
            #         return name_out, [n.strip(" ',\n") for n in opt.split("',")]
            #     elif opt[0] == '"':  # split to strings separated by '",' stripping ' ",\n'
            #         return name_out, [n.strip(' ",\n') for n in opt.split('",')]
            #     else:  # split to strings separated by ','
            #         return name_out, [n.strip() for n in opt.split(',')]
        if isinstance(opt, Mapping):
            if opt:
                opt_out = dict([val_type_fix(name[:-1] if name.endswith('lists') else name, n, v)
                            for n, v in opt.items()])
                # Note: global name_out was changed during val_type_fix()
            else:  # Call type_fix() only to set name_out
                name_out, val = type_fix(name[:-1] if name.endswith('lists') else name, None)
                opt_out = {}

            return name_out or name, opt_out

        if suffix == 'dict':
            # remove dict suffix when convert to dict type
            name_new = '_'.join(key_splitted[0:-1])
            if opt is None:
                # remove allowed key suffixes in the cfg name and return it with  value  #
                while True:
                    name_out, dict_fixed = type_fix(name_new, None)  # Temporary set opt = None because type_fix() not changes names if val is dict
                    if name_new == name_out:
                        break
                    name_new = name_out  # saving previous value for compare to exit cycle
                return name_out, {}  # None value to empty dict
            elif isinstance(opt, str):
                sep = '\n,' if ',\n' in opt else ','
                dict_fixed = dict([val_type_fix(
                    name_new, *n.strip().split(': ' if ': ' in n else ':', maxsplit=1)
                ) for n in opt.split(sep) if len(n)])
                return name_out, dict_fixed
            else:
                return type_fix(name_new, opt)  # ???


        if prefix == 'b':
            return name, literal_eval(opt)
        # if prefix == 'time':
        #     return oname, datetime.strptime(opt, '%Y %m %d %H %M %S')
        if prefix == 'dt':
            if suffix in {'days', 'seconds', 'microseconds', 'milliseconds', 'minutes', 'hours', 'weeks'}:
                name_out = '_'.join(key_splitted[:-1])
                if opt:
                    try:
                        opt = timedelta(**{suffix: float(opt)})
                    except TypeError as e:
                        raise KeyError(e.msg) from e  # changing error type to be not caught and accepted
                else:  # not need convert
                    opt = timedelta(0)
            else:  # do nothing
                name_out = name
            return name_out, opt
        b_trig_is_prefix = prefix in {'date', 'time'}
        if b_trig_is_prefix or suffix in {'date', 'time'}:
            if b_trig_is_prefix or prefix in {'min', 'max'}:  # not strip to 'min', 'max'
                name_out = name
                # = opt_new  #  use other temp. var instead name_out to keep name (see last "if" below)???
            else:
                name_out = '_'.join(key_splitted[0:-1])  # #oname = del suffix
                # name_out = opt_new  # will del old name (see last "if" below)???
            date_format = '%Y-%m-%dT'
            if '-' not in opt[:len(date_format)]:
                date_format = '%d.%m.%Y '
            try:  # opt has only date part?
                return name_out, datetime.strptime(opt, date_format[:-1])
            except ValueError:
                fmt_len = len(opt) - len(date_format) - 2
                n_hms = opt.count(':')
                if n_hms < 2:  # no seconds
                    time_format = '%H:%M'[:fmt_len] if fmt_len <= 5 else '%H:%M%z'
                else:
                    time_format = '%H:%M:%S%z'[:fmt_len]  # minus 2 because 2 chars of '%Y' corresponds 4 digits of year
                try:
                    tim = datetime.strptime(opt, f'{date_format}{time_format}')
                except ValueError:
                    if opt == 'NaT':  # fallback to None
                        tim = None
                    elif opt == 'now':
                        tim = datetime.now()
                    else:
                        raise
                return name_out, tim
        if suffix in {'int', 'integer', 'index'}:
            name_out = '_'.join(key_splitted[0:-1])
            return name_out, int(opt)
        if suffix == 'float':  # , 'percent'
            name_out = '_'.join(key_splitted[0:-1])
            return name_out, float(opt)
        if suffix in {'b', 'bool'}:
            name_out = '_'.join(key_splitted[0:-1])
            return name_out, literal_eval(opt)
        if suffix == 'chars':
            name_out = '_'.join(key_splitted[0:-1])
            return name_out, opt.replace('\\t', '\t').replace('\\ \\', ' ')
        if prefix in {'fixed', 'float', 'max', 'min'}:
            # this prefixes is at end because of 'max'&'min' which can be not for float,
            # so set to float only if have no other special format words
            return name, float(opt)

        if 'path' in {suffix, prefix}:
            return name, Path(opt) if opt not in ('Null', 'None') else None

        return name, opt
    except (TypeError, AttributeError, ValueError) as e:
        # do not try to convert not a str, also return None for "None"
        if not isinstance(opt, (str, dict)):
            return name_out if name_out else name, opt  # name_out is a replacement of oname
        elif opt == 'None':
            return name_out, None
        else:
            raise e


def ini2dict(arg_source: Union[Mapping[str, Any], str, PurePath, None] = None):
    """
    1. Loads configuration dict from *.ini file (if arg is not a dict already)
    2. Type conversion based on keys names. During this removes suffix type indicators but keeps prefiх.
    Prefiх/suffics type indicators (followed/preceded with "_"):
      - b
      - chars - to list of chars, use string "\\ \\" to specify space char
      - time
      - dt (prefix only) with suffixes: ... , minutes, hours, ... - to timedelta
      - list, (names - not recommended) - splitted on ',' but if first is "'" then on "'," - to allow "," char, then all "'" removed.
        If first list characters is " or ' then breaks list on " ',\n" or ' ",\n' correspondingly.
        before list can be other suffix to convert to
      - int, integer, index - to integer
      - float - to float

    :param arg_source: path of *.ini file. if None - use name of program called with
        ini extension.
    :return: dict - configuration parsed

    Uses only ".ini" extension nevertheless which was cpecified or was specified at all
    """

    config, arg_source, arg_ext = cfgfile2dict(arg_source)
    cfg = {key: {} for key in config}
    oname = '...'
    opt = None
    # convert specific fields data types
    try:
        for sname, sec in config.items():
            if sname[:7] == 'TimeAdd':
                d = {opt: float(opt) for opt in sec}
                cfg[sname] = timedelta(**d)
            else:
                if not hasattr(sec, 'items'):
                    continue
                opt_used = set()
                for oname, opt in sec.items():
                    new_name, val = type_fix(oname, opt)
                    # if new_name in sec:       # to implement this first set sec[new_name] to zero
                    #     val += sec[new_name]  #

                    if new_name in opt_used and cfg[sname][new_name] is not None:
                        if opt is None:  # val == opt if opt is None # or opt is Nat
                            continue
                        if isinstance(opt, Mapping):
                            if isinstance(cfg[sname][new_name], Mapping):
                                cfg[sname][new_name].update(opt)
                            else:  # Mapping will overwrite simple values
                                cfg[sname][new_name] = val
                        else:
                            cfg[sname][new_name] += val
                    else:
                        cfg[sname][new_name] = val
                        opt_used.add(new_name)
                # cfg[sname]= dict(config.items(sname)) # for sname in config.sections()]
    except Exception as e:  # ValueError, TypeError
        # l.exception(e)
        rerase(f'Error_in_config_parameter: [{sname}].{oname} = "{str(opt)}": {e.args[0]} > ', e)
        # e.with_traceback(e.__traceback__) from e

    set_field_if_no(cfg, 'in', {})
    cfg['in']['cfgFile'] = arg_source
    return cfg


def cfg_from_args_yaml(p, arg_add, **kwargs):
    """
    todo: version using Hydra
    Split sys.argv to ``arg_source`` for loading config (data of 2nd priority) and rest (data of 1st priority)
    arg_source = sys.argv[1] if it not starts with '--' else tries find file sys.argv[0] + ``.yaml`` or else ``.ini``
    and assigns arg_source to it.
    Loaded data (configargparse object) is converted to configuration dict of dicts

    1st priority (sys.argv[2:]) will overwrite all other
    See requirements for command line arguments p
    (argument_groups (sections) is top level dict)

    :param p: configargparse object of parameters. 1st command line parameter in it
     must not be optional. Also it may be callable with arguments to init configargparse.ArgumentParser()
    :param arg_add: list of string arguments in format of command line interface defined by p to add/replace parameters
     (starting from 2nd)
    :param kwargs: dicts for each section: to overwrite values in them (overwrites even high priority values, other values remains)
    :return cfg: dict with parameters
        will set cfg['in']['cfgFile'] to full name of used config file
        '<prog>' strings in p replaces with p.prog
    see also: my_argparser_common_part()
    """

    def is_option_name(arg):
        """
        True if arg is an option name (i.e. that is not a value, which must be followed)
        :param arg:
        :return:
        """
        return isinstance(arg, str) and arg.startswith('--')

    args = {}  # will be converted to dict cfg:
    cfg = None


def cfg_from_args(p, arg_add, **kwargs):
    """
    Split sys.argv to ``arg_source`` for loading config (data of 2nd priority) and rest (data of 1st priority)
    arg_source = sys.argv[1] if it not starts with '--' else tries find file sys.argv[0] + ``.yaml`` or else ``.ini``
    and assigns arg_source to it.
    Loaded data (configargparse object) is converted to configuration dict of dicts

    1st priority (sys.argv[2:]) will overwrite all other
    See requirements for command line arguments p
    (argument_groups (sections) is top level dict)

    :param p: configargparse object of parameters. 1st command line parameter in it
     must not be optional. Also it may be callable with arguments to init configargparse.ArgumentParser()
    :param arg_add: list of string arguments in format of command line interface defined by p to add/replace parameters
     (starting from 2nd)
    :param kwargs: dicts for each section: to overwrite values in them (overwrites even high priority values, other values remains)
    :return cfg: dict with parameters
        will set cfg['in']['cfgFile'] to full name of used config file
        '<prog>' strings in p replaces with p.prog
    see also: my_argparser_common_part()
    """
    from argparse import ArgumentError

    def is_option_name(arg):
        """
        True if arg is an option name (i.e. that is not a value, which must be followed)
        :param arg:
        :return:
        """
        return isinstance(arg, str) and arg.startswith('--')

    args = {}  # will be converted to dict cfg:
    cfg = None

    if arg_add:
        argv_save = sys.argv.copy()

        # # todo: exclude arguments that are not strings from argparse processing (later we add them to cfg)
        # # iterate though values
        # for i in range(start=(1 if is_option_name(arg_add[0]) else 0), end=len(arg_add), step=2):
        #     if not isinstance(arg_add[i], str):
        #         arg_add
        sys.argv[1:] = arg_add

    skip_config_file_parsing = False  # info argument (help, version) was passed?
    if len(sys.argv) > 1 and not is_option_name(sys.argv[1]):
        # name of config file (argument source) is specified
        arg_source = sys.argv[1]
    else:
        # auto search config file (source of arguments)
        exe_path = Path(sys.argv[0])
        cfg_file = exe_path.with_suffix('.yaml')
        if not cfg_file.is_file():  # do I need check Upper case letters for Linux?
            cfg_file = exe_path.with_suffix('.yml')
            if not cfg_file.is_file():
                cfg_file = exe_path.with_suffix('.ini')
                if not cfg_file.is_file():
                    print(f'note: default configuration file {exe_path.name}.ini|yml is not exist')
                    cfg_file = None
        sys.argv.insert(1, str(cfg_file))
        if cfg_file:
            print('using configuration from file:', cfg_file)
        arg_source = cfg_file

    if len(sys.argv) > 2:
        skip_config_file_parsing = sys.argv[2] in ["-h", "--help", "-v", "--version"]
        if skip_config_file_parsing:
            if callable(p):
                p = p(None)
            args = vars(p.parse_args())  # will generate SystemExit

    # Type suffixes '_list', '_int' ... (we need to remove if type was converted, for example config loaded from yaml)
    suffixes = {'_list', '_int', '_integer', '_index', '_float', '_b', '_bool', '_date', '_chars', '_dict'}
    re_suffixes = re.compile(f"({'|'.join(suffixes)})+$")

    # Load options from ini file
    config, arg_source, arg_ext = cfgfile2dict(arg_source)

    # todo: check this replacing of configargparse back to argparse to get rid of double loading of ini/yaml
    # try:
    #
    #     from configargparse import YAMLConfigFileParser, ArgumentError
    #
    #     if callable(p):
    #         p = p({'config_file_parser_class': YAMLConfigFileParser
    #                } if arg_ext.lower() in ('.yml', '.yaml') else None)
    # except ImportError:


    if callable(p):
        p = p()



    # Collect argument groups
    p_groups = {g.title: g for g in p._action_groups if
                g.title.split(' ')[-1] != 'arguments'}  # skips special argparse groups

    def get_or_add_sec(section_name, p_groups, sec_description=None):
        if section_name in p_groups:
            p_sec = p_groups[section_name]
        else:
            p_sec = p.add_argument_group(section_name, sec_description)
            p_groups[section_name] = p_sec
        return p_sec

    # Overwrite hardcoded defaults from ini in p: this is how we make it 2nd priority and defaults - 3rd priority
    if config:
        prefix = '--'
        for section_name, section in config.items():
            try:  # now "if not isinstance(section, dict)" is not works, "if getattr(section, 'keys')" still works but "try" is more universal
                ini_sec_options = set(section)  # same as set(section.keys())
            except Exception as e:
                continue

            p_sec = get_or_add_sec(section_name, p_groups)
            p_sec_hardcoded_list = [a.dest for a in p_sec._group_actions]
            ini_sec_options_new = ini_sec_options.difference(set(p_sec_hardcoded_list))
            ini_sec_options_same = ini_sec_options.difference(ini_sec_options_new)
            # get not hardcoded options from ini:
            for option_name in ini_sec_options_new:
                if not isinstance(option_name, str):
                    continue
                try:
                    # if parameter comes from yaml file and not contain type suffix it is not overridden by command line
                    # so override if in command line
                    if (len(sys.argv) > 1) and not isinstance(section[option_name], str):
                        for arg in sys.argv[2::2]:
                            if arg[2:].startswith(option_name) and option_name == re_suffixes.sub('', arg[2:]):
                                b_override = True
                                break
                        else:
                            # nothing to override
                            b_override = False
                        if b_override:
                            continue  # will use what is come in command line instead

                    # p_sec.set_defaults(**{'--' + option_name: config.get(section_name, option_name)})
                    p_sec.add(f'{prefix}{option_name}', default=section[option_name])
                except ArgumentError as e:
                    # Same options name but in other ini section
                    option_name_changed = f'{section_name}.{option_name}'
                    try:
                        p_sec.add(f'{prefix}{option_name_changed}', default=section[option_name])
                    except ArgumentError:
                        # Changed option name was hardcoded so replace defaults defined there
                        p_sec._group_actions[p_sec_hardcoded_list.index(option_name_changed)].default = section[
                            option_name]

            # overwriting
            for option_name in ini_sec_options_same:
                p_sec._group_actions[p_sec_hardcoded_list.index(option_name)].default = section[option_name]

    # Append arguments with my common options:
    p_sec = get_or_add_sec('program', p_groups, 'Program behaviour')
    try:
        p_sec.add_argument(
            '--b_interact', default='True',
            help='ask showing source files names before process them')
    except ArgumentError as e:
        pass  # option already exist - need no to do anything
    try:
        p_sec.add_argument(
            '--log', default=os_path.join('log', f'{this_prog_basename()}.log'),
            help='write log if path to existed file is specified')
    except ArgumentError as e:
        pass  # option already exist - need no to do anything
    try:
        p_sec.add_argument(
            '--verbose', '-V', type=str, default='INFO',  # nargs=1,
            choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
            help='verbosity of messages in log file')
    except ArgumentError as e:
        pass  # option already exist - need no to do anything

    # os_path.join(os_path.dirname(__file__), 'empty.yml')
    sys.argv[1] = ''  # do not parse ini file by configargparse already parsed by configparser
    try:
        args = vars(p.parse_args())
    except SystemExit as e:
        if skip_config_file_parsing:
            raise

        # Bad arguments found. Error message of unrecognized arguments displayed,
        # but we continue. To remove message add arguments to p before call this func
        pass

    # args = vars(p.parse_args())

    try:
        # Collect arguments dict to groups (ini sections) ``cfg``
        cfg = {}
        for section_name, gr in p_groups.items():
            keys = args.keys() & [a.dest for a in gr._group_actions]
            cfg_section = {}
            for key in keys:
                arg_cur = args[key]
                if isinstance(arg_cur, str):
                    arg_cur = arg_cur.replace('<prog>', p.prog).strip()
                if '.' in key:
                    key = key.split('.')[1]
                cfg_section[key] = arg_cur

            cfg[section_name] = cfg_section

        if arg_ext.lower() not in ('.yml', '.yaml'):
            # change types based on prefix/suffix
            cfg = ini2dict(cfg)
        # else type suffixes will be removed in cycle below

        # Convert cfg['re mask'] descendants and all str (or list of str but not starting from '') chields beginning with 're_' to compiled regular expression object
        # lists in 're_' are joined to strings before compile (useful as list allows in yaml to use aliases for part of re expression)

        for key_level0, v in cfg.items():
            if key_level0 == 're_mask':  # replace all chields to compiled re objects
                for key_level1, opt in v.items():
                    cfg['re_mask'][key_level1] = re.compile(opt)
            else:
                opt_used = set()
                for key_level1, opt in v.copy().items():
                    if key_level1.startswith('re_'):  # replace strings beginning with 're_' to compiled re objects
                        is_lst = isinstance(opt, list)
                        if is_lst:
                            if key_level1.endswith('_list'):  # type already converted, remove type suffix
                                new_key = key_level1[:-len('_list')]
                                v[new_key] = opt
                                del v[key_level1]
                                key_level1 = new_key

                            try:  # skip not list of str
                                if (not isinstance(opt[0], str)) or (not opt[0]):
                                    continue
                            except IndexError:
                                continue
                            v[key_level1] = re.compile(''.join(opt))
                        elif opt is not None:
                            v[key_level1] = re.compile(opt)
                    elif not (opt is None or isinstance(opt, (str, dict))):
                        # type already converted, remove type suffixes here only
                        new_name, n_rep = re_suffixes.subn('', key_level1)
                        if n_rep:
                            # exclusion for min_date and max_date
                            if not (key_level1.endswith(('_date', '_date_dict')) and new_name.startswith(('min', 'max', 'b_'))):
                                # todo: exclude all excisions that leave only special prefixes
                                v[new_name] = opt
                                del v[key_level1]

                        # for ends in suffixes:
                        #     if key_level1.endswith(ends):
                        #         new_name = key_level1[:-len(ends)]
                        #         if ends == '_date' and new_name.startswith(('min', 'max')):  # exclusion for min_date and max_date
                        #             break  # todo: exclude all excisions that leave only special prefixes
                        #         v[new_name] = opt
                        #         del v[key_level1]
                        #         break
                    else:
                        # type not converted, remove type suffixes and convert
                        new_name, val = type_fix(key_level1, opt)
                        if new_name == key_level1:               # if only type changed
                            if v[new_name] != val:
                                if v[new_name]:
                                    print(f'config value overwritten: {key_level0}.{new_name} = {v[new_name]} -> {val}')
                                v[new_name] = val
                        else:
                            # replace old key
                            if new_name not in v:  # if not str (=> not default) parameter without suffixes added already
                                v[new_name] = val
                                opt_used.add(new_name)  # to add dt with obtained from args of different suffixes: i.e. minutes to hours
                            elif new_name in opt_used:
                                if val:
                                    if isinstance(val, dict):
                                        v = dicts_values_addition(v, val)
                                    else:
                                        v[new_name] += val
                            elif v[new_name] and v[new_name] != val:
                                print(f'config {key_level1} value {val} ignored: {key_level0}.{new_name} = {v[new_name]} keeped')
                            del v[key_level1]   # already converted

        if kwargs:
            for key_level0, kwargs_level1 in kwargs.items():
                cfg[key_level0].update(kwargs_level1)

        cfg['in']['cfgFile'] = arg_source
        # config = configargparse.parse_args(cfg)
    except Exception as e:  # IOError
        print('Configuration ({}) error:'.format(arg_add), end=' ')
        print('\n==> '.join([s for s in e.args if isinstance(s, str)]))  # getattr(e, 'message', '')
        raise e
    finally:
        if arg_add:  # recover argv for possible outer next use
            sys.argv = argv_save

    return cfg


# class MyArgparserCommonPart(configargparse.ArgumentParser):
#     def init(self, default_config_files=[],
#              formatter_class=configargparse.ArgumentDefaultsRawHelpFormatter, epilog='',
#              args_for_writing_out_config_file=["-w", "--write-out-config-file"],
#              write_out_config_file_arg_help_message="takes the current command line arguments and writes them out to a configuration file the given path, then exits. But this file have no section headers. So to use this file you need to add sections manually. Sections are listed here in help message: [in], [out] ...",
#              ignore_unknown_config_file_keys=True, version='?'):
#         self.add('cfgFile', is_config_file=True,
#                  help='configuration file path(s). Command line parameters will overwrites parameters specified iside it')
#         self.add('--version', '-v', action='version', version=
#         f'%(prog)s version {version} - (c) 2017 Andrey Korzh <ao.korzh@gmail.com>.')
#
#         # Configuration sections
#
#         # All argumets of type str (default for add_argument...), because of
#         # custom postprocessing based of args names in ini2dict
#
#         '''
#         If "<filename>" found it will be sabstituted with [1st file name]+, if "<dir>" -
#         with last ancestor directory name. "<filename>" string
#         will be sabstituted with correspondng input file names.
#         '''
#
#         # p_program = p.add_argument_group('program', 'Program behaviour')
#         # p_program.add_argument(
#         #     '--verbose', '-V', type=str, default='INFO', #nargs=1,
#         #     choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
#         #     help='verbosity of messages in log file')
#         return (self)


def my_argparser_common_part(varargs, version='?'):  # description, version='?', config_file_paths=[]
    """
    Define configuration
    :param varargs: dict, containing configargparse.ArgumentParser parameters to set
        Note: '-' in dict keys will be replaced  to '_' in ArgumentParser
    :param version: value for `version` parameter
    :return p: configargparse object of parameters
    """
    varargs.setdefault('epilog', '')

    try:
        import configargparse
        # varargs.setdefault('default_config_files', [])
        # varargs.setdefault('formatter_class', configargparse.ArgumentDefaultsRawHelpFormatter)
        # formatter_class= configargparse.ArgumentDefaultsHelpFormatter,
        # varargs.setdefault('args_for_writing_out_config_file', ["-w", "--write-out-config-file"])
        # varargs.setdefault('write_out_config_file_arg_help_message',
        #                    "takes the current command line arguments and writes them out to a configuration file the given path, then exits. But this file have no section headers. So to use this file you need to add sections manually. Sections are listed here in help message: [in], [out] ...")
        # varargs.setdefault('ignore_unknown_config_file_keys', True)
        # p = configargparse.ArgumentParser(**varargs)
    except ImportError:
        from argparse import ArgumentParser
        p = ArgumentParser(**varargs)
        p.add = p.add_argument

        def with_alias_to_add_argument(fun, *args, **kwargs):
            """
            Makes alias to "add_argument" method of add_argument_group() result
            :param p:
            :return:
            """
            def fun_mod(*args, **kwargs):
                out = fun(*args, **kwargs)
                out.add = out.add_argument
                return out

            return fun_mod

        p.add_argument_group = with_alias_to_add_argument(p.add_argument_group)

    p.add(
        'cfgFile',  # is_config_file=True,
         help='configuration file path(s). Command line parameters will overwrites parameters specified inside it'
        )
    p.add('--version', '-v', action='version', version=
    '%(prog)s version {version} - (c) 2022 Andrey Korzh <ao.korzh@gmail.com>.')

    # Configuration sections

    # All argumets of type str (default for add_argument...), because of
    # custom postprocessing based of args names in ini2dict

    '''
    If "<filename>" found it will be substituted with [1st file name]+, if "<dir>" -
    with last ancestor directory name. "<filename>" string
    will be substituted with corresponding input file names.
    '''

    # p_program = p.add_argument_group('program', 'Program behaviour')
    # p_program.add_argument(
    #     '--verbose', '-V', type=str, default='INFO', #nargs=1,
    #     choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
    #     help='verbosity of messages in log file')

    return p


def pathAndMask(path: str, filemask=None, ext=None):
    """
    Depreciated!
    Find Path & Mask
    :param path:
    :param filemask:
    :param ext:
    :return: (dir, filemask)

    # File mask can be specified in "path" (for example full path) it has higher priority than
    # "filemask" which can include ext part which has higher priority than specified by "ext"
    # But if turget file(s) has empty name or ext then they need to be specified explisetly by ext = .(?)
    """
    path, fileN_fromCfgPath = os_path.split(path)
    if fileN_fromCfgPath:
        if '.' in fileN_fromCfgPath:
            fileN_fromCfgPath, cfg_path_ext = os_path.splitext(fileN_fromCfgPath)
            if cfg_path_ext:
                cfg_path_ext = cfg_path_ext[1:]
            else:
                cfg_path_ext = fileN_fromCfgPath[1:]
                fileN_fromCfgPath = ''
        elif '*' in fileN_fromCfgPath:
            return path, fileN_fromCfgPath
        else:  # wrong split => undo
            cfg_path_ext = ''
            path = os_path.join(path, fileN_fromCfgPath)
            fileN_fromCfgPath = ''
    else:
        cfg_path_ext = ''

    if filemask is not None:
        fileN_fromCfgFilemask, cfg_filemask_ext = os_path.splitext(filemask)
        if '.' in cfg_filemask_ext:
            if not cfg_path_ext:
                # possible use ext. from ['filemask']
                if not cfg_filemask_ext:
                    cfg_path_ext = fileN_fromCfgFilemask[1:]
                elif cfg_filemask_ext:
                    cfg_path_ext = cfg_filemask_ext[1:]

        if not fileN_fromCfgPath:
            # use name from ['filemask']
            fileN_fromCfgPath = fileN_fromCfgFilemask
    elif not fileN_fromCfgPath:
        fileN_fromCfgPath = '*'

    if not cfg_path_ext:
        # check ['ext'] exists
        if ext is None:
            cfg_path_ext = '*'
        else:
            cfg_path_ext = ext

    filemask = f'{fileN_fromCfgPath}.{cfg_path_ext}'
    return path, filemask


# ----------------------------------------------------------------------
def pairwise(iterable):
    """s -> (s0, s1), (s2, s3), (s4, s5), ..."""
    a = iter(iterable)
    return zip(a, a)


def generator_good_between(i_start=None, i_end=None):
    k = 0
    if i_start is not None:
        while k < i_start:
            yield False
            k += 1
    if i_end is not None:
        while k < i_end:
            yield True
            k += 1
        while True:
            yield False
    while True:
        yield True


def init_file_names(
        path=None, filemask=None, ext=None,
        b_search_in_subdirs=False,
        exclude_dirs_endswith=('bad', 'test'),
        exclude_files_endswith=None,
        start_file=0,
        end_file=None,
        b_interact=True,
        cfg_search_parent=None,
        msg_action='search for',
        **kwargs):
    """
      Fill cfg_files fields of file names: {'path', 'filemask', 'ext'}
    which are not specified.
      Searches for files with this mask. Prints number of files found.
      If any - asks user to proceed and if yes returns its names list.
      Else raises Ex_nothing_done exception.

    :param path: name of file
    :param filemask, ext: optional - path mask or its part
    :param exclude_files_endswith - additional filter for ends in file's names
    :param b_search_in_subdirs, exclude_dirs_endswith - to search in dirs recursively
    :param start_file, end_file - exclude files before and after this values in search list result
    :param b_interact: do ask user to proceed? If false proseed silently
    :param msg_action: object
    :param kwargs: not used
    :return: (paths, nfiles, path)
    - paths: list of full names of found files
    - nfiles: number of files found,
    - path:
    """
    # ensure absolute path:
    path = set_cfg_path_filemask(path, filemask, ext, cfg_search_parent)
    # Filter unused directories and files
    filt_dirCur = lambda f: bGood_dir(f, namesBadAtEdge=exclude_dirs_endswith)

    def skip_to_start_file(fun):
        if start_file or end_file:

            if isinstance(start_file, str) or isinstance(end_file, str):
                if end_file is None:
                    fun_skip = lambda name: name >= start_file
                elif start_file is None:
                    fun_skip = lambda name: end_file > name
                else:
                    fun_skip = lambda name: end_file > name >= start_file

                def call_skip(*args, **kwargs):
                    return fun(*args, **kwargs) and fun_skip(args[0])

            else:
                fun_skip = generator_good_between(start_file, end_file)

                def call_skip(*args, **kwargs):
                    return fun(*args, **kwargs) and next(fun_skip)

            return call_skip
        return fun

    def skip_files_endswith(fun):
        call_skip = lambda *args, **kwargs: fun(*args, namesBadAtEdge=exclude_files_endswith) if \
            exclude_files_endswith else fun(*args, namesBadAtEdge=('coef.txt',))
        return call_skip

    def print_file_name(fun):
        def call_print(*args, **kwargs):
            if fun(*args, **kwargs):
                print(args[0], end=' ')
                return True
            else:
                return False

        return call_print

    bPrintGood = True
    if not bPrintGood:
        print_file_name = lambda fun: fun

    @print_file_name
    @skip_files_endswith
    @skip_to_start_file
    def filt_file_cur(fname, mask, namesBadAtEdge):
        # if fnmatch(fname, mask) and bGood_NameEdge(fname, namesBadAtEdge):
        #     return True
        # return False
        return bGood_file(fname, mask, namesBadAtEdge, bPrintGood=False)

    print(f'{msg_action} {path}', end='')

    # Execute declared functions ######################################
    if b_search_in_subdirs:
        print(', including subdirs:', end=' ')
        paths = [f for f in dir_walker(
            path.parent, path.name,
            bGoodFile=filt_file_cur, bGoodDir=filt_dirCur)]
    else:
        print(':', end=' ')
        paths = [path.with_name(f) for f in sorted(os_listdir(path.parent))
                 if filt_file_cur(f, path.name)]
    nfiles = len(paths)
    nl1, nl0 = ('\n', '') if nfiles == 1 else ('', '\n')
    print(end=f"{nl0}- {nfiles} found")
    if nfiles == 0:
        print('!')
        raise Ex_nothing_done
    else:
        print(end='. ')
    if b_interact:
        s = input(f"{nl1}Process {'them' if nfiles > 1 else 'it'}? Y/n: ")
        if 'n' in s or 'N' in s:
            print('answered No')
            raise Ex_nothing_done
        else:
            print('wait... ', end='')
    else:
        print()

    """
    def get_vsz_full(inFE, vsz_path):
        # inFE = os_path.basename(in_full)
        inF = os_path.splitext(inFE)[0]
        vszFE = inF + '.vsz'
        return os_path.join(vsz_path, vszFE)

    def filter_existed(inFE, mask, namesBadAtEdge, bPrintGood, cfg_out):
        if cfg_out['fun_skip'].next(): return False

        # any([inFE[i] == strProbe for i in range(min(len(inFE), len(strProbe) + 1))])
        # in fnmatch.filter(os_listdir(root)
        if not cfg_out['b_update_existed']:
            # vsz file must not exist
            vsz_full = get_vsz_full(inFE, cfg_out['path'])
            if os_path.isfile(vsz_full):
                return False
        elif cfg_out['b_images_only']:
            # vsz file must exist
            vsz_full = get_vsz_full(inFE, cfg_out['path'])
            if not os_path.isfile(vsz_full):
                return False
        else:
            return bGood_file(inFE, mask, namesBadAtEdge, bPrintGood=True)


    """

    return paths, nfiles, path


# File management ##############################################################

def name_output_file(dir_path: PurePath, filenameB, filenameE=None, bInteract=True, fileSizeOvr=0
                     ) -> Tuple[PurePath, str, str]:
    """
    Depreciated!
    Name output file, rename or overwrite if output file exist.
    :param dir_path: file directoty
    :param filenameB: file base name
    :param filenameE: file extention. if None suppose filenameB is contans it
    :param bInteract: to ask user?
    :param fileSizeOvr: (bytes) bad files have this or smaller size. So will be overwrite
    :return: (path_out, sChange, msgFile):
    - path_out: PurePath, suggested output name. May be the same if bInteract=True, and user
                answer "no" (i.e. to update existed), or size of existed file <= fileSizeOvr
    - sChange: user input if bInteract else ''
    - msgFile: string about resulting output name
    """

    # filename_new= re_sub("[^\s\w\-\+#&,;\.\(\)']+", "_", filenameB)+filenameE

    # Rename while target exists and it hase data (otherwise no crime in overwriting)
    msgFile = ''
    m = 0
    sChange = ''
    str_add = ''
    if filenameE is None:
        filenameB, filenameE = os_path.splitext(filenameB)

    def append_to_filename(str_add):
        """
        Returns filenameB + str_add + filenameE if no file with such name in dir_path
        or its size is less than fileSizeOvr else returns None
        :param str_add: string to add to file name before extension
        :return: base file name or None
        """
        filename_new = f'{filenameB}{str_add}{filenameE}'
        full_filename_new = dir_path / filename_new
        if not full_filename_new.is_file():
            return filename_new
        try:
            if os_path.getsize(full_filename_new) <= fileSizeOvr:
                msgFile = 'small target file (with no records?) will be overwrited:'
                if bInteract:
                    print('If answer "no" then ', msgFile)
                return filename_new
        except Exception:  # WindowsError
            pass
        return None

    while True:
        filename_new = append_to_filename(str_add)
        if filename_new:
            break
        m += 1
        str_add = f'_({m})'

    if (m > 0) and bInteract:
        sChange = input('File "{old}" exists! Change target name to ' \
                        '"{new}" (Y) or update existed (n)?'.format(
            old=f'{filenameB}{filenameE}', new=filename_new))

    if bInteract and sChange in ['n', 'N']:
        # update only if answer No
        msgFile = 'update existed'
        path_out = dir_path / f'{filenameB}{filenameE}'  # new / overwrite
        writeMode = 'a'
    else:
        # change name if need in auto mode or other answer
        path_out = dir_path / filename_new
        if m > 0:
            msgFile += f'{str_add} added to name.'
        writeMode = 'w'
    dir_create_if_need(dir_path)
    return (path_out, writeMode, msgFile)


def set_cfg_path_filemask(path=None, filemask=None, ext=None,
                          cfg_search_parent: Optional[MutableMapping[str, Any]] = None):
    """
    absolute path based on input ``path``, sys.argv[0] if not absolute, and ``filemask`` and ``ext``

    :param path: 'path' or 'filemask' is required
    :param filemask: 'path' or 'filemask' is required
    :param ext, optional
    :param cfg_search_parent: dict with fields 'path' or 'db_path' (first extsted used) - used if :param path is't absolute to get parent dir
    :return: absolute path

    Note: Extension may be within :param path or in :param ext
    """
    path = Path(*pathAndMask(*[path, filemask, ext]))
    if not path.is_absolute():
        if cfg_search_parent:
            for field in ['path', 'db_path']:
                if field in cfg_search_parent and cfg_search_parent[field].is_absolute():
                    path = cfg_search_parent[field].parent / path
                    break
            else:
                path = Path(sys.argv[0]).parent / path
        try:
            dir = path.parent.resolve()
            # gets OSError "Bad file name" if do it directly for ``path`` having filemask symbols '*','?'
        except FileNotFoundError:
            dir_create_if_need(path.parent)
            dir = path.parent.resolve()
        return dir / path.name
    return path

def splitPath(path, default_filemask):
    """
    Depreciated!
    Split path to (D, mask, Dlast). Enshure that mask is not empty by using default_filemask.
    :param path: file or dir path
    :param default_filemask: used for mask if path is directory
    :return: (D, mask, Dlast). If path is file then D and mask adjacent parts of path, else
    mask= default_filemask
        mask: never has slash and is never empty
        D: everything leading up to mask
        Dlast: last dir name in D
    """
    D = os_path.abspath(path)
    if os_path.isdir(D):
        mask = default_filemask
        Dlast = os_path.basename(D)
    else:
        D, mask = os_path.split(D)
        if not os_path.splitext(mask)[1]:  # no ext => path is dir, no mask provided
            Dlast = mask
            D = os_path.join(D, Dlast)
            mask = default_filemask
        else:
            Dlast = os_path.basename(D)
    return D, mask, Dlast


def prep(args, default_input_filemask='*.pdf',
         msgFound_n_ext_dir='Process {n} {ext}{files} from {dir}'):
    """
    Depreciated!!!

    :param args: dict {path, out_path}
        *path: input dir or file path
        *out_path: output dir. Can contain
    <dir_in>: will be replased with last dir name in args['path']
    <filename>: not changed, but used to plit 'out_path' such that it is not last in outD
    but part of outF
    :param default_input_filemask:
    :param msgFound_n_ext_dir:
    :return: tuple (inD, namesFE, nFiles, outD, outF, outE, bWrite2dir, msgFile):
    inD             - input directory
    namesFE, nFiles - list of input files found and list's size
    outD            - output directory
    outF, outE      - output base file name and its extension ()
    bWrite2dir      - "output is dir" True if no extension specified.
    In this case outE='csv', outF='<filename>'
    msgFile         - string about numaber of input files found
    """

    # Input files

    inD, inMask, inDlast = splitPath(args['path'], default_input_filemask)
    try:
        namesFE = [f for f in os_path.os.listdir(inD) if fnmatch(f, inMask)]
    except WindowsError as e:
        raise Ex_nothing_done(f'{e.message} - No {inMask} files in "{inD}"?')
    nFiles = len(namesFE)

    if nFiles > 1:
        msgFile = msgFound_n_ext_dir.format(n=nFiles, dir=inD, ext=inMask, files=' files')
    else:
        msgFile = msgFound_n_ext_dir.format(n='', dir=inD, ext=inMask, files='')

    if nFiles == 0:
        raise Ex_nothing_done
    else:
        # Output dir
        outD, outMask, Dlast = splitPath(args['out_path'], '*.%no%')
        # can not replace just in args['out_path'] if inDlast has dots
        Dlast = Dlast.replace('<dir_in>', inDlast)
        outD = outD.replace('<dir_in>', inDlast)

        if '<filename>' in Dlast:  # ?
            outD = os_path.dirname(outD)
            outMask = outMask.replace('*', Dlast)

        outF, outE = os_path.splitext(outMask)
        bWrite2dir = outE.endswith('.%no%')
        if bWrite2dir:  # output path is dir
            outE = '.csv'
            if '<filename>' not in outF:
                outF = outF.replace('*', '<filename>')
    if not os_path.isdir(outD):
        os_path.os.mkdir(outD)
    return inD, namesFE, nFiles, outD, outF, outE, bWrite2dir, msgFile


def this_prog_basename(path=sys.argv[0]):
    return os_path.splitext(os_path.split(path)[1])[0]


@dataclass(repr=False)  # , slots=True
class Message:
    # __slots__ = ('fmt', 'args')
    fmt: str
    args: Tuple

    def __str__(self):
        try:
            return self.fmt.format(*self.args) if self.args else self.fmt
        except KeyError:  # allow dict argument
            try:
                return self.fmt.format_map(self.args[0])
            except IndexError:
                return self.fmt + '\n- Bad format string!\n' + (
                    f'Logging arguments: {self.args}' if len(self.args) else ''
                    )
        except (TypeError, IndexError):
            print('Logging error due to wrong format string:', self.fmt, 'for arguments:', self.args)
            raise


class LoggingContextFilter(logging.Filter):
    # https://docs.python.org/3/howto/logging-cookbook.html

    def filter(self, record):
        # First frame is the file in which this class is defined.
        frame = currentframe().f_back
        try:
            # Walk back through multiple levels of logging.
            while 'logging' in frame.f_code.co_filename or frame.f_code.co_name.startswith(('log', '<module>')):
                # print(frame.f_code.co_filename, frame.f_code.co_name)
                frame = frame.f_back
        except:
            pass
        # Create the overrides
        # record.filename = full_name
        record.funcName = frame.f_code.co_name
        record.lineno = frame.f_lineno
        return True


class LoggingStyleAdapter(logging.LoggerAdapter):
    """
    Uses str.format() style in logging messages. Usage:
    logger = LoggingStyleAdapter(logging.getLogger(__name__))
    also prepends message with [self.extra['id']]
    """
    def __init__(self, logger, extra=None):
        if isinstance(logger, str):
            logger = logging.getLogger(logger)
        f = LoggingContextFilter()
        logger.addFilter(f)

        self.message = Message('', ())
        super(LoggingStyleAdapter, self).__init__(logger, extra or {})

    def process(self, msg, kwargs):
        try:
            extra_id = self.extra['id']
        except KeyError:
            return msg, kwargs
        else:
            return f'[{extra_id}] {msg}', kwargs

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            self.message.fmt, kwargs = self.process(msg, kwargs)
            self.message.args = args
            self.logger._log(level, self.message, (), **kwargs)


class LoggingFilter_DuplicatesOption(logging.Filter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._message_lockup = {}

    def filter(self, record):
        try:
            log_interval = (msg_args0 := (msg := record.msg).args[0]).pop('filter_same')
            b_change_msg = False
        except KeyError:

            try:  # add number of repetition to the message string?
                log_interval = (msg_args0 := (msg := record.msg).args[0]).pop('add_same_counter')
                b_change_msg = True
            except KeyError:
                b_change_msg = False
                return True
            if log_interval is None or (log_interval == 1 and not b_change_msg):
                return True
        except (AttributeError, IndexError):
            return True

        current_log = (msg.fmt, tuple(msg_args0.items()))
        try:
            cnt = self._message_lockup[current_log]
        except KeyError:
            self._message_lockup[current_log] = 0
            return True
        cnt += 1
        self._message_lockup[current_log] = cnt
        if b_change_msg:
            record.msg.fmt += f' (repeated {cnt})'
        return not cnt % log_interval

        # self.last_log
        #     if current_log != getattr(self, "last_log", None):
        #         self.last_log = current_log
        #         return True



def my_logging(name, logger=None):
    logger = logging.getLogger(name)
    logger.addFilter(LoggingFilter_DuplicatesOption())
    return LoggingStyleAdapter(logger)


def init_logging(logger='', log_file=None, level_file='INFO', level_console=None):
    """
    Logging to file flogD/flogN.log and console with piorities level_file and levelConsole
    :param logger: name of logger or logger. Default: '' - name of root logger.
    :param log_file: name of log file. Default: & + "this program file name"
    :param level_file: 'INFO'
    :param level_console: 'WARN'
    :return: logging Logger

    Call example:
    l= init_logging('', None, args.verbose)
    l.warning(msgFile)
    """
    global l
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

    if logger is None:
        logger = sys._getframe(1).f_back.f_globals['__name__']  # replace with name of caller
    elif isinstance(logger, str) and __name__ == '__main__':
        logger = ''

    was_l = bool(l)
    if was_l:
        try:  # a bit more check that we already have logger
            l = logging.getLogger(logger) if isinstance(logger, str) else logger
        except Exception as e:
            pass
        if l and l.hasHandlers():
            l.handlers.clear()  # or if we have good handlers return l?
    else:
        l = logging.getLogger(logger) if isinstance(logger, str) else logger

    try:
        filename = Path(log_file)
        b_default_path = not filename.parent.exists()
    except FileNotFoundError:
        b_default_path = True
    if b_default_path:
        filename = Path(__file__).parent / 'logs' / filename.name

    # Create handlers if there no them in root
    if not l.root.hasHandlers():
        logging.basicConfig(filename=filename, format='%(asctime)s %(message)s', level=level_file)

        # set up logging to console
        console = logging.StreamHandler()
        console.setLevel(level_console if level_console else 'INFO' if level_file != 'DEBUG' else 'DEBUG')  # logging.WARN
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')  # %(name)-12s: %(levelname)-8s ...
        console.setFormatter(formatter)
        l.addHandler(console)

    # Or do not use root handlers:
    # l.propagate = not l.root.hasHandlers()  # to default

    if b_default_path:
        l.warning('Bad log path: %s! Using new path with default dir: %s', log_file, filename)

    return l


def name_output_and_log(out_path=None,
                        writeMode=None,
                        filemask=None,
                        min_size_to_overwrite=None,
                        logging=logging,
                        f_rep_filemask=lambda f: f,
                        bInteract=False,
                        log='log.log',
                        verbose=None):
    """
    path and splits it to fields
    'path', 'filemask', 'ext'
    Initialize logging and prints message of beginning to write

    :param cfg: dict of dicts, requires fields:
        'in' with fields
            'paths'
        'out' with fields
            'out_path'

    :param logging:
    :param bInteract: see name_output_file()
    :param f_rep_filemask: function f(path) modifying its argument
        To replase in 'filemask' string '<File_in>' with base of cfg['in']['paths'][0] use
    lambda fmask fmask.replace(
            '<File_in>', os_path.splitext(os_path.basename(cfg['in']['paths'][0])[0] + '+')
    :return: path, ext, l
    cfg with added fields:
        in 'out':
            'path'

            'ext' - splits 'out_path' or 'csv' if not found in 'out_path'
    """
    # find 'path' and 'ext' params for set_cfg_path_filemask()
    if out_path:
        path, ext = os_path.splitext(out_path)
        if not ext:
            ext = '.csv'
        path = f_rep_filemask(out_path)

        path = set_cfg_path_filemask(path, filemask, ext)

        # Check target exists
        path, writeMode, msg_name_output_file = name_output_file(
            path.parent, filemask, None,
            bInteract, min_size_to_overwrite)

        str_print = f"{msg_name_output_file} Saving all to {path.absolute()}:"

    else:
        if not out_path:
            path = '.'
        str_print = ''

    l = init_logging('', path.with_name(log), verbose)
    if str_print:
        l.warning(str_print)  # or use "a %(a)d b %(b)s", {'a':1, 'b':2}

    return path, ext, writeMode, l



class FakeContextIfOpen:
    """
    Context manager that does nothing if file is not str/PurePath or custom open function is None/False
    useful if instead file want use already opened file object
    # see better contextlib.nullcontext
    """

    def __init__(self,
                 fn_open_file: Optional[Callable[[Any], Any]] = None,
                 file: Optional[Any] = None,
                 opened_file_object = None):
        """
        :param fn_open_file: if bool(fn_open_file) is False then context manager will do nothing on exit
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


def open_csv_or_archive_of_them(filename: Union[PurePath, Iterable[Union[Path, str]]], binary_mode=False,
                                pattern='', encoding=None) -> Iterator[Union[TextIO, BinaryIO]]:
    """
    Opens and yields files from archive with name filename or from list of filenames in context manager (autoclosing).
    Note: Allows stop iteration over files in archive by assigning True to next() in consumer of generator
    Note: to can unrar the unrar.exe must be in path or set rarfile.UNRAR_TOOL
    :param filename: archive with '.rar'/'.zip' suffix or file name or Iterable of file names
    :param pattern: Unix shell style file pattern in the archive - should include directories if need search inside (for example place "*" at beginning)
    :return:
    Note: RarFile anyway opens in binary mode
    """
    read_mode = 'rb' if binary_mode else 'r'


    # not iterates inside many archives so if have iterator then just yield them opened
    if hasattr(filename, '__iter__') and not isinstance(filename, (str, bytes)):
        for text_file in filename:
            if pattern and not fnmatch(text_file, pattern):
                continue
            with open(text_file, mode=read_mode, encoding=encoding) as f:
                yield f
    else:
        filename_str = (
            filename.lower() if isinstance(filename, str) else
            str(filename).lower() if isinstance(filename, PurePath) else
            '')

        # Find arc_suffix ('.zip'/'.rar'/'') and pattern if it is in filename after suffix
        for arc_suffix in ('.zip', '.rar'):
            if arc_suffix in filename_str:
                filename_str_no_ext, pattern_parent = filename_str.split(arc_suffix, maxsplit=1)
                if pattern_parent:
                    pattern = str(PurePath(pattern_parent[1:]) / pattern)
                    filename_str = f'{filename_str_no_ext}{arc_suffix}'
                arc_files = [Path(filename_str).resolve().absolute()]
                break
            else:
                if arc_suffix in (pattern_lower := pattern.lower()):
                    pattern_arcs, pattern_lower = pattern_lower.split(arc_suffix, maxsplit=1)
                    pattern = pattern[-len(pattern_lower.lstrip('/\\')):]  # recover text case for pattern
                    arc_files = Path(filename_str).glob(f'{pattern_arcs}{arc_suffix}')
                    arc_files = list(arc_files)
                    if not arc_files:
                        if (arc_found := Path(filename_str) / f'{pattern_arcs}{arc_suffix}').is_file():
                            arc_files = [arc_found]
                        else:
                            print(f'"{arc_found}" not found!')
                            return None
                    break

        else:
            arc_suffix = ''

        if arc_suffix == '.zip':
            from zipfile import ZipFile as ArcFile
        elif arc_suffix == '.rar':
            import rarfile
            rarfile.UNRAR_TOOL = r"c:\Programs\_catalog\TotalCmd\Plugins\arc\UnRAR.exe"  # Set Your UnRAR executable
            ArcFile = rarfile.RarFile
            try:  # only try increase performance
                # Configure RarFile Temp file size: keep ~1Gbit free, always take at least ~20Mbit:
                # decrease the operations number as we are working with big files
                io.DEFAULT_BUFFER_SIZE = max(io.DEFAULT_BUFFER_SIZE, 8192 * 16)
                import tempfile, psutil
                rarfile.HACK_SIZE_LIMIT = max(20_000_000,
                                              psutil.disk_usage(Path(tempfile.gettempdir()).drive).free - 1_000_000_000
                                              )
            except Exception as e:
                l.warning('%s: can not update settings to increase peformance', standard_error_info(e))
            read_mode = 'r'  # RarFile need opening in mode 'r' (but it opens in binary_mode)
        if arc_suffix:
            for path_arc_file in arc_files:
                with ArcFile(str(path_arc_file), mode='r') as arc_file:
                    for text_file in arc_file.infolist():
                        arc_filename_cor_enc = None
                        if pattern and not fnmatch(text_file.filename, pattern):
                            # account for possible bad russian encoding
                            arc_filename_cor_enc = text_file.filename.encode('cp437').decode('CP866')
                            if fnmatch(arc_filename_cor_enc, pattern):
                                pass
                            else:
                                continue

                        with arc_file.open(text_file.filename, mode=read_mode) as f:
                            if arc_filename_cor_enc:
                                # return file object with correct encoded name and all properties same as of f
                                f.name = arc_filename_cor_enc
                            break_flag = yield (f if binary_mode else io.TextIOWrapper(
                                f, encoding=encoding, errors='replace', line_buffering=True))  # , newline=None
                            if break_flag:
                                print(f'exiting after opening archived file "{text_file.filename}":')
                                print(arc_file.getinfo(text_file))
                                break
        else:
            if pattern and not fnmatch(filename, pattern):
                return
            with open(filename, mode=read_mode) as f:
                yield f


def path_on_drive_d(path_str: str = '/mnt/D',
                    drive_win32: str = 'D:',
                    drive_linux: str = '/mnt/D'):
    """convert path location on my drive to current system (Linux / Windows)"""
    if path_str is None:
        return None
    linux_next_to_d = re.match(f'{drive_linux}(.*)', path_str)
    if linux_next_to_d:
        if sys.platform == 'win32':
            path_str = f'{drive_win32}{linux_next_to_d.group(1)}'
    elif sys.platform != 'win32':
        win32_next_to_d = re.match(f'{drive_win32}(.*)', path_str)
        path_str = f'{drive_linux}{win32_next_to_d.group(1)}'
    return Path(path_str)


def import_file(path: PurePath, module_name: str):
    """Import a python module from a path. 3.4+ only.

    Does not call sys.modules[full_name] = path
    """
    from importlib import util

    f = (path / module_name).with_suffix('.py')
    try:
        spec = util.spec_from_file_location(module_name, f)
        mod = util.module_from_spec(spec)

        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:  #(Exception),
        print(standard_error_info(e), '\n- Can not load module', f, 'here . Skipping!')
        mod = None
    return mod


def st(current: int, descr: Optional[str] = '') -> bool:
    """
    Says if need to execute current step.
    Note: executs >= one step beginnig from ``start``
    Attributes: start: int, end: int, go: Optional[bool] = True:
    start, end: int, step control limits
    go: default: True. If False or Sequence s with s[0] is False then returns False, prints "Stopped" and s[1].

    :param current: step#
    :param descr: step description to print
    :return desision: True if start <= current <= max(start, end)): allows one step if end <= start
    True => execute current st, False => skip
    """
    if st.start <= current <= max(st.start, st.end):
        if st.go is True:
            msg = f'Step {current}.\t{descr}'
            print(msg)
            print('-'*len(msg))
            st.current = current
            return True
        elif isinstance(st.go, Sequence) and st.go[0] is False:
            print(f'Step {current} skipped:', st.go[1])
        else:
            print(f'Step {current} skipped: stopped!')
    return False

st.start = 0
st.end = 1e9  # big value
st.go = True


def call_with_valid_kwargs(func: Callable[[Any], Any], *args, **kwargs):
    """
    Calls func with extracted valid arguments from kwargs
    inspired by https://stackoverflow.com/a/9433836
    :param func: function you're calling
    :param args:
    :param kwargs:
    :return:
    """
    valid_keys = kwargs.keys() & func.__code__.co_varnames[len(args):func.__code__.co_argcount]
    return func(*args, **{k: kwargs[k] for k in valid_keys})



@enum.unique
class ExitStatus(enum.IntEnum):
    """Portable definitions for the standard POSIX exit codes.
    https://github.com/johnthagen/exitstatus/blob/master/exitstatus.py
    """
    success = 0  # Indicates successful program completion
    failure = 1  # Indicates unsuccessful program completion in a general sense


if sys.platform == "win32":
    class GetMutex:
        """ Limits application to single instance
            Provides a method by which an application can ensure that only one
            instance of it can be running at any given time.
        Usage:
            if (app := GetMutex()).IsRunning():
                print("Application is already running")
                sys.exit()
        """

        def __init__(self):
            thisfile   = str(Path(sys.argv[0]).resolve()).replace('\\', '/')
            self.name  = f"{thisfile}_{{AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE}}"
            self.mutex = CreateMutex(None, False, self.name)
            self.error = GetLastError()

        def IsRunning(self):
            return (self.error == ERROR_ALREADY_EXISTS)

        def __del__(self):
            if self.mutex: CloseHandle(self.mutex)












"""
TRASH
# p_sec.set_defaults(**{'--' + option_name: config.get(section_name, option_name)})
# print('\n==> '.join([s for s in e.args if isinstance(s, str)]))
# if can loose sections
# par_ok, par_bad_list= p.parse_known_args()
# for a in par_bad_list:
#     if a.startswith('--'):
#         p.add(a)
"""
