#!/usr/bin/env python
# coding:utf-8
# , division
"""
Author:  Andrey Korzh <ao.korzh@gmail.com>
Purpose: update Veusz pattern by means of Veusz commands iterating over
list of data files. Saves changed vsz files and export images.
todo: Use hdf5 store to save dates of last modification of processed files
to have the ability not process twice same data on next calls.
Created: 02.09.2016
"""
# import ast
import logging
import re
from datetime import datetime
from os import chdir as os_chdir, getcwd as os_getcwd, environ as os_environ
from pathlib import Path, PurePath
from sys import platform as sys_platform, stdout as sys_stdout
from time import sleep
from typing import Any, Callable, Dict, Iterator, Iterable, Optional, Tuple, Sequence, Union
from itertools import dropwhile
import pandas as pd
from dateutil.tz import tzlocal, tzoffset

import asyncio
from functools import partial
import multiprocessing
from concurrent.futures.thread import ThreadPoolExecutor
# my
from .utils2init import (
    my_argparser_common_part,
    cfg_from_args,
    this_prog_basename,
    init_file_names,
    dir_from_cfg,
    init_logging,
    Ex_nothing_done,
    import_file,
    standard_error_info,
    LoggingStyleAdapter,
)
from . import h5

# Globals
to_mytz_offset = tzoffset(None, -tzlocal()._dst_offset.total_seconds())
# veusz = None  # 'veusz' variable must be corrected later
load_vsz = None  # must be corrected later

if __name__ != '__main__':
    lf = LoggingStyleAdapter(logging.getLogger(__name__))
else:
    lf = None  # will set in main()

default_veusz_path = PurePath(
    'C:\\Program Files\\Veusz' if sys_platform == 'win32' else '/home/korzh/.local/lib/python3.9/site-packages/veusz'  # try os_environ['PATH']?
)


def my_argparser():
    """
    Configuration parser
    - add here common options for different inputs
    - add help strings for them
    :return p: configargparse object of parameters
    """
    version = '0.1.0'
    p = my_argparser_common_part({'description': 'veuszPropagate version {}'.format(version) + """
----------------------------
Create vsz file for each source
file based on vsz pattern
----------------------------"""}, version)

    # Configuration sections
    s = p.add_argument_group('in',
                             'all about input files')
    s.add('--path', help='path to source file(s) to generate list of their names '
                         '(usually *.csv or *.txt, or *.vsz in b_images_only mode)')
    s.add('--pattern_path', help='path to ".vsz" file to use as pattern')
    s.add('--import_method',
          help='Veusz method to imort data in ".vsz" pattern')  # todo: read it from pattern
    s.add('--start_file', default="0",
          help='indexes begins from 0, optional, allows process only range of found files')
    s.add('--add_custom_list',
          help='custom definitions names for evaluation of expressions defined in add_custom_expressions_list')
    s.add('--add_custom_expressions_list',
          help='custom_expressions_list to add by Veusz AddCustom() function')
    s.add('--eval_list',
          help='string represented Veusz.Embed function call to eval')
    s.add('--data_yield_prefix',
          help='used to get from Veusz data having names started from this')
    s.add('--db_path',
          help='path to pytables hdf5 store having table_log (if it is specified to use instead files)')
    s.add('--tables_list',
          help='path to tables in db to find instead files')
    s.add('--table_log',
          help='name of log table - path to hdf5 table having intervals ("index" of type pd.DatetimeIndex and "DateEnd" of type pd.Datetime)')
    s.add('--min_time', help='%%Y-%%m-%%dT%%H:%%M:%%S, optional, allows range table_log rows')
    s.add('--max_time', help='%%Y-%%m-%%dT%%H:%%M:%%S, optional, allows range table_log rows')

    s = p.add_argument_group('out',
                             'all about output files')
    s.add('--export_pages_int_list', default='0',
          help='pages numbers to export, comma separated (1 is first), 0 = all, empty = none')
    s.add('--b_images_only', default='False',
          help='export only. If true then all output vsz must exist, they will be loaded and vsz not be updated')
    s.add('--b_update_existed', default='False',
          help='replace all existed vsz files else skip existed files. In b_images_only mode - skip exporting vsz files for which any files in "export_dir/*{vsz file stem}*. {export_format}" exist')
    s.add('--export_dir', default='images(vsz)',
          help='subdir relative to input path or absolute path to export images')
    s.add('--export_format', default='jpg',
          help='extension of images to export which defines format')
    s.add('--export_suffix', default='#{out_name}',
          help='will be added to each exported image before extension. {out_name} will be replaced with output vsz file name')
    s.add('--export_dpi_int_list', default='300',
          help='resolution (dpi) of images to export for all pages, defined in `export_pages_int_list`')
    s.add('--filename_fun', default='lambda tbl: tbl',
          help='function to modify output file name. Argument is input table name in hdf5')
    s.add('--add_to_filename', default='',
          help='string will be appended to output filenames. If input is from hdf5 table then filename is name of table, this will be added to it')

    # candidates to move out to common part
    s.add('--exclude_dirs_endswith_list', default='-, bad, test, TEST, toDel-',
          help='exclude dirs which ends with this srings. This and next option especially useful when search recursively in many dirs')
    s.add('--exclude_files_endswith_list', default='coef.txt, -.txt, test.txt',
          help='exclude files which ends with this srings')

    s = p.add_argument_group('program',
                             'program behaviour')
    s.add('--export_timeout_s_float', default='0',
          help='export asyncroniously with this timeout, s (tried 600s?)')
    s.add('--load_timeout_s_float', default='180',
          help='export asyncroniously with this timeout, s (tried 600s?)')
    s.add('--b_execute_vsz', default='False',
          help='instead of Load() read vsz and execute its content line by line')
    s.add('--hidden', default='False',
          help='set to True to not show embedded window')
    s.add('--veusz_path', default=default_veusz_path,
          help='directory of Veusz like /usr/lib64/python3.9/site-packages/veusz-2.1.1-py3.6-linux-x86_64.egg/veusz')
    s.add('--before_next_list', default=',',
          help=''' "Close()" - each time reopens pattern,
    "restore_config" - saves and restores initial configuration (may be changed in data_yield mode: see data_yield_prefix argument)''')
    s.add('--f_custom_in_cycle',
          help='''function evaluated in cycle: not implemented over command line''')
    s.add('--return', default='<end>',  # nargs=1,
          choices=['<cfg_from_args>', '<gen_names_and_log>', '<embedded_object>', '<end>'],
          help='<cfg_from_args>: returns cfg based on input args only and exit, <gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function... - see main()')

    return p


class SingletonTimeOut:
    """
    Timeout on a function call
    https://stackoverflow.com/a/63337662/2028147
    """
    pool = None
    @classmethod
    def run(cls, to_run: partial, timeout: float):
        """

        :param to_run: functools.partial, your function with arguments included
        :param timeout: seconds
        :return: your function output
        """
        pool = cls.get_pool()
        loop = cls.get_loop()
        try:
            task = loop.run_in_executor(pool, to_run)
            return loop.run_until_complete(asyncio.wait_for(task, timeout=timeout))
        except asyncio.TimeoutError as e:
            error_type = type(e).__name__  # TODO
            raise e

    @classmethod
    def get_pool(cls):
        if cls.pool is None:
            cls.pool = ThreadPoolExecutor(multiprocessing.cpu_count())
        return cls.pool

    @classmethod
    def get_loop(cls):
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
            # print("NEW LOOP" + str(threading.current_thread().ident))
            return asyncio.get_event_loop()


# ----------------------------------------------------------------------

def veusz_data(veusze, prefix: Union[str, Tuple[str]], suffix_prior: str = '') -> Dict[str, Any]:
    """
    Get data, loaded into the Veusz document filtered by prefix and suffix_prior
    :param veusze: Veusz embedded object
    :param prefix: include in output only datasets which name starts with this string(s). If tuple, and one of elements started with another, then put another later
    :param suffix_prior: string, if several datasets filtered by prefix diff only with this suffix, out only those which name ends with this, but rename keys to exclude (to be without) this suffix
    Returns: dict with found dataset which keys are its names excluding prefix and suffix
    Converts 'time'
    If return field name ends with 'starts' or 'ends' convert to int, and rename it just to corresp. 'starts' or 'ends' i.e. remove prefix if were no same named field yet
    """

    names = veusze.GetDatasets()

    names_filt = dict()
    names_to_check_on_step2 = dict()

    # Keep version that ends with suffix_prior, and remove version suffixes (to which delimiter is '_')
    # step1: separate max priority fields (names_filt) and others (names_to_check_on_step2)
    for name in names:
        if name.startswith(prefix):
            name_out = name if name in prefix else name[len(
                next(dropwhile(lambda pr: not name.startswith(pr), prefix))
                    if isinstance(prefix, tuple) else prefix):
                ].lstrip('_').split('_')[0]
            if name.endswith(suffix_prior):
                names_filt[name_out] = name
            else:
                names_to_check_on_step2[name_out] = name
    # step2: append other fields to names_filt that are not in already
    msg_names_skip = []
    if len(names_to_check_on_step2.keys()):
        for name_out, name in names_to_check_on_step2.items():
            if not name_out in names_filt:
                names_filt[name_out] = name
            else:
                msg_names_skip.append(name)
    if msg_names_skip:
        msg_names_skip = 'skip {} fields: '.format('|'.join(prefix)) + ','.join(msg_names_skip)
    else:
        msg_names_skip = ''
    lf.debug('\n'.join([msg_names_skip, ' load fields: {}'.format(names_filt)]))

    vsz_data = dict([(name_out, veusze.GetData(name)[0]) for name_out, name in names_filt.items()])
    if ('time' in vsz_data) and len(vsz_data['time']):
        vsz_data['time'] = pd.DatetimeIndex((vsz_data['time'] + 1230768000) * 1E+9, tz='UTC')

    for name in list(vsz_data.keys()):  # list() makes a copy (to a simpler type) to not change iterable
        if name.endswith(('starts', 'ends')):
            int_suffix = 'starts' if name.endswith('starts') else 'ends'
            tmp = vsz_data[name].astype('int32')
            if int_suffix in vsz_data.keys():
                vsz_data[name] = tmp
                continue
            vsz_data[int_suffix] = tmp
            del vsz_data[name]
    return vsz_data


def load_vsz_closure(veusz_path: PurePath=default_veusz_path,
                     load_timeout_s: Optional[int] = 120,
                     b_execute_vsz: bool = False,
                     hidden=False,
                     vsz_path_env=r'C:\Work\Python\AB_SIO_RAS\Veusz_plugins\func_vsz.py'
                     ) -> Callable[
    [Union[str, PurePath], Optional[str], Optional[str], Optional[str]], Tuple[Any, Optional[Dict[str, Any]]]]:
    """
    See load_vsz inside
    :param veusz_path: pathlib Path to directory of embed.py
    :param load_timeout_s: raises asyncio.TimeoutError if loads longer
    :param b_execute_vsz: can not use "Load" because of python expression in vsz is needed
    :param hidden: set to True to not show embedded window
    """

    # def import_veusz(veusz_path=u'C:\\Program Files\\Veusz'):
    #     if not os_path.isdir(veusz_path):
    #         return  # 'veusz' variable must be coorected later
    #     veusz_parent_dir, veusz_dir_name = os_path.split(veusz_path)
    #     sys_path.append(os_path.dirname(veusz_parent_dir))
    #     # import Veusz.embed as veusz  #
    #     import_file(full_name, path):
    #     #importlib.import_module('.embed', package=veusz_dir_name)
    #     sys_path.pop()
    #
    #     sep = ';' if sys_platform == 'win32' else ':'
    #     os_environ['PATH'] += sep + veusz_path
    #     return
    # not works:
    # sys_path.append(cfg['program']['veusz_path'])
    # sys_path.append(os_path.dirname(cfg['program']['veusz_path']))
    os_environ['VSZ_PATH'] = vsz_path_env  # variable that can be accessed in vsz by ENVIRON()
    if veusz_path:
        sep = ';' if sys_platform == 'win32' else ':'
        # to find Veusz executable (Windows only):
        os_environ['PATH'] += f'{sep}{veusz_path}'

        # for Linux set in ./bash_profile if not in Python path yet:
        # PYTHONPATH=$PYTHONPATH:$HOME/Python/other_sources/veusz
        # export PYTHONPATH
        # if you compile Veusz also may be you add there
        # VEUSZ_RESOURCE_DIR=/usr/share/veusz
        # export VEUSZ_RESOURCE_DIR
        veusz = import_file(veusz_path, 'embed')
    else:
        import veusz.embed as veusz

    # sys_path.append(os_path.dirname(cfg['program']['veusz_path']))

    def load_vsz(vsz: Union[str, PurePath, None] = None,
                 veusze: Optional[str] = None,
                 prefix: Optional[str] = None,
                 suffix_prior: Optional[str] = '_fbot') -> Tuple[veusz.Embedded, Optional[Dict[str, Any]]]:
        """
        Load (create) specifid data from '*.vsz' files
        :param vsz: full name of vsz or None. If not None and such file not found then create it
        :param veusze: veusz.Embedded object or None - will be created if None else reused
        :param prefix: only data started with this prefix will be loaded
        :param suffix_prior: high priority names suffix, removes other version of data if starts same but with no such suffix (see veusz_data())
        :return: (veusze, vsz_data):
            - veusze - veusz.Embedded object
            - vsz_data - loaded data if prefix is not None else None
        """
        if vsz is None:
            file_exists = False
            if veusze is None:
                title = 'empty'
                lf.debug('new embedded window')
            else:
                lf.debug('keep same embedded window')
        else:  # isinstance(vsz, (str, PurePath)):
            vsz = Path(vsz)
            file_exists = vsz.is_file()
            if file_exists:
                lf.debug(f'loading found vsz: {vsz}')
                title = f'{vsz} - was found'
            else:
                lf.debug(f'creatig vsz: {vsz}')
                title = f'{vsz} - was created'

        if veusze is None:
            # Veusz embedded window construction

            # Save right path in veusz.Embedded (closure)
            if __name__ != '__main__' and vsz:       # if this haven't done in main()
                path_prev = os_getcwd()      # to recover
                os_chdir(vsz.parent)   # allows veusze.Load(path) to work if _path_ is relative or relative paths is used in vsz
            try:
                veusze = veusz.Embedded(title, hidden=hidden)   # , hidden=True
            except ConnectionResetError:
                print('Last such problem I fased - my bad plagin: try to disconnect it or fix')
            # veusze.EnableToolbar()
            # veusze.Zoom('page')

            if __name__ != '__main__' and vsz:
                os_chdir(path_prev)          # recover
            was_in_use = False
        else:
            was_in_use = True

        if file_exists:
            if not b_execute_vsz:
                veusze.AddCustom('definition', 'BASENAME()', f"'{vsz.name}'", mode='replace')  # add absent in embedded
                if load_timeout_s:
                    # veusze.Load(str(vsz.name)) with timeout:  # not tried veusze.serv_socket.settimeout(60)
                    SingletonTimeOut.run(partial(veusze.Load, str(vsz.name)), load_timeout_s)
                    sleep(1)
                else:
                    veusze.Load(vsz.name)
            else:

                # import ast
                # def load_by_exec_new(_vsz, _veusze):
                #     code = ast.parse(vsz.read_text(encoding='utf-8'), mode='exec')
                #
                #     last = ast.Expression(code.body.pop().value)  # assumes last node is an expression
                #     for block in code.body:


                def load_by_exec(_vsz, _veusze):
                    """
                    Unsafe replacement for veusze.Load(vsz) to add variable argv
                    Runs any python commands before 1st Title command of
                    Updates ``argv``, not normally available in embedded mode, but only if it is loaded by "from sys import argv" command
                    :param _vsz:
                    :return:
                    """
                    # Variables of vsz-file will be loaded to context of this function. But in this function we have to
                    # use other variables. To minimize impact of them on executing vsz we remove them as soon as we can,
                    # to minimize chance of collision with vsz' variables we prefixed local variables with "_".
                    #os.environ['vsz_path'] = str(_vsz)  # not works!
                    with _vsz.open(encoding='utf-8') as _v:
                        # 1. Evaluate pure python lines at begin of file
                        _lines = []
                        _have_no_commands = False
                        for _line in _v:
                            # break code only when found Titled line without syntax error
                            if _line[:2].istitle():
                                try:
                                    exec(_line, {}, {})
                                    break
                                except SyntaxError:
                                    pass   # part of previous line?
                                except NameError as e:
                                    if _line.startswith('SetCompatLevel'):  # skip 'SetCompatLevel' line
                                        continue
                                    else:
                                        break
                            _lines.append(_line.removeprefix('from sys import argv'))
                        else:
                            _have_no_commands = True
                        # Dangerous for unknown vsz! We allow 1 time at beginning of file to use for known vsz.
                        loc_exclude = locals().copy()
                        del loc_exclude['_veusze']
                        loc = {
                            **_veusze.__dict__,
                            'argv': ['veusz.exe', str(_vsz)],
                            'BASENAME': (lambda: _vsz.name),
                        }
                        exec('\n'.join(_lines), loc, loc)
                        if _have_no_commands:
                            return

                        # 2. Evaluate Veusz commands (we restrict our format so that standard one-liner Veusz commands only left)

                        _basename_result = '\"{}\"'.format(loc['BASENAME']())

                        # Preparing to execute "AddCustom*" lines at first

                        def _not_custom_else_exec(line):
                            """
                            Indicate and execute "Custom definition" lines, which must start with "AddCustom".
                            Also correct "BASENAME()" in them by inserting needed result instead
                            :param line:
                            :return: True if line was not "AddCustom*" (i.e. not executed) else None
                            """
                            if line.startswith('AddCustom'):
                                if 'BASENAME()' in line:
                                    # line = line.replace("{", "{{").replace("}", "}}").replace(BASENAME(), '{}').format(_basename_result)

                                    line = line.replace(
                                        'BASENAME()', (
                                            _basename_result if '\"' not in line else
                                            _basename_result.replace('"', "'") if "\'" not in line else
                                            _basename_result.replace('"', "'''") if "'''" not in line else
                                            _basename_result.replace('"', '"""')
                                        )  # only removing BASENAME() helps in Custom Definitions expressions
                                    )
                                eval(f"""_veusze.{line}""", {}, loc)
                            else:
                                return True

                        loc.update(locals().copy())
                        for k in loc_exclude.keys():
                            del loc[k]

                        #_veusze.SetVerbose()

                        # eval line with previously found Veusz command
                        eval(f"""_veusze.{_line}""")
                        # eval AddCustom then other lines in their order
                        for _line in [_line for _line in _v if _not_custom_else_exec(_line)]:
                            try:
                                eval(f"""_veusze.{_line}""", {}, loc)
                                # if r := eval(f"""_veusze.{_line}""", {}, loc):
                                #     print(r)
                            except Exception as e:
                                lf.exception(f'Error eval({_line})')
                            # from ast import literal_eval
                            # params_dict = literal_eval(params.rsplit(')', maxsplit=1)[0])
                            # getattr(veusze, cmd)(**params_dict)
                    print('')  # loaded
                    return

                if was_in_use:
                    veusze.Wipe()  # todo: delete/update only vsz changes
                load_by_exec(vsz, veusze)
                # SingletonTimeOut.run(partial(load_by_exec, vsz, veusze)), load_timeout_s)


        if prefix is None:
            return veusze, None
        return veusze, veusz_data(veusze, prefix, suffix_prior)

    return load_vsz


# if _line[:2].istitle(): # if not re.match(' *#', _line):  #this checks are not needed if standard Veusz commands only left

# elif "ENVIRON.get('VSZ_PATH', FILENAME())" in _line:
#     print('found')
# cmd, params = _line.split('(', maxsplit=1)


def export_images(veusze, cfg_out, suffix, b_skip_if_exists=False):
    """
    Export pages
    :param veusze: Veusz embedded object
    :param cfg_out: must have fields 'export_pages' and 'export_dir': see command line arguments help
    :param suffix:
    :return:

    export_images(veusze, cfg['out'], log['out_name'])
    """
    # Export images
    if not cfg_out['export_pages']:
        return
    lf.debug('exporting {:s} {:s} images:', len(cfg_out['export_pages']), cfg_out['export_format'])
    dpi_list = cfg_out.get('export_dpi', [300])
    for i, key in enumerate(veusze.GetChildren(where='/'), start=1):
        if cfg_out['export_pages'][0] == 0 or (  # = b_export_all_pages
                i in cfg_out['export_pages']):
            file_name = Path(cfg_out['export_dir']) / f"{key}{suffix}.{cfg_out['export_format']}"
            if veusze.Get(key + '/hide') or (b_skip_if_exists and file_name.is_file()):  # skip hidden
                continue
            # Veusz partly blanks too big images on export - reduce
            dpi = dpi_list[min(i, len(dpi_list) - 1)]
            if int(re.sub('[^\d]', '', veusze.Get(key + '/width'))) > 400 and dpi > 200:
                dpi = 200

            try:
                veusze.Export(str(file_name), page=i - 1, dpi=dpi)
            except Exception as e:
                lf.error('Exporting error', exc_info=True)
            lf.debug('{},', i)


# try:
#     import async_timeout
#     import asyncio
#
#
#     def force_async(fn):
#         '''
#         turns a sync function to async function using threads
#         :param sync function:
#         :return async funciton:
#
#         run a sync task in a thread (returning a future object, coroutine.Future) and then turn this future object into a asyncio.Future object. Obivously, this future object will be resolve when the sync function is done, and also is awaitable.
#         '''
#         from concurrent.futures import ThreadPoolExecutor
#         pool = ThreadPoolExecutor()
#
#         def wrapper(*args, **kwargs):
#             future = pool.submit(fn, *args, **kwargs)
#             return asyncio.wrap_future(future)  # make it awaitable
#
#         return wrapper
#
#
#     @force_async
#     def export_images_a(veusze, cfg_out, suffix, b_skip_if_exists=False):
#         export_images(veusze, cfg_out, suffix, b_skip_if_exists)
#
#
#     async def export_images_timed(veusze, cfg, suffix, b_skip_if_exists=False) -> bool:
#         """
#         Asyncronous export_images(...)
#         :param veusze:
#         :param cfg: must have ['async']['export_timeout_s'] to skip on timeout, dict 'out' to call export_images(...)
#         :param suffix:
#         :param b_skip_if_exists:
#         :return:
#         """
#         async with async_timeout.timeout(cfg['async']['export_timeout_s']) as cm:
#             export_images_a(veusze, cfg['out'], suffix, b_skip_if_exists)
#         if cm.expired:
#             lf.warning('timeout on exporting. going to next file')
#         return cm.expired
#
# except Exception as e:
export_images_timed = None


def veusze_commands(veusze, cfg_in, file_name_r):
    """
    Execute Vieusz commands specified by special cfg_in fields
    :param veusze:
    :param cfg_in: dict, modify veusze if following fields specified:
        'add_custom' and 'add_custom_expressions' - lists of equal size
        'import_method', string - one of: 'ImportFile', 'ImportFileCSV' to import file_name_r file
        'eval', list of strings - patterns modified by format_map(cfg_in). file_name_r can be referred as cfg_in['nameRFE']
    :param file_name_r: relative name of data file
    :return:
    """
    if cfg_in is None:
        return
    if 'import_method' in cfg_in:
        if cfg_in['import_method'] == 'ImportFile':
            # veusze.call('ImportFile')
            veusze.ImportFile(f'{file_name_r}', cfg_in['c'], encoding='ascii', ignoretext=True, linked=True,
                              prefix=u'_')
            # veusze.ImportFile(f'{file_name_r}', u'`date`,Tim(time),Pres(float),Temp(float),Cond(float),'+
            # 'Sal(float),O2(float),O2ppm(floatadd_custom_expressions),pH(float),Eh(float),Turb(float)',
            #            encoding='ascii', ignoretext=True, linked=True, prefix=u'_')
        elif cfg_in['import_method'] == 'ImportFileCSV':
            veusze.ImportFileCSV(f'{file_name_r}', blanksaredata=True, dateformat=u'DD/MM/YYYY hh:mm:s',
                                 delimiter=b'\t', encoding='ascii', headermode='1st', linked=True, dsprefix=u'_',
                                 skipwhitespace=True)
    if ('add_custom_expressions' in cfg_in) and cfg_in['add_custom_expressions']:
        for name, expr in zip(cfg_in['add_custom'], cfg_in['add_custom_expressions']):
            veusze.AddCustom('definition', name, expr.format_map(cfg_in).strip(), mode='replace')

    if ('eval' in cfg_in) and cfg_in['eval']:
        cfg_in['nameRFE'] = file_name_r
        for ev in cfg_in['eval']:
            eval_str = ev.format_map(cfg_in).strip()
            if __debug__:
                lf.debug('eval: {}', eval_str)
            try:
                eval(f"veusze.{eval_str}")  # compile(, '', 'eval') or [], 'eval')
            except Exception as e:
                lf.error('error to eval "{:s}"', eval_str, exc_info=True)
    # veusze.AddCustom('constant', u'fileDataSource', f"u'{file_name_r}'", mode='replace')


def load_to_veusz(in_fulls: Iterable, cfg, veusze=None):
    """
    Generate Veusz embedded instances by opening vsz-file(s) and modify it by executing commands specified in cfg
    :param in_fulls: full names of source data file to load in veusz pattern (usually csv)
    :param cfg: dict with keys:
        out:
            path, paths - pattern paths
            b_images_only, b_update_existed - command line arguments - see my_argparser()
        in:
            before_next - modify Veusz pattern data by execute Veusz commands if have fields:
                'Close()' - reopen same pattern each cycle

            import_method, header if import_method == 'ImportFile',
            add_custom, add_custom_expressions - Veusz Castom Definitions
            eval - any Veusz command
    :param veusze: Veusz embedded object. If it is None creates new Veusz embedded object loading pattern path
    :yields (veusze, log)
        veusze: Veusz embedded object.
        log: dict, {'out_name': inF, 'out_vsz_full': out_vsz_full}

    Note 1: Uses global load_vsz function if veusze = None or cfg['out']['b_images_only']
    which is defined by call load_vsz_closure()
    Note 2: If 'restore_config' in cfg['program']['before_next'] then sets cfg['in']= cfg['in_saved']
    """
    global load_vsz

    filename_fun = eval(compile(cfg['out']['filename_fun'], '', 'eval'))

    for ifile, in_full in enumerate(in_fulls, start=1):
        in_full = Path(in_full)
        out_name = filename_fun(in_full.stem) + cfg['out']['add_to_filename']
        out_vsz_full = Path(cfg['out']['path']).with_name(out_name).with_suffix('.vsz')

        # do not update existed vsz/images if not specified 'b_update_existed'
        if not cfg['out']['b_update_existed']:
            if cfg['out']['b_images_only']:
                glob = Path(cfg['out']['export_dir']).glob(
                    f"*{out_name}*.{cfg['out']['export_format']}")
                if any(glob):
                    continue
            elif out_vsz_full.is_file():
                continue
            # yield (None, None)

        if in_full.stem != out_vsz_full.stem:
            lf.info('{:d}. {:s} -> {:s}, ', ifile, in_full.name, out_vsz_full.name)
        else:
            lf.info('{:d}. {:s}, ', ifile, in_full.name)
        sys_stdout.flush()
        log = {'out_name': out_name, 'out_vsz_full': out_vsz_full}

        if veusze:  # else  to do: try Wipe()
            try:
                b_closed = veusze.IsClosed()
            except AttributeError:  # 'NoneType' object has no attribute 'cmds'
                b_closed = True
            except Exception as e:
                lf.error('IsClosed() error', exc_info=True)
                b_closed = True
            if b_closed:
                veusze = None
            elif 'Close()' in cfg['program']['before_next']:  # or in_ext == '.vsz'
                veusze.Close()
                veusze = None



        def do_load_vsz(in_full, veusze, load_vsz):
            """
            loads
            If global cfg['out']['b_images_only'] then does load_vsz(in_full)
            :param in_full: vsz file or else any file list of paths cfg['out']['paths'][-1]
            :param veusze:
            :param load_vsz:
            :return:
            """
            in_ext = in_full.suffix.lower()
            # load same filePattern (last in list) if data file not "vsz":
            vsz_load = in_full if in_ext == '.vsz' else cfg['out']['paths'][-1]
            if cfg['out']['b_images_only']:
                veusze = load_vsz(vsz_load, veusze)[0]  # veusze.Load(in_full.with_suffix('.vsz'))
            else:
                if 'restore_config' in cfg['program']['before_next']:
                    cfg['in'] = cfg['in_saved'].copy()  # useful if need restore add_custom_expressions?
                if not veusze:
                    veusze = load_vsz(vsz_load)[0]  # , veusze=veusze
                    if cfg['program']['verbose'] == 'DEBUG':
                        veusze.SetVerbose()  # nothing changes
                # Relative path from new vsz to data, such as u'txt/160813_0010.txt'
                try:
                    file_name_r = in_full.relative_to(cfg['out']['path'].parent)
                except ValueError as e:
                    # lf.exception('path not related to pattern')
                    file_name_r = in_full
                veusze_commands(veusze, cfg['in'], file_name_r)
            return veusze

        try:
            veusze = do_load_vsz(in_full, veusze, load_vsz)
        except asyncio.TimeoutError as e:
            lf.warning('Recreating window because of {:s}', standard_error_info(e))
            veusze.remote.terminate()
            veusze.remote = None

            # veusze.Close()  # veusze.exitQt()                                              # not works
            # import socket
            # veusze.serv_socket.shutdown(socket.SHUT_RDWR)
            # veusze.serv_socket.close()
            # veusze.serv_socket, veusze.from_pipe = -1, -1

            # veusze.serv_socket.shutdown(socket.SHUT_RDWR); veusze.serv_socket.close()
            # veusze.startRemote()                                                           # no effect

            load_vsz = load_vsz_closure(
                cfg['program']['veusz_path'],
                cfg['program']['load_timeout_s'],
                cfg['program']['b_execute_vsz'],
                cfg['program']['hidden']
                )     # this is only unfreezes Veusz
            veusze = do_load_vsz(in_full, None, load_vsz)

        yield veusze, log


# -----------------------------------------------------------------------

def ge_names(cfg, f_mod_name=lambda x: x):
    """
    Yield all full file names
    :param cfg: dict with field ['in']['paths'], - list of parameters for f_mod_name
    :param f_mod_name: function(filename:str) returns new full file name: str
    :yields: f_mod_name(cfg['in']['paths'] elements)
    """
    for name in cfg['in']['paths']:

        try:  # Exclude names with only '0', '-'  '_' symbols
            next(re.finditer('[^0_-]', name.stem))
        except StopIteration:
            continue
        yield f_mod_name(name)



def co_savings(cfg: Dict[str, Any]) -> Iterator[None]:
    """
    Saves vsz, exports images and saves hdf5 log
    Corutine must receive:
        veusze: Veusz embedded object
        log: dict with parameters: 'out_name' - log's index, 'out_vsz_full' - vsz file name to save

    log parameters will be saved to pandas dataframe end then to hdf5 log cfg['program']['log'])[0]+'.h5'
    """
    with pd.HDFStore(Path(cfg['program']['log']).with_suffix('.h5'), mode='a') as storeLog:
        veusze = None
        if __name__ != '__main__':
            path_prev = os_getcwd()
            os_chdir(cfg['out']['path'].parent)
        print('Saving to {}'.format(cfg['out']['path'].parent.absolute()))
        try:
            while True:
                veusze, log = yield ()
                if not cfg['out']['b_images_only']:
                    veusze.Save(str(log['out_vsz_full']))
                    # Save vsz modification date
                    log['fileChangeTime'] = datetime.fromtimestamp(Path(
                        log['out_vsz_full']).stat().st_mtime)
                    df_log = pd.DataFrame.from_records(log, exclude=['out_name', 'out_vsz_full'],
                                                       index=[log['out_name']])
                    storeLog.append(Path(cfg['out']['path']).name, df_log, data_columns=True,
                                    expectedrows=cfg['in']['nfiles'], index=False, min_itemsize={'index': 30})

                export_suffix = cfg['out']['export_suffix'].format_map(log)
                if cfg['async']['loop']:
                    try:  # yield from     asyncio.ensure_future(
                        # asyncio.wait_for(, cfg['async']['export_timeout_s'], loop=cfg['async']['loop'])
                        b = cfg['async']['loop'].run_until_complete(
                            export_images_timed(veusze, cfg, export_suffix))
                    except asyncio.TimeoutError:
                        lf.warning('can not export in time')
                else:
                    export_images(veusze, cfg['out'], export_suffix)
        except GeneratorExit:
            print('Ok>')
        finally:
            if __name__ != '__main__':
                os_chdir(path_prev)
            if veusze and cfg['program']['return'] != '<embedded_object>':
                veusze.Close()
                lf.info('closing Veusz embedded object')
            # veusze.WaitForClose()


def co_send_data(gen_veusz_and_logs, cfg, cor_savings):
    """
    - sends loaded Veusz data to caller and recives parameter cfg_in for next step
    - executes Veusz commands by veusze_commands(veusze, cfg_in, ...)
    - sends (veusze, log) to cor_savings()

    :param gen_veusz_and_logs: load_to_veusz(ge_names(cfg), cfg)
    :param cfg: configuration dict. Must contain ['in']['data_yield_prefix'] field to specify which data to send
    :param cor_savings: corutine which receives (veusze, log) for example co_savings corutine

    Usage:
    cfgin_update = None
    while True:
        try:
            d, log = gen_data.send(cfgin_update)
        except (GeneratorExit, StopIteration):
            print('Ok>')
            break
        except Exception as e:
            print('There are error ', standard_error_info(e))
            continue

        # ... do some calcuations to prepare custom_expressions ...

        # if add_custom the same then update only add_custom_expressions:
        cfgin_update = {'add_custom_expressions': [expression1, expression2, ...]}

    Note: Courutine fubricated by this function is returned by veuszpropagate if it is called with --data_yield_prefix argument.
    """
    for veusze, log in gen_veusz_and_logs:
        if not veusze:
            continue
        vsz_data = veusz_data(veusze, cfg['in']['data_yield_prefix'])
        # caller do some processing of data and gives new cfg:
        cfgin_update = yield (vsz_data, log)  # to test here run veusze.Save('-.vsz')
        # cfg['in'].update(cfgin_update)  # only update of cfg.in.add_custom_expressions is tested

        file_name_r = Path(log['out_vsz_full']).relative_to(cfg['out']['path'].parent)
        veusze_commands(veusze, cfgin_update, file_name_r)
        cor_savings.send((veusze, log))


def main(new_arg=None, veusze=None, **kwargs):
    """
    Initialise configuration and runs or returns routines
    cfg:
        ['program']['log'],
        'out'
        'in'
        'async'
    globals:
        load_vsz
        lf

    :param new_arg:
    :param veusze: used to reuse veusz embedded object (thus to not leak memory)
    :return:
    """

    global lf, load_vsz
    cfg = cfg_from_args(my_argparser(), new_arg, **kwargs)
    if not cfg or not cfg['program'].get('return'):
        print('Can not initialise')
        return cfg
    elif cfg['program']['return'] == '<cfg_from_args>':  # to help testing
        return cfg

    lf = LoggingStyleAdapter(init_logging('', cfg['program']['log'], cfg['program']['verbose']))
    cfg['program']['log'] = lf.logger.root.handlers[0].baseFilename  # synchronize obtained absolute file name

    print('\n' + this_prog_basename(__file__), 'started', end=' ')
    __name__ = '__main__'  # indicate to other functions that they are called from main

    if cfg['out'].get('paths'):  #?
        if not cfg['out']['b_images_only']:
            raise NotImplementedError('Provided out in not "b_images_only" mode!')
        cfg['out']['nfiles'] = len(cfg['out']['paths'])
        cfg['out']['path'] = cfg['out']['paths'][0]
        print(end=f"\n- {cfg['out']['nfiles']} output files to export images...")
        pass
    else:
        # vsz-pattern needed
        if cfg['in']['pattern_path'] not in (None, Path()):
            # Using cfg['out'] to store pattern information
            if cfg['out']['b_images_only']:
                print('in images only mode. Output pattern: ')  # todo Export path: '
            else:
                print('. Output pattern and Data: ')
            try:
                if not Path(cfg['in']['pattern_path']).is_absolute():
                    cfg['in']['pattern_path'] = Path(cfg['in']['path']).with_name(str(cfg['in']['pattern_path']))
                cfg['out']['path'] = cfg['in']['pattern_path']
                cfg['out']['paths'], cfg['out']['nfiles'], cfg['out']['path'] = init_file_names(
                    **cfg['out'], b_interact=False)
            except Ex_nothing_done as e:
                if not cfg['out']['b_images_only']:
                    lf.warning(f'{e.message} - no pattern. Specify it or use "b_images_only" mode!')
                    return  # or raise FileNotFoundError?
        else:
            cfg['out']['path'] = cfg['in']['path'].with_name('*')
    if cfg['out']['b_images_only'] and cfg['out'].get('paths'):
        cfg['in']['paths'] = cfg['out']['paths']  # have all we need to export
    else:
        try:
            cfg['in']['paths'], cfg['in']['nfiles'], cfg['in']['path'] = init_file_names(
                **cfg['in'], b_interact=cfg['program']['b_interact'])
        except Ex_nothing_done as e:
            print(e.message)
            return  # or raise FileNotFoundError?
        except TypeError:  # expected str, bytes or os.PathLike object, not NoneType
            # cfg['in']['path'] is None. May be it is not need
            cfg['in']['paths'] = [cfg['in']['pattern_path']]  # dummy for compatibility
            cfg['in']['nfiles'] = 1

    cfg['out']['export_dir'] = dir_from_cfg(cfg['out']['path'].parent, cfg['out']['export_dir'])

    if 'restore_config' in cfg['program']['before_next']:
        cfg['in_saved'] = cfg['in'].copy()
    # Next is commented because reloading is Ok: not need to Close()
    # if cfg['out']['b_images_only'] and not 'Close()' in cfg['program']['before_next']:
    #     cfg['program']['before_next'].append(
    #         'Close()')  # usually we need to load new file for export (not only modify previous file)
    if cfg['program']['export_timeout_s'] and export_images_timed:
        cfg['async'] = {'loop': asyncio.get_event_loop(),
                        'export_timeout_s': cfg['program']['export_timeout_s']
                        }
    else:
        cfg['async'] = {'loop': None}

    load_vsz = load_vsz_closure(
        cfg['program']['veusz_path'],
        cfg['program']['load_timeout_s'],
        cfg['program']['b_execute_vsz'],
        cfg['program']['hidden']
        )
    cfg['load_vsz'] = load_vsz
    cfg['co'] = {}
    if cfg['in']['table_log'] and cfg['in']['db_path'] and not (
            cfg['out']['b_images_only'] and len(cfg['in']['paths']) > 1):
        # load data by ranges from table log rows
        in_fulls = h5.log_names_gen(cfg['in'])
    elif cfg['in']['tables']:

        def ge_names_from_hdf5_paths(cfg, f_file_name=lambda x: x):
            """
            Replacing for veuszPropagate.ge_names() to use tables instead files
            :param cfg: dict with field ['in']['tables'], - list of tables or list with regular expression path to find tables
            :return:
            """
            with pd.HDFStore(cfg['in']['db_path'], mode='r') as store:
                if len(cfg['in']['tables']) == 1:
                    cfg['in']['tables'] = h5.find_tables(store, cfg['in']['tables'][0])
            for tbl in cfg['in']['tables']:
                yield f_file_name(tbl)

        # tables instead files
        in_fulls = ge_names_from_hdf5_paths(cfg)
    else:  # switch to use found vsz as source if need only export images (even with database source)
        in_fulls = ge_names(cfg)

    cor_savings = co_savings(cfg)
    cor_savings.send(None)
    nfiles = 0
    try:  # if True:
        path_prev = os_getcwd()
        os_chdir(cfg['out']['path'].parent)
        if cfg['program']['return'] == '<corutines_in_cfg>':
            cfg['co']['savings'] = cor_savings
            cfg['co']['gen_veusz_and_logs'] = load_to_veusz(in_fulls, cfg)
            cfg['co']['send_data'] = co_send_data(load_to_veusz, cfg, cor_savings)
            return cfg  # return with link to generator function
        elif cfg['in'].get('data_yield_prefix'):
            # Cycle with obtaining Veusz data
            cfgin_update = None
            while True:  # for vsz_data, log in cor_send_data.send(cfgin_update):
                try:
                    vsz_data, log = co_send_data.send(cfgin_update)
                    nfiles += 1
                except (GeneratorExit, StopIteration, Ex_nothing_done):
                    break
                if 'f_custom_in_cycle' in cfg['program']:
                    cfgin_update = cfg['program']['f_custom_in_cycle'](vsz_data, log)
        else:
            # Cycle without obtaining Veusz data (or implemented by user's cfg['program']['f_custom_in_cycle'])
            for veusze, log in load_to_veusz(in_fulls, cfg, veusze):
                file_name_r = Path(log['out_vsz_full']).relative_to(cfg['out']['path'].parent)
                if cfg['program'].get('f_custom_in_cycle'):
                    cfgin_update = cfg['program']['f_custom_in_cycle'](veusze, log)
                    veusze_commands(veusze, cfgin_update, file_name_r)
                cor_savings.send((veusze, log))
                nfiles += 1
            cor_savings.close()
            if cfg['program']['return'] != '<embedded_object>':
                veusze = None  # to note that it is closed in cor_savings.close()
        print(f'{nfiles} processed. ok>')

        pass
    except Exception as e:
        lf.exception('Not good')
        return  # or raise FileNotFoundError?
    finally:
        if cfg['async']['loop']:
            cfg['async']['loop'].close()
        os_chdir(path_prev)
        if veusze and cfg['program']['return'] == '<end>':
            veusze.Close()
            try:
                veusze.WaitForClose()
            except AttributeError:  # 'NoneType' object has no attribute 'cmds'
                pass
            veusze = None
        elif cfg['program']['return'] == '<embedded_object>':
            cfg['veusze'] = veusze
            return cfg


if __name__ == '__main__':
    main()
