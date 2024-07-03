#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Save synchronized averaged data to hdf5 tables
  Created: 01.09.2021 - 2024

Load raw data and coefficients from hdf5 table (or group of tables)
Calculate new data (averaging by specified interval)
Combine this data to one new table
"""
import gc
import logging
import re
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import date as datetime_date
from datetime import datetime, timedelta
from functools import wraps, partial
from pathlib import Path
from time import sleep
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
import h5py
import dask.array as da
import dask.dataframe as dd
import hydra
import numpy as np
import pandas as pd
from numba import njit, prange
from numpy.polynomial.polynomial import polyval2d
from omegaconf import MISSING, OmegaConf, open_dict

# my:
# # allows to run on both my Linux and Windows systems:
# scripts_path = Path(f"{'D:' if sys.platform == 'win32' else '/mnt/D'}/Work/_Python3/And0K/h5toGrid/scripts")
# sys.path.append(str(scripts_path.parent.resolve()))
# sys.path.append( str(Path(__file__).parent.parent.resolve()) ) # os.getcwd()
# from utils2init import ini2dict
# from scripts.incl_calibr import calibrate, calibrate_plot, coef2str
# from other_filters import despike, rep2mean
from . import cfg_dataclasses as cfg_d
from . import h5  # v1
# v2
from .csv_load import search_correct_csv_files, load_from_csv_gen, pcid_from_parts
from .csv_specific_proc import parse_name
from .filters import b1spike, rep2mean
from .h5_dask_pandas import (
    cull_empty_partitions,
    dd_to_csv,
    filter_global_minmax,
    filter_local,
    i_bursts_starts,
    h5_load_range_by_coord
)
from .h5inclinometer_coef import dict_matrices_for_h5, h5copy_coef, rot_matrix_x, rotate_y
from .utils2init import (
    Ex_nothing_done,
    LoggingStyleAdapter,
    call_with_valid_kwargs,
    dir_create_if_need,
    set_field_if_no,
    this_prog_basename,
)

lf = LoggingStyleAdapter(logging.getLogger(__name__))
prog = 'incl_h5clc'
VERSION = '1.0.1'
hydra.output_subdir = 'cfg'

# version without hdf5 loading/saving
b_use_h5 = False
hdf5_suffixes = ('.h5', '.hdf5')
# ConfigIn_InclProc config parameters which are dicts with probes ID keys allows select different config value to
# process each probe data table. For this dict ID keys will be compared to regex group <n> returned by match this
# expression to table name (also matches tables with names having numbers joined by "_" of combined data):
re_probe_number = r'(?:(?:w|incl|i)_?(?P<n>[A-z]*\d+)_?)+'


@dataclass
class ConfigInCoefs_InclProc:
    """
    - Rz: rotation matrix
    - g0xyz: accelerometer values when inclination is zero (only relative difference is significant). If not
      None then
    we perform zero calibration and calculate and use new Rz coefficient, current/previous Rz will be ignored.
    New Rz will be saved only if saving raw data when loading from csv. Alternatively use
    ConfigIn_InclProc.time_ranges_zeroing to calculate new Rz.
    """
    Ag: Optional[List[List[float]]] = field(default_factory=lambda: [[0.00173, 0, 0], [0, 0.00173, 0], [0, 0, 0.00173]])
    Cg: Optional[Annotated[list[float], 3]] = field(default_factory=lambda: [10, 10, 10])
    Ah: Optional[List[List[float]]] = field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Ch: Optional[Annotated[list[float], 3]] = field(default_factory=lambda: [10, 10, 10])
    Rz: Optional[List[List[float]]] = field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    kVabs: Optional[List[float]] = field(default_factory=lambda: [10, -10, -10, -3, 3, 70])
    # kP: Optional[List[float]] = field(default_factory=lambda: [0, 1])  # default polynom will not change input
    P_t: Optional[List[List[float]]] = None
    azimuth_shift_deg: Optional[float] = 180
    g0xyz: Optional[Annotated[list[float], 3]] = None  # field(default_factory=lambda: [0, 0, 1])  #

    dates: Optional[Dict[str, str]] = field(default_factory=dict)
    date: Optional[str] = None

@dataclass
class ConfigIn_InclProc():  # cfg_d.ConfigInCsv
    """Parameters of input files
    - path: can point to load data from:
      - text file: should contain {prefix:} and {number:0>2}-like placeholders (0>2 is to fill left with "0"
        for 2 digits)
        This allows separate data from found device types and numbers to corresponded different tables (else
        all found data will be joined). Prefix is a ``prefix`` parameter if not None else text before number.
        Number will be found as digits after text. If pass dir only then default '{prefix:}{id}*.[tT][xX][tT]'
        will be used to search input files, where {id} will be replaced with allowed model char/0 and digits:
        "[0bdp]{\\d:0>2}"
      - pytables hdf5 store: may use patterns in Unix shell style (usually *.h5) or / and
    - tables: list of tables names or regex patterns to search. Note: regex search is switched on only if you
    include "*" symbol somewhere in the pattern.
    - coefs: coefficients. Note: only items different from default items of `ConfigInCoefs_InclProc` will
    have more priority than ones loaded from `coefs_path`
    - coefs_path: coefficients will be copied from this hdf5 store to output hdf5 store. Used only if no coefs
    provided
    - azimuth_add_float: degrees, adds this value to velocity direction (addition to coef.azimuth_shift_deg')
    - time_ranges_zeroing: list of time str, if specified then rotate data in this interval such that it
    will have min mean pitch and roll, display "info" warning about')
    Note: if loading data from csv then saving coef works if csv loaded in chunks has time_ranges_zeroing not
    in 1st chunk.
    New Rz coefficient will be calculated and used instead current/previous Rz. New Rz will be saved only if
    saving raw data when loading from csv. Alternatively keep time_ranges_zeroing None and use coefs.g0xyz
    for new Rz.
    - dt_from_utc: time shift setting in seconds from UTC. Other fields with same prefix and "_hours" and
    other time suffixes can be used to get sum shift.
    - date_to_from: alternative time shift setting, can be converted to dt_from_utc_hours by expression:
    diff(array(date_to_from, 'M8[ms]')).astype('float')/3600000
    - max_incl_of_fit_deg: inclination where calibration curve y = Vabs(x=inclination) became bend down
    and we replace it with line so {\\Delta}^{2}y ≥ 0 for x > max_incl_of_fit_deg
    - calc_version: string, default=, variant of processing Vabs(inclination):',
    choices=['trigonometric(incl)', 'polynom(force)'])

    Note: str type for time is used (and for other special types which are not supported by Hydra)
    """
    # path or paths must be defined
    path: Optional[str] = None         # dir/mask
    paths: Optional[List[str]] = None  # exact paths list without masks, not for csv. If set then ignores `path`
    tables: List[str] = field(default_factory=lambda: ['incl.*'])  # table names in hdf5 store to get data. Uses regexp if only one table name
    tables_log: List[str] = field(default_factory=lambda: ['{}/logFiles'])
    # dt_min_binning_proc: Optional[int] = 2  # seconds, switch to input processed noAvg data when bin is >
    # (uncomment to not use hard coded 2s default)

    # params needed to process data when loading directly from csv:
    prefix: Optional[str] = 'I*_'  # 'INKL_'
    text_type: Optional[str] = None  # corresponds to predefined text_line_regex. If it and text_line_regex is None then
    # will be guessed by file name
    text_line_regex: Optional[str] = None  # if not None then ignore text_type else will be selected based on text_type

    coefs: Optional[ConfigInCoefs_InclProc] = field(default_factory=ConfigInCoefs_InclProc)  # None
    coefs_path: Optional[str] = r'd:\WorkData\~configuration~\inclinometer\190710incl.h5'
    date_to_from: Optional[List[str]] = None
    dt_from_utc: Optional[int] = 0
    min_date: Optional[str] = None  # input data time range minimum
    max_date: Optional[str] = None  # input data time range maximum
    time_ranges: Optional[List[str]] = None
    coordinates: Optional[List[float]] = None  # add magnetic declination at coordinates [Lat, Lon], degrees

    time_ranges_zeroing: Optional[List[str]] = field(default_factory=list)
    azimuth_add: float = 0
    max_incl_of_fit_deg: Optional[float] = None
    calc_version: str = 'trigonometric(incl)'

    # minimum time between blocks, required in filt_data_dd() for data quality control messages:
    # default: nearly infinity value to not use bursts, None to auto-find and repartition
    dt_between_bursts: Optional[float] = timedelta.max.total_seconds()
    dt_hole_warning: Optional[int] = 600


@dataclass
class ConfigOut_InclProc(cfg_d.ConfigOutSimple):
    """
    "out": parameters of output files
    #################################
    :param not_joined_db_path: If set then save processed not averaged data for each probe individually to this path. If set to True then path will be out.db_path with suffix ".proc_noAvg.h5". Table names will be the same as input data.
    :param table: table name in hdf5 store to write combined/averaged data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in auto numbered locations (see dask to_hdf())
    :param split_period: string (pandas offset: D, H, 2S, ...) to process and output in separate blocks. If saving csv for data with original sampling period is set then that csv will be splitted with by this length (for data with no bin averaging  only)
    :param dt_bins: list of int seconds to bin average data. This can greatly reduce sizes of output hdf5/text files. Frequently used: 0, 2, 600, 7200
    :param text_path: path to save text files with processed velocity (each probe individually). No file if empty, "text_output" if ``out.dt_bins`` is set. Relative paths are from out.db_path
    :param text_date_format: default '%Y-%m-%d %H:%M:%S.%f', format of date column in output text files. (.%f will be removed from end for bins in dt_bins > 1s). Can use float or string representations
    :param text_columns_list: if not empty then saved text files will contain only columns here specified
    :param b_all_to_one_col: concatenate all data in same columns in out DB. both separated and joined text files will be written
    :param b_del_temp_db: default='False', temporary hdf5 file will be automatically deleted after operation. If false it will be remained. Delete it manually. Note: it may contain useful data for recovery

    """
    not_joined_db_path: Any = None
    raw_db_path: Any = 'auto'  # will be the parent of input.path dir for text files
    table: str = ''
    tables_log: List[str] = field(default_factory=lambda: ['{}/logFiles'])  # overwrites default
    dt_bins: Optional[List[int]] = field(
        default_factory=lambda: [0, 2, 600, 3600, 7200]
    )  # int will be auto converted to timedelta [s]
    dt_bins_min_save_text: Optional[int] = 1  # s: Save to text only averaged data with bin >= 1s
    split_period: str = ''
    text_path: Optional[str] = 'text_output'
    text_date_format: str = '%Y-%m-%d %H:%M:%S.%f'
    text_columns: List[str] = field(default_factory=list)
    b_split_by_time_ranges: bool = False  # split data by cfg['in']['time_ranges']. Default False: merge them
    b_all_to_one_col: bool = False
    b_del_temp_db: bool = False
    b_overwrite: bool = True  # default is not to add new data to previous


@dataclass
class ConfigFilter_InclProc:
    """
    "filter": excludes some data:
    :param min_dict, max_dict: Filter out (set to NaN) data of ``key`` columns if it is below/above ``value``
    Possible keys are: input parameters: ... and intermediate parameters:
    - g_minus_1: sets Vabs to NaN if module of acceleration is greater
    - h_minus_1: sets Vdir to zero if module of magnetic field is greater,

    :param dates_min_dict: List with items in "key:value" format. Start of time range for each probe: (used instead common for each probe min_dict["Time"]) ')
    :param dates_max_dict: List with items in "key:value" format. End of time range for each probe: (see dates_min_dict)
    :param bad_p_at_bursts_starts_period: pandas offset string. If set then marks each 2 samples of Pressure at start of burst as bad')
    :param date_to_from: used to shift time if loading from csv

    """
    min: Optional[Dict[str, float]] = field(default_factory=dict)
    max: Optional[Dict[str, float]] = field(default_factory=lambda: {'g_minus_1': 1, 'h_minus_1': 8})
    bad_p_at_bursts_starts_period: str = ''

@dataclass
class ConfigInMany_InclProc:
    """
    Processing parameters for many probes: specify probe numbers as keys in parameters. Probe id will be infer from table name. For single probes these fields mostly defined in `ConfigIn_InclProc`
    - bad_p_at_bursts_starts_period: see the field in `ConfigFilter_InclProc`
    """
    # Fields have keys and meaning same that can be found in ConfigIn/ConfigFilter fields but values are
    # dicts {pcid: value} which specifies different values for each Probe output Column ID (pcid):
    # {type}_{model}{number} or {type}{number} if no model: type may be i/w for inclinometer/wave gage.
    # Inclinometers with pressure sensor currently have only model "p"
    min_date: Dict[str, Any] = field(default_factory=dict)  # Any is for List[str] but hydra not supported
    max_date: Dict[str, Any] = field(default_factory=dict)  # - // -
    time_ranges: Optional[Dict[str, Any]] = field(default_factory=dict)
    coordinates: Optional[Dict[str, Any]] = field(default_factory=dict)
    time_ranges_zeroing: Optional[Dict[str, List[str]]] = field(default_factory=dict)

    date_to_from: Optional[Dict[str, Any]] = field(default_factory=dict)
    # bad_p_at_bursts_starts_period: Optional[Dict[str, Any]] = field(default_factory=dict)

    # skip_if_no_coef: Optional[bool] = False  # If True then if coefs_path specified, but no coef there found then
    #                                          # continue using ``coefs`` else - error (default)

@dataclass
class ConfigProgram_InclProc(cfg_d.ConfigProgram):
    """
    return_ may have values:
    - "<cfg_before_cycle>" to return config before loading data, or other any nonempty
    - "<cfg_input_files_list>",
    - "<cfg_input_files_list_to_yaml>",
    - "<cfg_input_files_meta>",
    - "<cfg_input_files_meta_to_yaml>"

    threads - good for numeric code that releases the GIL (like NumPy, Pandas, Scikit-Learn, Numba, …)
    """
    dask_scheduler: str = ''  # variants: synchronous, threads, processes, single-threaded, distributed
    sleep_s: float = 0.5  # to run slower, helps for system memory management?'

    # cruise directories to search in in.db_path to set path of out.db_path under it if out.db_path is not absolute:
    # raw_dir_words_list: Optional[List[str]] = field(
    #     default_factory=lambda: cfg_d.ConfigInCsv().raw_dir_words)
    # b_incremental_update: bool = False  # not used  # todo: use



cs_store_name = Path(__file__).stem
cs, ConfigType = cfg_d.hydra_cfg_store(
    cs_store_name, {
        'input': [ConfigIn_InclProc],  # Load the config "in_hdf5" from the config group "input"
        'out': [ConfigOut_InclProc],  # Set as MISSING to require the user to specify a value on the command line.
        'filter': [ConfigFilter_InclProc],
        'in_many': [ConfigInMany_InclProc],
        'program': [ConfigProgram_InclProc]
        # 'probes': ['probes__incl_proc'],
        # 'search_path': 'empty.yml' not works
    },
    module=sys.modules[__name__]
    )


RT = TypeVar('RT')  # return type


def allow_dask(wrapped: Callable[..., RT]) -> Callable[..., RT]:
    """
    Use dask.Array functions instead of numpy if first argument is dask.Array
    :param wrapped: function that use methods of numpy each of that existed in dask.Array
    :return:
    """

    @wraps(wrapped)
    def _func(*args):
        if isinstance(args[0], (da.Array, dd.DataFrame)):
            np = da
        return wrapped(*args)

    return _func


@allow_dask
def f_pitch(Gxyz):
    """
    Pitch calculating
    :param Gxyz: shape = (3,len) Accelerometer data
    :return: angle, radians, shape = (len,)
    """
    return -np.arctan(Gxyz[0, :] / np.linalg.norm(Gxyz[1:, :], axis=0))
    # =arctan2(Gxyz[0,:], sqrt(square(Gxyz[1,:])+square(Gxyz[2,:])) )')


@allow_dask
def f_roll(Gxyz):
    """
    Roll calculating
    :param Gxyz: shape = (3,len) Accelerometer data
    :return: angle, radians, shape = (len,)
    """
    return np.arctan2(Gxyz[1, :], Gxyz[2, :])


# @njit - removed because numba gets long bytecode dump without messages/errors
def fIncl_rad2force(incl_rad: np.ndarray):
    """
    Theoretical force from inclination
    :param incl_rad:
    :return:
    """
    return np.sqrt(np.tan(incl_rad) / np.cos(incl_rad))


@allow_dask
def fIncl_deg2force(incl_deg):
    return fIncl_rad2force(np.radians(incl_deg))


# no @jit - polyval not supported
def fVabsMax0(x_range, y0max, coefs):
    """End point of good interpolation"""
    x0 = x_range[np.flatnonzero(np.polyval(coefs, x_range) > y0max)[0]]
    return (x0, np.polyval(coefs, x0))


# no @jit - polyval not supported
def fVabs_from_force(force, coefs, vabs_good_max=0.5):
    """

    :param force:
    :param coefs: polynom coef ([d, c, b, a]) to calc Vabs(force)
    :param vabs_good_max: m/s, last (max) value of polynom to calc Vabs(force) next use line function
    :return:
    """

    # preventing "RuntimeWarning: invalid value encountered in less/greater"
    is_nans = np.isnan(force)
    force[is_nans] = 0

    # last force of good polynom fitting
    x0, v0 = fVabsMax0(np.arange(1, 4, 0.01), vabs_good_max, coefs)

    def v_normal(x):
        """
            After end point of good polinom fitting use linear function
        :param x:
        :return:
        """
        v = np.polyval(coefs, x)
        return np.where(x < x0, v, v0 + (x - x0) * (v0 - np.polyval(coefs, x0 - 0.1)) / 0.1)

    # Using fact that 0.25 + eps = fIncl_rad2force(0.0623) where eps > 0 is small value
    incl_good_min = 0.0623
    incl_range0 = np.linspace(0, incl_good_min, 15)
    force_range0 = fIncl_rad2force(incl_range0)

    def v_linear(x):
        """
        Linear(incl) function crossed (0,0) and 1st good polinom fitting force = 0.25
        where force = fIncl_rad2force(incl)
        :param x:
        :return: array with length of x

        """
        return np.interp(x, force_range0, incl_range0) * np.polyval(coefs, 0.25) / incl_good_min

    force = np.where(force > 0.25, v_normal(force), v_linear(force))
    force[is_nans] = np.nan
    return force


@njit(fastmath=True)
def trigonometric_series_sum(r, coefs):
    """

    :param r:
    :param coefs: array of even length
    :return:
    Not jitted version was:
    def trigonometric_series_sum(r, coefs):
        return coefs[0] + np.nansum(
            [(a * np.cos(nr) + b * np.sin(nr)) for (a, b, nr) in zip(
                coefs[1::2], coefs[2::2], np.arange(1, len(coefs) / 2)[:, None] * r)],
            axis=0)
    """
    out = np.empty_like(r)
    out[:] = coefs[0]
    for n in prange(1, (len(coefs)+1) // 2):
        a = coefs[n * 2 - 1]
        b = coefs[n * 2]
        nr = n * r
        out += (a * np.cos(nr) + b * np.sin(nr))
    return out


@njit(fastmath=True)
def rep_if_bad(checkit, replacement):
    return checkit if (checkit and np.isfinite(checkit)) else replacement


@njit(fastmath=True)
def f_linear_k(x0, g, g_coefs):
    replacement = np.float64(10)
    return min(rep_if_bad(np.diff(g(x0 - np.float64([0.01, 0]), g_coefs)).item() / 0.01, replacement),
               replacement)


@njit(fastmath=True)
def f_linear_end(g, x, x0, g_coefs):
    """
    :param g, x, g_coefs: function and its arguments to calc g(x, *g_coefs)
    :param x0: argument where g(...) replace with linear function if x > x0
    :return:
    """
    g0 = g(x0, g_coefs)
    return np.where(x < x0, g(x, g_coefs), g0 + (x - x0) * f_linear_k(x0, g, g_coefs))


@njit(fastmath=True)
def v_trig(r, coefs):
    squared = np.sin(r) / trigonometric_series_sum(r, coefs)
    # with np.errstate(invalid='ignore'):  # removes warning of comparison with NaN
    # return np.sqrt(squared, where=squared > 0, out=np.zeros_like(squared)) to can use njit replaced with:
    return np.where(squared > 0, np.sqrt(squared), 0)


def v_abs_from_incl(incl_rad: np.ndarray, coefs: Sequence, calc_version='trigonometric(incl)', max_incl_of_fit_deg=None) -> np.ndarray:
    """
    Vabs = np.polyval(coefs, Gxyz)

    :param incl_rad:
    :param coefs: coefficients.
    Note: for 'trigonometric(incl)' option if not max_incl_of_fit_deg then it is in last coefs element
    :param calc_version: 'polynom(force)' if this str or len(coefs)<=4 else if 'trigonometric(incl)' uses trigonometric_series_sum()
    :param max_incl_of_fit_deg:
    :return:
    """
    if len(coefs)<=4 or calc_version == 'polynom(force)':
        lf.warning('Old coefs method "polynom(force)"')
        if not len(incl_rad):   # need for numba njit error because of dask calling with empty arg to check type if no meta?
            return incl_rad     # empty of same type as input
        force = fIncl_rad2force(incl_rad)
        return fVabs_from_force(force, coefs)

    elif calc_version == 'trigonometric(incl)':
        if max_incl_of_fit_deg:
            max_incl_of_fit = np.radians(max_incl_of_fit_deg)
        else:
            max_incl_of_fit = np.radians(coefs[-1])
            coefs = coefs[:-1]

        with np.errstate(invalid='ignore'):                         # removes warning of comparison with NaN
            return f_linear_end(g=v_trig, x=incl_rad,
                                x0=np.atleast_1d(max_incl_of_fit),
                                g_coefs=np.float64(coefs))          # atleast_1d, float64 is to can use njit
    else:
        raise NotImplementedError(f'Bad calc method {calc_version}', )


# @overload(np.linalg.norm)
# def jit_linalg_norm(x, ord=None, axis=None, keepdims=False):
#     # replace notsupported numba argument
#     if axis is not None and (ord is None or ord == 2):
#         s = (x.conj() * x).real
#         # original implementation: return sqrt(add.reduce(s, axis=axis, keepdims=keepdims))
#         return np.sqrt(s.sum(axis=axis, keepdims=keepdims))


@allow_dask
def fInclination(Gxyz: np.ndarray):
    return np.arctan2(np.linalg.norm(Gxyz[:-1, :], axis=0), Gxyz[2, :])


# @allow_dask not need because explicit np/da lib references are not used
#@njit("f8[:,:](f8[:,:], f8[:,:], f8[:,:])") - failed for dask array
def fG(Axyz: Union[np.ndarray, da.Array],
       Ag: Union[np.ndarray, da.Array],
       Cg: Union[np.ndarray, da.Array]) -> Union[np.ndarray, da.Array]:
    """
    Allows use of transposed Cg
    :param Axyz:
    :param Ag: scaling coefficient
    :param Cg: shift coefficient
    :return:
    """
    assert Ag.any(), 'Ag coefficients all zeros!!!'
    if Cg.ndim < 2:
        Cg = Cg.reshape(-1, 1)
    return Ag @ (Axyz - Cg)


# @overload(fG)
# def jit_fG(Axyz: np.ndarray, Ag: np.ndarray, Cg: np.ndarray):
#     # replace notsupported numba int argument
#     if isinstance(Axyz.dtype, types.Integer):
#         return Ag @ (Axyz.astype('f8') - (Cg if Cg.shape[0] == Ag.shape[0] else Cg.T))


# def polar2dekart_complex(Vabs, Vdir):
#     return Vabs * (da.cos(da.radians(Vdir)) + 1j * da.sin(da.radians(Vdir)))
@allow_dask
def polar2dekart(Vabs, Vdir) -> List[Union[da.Array, np.ndarray]]:
    """

    :param Vabs:
    :param Vdir: degrees
    :return: list [v, u] of (north, east) components. List (not tuple) is used because it is more convenient to concatenate it with other data
    """
    return [Vabs * np.cos(np.radians(Vdir)), Vabs * np.sin(np.radians(Vdir))]


# @allow_dask
# def dekart2polar(v_en):
#     """
#     Not Tested
#     :param u:
#     :param v:
#     :return: [Vabs, Vdir]
#     """
#     return np.linalg.norm(v_en, axis=0), np.degrees(np.arctan2(*v_en))

def dekart2polar_df_uv(df, **kwargs):
    """

    :param d: if no columns u and v remains unchanged
    :**kwargs :'inplace' not supported in dask. dumn it!
    :return: [Vabs, Vdir] series
    """

    # why da.linalg.norm(df.loc[:, ['u','v']].values, axis=1) gives numpy (not dask) array?
    # da.degrees(df.eval('arctan2(u, v)')))

    if 'u' in df.columns:

        kdegrees = 180 / np.pi

        return df.eval(f"""
        Vabs = sqrt(u**2 + v**2)
        Vdir = arctan2(u, v)*{kdegrees:.20}
        """, **kwargs)
    else:
        return df


def incl_calc_velocity_nodask(
        a: pd.DataFrame,
        Ag, Cg, Ah, Ch,
        kVabs: Sequence = (1, 0),
        azimuth_shift_deg: float = 0,
        cfg_filter: Mapping[str, Any] = None,
        cfg_proc=None):
    """

    :param a:
    :param Ag:
    :param Cg:
    :param Ah:
    :param Ch:
    :param azimuth_shift_deg:
    :param cfg_filter: dict. cfg_filter['max_g_minus_1'] useed to check module of Gxyz, cfg_filter['max_h_minus_1'] to set Vdir=0
    :param cfg_proc: 'calc_version', 'max_incl_of_fit_deg'
    :return: dataframe withcolumns ['Vabs', 'Vdir', col, 'inclination'] where col is additional column in _a_, or may be absent
    """
    da = np
    # da.from_array = lambda x, *args, **kwargs: x
    lf.info('calculating V')
    if kVabs == (1, 0):
        lf.warning('kVabs == (1, 0)! => V = sqrt(sin(inclination))')
    #
    # old coefs need transposing: da.dot(Ag.T, (Axyz - Cg[0, :]).T)
    # fG = lambda Axyz, Ag, Cg: da.dot(Ag, (Axyz - Cg))
    # fInclination = lambda Gxyz: np.arctan2(np.sqrt(np.sum(np.square(Gxyz[:-1, :]), 0)), Gxyz[2, :])

    try:
        Gxyz = fG(a.loc[:, ('Ax', 'Ay', 'Az')].to_numpy().T, Ag, Cg)  # lengths=True gets MemoryError   #.to_dask_array()?, dd.from_pandas?
        # .rechunk((1800, 3))
        # filter
        GsumMinus1 = da.linalg.norm(Gxyz, axis=0) - 1  # should be close to zero
        incl_rad = fInclination(Gxyz)

        if cfg_filter and ('max_g_minus_1' in cfg_filter):
            bad_g = np.fabs(GsumMinus1.compute()) > cfg_filter['max_g_minus_1']
            bad_g_sum = bad_g.sum(axis=0)
            if bad_g_sum > 0.1 * len(GsumMinus1):
                print('Acceleration is too high (>{}) in {}% points!'.format(
                    cfg_filter['max_g_minus_1'], 100 * bad_g_sum / len(GsumMinus1))
                )
            incl_rad[bad_g] = np.nan

        Vabs = v_abs_from_incl(incl_rad, kVabs, cfg_proc['calc_version'], cfg_proc['max_incl_of_fit_deg'])

        Hxyz = fG(a.loc[:, ('Mx', 'My', 'Mz')].to_numpy().T, Ah, Ch)
        Vdir = azimuth_shift_deg - da.degrees(da.arctan2(
            (Gxyz[0, :] * Hxyz[1, :] - Gxyz[1, :] * Hxyz[0, :]) * (GsumMinus1 + 1),
            Hxyz[2, :] * da.square(Gxyz[:-1, :]).sum(axis=0) - Gxyz[2, :] * (
                (Gxyz[:-1, :] * Hxyz[:-1, :]).sum(axis=0))
            ))

        col = 'Pressure' if ('Pressure' in a.columns) else 'Temp' if ('Temp' in a.columns) else []
        columns = ['Vabs', 'Vdir', 'v', 'u', 'inclination']
        arrays_list = [Vabs, Vdir] + polar2dekart(Vabs, Vdir) + [da.degrees(incl_rad)]
        a = a.assign(**{c: ar for c, ar in zip(columns, arrays_list)})  # a[c] = ar

        # df = pd.DataFrame.from_records(dict(zip(columns, [Vabs, Vdir, np.degrees(incl_rad)])), columns=columns, index=a.index)  # no sach method in dask
        return a[columns + [col]]
    except Exception as e:
        lf.exception('Error in incl_calc_velocity():')
        raise


def norm_field(raw3d, coef_a2d, coef_c, raw3d_helps_recover=None):
    """

    :param raw3d: data
    :param coef_a2d: multiplier part of coef
    :param c: shift part of coef
    :param raw3d_helps_recover: use sign of raw3d_helps_recover
    # todo: get initial sign such that most of raw3d_helps_recover sign will match recovering channel data
    :return:
    """
    if coef_c.ndim < 2:
        coef_c = coef_c.reshape(-1, 1)
    # Apply coefs
    s = coef_a2d @ (raw3d - coef_c)

    # If gain for some channel is zero
    i_ch_bad = np.flatnonzero(coef_a2d.diagonal() == 0)  # diagonal elements: coef_a2d[[0, 1, 2], [0, 1, 2]]
    if i_ch_bad.size:
        # Recover channel
        i_ch_ok = [i for i in range(3) if i != i_ch_bad]
        s[i_ch_bad] = np.square(1 - (s[i_ch_ok] ** 2).sum(axis=0))
        if (s[i_ch_bad].imag != 0).any():
            s[i_ch_bad] = s[i_ch_bad].real

        # Sign of recovering channel
        if raw3d_helps_recover is not None:
            s[i_ch_bad] *= np.sign(raw3d_helps_recover[i_ch_bad])
            return s

        # Select sign in inverse gradient points so that we minimise changes of gradient
        s_dif = np.ediff1d(s[i_ch_bad], to_begin=0)
        # where gradient reverse
        b_reversed = (s_dif < 0)  # signal decreases
        b_reversed &= np.append(~b_reversed[1:], False)  # and signal increases in next point
        # Keep reversed where diff(s_dif) of inverted signal will be less than of not inverted (where gradient reversed)
        # diff of reversed signal (after b_reversed points)
        _ = s[i_ch_bad, b_reversed] + s[i_ch_bad, np.roll(b_reversed, 1)]
        # diff in point before gradient reverse:
        s_dif_prev = s_dif[np.roll(b_reversed, -1)]
        # select sign of signal with minimum change of |gradient|
        b_reversed[b_reversed] = np.abs(s_dif_prev - _) < np.abs(s_dif_prev - s_dif[b_reversed])

        # Consequently invert signal in b_reversed points
        s_rev_sign = np.zeros_like(s_dif);
        s_rev_sign[0] = 1
        n_reversed = sum(b_reversed)
        s_rev_sign[b_reversed] = np.tile([-2, 2], int(np.ceil(n_reversed / 2)))[:n_reversed]
        s_rev_sign = np.cumsum(s_rev_sign)
        s[i_ch_bad] = s_rev_sign * s[i_ch_bad]
    return s


@njit
def recover_x__sympy_lambdify(y, z, Ah, Ch, mean_Hsum):
    """

    :param y: channel data
    :param z: channel data
    :param Ah: scaling coef
    :param Ch: shifting coef
    :param mean_Hsum:
    :return:
    Note:  After sympy added abs() under sqrt() to exclude complex values
    """

    [a00, a01, a02] = Ah[0]
    [a10, a11, a12] = Ah[1]
    [a20, a21, a22] = Ah[2]
    [c00, c10, c20] = np.ravel(Ch)
    return (
        a00 ** 2 * c00 + a00 * a01 * c10 - a00 * a01 * y + a00 * a02 * c20 - a00 * a02 * z + a10 ** 2 * c00 +
        a10 * a11 * c10 - a10 * a11 * y + a10 * a12 * c20 - a10 * a12 * z + a20 ** 2 * c00 + a20 * a21 * c10 -
        a20 * a21 * y + a20 * a22 * c20 - a20 * a22 * z - np.sqrt(np.abs(
               -a00 ** 2 * a11 ** 2 * c10 ** 2 + 2 * a00 ** 2 * a11 ** 2 * c10 * y - a00 ** 2 * a11 ** 2 * y ** 2 - 2 * a00 ** 2 * a11 * a12 * c10 * c20 + 2 * a00 ** 2 * a11 * a12 * c10 * z + 2 * a00 ** 2 * a11 * a12 * c20 * y - 2 * a00 ** 2 * a11 * a12 * y * z - a00 ** 2 * a12 ** 2 * c20 ** 2 + 2 * a00 ** 2 * a12 ** 2 * c20 * z - a00 ** 2 * a12 ** 2 * z ** 2 - a00 ** 2 * a21 ** 2 * c10 ** 2 + 2 * a00 ** 2 * a21 ** 2 * c10 * y - a00 ** 2 * a21 ** 2 * y ** 2 - 2 * a00 ** 2 * a21 * a22 * c10 * c20 + 2 * a00 ** 2 * a21 * a22 * c10 * z + 2 * a00 ** 2 * a21 * a22 * c20 * y - 2 * a00 ** 2 * a21 * a22 * y * z - a00 ** 2 * a22 ** 2 * c20 ** 2 + 2 * a00 ** 2 * a22 ** 2 * c20 * z - a00 ** 2 * a22 ** 2 * z ** 2 + a00 ** 2 * mean_Hsum ** 2 + 2 * a00 * a01 * a10 * a11 * c10 ** 2 - 4 * a00 * a01 * a10 * a11 * c10 * y + 2 * a00 * a01 * a10 * a11 * y ** 2 + 2 * a00 * a01 * a10 * a12 * c10 * c20 - 2 * a00 * a01 * a10 * a12 * c10 * z - 2 * a00 * a01 * a10 * a12 * c20 * y + 2 * a00 * a01 * a10 * a12 * y * z + 2 * a00 * a01 * a20 * a21 * c10 ** 2 - 4 * a00 * a01 * a20 * a21 * c10 * y + 2 * a00 * a01 * a20 * a21 * y ** 2 + 2 * a00 * a01 * a20 * a22 * c10 * c20 - 2 * a00 * a01 * a20 * a22 * c10 * z - 2 * a00 * a01 * a20 * a22 * c20 * y + 2 * a00 * a01 * a20 * a22 * y * z + 2 * a00 * a02 * a10 * a11 * c10 * c20 - 2 * a00 * a02 * a10 * a11 * c10 * z - 2 * a00 * a02 * a10 * a11 * c20 * y + 2 * a00 * a02 * a10 * a11 * y * z + 2 * a00 * a02 * a10 * a12 * c20 ** 2 - 4 * a00 * a02 * a10 * a12 * c20 * z + 2 * a00 * a02 * a10 * a12 * z ** 2 + 2 * a00 * a02 * a20 * a21 * c10 * c20 - 2 * a00 * a02 * a20 * a21 * c10 * z - 2 * a00 * a02 * a20 * a21 * c20 * y + 2 * a00 * a02 * a20 * a21 * y * z + 2 * a00 * a02 * a20 * a22 * c20 ** 2 - 4 * a00 * a02 * a20 * a22 * c20 * z + 2 * a00 * a02 * a20 * a22 * z ** 2 - a01 ** 2 * a10 ** 2 * c10 ** 2 + 2 * a01 ** 2 * a10 ** 2 * c10 * y - a01 ** 2 * a10 ** 2 * y ** 2 - a01 ** 2 * a20 ** 2 * c10 ** 2 + 2 * a01 ** 2 * a20 ** 2 * c10 * y - a01 ** 2 * a20 ** 2 * y ** 2 - 2 * a01 * a02 * a10 ** 2 * c10 * c20 + 2 * a01 * a02 * a10 ** 2 * c10 * z + 2 * a01 * a02 * a10 ** 2 * c20 * y - 2 * a01 * a02 * a10 ** 2 * y * z - 2 * a01 * a02 * a20 ** 2 * c10 * c20 + 2 * a01 * a02 * a20 ** 2 * c10 * z + 2 * a01 * a02 * a20 ** 2 * c20 * y - 2 * a01 * a02 * a20 ** 2 * y * z - a02 ** 2 * a10 ** 2 * c20 ** 2 + 2 * a02 ** 2 * a10 ** 2 * c20 * z - a02 ** 2 * a10 ** 2 * z ** 2 - a02 ** 2 * a20 ** 2 * c20 ** 2 + 2 * a02 ** 2 * a20 ** 2 * c20 * z - a02 ** 2 * a20 ** 2 * z ** 2 - a10 ** 2 * a21 ** 2 * c10 ** 2 + 2 * a10 ** 2 * a21 ** 2 * c10 * y - a10 ** 2 * a21 ** 2 * y ** 2 - 2 * a10 ** 2 * a21 * a22 * c10 * c20 + 2 * a10 ** 2 * a21 * a22 * c10 * z + 2 * a10 ** 2 * a21 * a22 * c20 * y - 2 * a10 ** 2 * a21 * a22 * y * z - a10 ** 2 * a22 ** 2 * c20 ** 2 + 2 * a10 ** 2 * a22 ** 2 * c20 * z - a10 ** 2 * a22 ** 2 * z ** 2 + a10 ** 2 * mean_Hsum ** 2 + 2 * a10 * a11 * a20 * a21 * c10 ** 2 - 4 * a10 * a11 * a20 * a21 * c10 * y + 2 * a10 * a11 * a20 * a21 * y ** 2 + 2 * a10 * a11 * a20 * a22 * c10 * c20 - 2 * a10 * a11 * a20 * a22 * c10 * z - 2 * a10 * a11 * a20 * a22 * c20 * y + 2 * a10 * a11 * a20 * a22 * y * z + 2 * a10 * a12 * a20 * a21 * c10 * c20 - 2 * a10 * a12 * a20 * a21 * c10 * z - 2 * a10 * a12 * a20 * a21 * c20 * y + 2 * a10 * a12 * a20 * a21 * y * z + 2 * a10 * a12 * a20 * a22 * c20 ** 2 - 4 * a10 * a12 * a20 * a22 * c20 * z + 2 * a10 * a12 * a20 * a22 * z ** 2 - a11 ** 2 * a20 ** 2 * c10 ** 2 + 2 * a11 ** 2 * a20 ** 2 * c10 * y - a11 ** 2 * a20 ** 2 * y ** 2 - 2 * a11 * a12 * a20 ** 2 * c10 * c20 + 2 * a11 * a12 * a20 ** 2 * c10 * z + 2 * a11 * a12 * a20 ** 2 * c20 * y - 2 * a11 * a12 * a20 ** 2 * y * z - a12 ** 2 * a20 ** 2 * c20 ** 2 + 2 * a12 ** 2 * a20 ** 2 * c20 * z - a12 ** 2 * a20 ** 2 * z ** 2 + a20 ** 2 * mean_Hsum ** 2))
           ) / (a00 ** 2 + a10 ** 2 + a20 ** 2)


def recover_magnetometer_x(Mcnts, Ah, Ch, max_h_minus_1, len_data):
    Hxyz = fG(Mcnts, Ah, Ch)  # x.rechunk({0: -1, 1: 'auto'}, block_size_limit=1e8)
    HsumMinus1 = da.linalg.norm(Hxyz, axis=0) - 1  # should be close to zero

    # Channel x recovering
    bad = da.isnan(Mcnts[0, :])
    need_recover_mask = da.isfinite(Mcnts[1:, :]).any(axis=0)  # where other channels ok
    #sleep(cfg_program['sleep_s'])
    can_recover = need_recover_mask.sum(axis=0).compute()
    if can_recover:
        Mcnts_list = [[], [], []]
        need_recover_mask &= bad  # only where x is bad
        need_recover = need_recover_mask.sum(axis=0).compute()
        if need_recover:  # have points where recover is needed and is possible
            lf.info(
                'Magnetometer x channel recovering where {:d} it is bad and y&z is ok (y&z ok in {:d}/{:d})',
                need_recover, can_recover, len_data
            )
            # Try to recover mean_Hsum (should be close to 1)
            mean_HsumMinus1 = np.nanmedian((HsumMinus1[HsumMinus1 < max_h_minus_1]).compute())
            if np.isnan(mean_HsumMinus1) or (np.fabs(mean_HsumMinus1) > 0.5 and need_recover / len_data > 0.95):
                # need recover all x points because too small points with good HsumMinus1
                lf.warning('mean_Hsum is mostly bad (mean={:g}), most of data need to be recovered ({:g}%) so no trust it'
                          ' at all. Recovering all x-ch.data with setting mean_Hsum = 1',
                          mean_HsumMinus1, 100 * need_recover / len_data)
                bad = da.ones_like(HsumMinus1, dtype=np.bool_)
                mean_HsumMinus1 = 0
            else:
                lf.warning('calculated mean_Hsum - 1 is good (close to 0): mean={:s}', mean_HsumMinus1)

            # Recover magnetometer's x channel

            # todo: not use this channel but recover dir instantly
            # da.apply_along_axis(lambda x: da.where(x < 0, 0, da.sqrt(abs(x))),
            #                               0,
            #                  mean_Hsum - da.square(Ah[2, 2] * (rep2mean_da(Mcnts[2,:], Mcnts[2,:] > 0) - Ch[2])) -
            #                               da.square(Ah[1, 1] * (rep2mean_da(Mcnts[1,:], Mcnts[1,:] > 0) - Ch[1]))
            #                               )

            #Mcnts_x_recover = recover_x__sympy_lambdify(Mcnts[1, :], Mcnts[2, :], Ah, Ch, mean_Hsum=mean_HsumMinus1 + 1)
            Mcnts_x_recover = da.map_blocks(  # replaced to this to use numba:
                recover_x__sympy_lambdify, Mcnts[1, :], Mcnts[2, :],
                Ah=Ah, Ch=Ch, mean_Hsum=mean_HsumMinus1 + 1, dtype=np.float64, meta=np.float64([])
            )
            Mcnts_list[0] = da.where(need_recover_mask, Mcnts_x_recover, Mcnts[0, :])
            bad &= ~need_recover_mask

            # other points recover by interp
            Mcnts_list[0] = da.from_array(rep2mean_da2np(Mcnts_list[0], ~bad), chunks=Mcnts_list[0].chunks,
                                          name='Mcnts_list[0]')
        else:
            Mcnts_list[0] = Mcnts[0, :]

        lf.debug('interpolating magnetometer data using neighbor points separately for each channel...')
        need_recover_mask = da.ones_like(HsumMinus1, dtype=np.bool_)  # here save where Vdir can not recover
        for ch, i in [('x', 0), ('y', 1), ('z', 2)]:  # in ([('y', 1), ('z', 2)] if need_recover else
            print(ch, end=' ')
            if (ch != 'x') or not need_recover:
                Mcnts_list[i] = Mcnts[i, :]
            bad = da.isnan(Mcnts_list[i])
            n_bad = bad.sum(axis=0).compute()  # exits with "Process finished with exit code -1073741819 (0xC0000005)"!
            if n_bad:
                n_good = HsumMinus1.shape[0] - n_bad
                if n_good / n_bad > 0.01:
                    lf.info(f'channel {ch}: bad points: {n_bad} - recovering using nearest good points ({n_good})')
                    Mcnts_list[i] = da.from_array(rep2mean_da2np(Mcnts_list[i], ~bad), chunks=Mcnts_list[0].chunks,
                                                  name=f'Mcnts_list[{ch}]-all_is_finite')
                else:
                    lf.warning(
                        f'channel {ch}: bad points: {n_bad} - will not recover because too small good points ({n_good})'
                    )
                    Mcnts_list[i] = np.nan + da.empty_like(HsumMinus1)
                    need_recover_mask[bad] = False

        Mcnts = da.vstack(Mcnts_list)
        Hxyz = fG(Mcnts, Ah, Ch)  # #x.rechunk({0: -1, 1: 'auto'}, block_size_limit=1e8)

    else:
        lf.info('Magnetometer can not be recovered')
        need_recover_mask = None

    return Hxyz, need_recover_mask


def rep2mean_da(y: da.Array, bOk=None, x=None, overlap_depth=None) -> da.Array:
    """
    Interpolates bad values (inverce of bOk) in each dask block.
    Note: can leave NaNs if no good data in block
    :param y:
    :param bOk:
    :param x:
    :param overlap_depth:
    :return: dask array of np.float64 values

    g = da.overlap.overlap(x, depth={0: 2, 1: 2}, boundary={0: 'periodic', 1: 'periodic'})
    todo:
    g2 = g.map_blocks(myfunc)
    result = da.overlap.trim_internal(g2, {0: 2, 1: 2})
    """
    if x is None:  # dask requires "All variadic arguments must be arrays"
        return da.map_overlap(rep2mean, y, bOk, depth=overlap_depth, dtype=np.float64, meta=np.float64([]))
    else:
        return da.map_overlap(rep2mean, y, bOk, x, depth=overlap_depth, dtype=np.float64, meta=np.float64([]))
    #y.map_blocks(rep2mean, bOk, x, dtype=np.float64, meta=np.float64([]))


def rep2mean_da2np(y: da.Array, bOk=None, x=None) -> np.ndarray:
    """
    same as rep2mean_da but also replaces bad block values with constant (rep2mean_da can not replace bad if no good)
    :param y:
    :param bOk:
    :param x: None, other not implemented
    :return: numpy array of np.float64 values

    """

    y_last = None
    y_out_list = []
    for y_bl, b_ok in zip(y.blocks, bOk.blocks):
        y_bl, ok_bl = da.compute(y_bl, b_ok)
        y_bl = rep2mean(y_bl, ok_bl)
        ok_in_replaced = np.isfinite(y_bl[~ok_bl])
        if not ok_in_replaced.all():            # have bad
            assert not ok_in_replaced.any()     # all is bad
            if y_last:                          # This only useful to keep infinite/nan values considered ok:
                y_bl[~ok_bl] = y_last          # keeps y[b_ok] unchanged - but no really good b_ok if we here
            print('continue search good data...')
        else:
            y_last = y_bl[-1]                  # no more y_last can be None
        y_out_list.append(y_bl)
    n_bad_from_start = y.numblocks[0] - len(y_out_list)  # should be 0 if any good in first block
    for k in range(n_bad_from_start):
        y_out_list[k][:] = y_out_list[n_bad_from_start+1][0]
    return np.hstack(y_out_list)

    # # This little simpler version not keeps infinite values considered ok:
    # y_rep = rep2mean_da(y, bOk, x)
    # y_last = None
    # y_out_list = []
    # for y_bl in y_rep.blocks:
    #     y_bl = y_bl.compute()
    #     ok_bl = np.isfinite(y_bl)
    #     if not ok_bl.all():
    #         if y_last:
    #             y_bl[~ok_bl] = y_last
    #     else:
    #         y_last = y_bl[-1]                  # no more y_last can be None
    #     y_out_list.append(y_bl)
    # n_bad_from_start = len(y.blocks) - len(y_out_list)
    # for k in range(n_bad_from_start):
    #     y_out_list[k][:] = y_out_list[n_bad_from_start+1][0]
    # return np.hstack(y_out_list)

# Next function calculates this columns:
incl_calc_velocity_cols = ('Vabs', 'Vdir', 'v', 'u', 'inclination')

def incl_calc_velocity(
    a: dd.DataFrame,
    filt_max: Optional[Mapping[str, float]] = None,
    cfg_proc: Optional[Mapping[str, Any]] = None,
    Ag: Optional[np.ndarray] = None, Cg: Optional[np.ndarray] = None,
    Ah: Optional[np.ndarray] = None, Ch: Optional[np.ndarray] = None,
    kVabs: Optional[np.ndarray] = None, azimuth_shift_deg: Optional[float] = 0,
    calc_version='trigonometric(incl)',
    max_incl_of_fit_deg=None,
    cols_prepend: Optional[Sequence[str]] = incl_calc_velocity_cols,
    **kwargs
    ) -> dd.DataFrame:
    """
    Calculates velocity from raw accelerometer/magnetometer data. Replaces raw columns with velocity vector
    parameters.
    :param a: dask dataframe with columns (at least):
    - 'Ax','Ay','Az': accelerometer channels
    - 'Mx','My','Mz': magnetometer channels
    Coefficients:
    :param Ag: coef
    :param Cg:
    :param Ah:
    :param Ch:
    :param kVabs: if None then will not try to calc velocity
    :param azimuth_shift_deg:
    :param calc_version: see `ConfigIn_InclProc` config,
    :param max_incl_of_fit_deg: see `ConfigIn_InclProc` config,
    :param filt_max: dict with fields:
    - g_minus_1: mark bad points where |Gxyz| is greater, if any then its number will be logged,
    - h_minus_1: to set Vdir=0 and...
    :param cols_prepend: parameters that will be calculated hare and prepended to the columns of `a`. If None
        then ('Vabs', 'Vdir', 'v', 'u', 'inclination'): the only columns that will be keeped if we have only
        raw accelerometer and magnetometer data fields because source fields are discarded. Use any from these
        elements in needed order.
    :param kwargs: not affects calculation
    :return: dataframe with prepended columns ``cols_prepend`` and removed accelerometer and magnetometer raw
    data
    """

    # old coefs need transposing: da.dot(Ag.T, (Axyz - Cg[0, :]).T)
    # fG = lambda Axyz, Ag, Cg: da.dot(Ag, (Axyz - Cg))  # gets Memory error
    # def fInclination_da(Gxyz):  # gets Memory error
    #     return da.arctan2(da.linalg.norm(Gxyz[:-1, :], axis=0), Gxyz[2, :])

    # f_pitch = lambda Gxyz: -np.arctan2(Gxyz[0, :], np.sqrt(np.sum(np.square(Gxyz[1:, :]), 0)))
    # f_roll = lambda Gxyz: np.arctan2(Gxyz[1, :], Gxyz[2, :])
    # fHeading = lambda H, p, r: np.arctan2(H[2, :] * np.sin(r) - H[1, :] * np.cos(r), H[0, :] * np.cos(p) + (
    #        H[1, :] * np.sin(r) + H[2, :] * np.cos(r)) * np.sin(p))

    # Gxyz = a.loc[:,('Ax','Ay','Az')].map_partitions(lambda x,A,C: fG(x.values,A,C).T, Ag, Cg, meta=('Gxyz', float64))
    # Gxyz = da.from_array(fG(a.loc[:,('Ax','Ay','Az')].values, Ag, Cg), chunks = (3, 50000), name='Gxyz')
    # Hxyz = da.from_array(fG(a.loc[:,('Mx','My','Mz')].values, Ah, Ch), chunks = (3, 50000), name='Hxyz')
    lengths = tuple(a.map_partitions(len).compute())  # or True to autocalculate it
    length = sum(lengths)
    if hasattr(a, 'length'):
        assert length == a.length
    else:
        a.length = length

    if kVabs is not None and 'Ax' in a.columns:
        lf.debug('calculating V')
        try:    # lengths=True gets MemoryError   #.to_dask_array()?, dd.from_pandas?
            Hxyz = a.loc[:, ('Mx', 'My', 'Mz')].to_dask_array(lengths=lengths).T
            for i in range(3):
                _ = da.map_overlap(b1spike, Hxyz[i], max_spike=100, depth=1)
                Hxyz[i] = rep2mean_da(Hxyz[i], ~_)
            Hxyz, need_recover_mask = recover_magnetometer_x(Hxyz, Ah, Ch, filt_max['h_minus_1'], a.length)

            Gxyz = a.loc[:, ('Ax', 'Ay', 'Az')].to_dask_array(lengths=lengths).T
            for i in range(3):
                _ = da.map_overlap(b1spike, Gxyz[i], max_spike=1500, depth=1)
                Gxyz[i] = rep2mean_da(Gxyz[i], ~_)

            Gxyz = norm_field(Gxyz, Ag, Cg, Hxyz)
            # Gxyz = fG(a.loc[:, ('Ax', 'Ay', 'Az')].to_dask_array(lengths=lengths).T, Ag, Cg)

            # .rechunk((1800, 3))
            # filter
            GsumMinus1 = da.linalg.norm(Gxyz, axis=0) - 1  # should be close to zero
            incl_rad = fInclination(Gxyz)  # .compute()

            if 'g_minus_1' in filt_max:
                bad = np.fabs(GsumMinus1) > filt_max['g_minus_1']  # .compute()
                bad_g_sum = bad.sum(axis=0).compute()
                if bad_g_sum:
                    # 1. recover where we have over scaled 1 channel a time
                    b_one_ch_is_over_scaled = (
                            (a.loc[:, ('Ax', 'Ay', 'Az')] == -32624).sum(axis=1) == 1
                    ).to_dask_array(lengths=lengths)
                    if b_one_ch_is_over_scaled.any().compute():
                        n_one_ch_is_over_scaled = (a.loc[:, ('Ax', 'Ay', 'Az')] == -32624).sum(axis=0).compute()
                        n_one_ch_is_over_scaled = n_one_ch_is_over_scaled[n_one_ch_is_over_scaled > 0]
                        lf.warning(
                            'Raw acceleration over scaled (= -32624 counts) {}% where 2 channels '
                            'are good! Recovering assuming ideal calibration (|Gxyz_i| = 1)...',
                            (100 * n_one_ch_is_over_scaled / a.length).to_dict()
                        )
                        g_max_of_two_ch = []
                        for ch in n_one_ch_is_over_scaled.keys():
                            cols = ['Ax', 'Ay', 'Az']
                            i = cols.index(ch); cols.pop(i)  # keep other cols only
                            i_other = [0, 1, 2]; i_other.remove(i)
                            b_i_ch_is_over_scaled = b_one_ch_is_over_scaled & (
                                a[ch] == -32624).to_dask_array(lengths=lengths)
                            g_other_sqear = Gxyz[i_other, b_i_ch_is_over_scaled]**2
                            g_other_sqear.compute_chunk_sizes()
                            g_other_sqear = g_other_sqear.sum(axis=0)
                            n_g_bigger_one = 100 * (g_other_sqear > 1).sum().compute() / a.length
                            if n_g_bigger_one > 0.1:
                                # take too_big_g_max as norm of 3 channels if it is not a spike else we choose mean
                                too_big_g_mean, too_big_g_max = da.compute(
                                    g_other_sqear[g_other_sqear > 1].mean(), g_other_sqear.max()
                                )
                                if too_big_g_max == too_big_g_mean:  # big values are same => strange => ignore them
                                    max_g = 1
                                    g_other_sqear[g_other_sqear > 1] = max_g  # for sqrt below without warnings
                                elif (too_big_g_max - too_big_g_mean) < (too_big_g_max - 1) / 2:
                                    max_g = too_big_g_mean
                                    g_other_sqear[g_other_sqear > 1] = max_g  # for sqrt below without warnings
                                else:
                                    max_g = too_big_g_max
                                lf.warning(
                                    'But two other channels already bad scaled too in {} points (where |{}| > 1: mean: {}, max: {})! So removing from {}', n_g_bigger_one, cols, too_big_g_mean, too_big_g_max, max_g
                                )
                            else:
                                max_g = 1
                                if n_g_bigger_one:
                                    g_other_sqear[g_other_sqear > 1] = max_g  # for sqrt below without warnings

                            _ = da.sqrt(max_g - g_other_sqear)
                            _.compute_chunk_sizes()  # not need .rechunk(1, b_one_ch_is_over_scaled.chunks)?
                            Gxyz[i, b_i_ch_is_over_scaled] = _
                        bad_g_sum -= b_one_ch_is_over_scaled.sum()
                        if bad_g_sum > 0:
                            bad[b_one_ch_is_over_scaled] = False
                        else:
                            bad_g_sum = False
                if bad_g_sum:
                    # 2. delete bad
                    bad_g_sum = 100 * bad_g_sum / a.length  # to %
                    if bad_g_sum > 1:
                        lf.warning(
                            'Acceleration is too high (>{}) in {:g}% points! => data nulled',
                            filt_max['g_minus_1'], bad_g_sum
                        )
                    incl_rad[bad] = np.nan
            # else:
            #     bad = da.zeros_like(GsumMinus1, np.bool_)

            # lf.debug('{:.1g}Mb of data accumulated in memory '.format(dfs_all.memory_usage().sum() / (1024 * 1024)))

            # sPitch = f_pitch(Gxyz)
            # sRoll = f_roll(Gxyz)
            # Vdir = np.degrees(np.arctan2(np.tan(sRoll), np.tan(sPitch)) + fHeading(Hxyz, sPitch, sRoll))

            # Velocity absolute value

            #Vabs = da.map_blocks(incl_rad, kVabs, )  # , chunks=GsumMinus1.chunks
            Vabs = incl_rad.map_blocks(
                v_abs_from_incl,
                coefs=kVabs,
                calc_version=calc_version,
                max_incl_of_fit_deg=max_incl_of_fit_deg,
                dtype=np.float64, meta=np.float64([])
            )
            # Vabs = np.polyval(kVabs, np.where(bad, np.nan, Gxyz))
            # v = Vabs * np.cos(np.radians(Vdir))
            # u = Vabs * np.sin(np.radians(Vdir))


            if need_recover_mask is not None:
                HsumMinus1 = da.linalg.norm(Hxyz, axis=0) - 1  # should be close to zero
                Vdir = 0  # default value
                # bad = ~da.any(da.isnan(Mcnts), axis=0)
                Vdir = da.where(da.logical_or(need_recover_mask, HsumMinus1 < filt_max['h_minus_1']),
                                azimuth_shift_deg - da.degrees(da.arctan2(
                                    (Gxyz[0, :] * Hxyz[1, :] - Gxyz[1, :] * Hxyz[0, :]) * (GsumMinus1 + 1),
                                    Hxyz[2, :] * da.square(Gxyz[:-1, :]).sum(axis=0) - Gxyz[2, :] * (
                                            Gxyz[:-1, :] * Hxyz[:-1, :]).sum(axis=0)
                                    )),
                                Vdir  # default value
                                )
            else:  # Set magnetometer data as a function of accelerometer data - worst case: relative direction recovery
                lf.warning(
                    'Bad magnetometer data => Assign direction inversely proportional to toolface angle (~ relative angle if no rotations around device axis)')
                Vdir = azimuth_shift_deg - da.degrees(da.arctan2(Gxyz[0, :], Gxyz[1, :]))
            Vdir = Vdir.flatten()

            # Combine calculated data with existed in ``a`` except raw accelerometer and magnetometer columns.
            a = a.drop(['Ax', 'Ay', 'Az', 'Mx', 'My', 'Mz'], axis='columns')
            cols_remains = a.columns.to_list()
            # Placing columns at first, defined in ``cols_prepend`` in their order, and keep remained in
            # previous order
            _ = zip(incl_calc_velocity_cols, [Vabs, Vdir] + polar2dekart(Vabs, Vdir) + [da.degrees(incl_rad)])
            a = a.assign(
                **{c: (
                    ar if isinstance(ar, da.Array) else
                    da.from_array(ar, chunks=GsumMinus1.chunks)
                    ).to_dask_dataframe(index=a.index) for c, ar in _ if c in cols_prepend
                }
                )[list(cols_prepend) + cols_remains]  # reindex(, axis='columns')  # a[c] = ar
        except Exception as e:
            lf.exception('Error in incl_calc_velocity():')
            raise
    return a



def calc_pressure(a: dd.DataFrame,
        P_t=None,
        **kwargs
) -> dd.DataFrame:
    """
    replaces P column by Pressure applying polyval() and temperature and battery corrections if such coefs exists to P.
    :param a:
    :param P_t: conversion of raw Pressure to physical units polynom coefficients with temperature compensation
    :param kwargs:
    :return:
    """
    if P_t is None:
        return a
    for col_p in ['P', 'P_counts']:
        if col_p in a.columns:
            break
    else:
        return a
    lf.debug('calculating P')
    meta = ('Pressure', 'f8')
    lengths = tuple(a.map_partitions(len, enforce_metadata=False).compute())
    len_data = sum(lengths)
    a = a.rename(columns={col_p: 'Pressure'})
    a['Pressure'] = a['Pressure'].astype(float)  # pressure is in integer counts and we will add float correction

    #
    # Calculate pressure using P polynom with compensation for temperature
    arr = da.map_blocks(
        polyval2d,
        a.Pressure.to_dask_array(lengths=lengths),
        a.Temp.to_dask_array(lengths=lengths),
        P_t
    )
    a.Pressure = arr.to_dask_dataframe(index=a.index)
    return a


def calc_pressure_old(
        a: dd.DataFrame,
        bad_p_at_bursts_starts_period: Optional[str] = None,
        P=None,
        PTemp=None,
        PBattery=None,
        PBattery_min=None,
        Battery_ok_min=None,
        **kwargs
    ) -> dd.DataFrame:
    """
    replaces P column by Pressure applying polyval() and temperature and battery corrections if such coefs exists to P.
    :param a:
    :param bad_p_at_bursts_starts_period:
    :param P: conversion of raw Pressure to physical units polynom coefficents
    :param PTemp: polynom coefficients for temperature compensation that will be added to raw Pressure
    :param PBattery: raw P battery compensation coefficients
    :param PBattery_min: if is not None then:
      - where Battery > PBattery_min only add constant polyval(PBattery, PBattery_min) to Pressure
      - where Battery < PBattery_min add polyval(PBattery, Battery)
      else adds polyval(PBattery, Battery) everywhere
    :param Battery_ok_min:
    :param kwargs:
    :return:
    """
    if P is None:
        return a
    for col_p in ['P', 'P_counts']:
        if col_p in a.columns:
            break
    else:
        return a

    meta = ('Pressure', 'f8')
    lengths = tuple(a.map_partitions(len, enforce_metadata=False).compute())
    len_data = sum(lengths)
    a = a.rename(columns={col_p: 'Pressure'})
    a['Pressure'] = a['Pressure'].astype(float)  # pressure is in integer counts and we will add float correction

    # Compensate for Temperature
    if PTemp is not None:
        a, lengths = cull_empty_partitions(a, lengths)  # removing empty partitions need for no empty chunks for rep2mean_da

        arr = a.Temp.to_dask_array(lengths=lengths)
        # Interpolate Temp jaggies

        # where arr changes:
        bc = (da.ediff1d(arr, to_begin=1000) != 0).rechunk(chunks=arr.chunks)  # diff get many 1-sized chunks

        def f_st_en(x):
            b = np.flatnonzero(x)
            if b.size:
                return b[[0, -1]]
            else:
                return np.int64([-1, -1])

        st_en_use = bc.map_blocks(f_st_en, dtype=np.int64).compute()
        i_ok = np.append(
            (np.append(0, np.cumsum(bc.chunks[0][:-1])).repeat(2) + st_en_use)[st_en_use >= 0],
            len_data
            )
        i_ok = i_ok[np.ediff1d(i_ok, to_begin=i_ok[0]) > 100]
        d_ok = np.ediff1d(i_ok, to_begin=i_ok[0])
        assert d_ok.sum() == len_data
        d_ok = (tuple(d_ok),)
        # interpolate between change points:
        arr_smooth = rep2mean_da(arr.rechunk(chunks=d_ok), bOk=bc.rechunk(chunks=d_ok), overlap_depth=1)

        a_add = arr_smooth.rechunk(chunks=arr.chunks).map_blocks(
            lambda x: np.polyval(PTemp, x), dtype=np.float64, meta=np.float64([])
            ).to_dask_dataframe(index=a.index)
        a.Pressure += a_add  #

    # Compensate for Battery
    if PBattery is not None:
        arr = a.Battery.to_dask_array(lengths=lengths)

        # Interpolate Battery bad region (near the end where Battery is small and not changes)
        if Battery_ok_min is not None:
            i_0, i_1, i_st_interp = da.searchsorted(  # 2 points before bad region start and itself
                -arr, -da.from_array(Battery_ok_min + [0.08, 0.02, 0.001]), side='right'
                ).compute()
            # if have bad region => the number of source and target points is sufficient:
            if i_0 != i_1 and i_st_interp < len_data:
                arr[i_st_interp:] = da.arange(len_data - i_st_interp) * \
                             ((arr[i_1] - arr[i_0]) / (i_1 - i_0)) + arr[i_st_interp]

        # Compensation on Battery after Battery < PBattery_min, before add constant polyval(PBattery, PBattery_min)
        if PBattery_min is not None:
            i_st_compensate = da.searchsorted(-arr, da.from_array(-PBattery_min)).compute().item()
            arr[:i_st_compensate] = np.polyval(PBattery, PBattery_min)
            arr[i_st_compensate:] = arr[i_st_compensate:].map_blocks(
                lambda x: np.polyval(PBattery, x), dtype=np.float64, meta=np.float64([]))
        else:
            arr = arr.map_blocks(lambda x: np.polyval(PBattery, x), dtype=np.float64, meta=np.float64([]))

        a.Pressure += arr.to_dask_dataframe(index=a.index)

    # Calculate pressure using P polynom
    if bad_p_at_bursts_starts_period:   # '1h'
        # with marking bad P data in first samples of bursts (works right only if bursts is at hours starts!)
        p_bursts = a.Pressure.repartition(freq=bad_p_at_bursts_starts_period)

        def calc_and_rem2first(p: pd.Series) -> pd.Series:
            """ mark bad data in first samples of burst"""
            # df.iloc[0:1, df.columns.get_loc('P')]=0  # not works!
            pressure = np.polyval(P, p.values)
            pressure[:2] = np.nan
            p[:] = pressure
            return p

        a.Pressure = p_bursts.map_partitions(calc_and_rem2first, meta=meta)
    else:
        a.Pressure = a.Pressure.map_partitions(lambda x: np.polyval(P, x), meta=meta)

    return a


def coef_zeroing_rotation(g_xyz_0, Ag_old, Cg) -> Union[np.ndarray, int]:
    """
    Get zeroing rotation matrix `Rz` that can be used to correct old rotation matrices
    `coef_rotate(Ag_old, Ah_old, Rz)` based on accelerometer values when inclination is zero.
    :param g_xyz_0: 3x1 values of columns 'Ax','Ay','Az' (it is practical to provide mean values of some time
    range)
    :param Ag_old, Cg: numpy.arrays, rotation matrix and shift for accelerometer
    :return Rz: 3x3 numpy.array , corrected rotation matrices
    Todo: use new rotate algorithm

    Method of calculation of ``g_xyz_0`` from dask dataframe data:
    g_xyz_0 = da.atleast_2d(da.from_delayed(
        a_zeroing.loc[:, ('Ax', 'Ay', 'Az')].mean(
            ).values.to_delayed()[0], shape=(3,), dtype=np.float64, name='mean_G0'))
    """

    if not len(g_xyz_0):
        print(f'zeroing(): no data {g_xyz_0}, no rotation is done')
        return 1
    if g_xyz_0.shape[0] != 3:
        raise ValueError('Bad g_xyz_0 shape')

    old_g_xyz_0 = fG(g_xyz_0, Ag_old, Cg)
    old_pitch = f_pitch(old_g_xyz_0)[0]
    old_roll = f_roll(old_g_xyz_0)[0]
    lf.info('Zeroing pitch = {}°, roll = {}°', *[
        np.format_float_positional(x, precision=5) for x in np.rad2deg([old_pitch, old_roll])
    ])
    return rotate_y(rot_matrix_x(np.cos(old_roll), np.sin(old_roll)), angle_rad=old_pitch)


def coef_rotate(*A, Z):
    return [Z @ a for a in A]


def coef_zeroing_rotation_from_data(
        df: dd.DataFrame | pd.DataFrame,
        Ag, Cg,
        time_ranges=None,
        **kwargs) -> np.ndarray:
    """
    Zeroing rotation matrix Rz that can be used to correct old rotation matrices
    based on accelerometer raw data from time range
    :param df: pandas or dask DataFrame with columns 'Ax', 'Ay', 'Az' - raw accelerometer data
    :param time_ranges: optional data index time ranges to select data splitted here to pairs of start and end
    Old coefficients:
    :param Ag:
    :param Cg:
    :param kwargs: not used
    :return: Ag, Ah - updated coefficients
    """

    if time_ranges:
        time_ranges = pd.to_datetime(time_ranges, utc=True)
        if len(time_ranges) <= 2:
            idx = slice(*time_ranges)
        else:
            # Create an iterator and use zip to pair up start and end times
            time_pairs = (lambda x=iter(time_ranges): list(zip(x, x)))()
            # Create a boolean mask for rows where the index is within any of the intervals
            idx = pd.Series(False, index=df.index)
            for start, end in time_pairs:
                idx |= (df.index >= start) & (df.index <= end)  # or df.index.to_series().between(start, end)
    else:
        idx = None
    df = df.loc[idx, ("Ax", "Ay", "Az")]
    if df.empty():
        lf.info('Zeroing data: average {:d} points in interval {:s} - {:s}', len(df),
                *getattr(df, 'divisions' if (b_dd := isinstance(df, dd.DataFrame)) else 'index')[[0, -1]])
        g_xyz_0 = np.atleast_2d(df.mean().values.compute() if b_dd else df.mean().values).T
        return coef_zeroing_rotation(g_xyz_0, Ag, Cg)
    else:
        return None


def coef_zeroing(g_xyz_0, Ag_old, Cg, Ah_old) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zeroing: correct Ag_old, Ah_old
    :param g_xyz_0: 1x3 values of columns 'Ax','Ay','Az' (it is practical to provide mean values of some time range)
    :param Ag_old, Cg: numpy.arrays, rotation matrix and shift for accelerometer
    :param Ah_old: numpy.array 3x3, rotation matrix for magnetometer
    return (Ag, Ah): numpy.arrays (3x3, 3x3), corrected rotation matrices

    Calculation method of ``g_xyz_0`` from dask dataframe data can be:
    g_xyz_0 = da.atleast_2d(da.from_delayed(
        a_zeroing.loc[:, ('Ax', 'Ay', 'Az')].mean(
    ).values.to_delayed()[0], shape=(3,), dtype=np.float64, name='mean_G0'))
    """
    lf.warning('coef_zeroing() depreciated. Use coef_zeroing_rotation() instead and save rotation matrix for future use')
    Z = coef_zeroing_rotation(g_xyz_0, Ag_old, Cg, Ah_old)
    Ag = Z @ Ag_old
    Ah = Z @ Ah_old
    lf.debug('calibrated Ag = {:s},\n Ah = {:s}', Ag, Ah)
    # # test: should be close to zero:
    # Gxyz0 = fG(g_xyz_0, Ag, Cg)
    # #? Gxyz0mean = np.transpose([np.nanmean(Gxyz0, 1)])

    return Ag, Ah

    # filter temperature
    # if 'Temp' in a.columns:
    # x = a['Temp'].map_partitions(np.asarray)
    # blocks = np.diff(np.append(i_starts, len(x)))
    # chunks = (tuple(blocks.tolist()),)
    # y = da.from_array(x, chunks=chunks, name='tfilt')
    #
    # def interp_after_median3(x, b):
    #     return np.interp(
    #         da.arange(len(b_ok), chunks=cfg_out['chunksize']),
    #         da.flatnonzero(b_ok), median3(x[b]), da.nan, da.nan)
    #
    # b = da.from_array(b_ok, chunks=chunks, meta=('Tfilt', 'f8'))
    # with ProgressBar():
    #     Tfilt = da.map_blocks(interp_after_median3(x, b), y, b).compute()

    # hangs:
    # Tfilt = dd.map_partitions(interp_after_median3, a['Temp'], da.from_array(b_ok, chunks=cfg_out['chunksize']), meta=('Tfilt', 'f8')).compute()

    # Tfilt = np.interp(da.arange(len(b_ok)), da.flatnonzero(b_ok), median3(a['Temp'][b_ok]), da.nan,da.nan)
    # @+node:korzh.20180524213634.8: *3* main
    # @+others
    # @-others


def format_timedelta(dt: timedelta) -> str:
    days = dt.days
    str_parts = [f"{days} days"] if days else []

    seconds_remainder = dt.seconds
    if seconds_remainder:
        hours, seconds_remainder = divmod(seconds_remainder, 3600)
        minutes, seconds = divmod(seconds_remainder, 60)
        str_parts += [f"{hours:02}:{minutes:02}:{seconds:02}"]

    return ' '.join(str_parts) if str_parts else 'no'


def year_fraction(date: datetime) -> float:
    """
    datetime dates to decimal years
    https://stackoverflow.com/a/36949905/2028147
    :param date:
    :return:
    Note: calculates the fraction based on the start of the day, so December 31 will be 0.997, not 1.0.

    >>> print year_fraction(datetime.today())
    2016.32513661
    """
    start = datetime_date(date.year, 1, 1).toordinal()
    year_length = datetime_date(date.year + 1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length


def mag_dec(lat, lon, time: datetime, depth: float = 0):
    """
        Returns magnetic declination using wmm2020 library

    :param lat, lon: coordinates in degrees WGS84
    :param time: # ='2020,09,20'
    :param depth: in meters (negative below sea surface)
    """
    import wmm2020 as wmm
    yeardec = year_fraction(time)
    mag = wmm.wmm(lat, lon, depth / 1000, yeardec)
    # .item()
    return mag.decl if (isinstance(lat, Iterable) or isinstance(lon, Iterable)) else mag.decl.item(0)


def filt_data_dd(
    a, dt_between_bursts=None, dt_hole_warning: Optional[np.timedelta64] = None, cfg_filter=None
) -> Tuple[dd.DataFrame, np.array]:
    """
    Filter and get burst starts (i.e. finds gaps in data)
    :param a:
    :param dt_between_bursts: minimum time interval between blocks to detect them and get its starts (i_burst),
    also repartition on found blocks if can i.e. if known_divisions. If None then set as defined in i_bursts_starts():
    greater than min of two first intervals + 1s
    :param dt_hole_warning: numpy.timedelta64
    :param cfg_filter: if set then filter by removing rows
    :return: (a, i_burst) where:
     - a: filtered,
     - i_burst: array with 1st elem 0 and other - starts of data after big time holes

    """
    if True:  # try:
        # filter

        # this is will be done by filter_global_minmax() below
        # if 'P' in a.columns and min_p:
        #     print('restricting time range by good Pressure')
        #     # interp(NaNs) - removes warning 'invalid value encountered in less':
        #     a['P'] = filt_blocks_da(a['P'].values, i_burst, i_end=len(a)).to_dask_dataframe(['P'], index=tim)
        #     # todo: make filt_blocks_dd and replace filt_blocks_da: use a['P'] = a['P'].repartition(chunks=(tuple(np.diff(i_starts).tolist()),))...?

        # decrease interval based on ini date settings and filtering and recalculate bursts
        a = filter_global_minmax(a, cfg_filter=cfg_filter)
        tim = a.index.compute()  # History: MemoryError((6, 10868966), dtype('float64'))
        i_burst, mean_burst_size, max_hole = i_bursts_starts(tim, dt_between_blocks=dt_between_bursts)
        # or use this and check dup? shift?:
        # i_good = np.search_sorted(tim, a.index.compute())
        # i_burst = np.search_sorted(i_good, i_burst)

        if not a.known_divisions:  # this is usually required for next op
            divisions = tuple(tim[np.append(i_burst, len(tim) - 1)])
            a.set_index(a.index, sorted=True).repartition(divisions=divisions)
            # a = a.set_index(a.index, divisions=divisions, sorted=True)  # repartition? (or reset_index)

        if max_hole and dt_hole_warning and max_hole > dt_hole_warning:
            lf.warning(f'max time hole: {max_hole.astype(datetime)*1e-9}s')
        return a, i_burst


def stem_proc_db(proc_dir: Path) -> Path:
    proc_db_stem = (
        proc_dir.name if proc_dir.name[0].isdigit() else
        proc_dir.parent.name if proc_dir.parent.name[0].isdigit() or
        proc_dir.name.startswith('inclinometer') else
        proc_dir.name
    ).replace('@', '_', 1).split('_')[0]
    return proc_db_stem


def get_full_raw_db_path(raw_db_path, path_in) -> Tuple[Path, Path]:
    """
    Determines the full path for raw database files (`raw_db_path`) and the directory for processed data
    (`proc_dir`) based on the given configuration settings.
    :param raw_db_path: may be
    - absolute path or falsy: this value will be used for returned `raw_db_path` as is
    - "auto": construct output `raw_db_path` from a directory named "_raw" (which is looked under `path_in` if
    it is a name of processed data database or else above `path_in`) and a name of first parent dir that
    starts with a digit, removing all symbols beginning from 1st found "_" or "@" in it.
    :param path_in: The configuration setting for the input path, which can be
    - a path to raw text or
    - HDF5 data (usually under "_raw" directory) or
    - processed HDF5 data (usually just above the "_raw" directory).
    :return: (raw_db_path, proc_dir):
    - raw_db_path: The absolute path to the raw database.
    - proc_dir: The absolute path to the directory for processed data, which is either one level higher than the raw DB or the same directory if loading from files like *.proc.h5:
      - If `path_in` includes "_raw", `proc_dir` is set to one level higher than the "_raw" directory.
      - If the input path includes file suffixes like '.proc' or '.proc_noAvg', `proc_dir` is set to the same
      directory as the input path.
      - Otherwise, it defaults to navigating up the directory tree to find a
      suitable directory, considering directories named "inclinometer".
    """
    parents = path_in.parents
    idx_raw_dir = None
    try:
        idx_proc_dir = -path_in.parts.index('_raw')
        proc_dir = parents[idx_proc_dir]
        idx_raw_dir = idx_proc_dir - 1
    except ValueError:  # no "_raw" dirs
        if '.proc' in path_in.suffixes or '.proc_noAvg' in path_in.suffixes:
            proc_dir = path_in.parent
            p_parent = proc_dir / '_raw'
        else:
            # If in.paths under archive we need go up several levels
            for p_parent, proc_dir in zip(*(lambda p: (p, p[1:]))(list(parents))):
                if p_parent.is_dir():
                    if p_parent.startswith('inclinometer'):
                        proc_dir, p_parent = p_parent, proc_dir / '_raw'
                        break
                    break

    if (not raw_db_path) or raw_db_path.is_absolute():
        return raw_db_path, proc_dir

    raw_db_name = raw_db_path.name
    b_auto = raw_db_name == 'auto'
    if b_auto:
        if idx_raw_dir:
            try:
                raw_subdir = parents[idx_raw_dir - 1]
            except IndexError:
                raw_db_name = None
            else:
                _ = raw_subdir.stem
                raw_db_name = (
                    _.replace('@', '_', 1).split('_')[0] if raw_subdir.is_dir() and _[0].isdigit() else None
                    )
        else:
            raw_db_name = None
        # If no "_raw" dir was found then try get DB name from cruise dir: parent dir that starts from digits
        if not raw_db_name:
            raw_db_name = stem_proc_db(proc_dir)

    raw_db_path = ((parents[idx_raw_dir] if idx_raw_dir else p_parent) / raw_db_name).with_suffix('.raw.h5')
    return raw_db_path, proc_dir


def set_full_paths_and_h5_out_fields(cfg_out: MutableMapping, path, **cfg_in) -> List[Path]:
    """
    Sets ouput database parameters in cfg_out (wraps `h5.out_init()`).
    :param path: input path. It can be relative to `proc_dir` if one of cfg_out fields: "db_path",
    not_joined_db_path", "text_path" are full path (`proc_dir` is a common parent of them)
    :param cfg_in: other input configuraton parameters to call `h5.out_init()`
    :param cfg_out: should contain
    - raw_db_path
    :updates `cfg_out` fields:
    - raw_db_path
    - not_joined_db_path
    - db_path: switch to averaging DB (*proc.h5) if loading from processed noAvg DB
    - text_path
    - other fields, modified by `h5.out_init()`
    return path: absoulute input path
    """

    proc_dir = None
    if not path.is_absolute():
        for p in [cfg_out["db_path"], cfg_out["not_joined_db_path"], cfg_out["text_path"]]:
            if p.is_absolute():
                proc_dir = p.parent
                break
        path = proc_dir / str(path)

    # raw DB full path
    cfg_out["raw_db_path"], proc_dir = get_full_raw_db_path(
        cfg_out["raw_db_path"], path_in=path if proc_dir is None else path
        )

    # .proc DB full path
    if not cfg_out["db_path"].is_absolute():
        out_db_path_stem = cfg_out["db_path"].stem
        if not out_db_path_stem:
            out_db_path_stem = stem_proc_db(proc_dir)
        cfg_out["db_path"] = (proc_dir / out_db_path_stem).with_suffix("").with_suffix(".proc.h5")

    # .proc_noAvg DB full path
    not_joined_db_path = path if path.suffix[0].startswith(".proc_noAvg") else None
    if not_joined_db_path:  # If input path is .proc_noAvg DB then use it
        cfg_out["not_joined_db_path"] = not_joined_db_path
    elif cfg_out["not_joined_db_path"] is True or not cfg_out["not_joined_db_path"]:  # Need to set to default
        cfg_out["not_joined_db_path"] = (
            cfg_out["db_path"].with_suffix("").with_suffix("").with_suffix(".proc_noAvg.h5")
        )  # old: raw_db_path.parent.with_name(raw_db_path.stem).with_suffix(".proc_noAvg.h5")
    elif not cfg_out["not_joined_db_path"].is_absolute():  # Need to set absolute
        cfg_out["not_joined_db_path"] = proc_dir / cfg_out["not_joined_db_path"]

    # Output text data full path
    if cfg_out["text_path"] is not None and not cfg_out["text_path"].is_absolute():
        cfg_out["text_path"] = proc_dir / cfg_out["text_path"]


    # Init hdf5 data loading parameters
    cfg_out_table, tables_log = cfg_out["tables"], cfg_out["tables_log"]  # not need fill
    # we set tables as not want auto set it but it sets "tables_log", [f"{cfg_out['table']}/logFiles"]) if not exist
    call_with_valid_kwargs(h5.out_init, cfg_out, **cfg_in, table='prevent_auto')
    cfg_out["tables"], cfg_out["tables_log"] = cfg_out_table, tables_log

    # # todo: To can use You need provide info about chunks that will be loaded:
    # cfg_out['b_incremental_update'] = False

    return path


def get_input_db(path_in, dt_bins, raw_db_path, not_joined_db_path, dt_min_binning_proc, **kwargs):
    """
    Switches current input source between raw DB and not avg DB:
    This allows to load previously saved data in subsequent runs, but if raw DB not
    exist then raise.
    :param path_in: The configuration setting for the input path, which can be
    - a path to raw text or
    - HDF5 data (usually under "_raw" directory) or
    - processed HDF5 data (usually just above the "_raw" directory).
    :param dt_bins, raw_db_path, not_joined_db_path, dt_min_binning_proc: output config parameters (see
    `ConfigOut_InclProc`)
    :param kwargs: not used
    :return path_in: full input path
    """
    need_load_counts = True  # raw DB by default
    p_sfx = path_in.suffixes[-2:-1]
    if p_sfx and p_sfx[0].startswith(".proc"):
        need_load_counts = any((not bin) or (bin <= dt_min_binning_proc) for bin in dt_bins)
    for path in ([] if need_load_counts else [not_joined_db_path]) + [raw_db_path]:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Not found stored data {not_joined_db_path} or {raw_db_path}")



def load_coefs(store: pd.HDFStore|Path, tbl: str):
    """
    Finds up to 2 levels of coefficients, collects to 1 level, accepts any coef paths
    :param store: hdf5 file Path or opend pd.HDFStore with group "``tbl``/coef"
    :param tbl:
    :return: dict with loaded coefs and 'dates' field having dict of the each parameter 'date' atrribute. None
    if `tbl` node not found
    Example
    for coefs nodes:
    ['coef/G/A', 'coef/G/C', 'coef/H/A', 'coef/H/C', 'coef/H/azimuth_shift_deg', 'coef/Vabs0'])
    naming rule gives coefs this names:
    ['Ag', 'Cg', 'Ah', 'Ch', 'azimuth_shift_deg', 'kVabs'],
    """
    with nullcontext(store) if isinstance(store, pd.HDFStore) else pd.HDFStore(store, mode="r") as store:
        node_coef = store.get_node(f'{tbl}/coef')
        if node_coef is None:
            return
        else:
            coefs_dict = {'dates': {}}
            for node_name in node_coef.__members__:
                node_coef_l2 = node_coef[node_name]
                if getattr(node_coef_l2, '__members__', False):  # node_coef_l2 is group
                    for node_name_l2 in node_coef_l2.__members__:
                        name = f'{node_name_l2}{node_name.lower() if node_name_l2[-1].isupper() else ""}'
                        coefs_dict[name] = node_coef_l2[node_name_l2].read()
                        try:
                            coefs_dict['dates'][name] = node_coef_l2[node_name_l2].attrs['timestamp']
                            pass
                        except KeyError:
                            pass
                else:  # node_coef_l2 is value
                    name = node_name if node_name != 'Vabs0' else 'kVabs'
                    coefs_dict[name] = node_coef_l2.read()
                    try:
                        coefs_dict['dates'][name] = node_coef_l2.attrs['timestamp']
                        pass
                    except KeyError:
                        pass
    try:  # Get last date record from old format of list of dates in microseconds since the Unix epoch
        if isinstance(coefs_dict["date"].item(-1), (bytes, str)):
            coefs_dict["date"] = np.nanmax(np.array(coefs_dict["date"], 'M8[s]'))
        else:
            coefs_dict["date"] = np.datetime64(int(np.nanmax(coefs_dict["date"])), "us")  # need?
    except (KeyError, TypeError):  # No or has been converted already
        pass

    if coefs_dict["dates"]:  # If no date attributes in DB, as for old coefs, then keep 'date' undefined
        coefs_dict["date"] = max(
            [coefs_dict["date"]] + [np.datetime64(d) for d in coefs_dict["dates"].values()]
        )
    return coefs_dict


def get_coefs(
    coefs_paths: Sequence[str | pd.HDFStore], tbl: str, coefs_ovr: Optional[Mapping[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Load coefficients and overlay with optional overrides. Convert lists to arrays. Add 'date' field equal to
    last found date or now

    :param coefs_paths: Paths to input coefficients or opened store. 1st found will be used.
    :param tbl: Table name for coefficients.
    :param coefs_ovr: Overrides for coefficients with optional 'date' field.
    :return: Coefficients dictionary with lists converted to arrays and 'date' field processed: None if no
        `coef_ovr`, if exist but has no 'date', then current date.
    """
    # Fields that by default are not None
    defaults = {
        k: v_def
        for k, v in ConfigInCoefs_InclProc.__dataclass_fields__.items()
        if (v_def := cfg_d.get_field_default(v)) is not None
        and not (isinstance(v_def, (list, dict)) and ((not v_def) or not any(lst != [] for lst in v_def)))
    }
    now = datetime.now()
    coefs_ovr_dates = {}
    if coefs_ovr:  # Always True in this module so detect whether it is not the default
        # Fields that also not redefined by `coefs_ovr`
        not_ovr = [
            k for k, v_def in defaults.items() if (
                (v_ovr := coefs_ovr.get(k)) == v_def or
                (isinstance(v_ovr, list) and ((not v_ovr) or not any(lst != [] for lst in v_ovr)))
            )
        ]

        if len(not_ovr) < len(defaults):    # Some coefs redefined
            if not not_ovr:
                # All coefs redifined by `coefs_ovr`
                coefs_paths = []  # Not need load coefs
                if OmegaConf.is_config(coefs_ovr):
                    coefs_ovr = OmegaConf.to_container(coefs_ovr)

            # If ovr dates is not set, then set it to "date" or now
            coefs_ovr_dates, _ = coefs_ovr.get("dates", coefs_ovr_dates), coefs_ovr.get("date", now)
            coefs_ovr_dates = {
                k: coefs_ovr_dates.get(k) or now for k in coefs_ovr if k in defaults and k not in not_ovr
            }
    else:
        not_ovr = list(defaults)

    # Load coefs from HDF5 and overwrite them with (not default) `coefs_ovr`
    if coefs_paths:
        for coefs_path in coefs_paths:
            coefs_load = load_coefs(coefs_path, tbl)
            if coefs_load is not None:
                break
        if coefs_load is None:
            lf.error(
                'Not found coefficients table "{:s}" in {}, {:s} redefined from current run config!',
                tbl,
                coefs_paths,
                "but all are" if not not_ovr else
                f"and {not_ovr} are not" if len(not_ovr) < len(defaults) else
                "and no one are"
            )
            if len(not_ovr) == len(defaults):
                raise ValueError("No coefficients provided / found")
            coefs_load = {**coefs_ovr, "dates": coefs_ovr_dates}
        else:
            coefs_load_dates = coefs_load.get("dates", coefs_ovr_dates)
            for k, v in coefs_ovr.items():
                if v is not None and k in defaults and k not in not_ovr:
                    coefs_load[k] = v
                    coefs_load_dates[k] = coefs_ovr_dates[k]  # Overwrite loaded dates
            coefs_load["dates"] = coefs_load_dates
    else:
        coefs_load = {**coefs_ovr, "dates": coefs_ovr_dates}

    # Convert lists to arrays
    if coefs_load:
        coefs_load = {
            k: np.float64(v) if (isinstance(v, list) and v is not None and k != "dates") else v
            for k, v in coefs_load.items()
        }

    # Update 'date' to max found date
    out_dates = [
        datetime.fromisoformat(d) if isinstance(d, str) else d
        for d in [coefs_load.get("date")] + list(coefs_load["dates"].values())
        if d
    ]
    if out_dates:
        coefs_load["date"] = max(out_dates)
    return coefs_load


def coefs_format_for_h5(
    coef: Mapping[str, Any], pcid: str = None, date: Optional[str] = None
) -> Mapping[str, Any]:
    """
    Create coefficients dict to save to hdf5:
        - A: for accelerometer:
        - A: A 3x3 scale and rotation matrix
        - C: U_G0 accelerometer 3x1 channels shifts
    - M: for magnetometer:
        - A: M 3x3 scale and rotation matrix
        - C: U_B0 3x1 channels shifts
        - azimuth_shift_deg: Psi0 magnetometer direction shift to the North, radians
    - Vabs0: for calculation of velocity magnitude from inclination - 6 element vector:
    Copy other coefs:
    - Rz: zero calibration for accelerometer and magnetometer rotation matrix
    - Pt: pressure with temperature compensation polynom matrix
    :param coef:
    :param pcid: to exclude writing not needed defaults and to write it as metadata
    :return:
    """
    if coef is None:
        # making default coef matrix
        coef = ConfigInCoefs_InclProc().__dict__
        del coef['g0xyz']
        if not pcid.split("_")[-1].startswith('p'):  # device with pressure sensor
            del coef['P_t']
    elif 'Rz' not in coef:  # added for compability with old coefs (todo: better get by diagonalize G)
        coef['Rz'] = np.eye(3)

    coef_no_rename = ['Rz']
    if 'P_t' in coef:
        coef_no_rename += ['P_t']
    return {
        **{
            f'//coef//{ch_u}//{m}': coef[f'{m}{ch}']
            for ch, ch_u in (('h', 'H'), ('g', 'G')) for m in ('A', 'C')
        },
        '//coef//H//azimuth_shift_deg': coef['azimuth_shift_deg'],
        '//coef//Vabs0': coef['kVabs'],
        **{f'//coef//{p}': coef[p] for p in coef_no_rename},
        '//coef//pid': pcid,
        '//coef//date': date or datetime.now().replace(microsecond=0).isoformat()
    }

def separate_cfg_in_for_probes(cfg_in, in_many={}, probes=None
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    """
    Convert form of config with probes keys in parameters to separate dicts for each key:
    {parameter1: {probe1: value1, probe2: value2, ...}} to form with singular parameter for each probe: {
    probe1: {parameter1: value1, parameter2:value2},
    probe2: {parameter1: value1, parameter2: value2}, ...}
    :param cfg_in: dict from `ConfigIn_InclProc` class, nessesary fields:
    - paths:
    - tables:
    :param in_many: dict with optional fields containing dicts for each probe with values that have same
    meaning as fields of cfg['in'/'filter'] but used to set parameter for each probe. Originated from
    `ConfigInMany_InclProc` class:
    :param probes: output grouped_cfg keys. If None, then auto infered from `tables`, not used in `incl_h5clc`
    module
    :return: (grouped_cfg, cfg_in_common)
    - cfg_in_common dict: inmput parameters with removed plural fields
    - grouped_cfg: dict with dicts for each probe, i.e.:
      - keys: groups: probe identificators or '*' if no found,
      - values: dicts with:
        - keys: singular named keys obtained from found plural named arguments/in_many keys,
        - values: have value of argument/in_many values for each key
    """
    # # Combine all input parameters different for different probes we need to convert to this dict, renaming
    # # fields to singular form:
    # cfg_many = {}
    # cfg_in_common = in_many.copy()  # temporary
    # for k in [  # single_name_to_many_vals
    #     'min_date',
    #     'max_date',
    #     'time_ranges',
    #     'time_ranges_zeroing',
    #     'date_to_from',
    #     'bad_p_at_bursts_starts_period',
    #     'coordinates'
    # ]:
    #     try:
    #         cfg_many[k] = cfg_in_common.pop(k)
    #     except KeyError:
    #         continue
    # cfg_in_common.update(cfg_in.copy())
    cfg_in_common = cfg_in.copy()
    cfg_many = in_many.copy()

    def delistify(vals):
        # Here only gets table item from 'tables' parameter (that is of type list with len=1)
        return vals[0] if isinstance(vals, Iterable) and len(vals) == 1 else vals

    def group_dict_vals(param_dicts, probes=None) -> Dict[str, Dict[str, str]]:
        """
        Convert form of config with plural named keys ``parameter_pl``
        {parameter_pl: {probe1: value1, probe2: value2, ...}}
        to form with singular parameter for each probe: {
        probe1: {parameter1: value1, parameter2: value2},
        probe2: {parameter1: value1, parameter2: value2}, ...}

        :param param_dicts: dict {parameter: {probe: value}} or {parameter: value} to use this value for all
            probes
        :param probes: probes id list for value in last form of param_dicts. Should have same length as value
            to assign each item to each probe. Not used for config defined in this file (no cfg['in']['probe']
            parameter here)
        :return: dict of dicts with singular named keys
        """

        cfg_probes = {}
        params_common = {}
        if isinstance(probes, list):
            if len(probes) == 1:
                probes_ptn = probes[0]
            else:
                probes_ptn = None
        else:
            probes_ptn = probes
        if probes_ptn and probes_ptn.startswith('*'):
            probes_ptn = f'.{probes_ptn}'  # make valid regex

        for param, probe_vals in param_dicts.items():
            # if probe_vals is None:
            #     continue
            if (not probes_ptn) and isinstance(probe_vals, list) and len(probe_vals) == len(probes):
                probe_vals = dict(zip(probes, probe_vals))
            if isinstance(probe_vals, Mapping):
                for probe, val in probe_vals.items():
                    probe = to_pcid_from_name(probe)
                    # Filter out not needed probes (result keeps only needed to iterate and load)
                    if probe in probes or (probes_ptn and re.match(probes_ptn, probe)):
                        if probe in cfg_probes:
                            cfg_probes[probe][param] = val
                        else:
                            cfg_probes[probe] = {param: val}
            else:
                params_common[param] = delistify(probe_vals)

        if cfg_probes:  # dicts specific to each probe, each includes copy of params_common
            return {probe: {**params_common, **probe_vals} for probe, probe_vals in cfg_probes.items()}
        else:  # only common parameters were specified
            return {'*': {param: delistify(probe_vals) for param, probe_vals in param_dicts.items()}}

    if len(cfg_in['tables']) > 0 and not probes:
        probes = [to_pcid_from_name(tbl) for tbl in cfg_in["tables"]]  # tbl.rsplit('_', 1)[-1]
    return cfg_in_common, group_dict_vals(cfg_many, probes)


def pcid_to_raw_name(pcid: str):
    """
    :param pcid: Probe output Column ID (pcid):
    # {type}_{model}{number} or if no model then {type}{number}:  may be i/w for inclinometer/wave gage.
    # Inclinometers with pressure sensor currently have only model "p"
    :return: table name in **raw** or **not averaged** data DB
    """
    return f"incl{pcid[1:]}" if pcid[0] == "i" else pcid


def to_pcid_from_name(probe_name: str|int, probe_type: str = "i"):
    """
    Get Probe output Column ID (pcid)
    :param probe_name: any of
    - output column name
    - raw table name
    - raw file stem search pattern/name
    - pid str of 3 chars probe id or just number
    :param probe_type: probe type ('i' for inclinometers), used if `probe_name` is digit
    :return: pcid: our standartisied probe name, used in
    - output column name in Avg/noAvg db
    - column suffix in combined table
    - name in specific for this probe hydra conig `ConfigInMany_InclProc`
    """
    if (isinstance(probe_name, str) and probe_name[0].isdigit()) or isinstance(probe_name, int):
        return f"{probe_type}{probe_name:0>2}"

    # pid to pcid
    if isinstance(probe_name, str) and len(probe_name) == 3 and probe_name[1:].isdigit():
        if probe_name[0] == probe_type == "i":
            return probe_name
        elif probe_name[0].isalpha():
            return f"{probe_type}_{probe_name}"

    pattern_name_parts = parse_name(probe_name.replace('.', ''))
    if pattern_name_parts:
        return call_with_valid_kwargs(pcid_from_parts, **pattern_name_parts)
    else:
        return '*'


def cur_cfg(
        pcid: str,
        cfg_in_for_probes: Mapping[str, Any], cfg_in_common: Mapping[str, Any], cfg: Mapping[str, Any]
) -> Tuple[Mapping[str, Any], str, str]:
    """
    Get a copy of all required processing configuration for probe `pcid`
    :param pcid: Probe output Column ID
    :param cfg_in_for_probes: config for current probe
    :param cfg_in_common: input config common to all probes
    :param cfg:
    :return: cfg1: dict, config for 1 probe with
      - copy of `cfg` fields that can be changed (to can change them without modifying `cfg`)
      - coefs
    """
    # Combine fields from cfg, cfg_in_common, cfg_in_cur to processing configuration for current probe
    cfg1 = {
        'in': {**cfg_in_common.copy(), **cfg_in_for_probes.get(pcid, {})},
        'out': cfg["out"].copy(),  # actually may not need copy() here
        'filter': cfg['filter'].copy()
    }
    # Load coefs
    cfg1["in"]["coefs"] = get_coefs(
        ([cfg["in"]["path"]] if '.raw' in cfg["in"]["path"].suffixes[-2:-1] else []
        ) + [cfg["in"]["coefs_path"]],
        pcid_to_raw_name(pcid), coefs_ovr=cfg1["in"]["coefs"]
    )
    return cfg1


def get_coef_azimuth_shift(azimuth_add, coordinates, azimuth_shift_deg, data_date=datetime.now(), **kwargs):
    """
    Shift azimuth if user configured angle or coordinates to find it from inclination there

    """
    # Azimuth correction `azimuth_shift_deg` from `azimuth_add` and `coordinates`
    if azimuth_add or coordinates:
        msgs = ["(coef. {:g})".format(azimuth_shift_deg.item())]
        if azimuth_add:
            # individual or the same correction for each table:
            msgs.append("(azimuth_shift_deg {:g})".format(azimuth_add))
            azimuth_shift_deg += azimuth_add
        if coordinates:  # add magnetic declination at coordinates [Lat, Lon]
            mag_decl = mag_dec(
                *coordinates,
                data_date,
                depth=-1,  # depth has negligible effect if it is less than several km
            )
            msgs.append("(magnetic declination {:g})".format(mag_decl))
            azimuth_shift_deg += mag_decl

        lf.warning("Azimuth correction updated to {:g} = {}°", azimuth_shift_deg.item(), " + ".join(msgs))
    return azimuth_shift_deg


def get_coef_zeroing_matrix(
    Rz=None, g0xyz=None, Ag=None, Cg=None, **kwargs
):
    """
    g0xyz overwrites Rz
    """
    if g0xyz is not None:
        Rz = coef_zeroing_rotation(g0xyz[:, None], np.float64(Ag), Cg)
        msg_rotated = "with new rotation to user defined zero point (g0xyz) "
    elif Rz is not None and (Rz != np.eye(3)).any():  # need?
        msg_rotated = ""
    else:
        Rz, msg_rotated = None, ""
    return Rz, msg_rotated


def coef_prepare(coefs, time_ranges_zeroing, azimuth_add, coordinates, data, data_date):
    """
    Add dates and absent fields to new coef from previous preseriving reference and save to db
    coefs: dict of ndarrays, original loaded coefs
    """

    coefs_new = {
        "azimuth_shift_deg": get_coef_azimuth_shift(
            azimuth_add, coordinates, coefs["azimuth_shift_deg"], data_date
        )
    }
    # Get rotation coefficient Rz (if data time range for zeroing is specified)
    msg_zeroed = ""
    if time_ranges_zeroing:
        _ = coef_zeroing_rotation_from_data(data, **coefs, time_ranges=time_ranges_zeroing)
        if _ is None:
            lf.debug("time_ranges_zeroing not in current data range")
        else:
            coefs_new["Rz"] = _
            msg_zeroed = "with new rotation to direction averaged on configured time_ranges_zeroing interval "

    # Get rotation coefficient from other coefs config and save
    coef_zeroing_matrix, msg_rotated = get_coef_zeroing_matrix(**coefs)

    # Prepare to save coefficient dates (some of them can be updated below)
    try:
        dates = coefs["dates"]
    except (KeyError, TypeError, AttributeError):  # why AttributeError?
        dates = {}
    # Set `dates` items to True for changed and new coef to be replaced with current date in `h5copy_coef()`
    for k, v in coefs_new.items():
        try:
            cur_prev = coefs[k]
            if all(cur_prev == v) if isinstance(cur_prev, np.ndarray) else cur_prev == v:
                continue
        except (KeyError, TypeError):
            pass
        else:
            continue
        dates[k] = True

    # Return old coef updated with new
    return {**coefs, **coefs_new}, coef_zeroing_matrix, dates, msg_zeroed + msg_rotated


def binning(d, dt_bin, repartition_freq=None, print_binning="{}-binning, "):
    """
    Do binning, repartition, persist
    :param dt_bin: resulting resampled period in seconds
    """
    b_binning = dt_bin and dt_bin > timedelta(0)  # else repartition and persist only
    if b_binning:
        if print_binning:
            print(print_binning.format(format_timedelta(dt_bin)), end="")

        # If period > data time span then convert to one row dask dataframe with index = mean index
        if dt_bin > np.diff(d.divisions[0 :: (len(d.divisions) - 1)]).item():
            df = d.mean().compute().to_frame().transpose()
            df.index = [d.index.compute().mean()]
            if df.empty:
                return None
            return dd.from_pandas(df, npartitions=1)

        # Do binning and remove all bins with less than 10% of values

        # Number of existed values in each bin (using 1st column)
        n_good = (~d.iloc[:, 0].isna()).resample(dt_bin).count()

        d = d.resample(
            dt_bin,
            # closed='right' if 'Pres' in cfg['in']['path'].stem else 'left'
            # 'right' for burst mode
            # because the last value of interval used in wavegauges is round - not true?
        ).mean()
        d = d.where(n_good > n_good[n_good > 0].mean() / 10).dropna(how="all")  # axis=0 is default

    if repartition_freq:
        try:  # repartition for ascii splitting (helps to prevent MemoryError also)
            d = d.repartition(freq=repartition_freq)
        except TypeError as e:
            return None

    # Persist to greatly speedup next calculations
    try:
        d.persist()  # excludes missed values?
    except MemoryError:
        lf.info(
            "Skipped persisting to speedup: not enough memory for {}-aggregated data...",
            dt_bin
        )

    return d


def to_physical(
    df_raw: pd.DataFrame | dd.DataFrame,
    coefs,
    coef_zeroing_matrix,
    cfg_filter,
    dt_bins: Sequence[timedelta],
    dt_min_binning_proc: timedelta = timedelta(seconds=2),
    tbl='',
    dt_dask_partition=pd.Timedelta(1, 'D'),
    df_physical=None
) -> Tuple[Optional[dd.DataFrame], List[dd.DataFrame]]:
    """
    Calculate physical parameters and binning.

    :param df_raw:
    :param coefs: `incl_calc_velocity` and `calc_pressure` keyward arguments (which are mostly coefs)
    :param coef_zeroing_matrix:
    :param cfg_filter:
    :param dt_bins:
    :param dt_min_binning_proc: There is should be no more than one `dt_bin` (except 0s-bin) that is <= than this param
    :param tbl: table name for more informative logging error messages if will be error
    :param dt_dask_partition:
    :param df_physical: ready not avg physical data to use or output instead calc here
    :return: list of physical params dataframes for each of `dt_bins`, if no 0s-bin required then 1st is None
    d, d_avg = to_physical(
        ...
        dt_dask_partition=cfg["out"]["split_period"]
    )
    """
    if not dt_bins:
        return None, []
    dt_bin_min = min(dt_bins)
    if dt_dask_partition is None and dt_bin_min > timedelta(0):
        # By default, we restrict dask partition size to 100000 bins to prevent memory overflow
        dt_dask_partition = 100000 * dt_bin_min

    # Raw data processing
    df_raw = filter_local(df_raw, cfg_filter, ignore_absent={"h_minus_1", "g_minus_1"})
    # d_raw[['Mx','My','Mz']] = d_raw[['Mx','My','Mz']].mask(lambda x: x>=4096)
    # todo: message of filtered especially if all was filt out

    # Apply loaded or calculated rotation to Ag & Ah coefficients (with copying to new coef object to not cumulate)
    if coef_zeroing_matrix:
        coefs = {**coefs, **{c: coef_zeroing_matrix @ coefs[c] for c in ["Ag", "Ah"]}}
    else:
        lf.info("No rotation (Rz) in loaded coefficient. Suppose rotations already applied")

    if isinstance(df_raw, pd.DataFrame):
        d_raw = dd.from_pandas(df_raw, npartitions=1)
        d_raw.length = df_raw.shape[0]  # save length because getting it is a long operation in dask
    else:
        d_raw = df_raw

    need_d_out = dt_bins[0] == timedelta(0)  # timedelta(0) in dt_bins
    if df_physical is None:
        d = None
        need_d = need_d_out or any(dt > dt_min_binning_proc for dt in dt_bins)  # need creterea
    else:
        d = dd.from_pandas(df_physical, npartitions=1)  # chunksize | if None will be calc if needed
        if dt_bins and need_d_out:
            dt_bins = dt_bins[1:]
        need_d = False

    # Binning and physical parameters calculation
    d_avgs = []
    for dt_bin in dt_bins:
        # Binning before physical parameters calculation (if bin <= 2s by default)
        bin_before_physical = dt_bin <= dt_min_binning_proc
        if bin_before_physical:
            d_raw = binning(d_raw, dt_bin, repartition_freq=dt_dask_partition)
            if d_raw is None:
                lf.error("No data after raw {} {}-binning  => Skipping. Continue...", tbl, dt_bin)
                continue

        if bin_before_physical or need_d:

            # Velocity
            d_avg = incl_calc_velocity(d_raw, filt_max=cfg_filter["max"], **coefs)

            # Pressure
            d_avg = calc_pressure(d_avg, **coefs)
            # **{(pb := 'bad_p_at_bursts_starts_period'): cfg['filter'][pb]},

            if need_d:
                d = d_avg
                need_d = False
                if dt_bin == timedelta(0):
                    continue
            else:  # bin_before_physical
                d_avgs.append(d_avg)
                continue

        # Binning after physical parameters calculation
        if not bin_before_physical:
            d_avg = binning(
                d, dt_bin, repartition_freq=dt_dask_partition, print_binning="{}-binning. "
            )
            if d_avg is None:
                lf.error("No data after processed {} {}-binning  => Skipping. Continue...", tbl, dt_bin)
            d_avgs.append(d_avg)
        else:
            print(end='. ')
    print()
    return [d] + d_avgs


def gen_physical(
    cfg: MutableMapping[str, Any],
    incl_calc_kwargs={},
    **in_many
) -> Iterator[Tuple[dd.DataFrame, Dict[str, np.array], str, int]]:
    """
    Yields parameters to process many paths, tables, dates_min and dates_max. For hdf5 data uses
    `h5.dispenser_and_names_gen(fun_gen)`, for text input - `load_csv.load_from_csv_gen()`

    :param cfg: dict with fields
    - in:
        - table: pattern to search input tables (any input to `to_pcid_from_name`)
          letter type prefix (incl) for data in counts else one letter (i). After is porobe identificator separated with "_" if not start with "i" else just 2 digits    - out: dict with fields
    - out:
        - table: pattern to construct output table based on input tables found using `in.table` search pattern
        - dt_bins: list of bins to average
    - ...
    - program: if this dict has field 'return_' equal to '<cfg_input_files>' then returned `d` will have only
    edge rows
    :param: in_many: fields originated from ConfigInMany_InclProc: these fields will be replaced by single valued fields like in ConfigIn_InclProc:
    - path: Union[str, Path], str/path of real path or multiple paths joined by '|' to be split in list cfg['paths'],

    - min_date:
    - max_date:
    - dt_between_bursts, dt_hole_warning - optional - arguments
    and optional plural named fields - same meaning as keys of cfg['in'/'filter'] but for many probes:

    :return: Iterator((d, (cfg1, tbl, pcid, i_tbl_part))), where:
    - d - data
    - cfg1.coefs, tbl, pcid - fun_gen output:
    tbl: name of tables in coefs database (same as in .raw.h5) and, in proc_noAvg.h5, or in proc.h5
    - i_tbl_part - part number if loading many parts in same output table

    --------------------------
    for d, (tbl, pcid, probe_continues) in gen_physical(...):

    """

    # Separate config for each probe from cfg['in_many'] fields
    cfg_in_common, cfg_in_for_probes = separate_cfg_in_for_probes(cfg["in"], in_many)

    # Prepare to save coefs to DB
    save_coef_funcs = []  # container for functions that saves coefs

    def factory_save_coef(is_db_a_pandas_store=True, **kwards):
        """Save coefs function factory
        :param is_db_a_pandas_store: set to False if you've open `db` by h5py
        :param kwards: tbl_raw, dict_matrices, dates
        """
        def h5copy_coef_to_db(db):
            # can't create dataset using h5py in the opened pandas store, so close and reopen
            if is_db_a_pandas_store:
                db.close()
            table_written = h5copy_coef(None, db.filename if is_db_a_pandas_store else db, **kwards)
            if is_db_a_pandas_store:
                db.open()
            return table_written

        return h5copy_coef_to_db

    pcid_prev = None
    pcid_part = 0

    def calc(
        df_raw,
        cfg_in,
        cfg_filter,
        pcid,
        b_raw_data_from_db=None,
        df_physical=None,
        dt_bins=cfg["out"]["dt_bins"],
        dt_dask_partition=cfg["out"]["split_period"]
    ):
        """
        Prepare coef, calculate and average physical parameters data (calls `coef_prepare`, `to_physical`)
        :param b_raw_data_from_db: if set then do not save coefs to DB as data (with coefs) is there already
        """
        nonlocal pcid_part, pcid_prev

        probe_continues = (pcid == pcid_prev)
        if probe_continues:
            pcid_part += 1  # next part of same csv
        else:
            pcid_prev = pcid
            pcid_part = 0

        tbl_raw = pcid_to_raw_name(pcid)
        lf.warning(
            "Raw {:s}{:s}{:s} data loaded. Processing...",
            tbl_raw,
            "" if b_raw_data_from_db is None else
            (" raw db" if df_physical is None else " raw/noAvg db") if b_raw_data_from_db else " csv",
            "" if not pcid_part else f" (part {pcid_part})"
        )
        # lf.warning(
        #     "{: 2d}. {:s} from {:s}{:s}",
        #     ipid,
        #     pcid,
        #     tbl_in,
        #     msg,
        # )
        coefs, coef_zeroing_matrix, dates, msg_coefs = call_with_valid_kwargs(
            coef_prepare,
            **cfg_in,
            data=df_raw,
            data_date=getattr(df_raw, 'divisions' if isinstance(df_raw, dd.DataFrame) else 'index')[0]
        )

        # Save coeff. (postpone save by h5py to raw db as now it can be opened by pandas)
        if not (b_raw_data_from_db or probe_continues):
            lf.info(f"Coefficients will be saved {msg_coefs}to {tbl_raw}")
            save_coef_funcs.append(factory_save_coef(
                is_db_a_pandas_store=False,
                dict_matrices=coefs_format_for_h5(coefs, pcid),
                tbl=tbl_raw,
                dates=dates,
                )
            )

        d_avgs = to_physical(
            df_raw,
            {
                **coefs,
                **incl_calc_kwargs,
                "max_incl_of_fit_deg": cfg_in["max_incl_of_fit_deg"],
                "calc_version": cfg_in["calc_version"],
            },
            coef_zeroing_matrix,
            cfg_filter=cfg_filter,
            dt_bins=dt_bins,
            dt_min_binning_proc=cfg_in["dt_min_binning_proc"],
            tbl=tbl_raw,
            dt_dask_partition=dt_dask_partition,
            df_physical=df_physical,
        )
        return d_avgs, probe_continues


    b_input_is_h5 = cfg["in"]["paths"][0].suffix in hdf5_suffixes
    if b_input_is_h5:
        # Loading from HDF5
        ###################

        # Determine whether we need input raw data or averaged data
        cfg["in"]["path"] = get_input_db(
            cfg["in"]["path"], dt_min_binning_proc=cfg["in"]["dt_min_binning_proc"], **cfg["out"]
        )
        b_raw = '.raw' in cfg["in"]["path"].suffixes[-2:-1]

        def gen_raw(skip_for_meta: Callable[[Any], Any]):
            """
            Search and generate data from raw / noAvg DB
            For use as argument of `h5.append_through_temp_db_gen()` to update noAvg / Avg DB through temp DB
            :param skip_for_meta: function to send meta to the caller `h5.append_through_temp_db_gen` that
            determines wherther to reuse saved noAvg data instead of this function output to calc it
            yields df_raw_or_None, (df_raw, meta_raw_db) where:
            - df_raw_or_None: raw data or None if skip_for_meta,
            - df_raw: raw data,
            - meta_raw_db: (i1pid, pcid, path_csv), rec_raw, tbl_raw
            Call in calc phys. cycle inside `h5.append_through_temp_db_gen` that can reuse data from noAvg DB
            """

            n_yields = 1

            def yielding(d_raw, df_log, tbl_raw, msg=""):
                """
                return (df, (df_raw, (i1pid, pcid, path_csv), rec_raw, tbl_raw)): compatible output with
                `gen_avg_updating_no_avg_db` input equal to output of other `gen_raw` version below
                """
                time_st, time_en = d_raw.divisions[:: len(d_raw.divisions) - 1]
                df_log_raw = df_log[(time_en >= df_log["DateEnd"])&(df_log.index >= time_st)]
                meta_raw_db = ((msg, tbl_raw.replace("incl", "i", 1), []), df_log_raw, tbl_raw)  # `to_record_avg_db` compartible
                # Send meta to the caller `table_from_meta`, `meta_to_record` to reuse saved noAvg data
                b_skip_dt0 = skip_for_meta(meta_raw_db)
                return None if b_skip_dt0 else d_raw, (d_raw, meta_raw_db)


            # Output table pattern to get out tables from input tables names (set None if len > 1 to not use)
            table_out_ptns = {
                key: (lambda t: val_default if not t else (t[0] or val_default) if len(t) == 1 else None)(
                    cfg["out"].get(key)
                )
                for key, val_default in [("tables", "{}"), ("tables_log", "{}/logFiles")]
            }

            # Tables fill parameter cycle (to fill patten with it, if has "*" then later with search results)
            for fill, cfg_in_cur in cfg_in_for_probes.items():

                # Fill input table name pattern
                try:
                    tbl_raw_ptn = cfg_in_cur.pop("table")
                except KeyError:
                    # continue
                    tbl_raw_ptn = '{}'

                # Convert out table name to input table to fill in search patten
                if fill.startswith("*"):
                    fill = f".{fill}"  # make valid regex
                tbl_raw_ptn = tbl_raw_ptn.format(fill)
                if tbl_raw_ptn[0] == "i" and "incl" not in tbl_raw_ptn:
                    tbl_raw_ptn = f"incl{tbl_raw_ptn[1:]}"
                tbl_raw_ptn = tbl_raw_ptn.replace(".*.*", ".*")  # removes redundant regex

                # Fill output table name (if fill is not pattern(found table))
                if '*' not in fill:
                    for key in ["tables", "tables_log"]:
                        if table_out_ptns[key]:
                            cfg["out"][key] = table_out_ptns[key].replace(".*", "").format(fill)
                elif "{" in tbl_raw_ptn and "{" not in table_out_ptns["table"]:
                    lf.warning(  # multiple input tables to one otput table warning
                        f"All {fill} found tables by pattern {tbl_raw_ptn} will be loaded to table "
                        f"{cfg['out']['tables'][0]} based on current config group name. Consider to use {{}} "
                        "in table name search pattern to fill each input group output name with input table "
                        "name instead"
                    )

                path_in = cfg_in_cur.get("path") or cfg_in_common["path"]
                with pd.HDFStore(path_in, mode="r") as db:
                    tables = h5.find_tables(db, tbl_raw_ptn)
                    # Tables cycle for current probe, but change current probe if it is a pattern(table)
                    for tbl_in in tables:
                        # Change current probe
                        if '*' in fill:
                            tbl_out = tbl_in.replace("incl", "i", 1)
                            cfg_in_cur = cfg_in_for_probes.get(tbl_out, cfg_in_cur)

                            for key in ["tables", "tables_log"]:
                                if table_out_ptns[key]:
                                    cfg["out"][key] = table_out_ptns[key].replace(".*", "").format(tbl_out)

                        df_log = h5.read_db_log(db, f"{tbl_in}/logFiles")

                        # Config to load raw data from current table
                        cfg_in_cur["table"] = tbl_in

                        # Init ``time_ranges``
                        if cfg_in_cur["time_ranges"] is None and (
                            cfg_in_cur["min_date"] or cfg_in_cur["max_date"]
                        ):
                            cfg_in_cur["time_ranges"] = [
                                cfg_in_cur["min_date"] or "2000-01-01",
                                cfg_in_cur["max_date"] or pd.Timestamp.now(),
                            ]

                        # Query data by time interval(s) if loading from raw DB
                        if cfg_in_cur["time_ranges"] and (
                            cfg["out"]["b_split_by_time_ranges"] or ".raw" in path_in.suffixes[-2:-1]
                        ):
                            try:
                                index_range, i0range, iq_edges = h5.coords(
                                    cfg_in_cur["db"], tbl_in, q_time=cfg_in_cur["time_ranges"] or None
                                )  # time range index and numeric indexes range of data
                            except TypeError:  # skip empty nodes
                                # cannot create a storer if the object is not existing nor a value are passed
                                lf.warning("Skipping {} without data table found", tbl_in)
                                continue
                            n_parts = round(len(iq_edges) / 2)
                            if n_parts < 1:
                                lf.warning(
                                    "Skipping {0}: no data found{1}{2}{4}{3}",
                                    tbl_in,
                                    *(
                                        (" in range ", *cfg_in_cur["time_ranges"], " - ")
                                        if cfg_in_cur["time_ranges"]
                                        else [""] * 4
                                    ),
                                )
                                continue
                            for i_part, iq_edges_cur in enumerate(zip(iq_edges[::2], iq_edges[1::2])):
                                n_rows_to_load = -np.subtract(*iq_edges_cur)
                                if n_rows_to_load == 0:  # empty interval - would get errors in main circle
                                    continue  # ok when used config with many intervals from different db
                                ddpart = h5_load_range_by_coord(
                                    db_path=path_in, **cfg_in_cur, range_coordinates=iq_edges_cur
                                )
                                # Filter by 'min'/'max' params if data is not proc. (that should've been filtered)
                                if b_raw:
                                    d, i_burst = filt_data_dd(
                                        ddpart,
                                        cfg_in_cur.get("dt_between_bursts"),
                                        cfg_in_cur.get("dt_hole_warning"),
                                        cfg_in_cur,
                                    )
                                else:
                                    d = ddpart
                                if index_range is not None:
                                    yield yielding(
                                        d,
                                        df_log,
                                        tbl_in,
                                        msg="{}: {:%Y-%m-%d %H:%M:%S} – {:%m-%d %H:%M:%S}".format(
                                            f" part {i_part}" if n_parts > 1 else "",
                                            *index_range[iq_edges_cur - i0range],
                                        ),
                                    )
                                else:
                                    yield yielding(d, df_log, tbl_in)
                                n_yields += 1
                        else:
                            # Process full range data
                            d = h5_load_range_by_coord(**cfg_in_cur, db_path=path_in, range_coordinates=None)
                            yield yielding(d, df_log, tbl_in)
    else:
        # Load CSV (meta)data
        #####################
        if cfg["in"]["path"]:
            cfg["in"]["paths"] = []  # ignore
        if cfg["out"]["dt_bins"]:
            ...

        # Load only metadata: probes list, optionally with 1st and last data row for each probe
        raw_corrected = search_correct_csv_files(cfg_in_common, cfg["program"])

        gen_csv = partial(
            load_from_csv_gen,
            raw_corrected=raw_corrected,
            cfg_in=cfg_in_common,
            cfg_in_probe=cfg_in_for_probes,
        )

        if cfg["program"]["return_"] and cfg["program"]["return_"].startswith("<cfg_input_files"):
            # Yield metadata:
            # cfg1: calc. parameters (including dataframes with edge rows if cfg["program"]["return_"] set so)
            # pcid: output col name suffix
            # tbl_raw: same as pcid with i replaced to "incl"

            for df_raw, (ipid, pcid, path_csv) in gen_csv(
                return_=cfg["program"]["return_"],
            ):
                # Configuration for current input pcid
                cfg1 = cur_cfg(pcid, cfg_in_for_probes, cfg_in_common, cfg)
                cfg1["in"]["path"] = path_csv  # corrected raw txt
                if df_raw:
                    cfg1["time_ranges"] = [dt.isoformat() for dt in df_raw.index]
                # output pcid
                if cfg["out"]["table"]:
                    pcid = to_pcid_from_name(cfg["out"]["table"])
                yield cfg1, (False, pcid, None)
            return



        # CSV to raw DB (`*.raw.h5`)

        def to_record_raw(meta):
            """
            :param meta: tuple (ipid, pcid, csv) - `gen_csv()` metadata output
            """
            return h5.file_name_and_time_to_record(
                meta[-1], *(cfg["out"].get("logfield_fileName_len") or (),)
            )

        def gen_raw(skip_for_meta: Callable[[Any], Any]):
            """
            Generate raw data and update raw DB (`*.raw.h5`) through temporary (`*.raw_not_sorted.h5`),
            skipping to save the existed up to date text data in raw DB (in that case load it).
            For use in `h5.append_through_temp_db_gen`
            :param skip_for_meta: function to send meta to the caller `h5.append_through_temp_db_gen` that
            determines wherther to reuse saved noAvg data instead of this function output to calc it
            yields df_raw_or_None, (df_raw, metacsv_rec_tblraw) where:
            - df_raw_or_None raw data or None if this data is not needed,
            - df_raw: raw data,
            - metacsv_rec_tblraw: tuple where `metacsv` is (ipid, pcid, csv) yelded by `load_from_csv_gen()`
            Call in calc phys. cycle inside `h5.append_through_temp_db_gen` that can reuse data from noAvg DB
            """
            for df_raw, metacsv_rec_tblraw in h5.append_through_temp_db_gen(
                gen_csv,
                db_path=cfg["out"]["raw_db_path"],
                # temp_db_path=cfg["out"]["temp_db_path"],  # need?
                table_from_meta=(lambda meta_csv: pcid_to_raw_name(meta_csv[1])),
                skip_fun=(
                    partial(h5.keep_recorded_file, keep_newer_records=False)
                    if cfg["out"]["b_incremental_update"]
                    else (lambda cur, existing: False)
                ),
                meta_to_record=to_record_raw,
            ):
                # Send meta to the __caller__ `table_from_meta`, `meta_to_record` to reuse saved noAvg data
                b_skip_dt0 = skip_for_meta(metacsv_rec_tblraw)
                yield None if b_skip_dt0 else df_raw, (df_raw, metacsv_rec_tblraw)

            # Write coefs to raw db (without temporary db)
            if save_coef_funcs:
                tables_written = []
                with h5py.File(cfg["out"]["raw_db_path"], "a") as h5dest:
                    for fun_save_coef in save_coef_funcs:
                        tables_written += fun_save_coef(h5dest)
                print(f"coefs written to raw db: {tables_written}")





    # Raw to noAvg DB (`*.proc_noAvg.h5`)
    cfg1 = None  # configuration for current probe in cycle with copy of cfg fields that can be changed
    pcid = None  # name of probe in `cfg_many` config (Probe output Column ID in output composed table)

    def to_record_avg_db(meta):
        """
        Extract noAvg DB log record, and init `cfg1` with coefficients as their date used here
        :param meta: tuple (i1pid_pid_csv, csv_record, tbl_raw)
        By the way we get `cfg1` and `pcid`
        """
        nonlocal cfg1, pcid
        i1pid_pid_csv, csv_record, tbl_raw = meta

        # Get coefficients time for metadata record and, by the way, vars needed later
        pcid = to_pcid_from_name(tbl_raw)
        cfg1 = cur_cfg(pcid, cfg_in_for_probes, cfg_in_common, cfg)

        # Set metadata record with fileChangeTime = max time of sources that can affect result
        record_file, record_max = (
            [csv_record[k] for k in ["fileName", "fileChangeTime"]]
            if isinstance(csv_record, Mapping)  # when not from raw db
            else csv_record.loc[:, ["fileName", "fileChangeTime"]].values[0]
        )
        # to_record_raw(i1pid_pid_csv)["fileName"] if i1pid_pid_csv
        return {
            "fileName": record_file,
            "fileChangeTime": max(
                (d for d in [cfg1["in"]["coefs"].get("date"), record_max] if d is not None)
            )
        }

    def gen_avg_updating_no_avg_db(skip_for_meta: Callable[[Any], Any]):
        """
        Gen processed data (with updating noAvg DB). For use in `h5.append_through_temp_db_gen`
        :param skip_for_meta: function defined inside `h5.append_through_temp_db_gen` to skip
            averaging and load saved data from Avg DB and `table_from_meta` and `meta_to_record` args
        yields (d_avg, meta_avg_db_bin):
        - d_avg: processed data bin averaged with `dt_bin` (which can be 0 if need proc. not avg more)
        - cont_pcid_bin_tbl_rec: metadata (probe_continues, pcid, tbl_raw, dt_bin, tbl, rec)

        """
        for df, ((df_raw, metacsv_rec_tblraw), rec_noavg, tbl_noavg) in h5.append_through_temp_db_gen(
            gen_raw,
            db_path=cfg["out"]["not_joined_db_path"],
            table_from_meta=lambda metacsv_rec_tblraw: metacsv_rec_tblraw[-1].replace("incl", "i", 1),
            meta_to_record=to_record_avg_db,
        ):
            (i1pcid, pcid, path_csv), rec_raw, tbl_raw = metacsv_rec_tblraw

            # `df` can be data from noAvg DB or raw data equal to `df_raw` have used to get `record`
            if df.columns.to_list() == df_raw.columns.to_list():
                df = None
            #     rec_out = rec_raw
            # else:
            #     rec_out = rec_noavg

            d_avgs, probe_continues = calc(
                df_raw, cfg1["in"], cfg1["filter"], pcid,
                b_raw_data_from_db=None if path_csv is None else isinstance(path_csv, list), df_physical=df
            )  # `load_from_csv_gen` yields `path_csv` as a list if skips for the reuse of existed db data
            dt_bins = cfg["out"]["dt_bins"]
            i_avg_st = None if not dt_bins else (dt_bins[0] > timedelta(0))  # index of 1st averaging data
            for d, dt_bin in zip(d_avgs[i_avg_st:], dt_bins[i_avg_st:]):
                bin = int(dt_bin.total_seconds())
                if not bin:
                    continue

                tbl_avg = f"{tbl_noavg}bin{bin}s"

                # Send meta to the caller `table_from_meta`, `meta_to_record` to reuse saved Avg data
                b_skip = skip_for_meta([rec_raw, tbl_avg])

                yield None if b_skip else d, [probe_continues, pcid, dt_bin] #, tbl_avg, rec_out


    if cfg["out"]["db_path"]:
        # Configured writing csv data to raw (if csv), raw to noAvg DBs (through temporary DBs) is on

        # Gen processed data separatly for each averaging bin (with updating Avg DB)
        for d, (cont_pcid_bin, rec_avg, tbl_avg) in h5.append_through_temp_db_gen(
            gen_avg_updating_no_avg_db,
            db_path=cfg["out"]["not_joined_db_path"].parent
            / cfg["out"]["not_joined_db_path"].name.replace("_noAvg", "_Avg", 1),
            table_from_meta=(lambda rec_tbl: rec_tbl[1]),
            meta_to_record=(lambda rec_tbl: rec_tbl[0]),
        ):  # d can be pandas dataframe from Avg DB or dask dataframe obtained now
            yield d, cont_pcid_bin

    else:
        # Not writing to DB / not reading DB data here (except config coefs)

        for df_raw, (i1pcid, pcid, path_csv) in gen_csv():
            # Configuration for pcid: `cfg1` with copy of cfg fields that can be changed in cycle
            cfg1 = cur_cfg(pcid, cfg_in_for_probes, cfg_in_common, cfg)  # get coefs
            d_avgs, probe_continues = calc(
                df_raw, cfg1["in"], cfg1["filter"], pcid, b_raw_data_from_db=False
            )
            if cfg["out"]["table"]:
                pcid = to_pcid_from_name(cfg["out"]["table"])
            dt_bins = cfg["out"]["dt_bins"]
            for d, dt_bin in zip(d_avgs[(dt_bins[0] > timedelta(0)):], dt_bins):
                bin = int(dt_bin.total_seconds())
                if not bin:
                    continue

                yield d_avgs, [probe_continues, pcid, dt_bin]


# -----------------------------------------------------------------------------------------------------------

@hydra.main(config_name=cs_store_name, config_path='cfg', version_base='1.3')
def main(config: ConfigType) -> Union[None, Mapping[str, Any], pd.DataFrame]:
    """
    Load data from raw csv file or hdf5 table (or group of them)
    Calculate new data or average by specified interval
    Combine this data to new table
    :param config:
    Set to use raw input data for output bins < 2s by default else (not averaged) physical values
    :return:
    """
    global cfg
    if config.input.path is None and config.input.paths is None:
        raise ValueError('Input `path` or `paths` must be provided.')
    cfg = cfg_d.main_init(config, cs_store_name, __file__=None)
    cfg['input']['paths'] = (
        [Path(config.input['path'])] if config.input['paths'] is None
        else [Path(p) for p in config.input['paths']])
    cfg = cfg_d.main_init_input_file(cfg, cs_store_name, msg_action='Loading data from', in_file_field='path')
    lf.info(
        "Begin {:s} {:s}",
        this_prog_basename(__file__),
        "(no averaging)"
        if cfg["out"]["dt_bins"] == [timedelta(0)]
        else "(bins: {})".format(', '.join(format_timedelta(dt) for dt in cfg["out"]["dt_bins"]))
    )
    cfg["in"]["path"] = set_full_paths_and_h5_out_fields(cfg["out"], **cfg["in"])
    cfg['in'].setdefault('dt_min_binning_proc', pd.Timedelta('2s'))

    if cfg['program']['return_']:

        # Return / save configuration

        if cfg['program']['return_'] == '<cfg_before_cycle>':
            return cfg  # common config to help testing
        if cfg['program']['return_'].startswith('<cfg_input_files'):
            # Return configuration for each input file and optionally save them
            if cfg['program']['return_'].endswith('to_yaml>'):
                dir_cfg_proc = cfg["out"]["raw_db_path"].parent / "cfg_proc" / "defaults"
                dir_cfg_proc.mkdir(parents=True, exist_ok=True)
                file_name_pattern = f"{{id}}_{datetime.now():%Y%m%d_%H%M%S}.yaml"
                lf.info(f"Saving configuration parameters mode: to {file_name_pattern} for each probe ID")
            else:
                dir_cfg_proc = None
            out_dicts = {}
            file_stems = set()
            lf.info('Saving yaml configs')
            for cfg1, (probe_continues, pcid, _) in gen_physical(cfg, **cfg['in_many']):
                if dir_cfg_proc:
                    # Save calculating parameters
                    # rename incompatible to OmegaConf "in" name, remove paths as we save for each file
                    cfg1 = {"input": {**cfg1.pop("in"), "paths": None}, **cfg1}
                    cfg1["out"]["dt_bins"] = cfg["out"]["dt_bins"]
                    # Fields that added here and are not in structured config
                    del cfg1["out"]["tables"]
                    del cfg1["out"]["nfiles"]
                    del cfg1["out"]["b_del_temp_db"]
                    del cfg1["out"]["temp_db_path"]
                    del cfg1["out"]["b_incremental_update"]

                    del cfg1["in"]["dt_min_binning_proc"]
                    del cfg1["in"]["b_insert_separator"]
                    del cfg1["in"]["cfgFile"]

                    conf_ = cfg_d.to_omegaconf_merge_compatible(cfg1, ConfigType)
                    conf = OmegaConf.create(conf_)

                    # Save to `defaults` dir so we can use them as configs with new defaults to run later
                    while True:  # Make unique file name (can gen_physical yield same pcid?)
                        file_stem = file_name_pattern.format(id=pcid)
                        if file_stem in file_stems:  # change time-based name
                            file_name_pattern = f"{{id}}_{datetime.now():%Y%m%d_%H%M%S}.yaml"
                            continue
                        file_stems.add(file_stem)
                        break
                    print(f"{pcid}: {file_stem}", end=", ")
                    with (dir_cfg_proc / file_stem).with_suffix('.yaml') as fp:
                        OmegaConf.save(conf, fp)  # or pickle.dump(conf, fp)
                        # fp.flush()
                out_dicts[str(cfg1['input']['path'])] = cfg1  # [tbl]? cfg_?
            print("\nOk.", end=" ")
            return out_dicts

    if not (cfg["out"]["not_joined_db_path"] or cfg["out"]["text_path"]):
        lf.error("Neither output hdf5 store nor text output was requested. The end")
        return

    # Set columns from incl_calc_velocity() we need to calculate: all calculated cols will be prepend if
    # save txt and without Vabs/dir columns if not txt needed
    cols_out_h5 = ['v', 'u', 'Pressure', 'Temp', 'inclination']  # absent here cols will be ignored

    # Filtering config [min/max][M] could be specified with just key M to set same value for keys Mx My Mz
    for lim in ["min", "max"]:
        if "M" in cfg["filter"][lim]:
            for ch in ("x", "y", "z"):
                set_field_if_no(cfg["filter"][lim], f"M{ch}", cfg["filter"][lim]["M"])

    if cfg["program"]["dask_scheduler"] == "distributed":
        from dask.distributed import Client, progress

        # cluster = dask.distributed.LocalCluster(n_workers=2, threads_per_worker=1, memory_limit="5.5Gb")
        client = Client(processes=False)
        # navigate to http://localhost:8787/status to see the diagnostic dashboard if you have Bokeh installed
        # processes=False: avoide inter-worker communication for computations releases the GIL (numpy, da.array)  # without is error
    else:
        if cfg["program"]["dask_scheduler"] == "synchronous":
            lf.warning('using "synchronous" scheduler for debugging')
        import dask

        dask.config.set(scheduler=cfg["program"]["dask_scheduler"])
        progress = None
        client = None

    def map_to_suffixed(names, suffix):
        """Adds tbl suffix to output columns before accumulate in cycle for different tables"""
        return {col: f"{col}_{suffix}" for col in names}

    # Data processing cycle
    dfs_all_bins: Mapping[str, List[pd.DataFrame]] = {
        dt: [] for dt in cfg["out"]["dt_bins"] if dt > timedelta(0)}
    tbls = {k: [] for k in dfs_all_bins.keys()}  # tables to add in log table, to joined suffix to comb. csv
    cfg["out"]['tables_written'] = set()
    probe_continue_file_name = None  # this line actually not needed, used just for Pylance
    for d, (probe_continues, pcid, dt_bin) in gen_physical(
        cfg,
        **cfg["in_many"],
        incl_calc_kwargs={"cols_prepend": [c for c in cols_out_h5 if c not in ("Pressure", "Temp")]}
        if cfg["out"]["text_path"]
        else {},  # keep min cols if not need to save txt: v, u, inclination
    ):
        if d is None:
            continue
        bin = int(dt_bin.total_seconds())

        # Save csv (splitted by configured period)
        if dt_bin >= cfg["out"]["dt_bins_min_save_text"]:
            if isinstance(d, pd.DataFrame):
                d = dd.from_pandas(d, npartitions=1)
            if cfg["out"]["split_period"]:
                d = d.repartition(freq=cfg["out"]["split_period"])
            probe_continue_file_name = dd_to_csv(
                d,
                cfg["out"]["text_path"],
                text_date_format=(  # Shorter out time format if seconds fractional part not needed
                    (lambda dt: dt[: -len(".%f")] if dt.endswith(".%f") and bin % 1 == 0 else dt)(
                        cfg["out"]["text_date_format"]
                    )
                ),
                text_columns=cfg["out"]["text_columns"],
                suffix=("bin{bin}s".format(bin=bin) if bin else "") + f"@{pcid}",
                single_file_name=(
                    probe_continue_file_name if probe_continues else not cfg["out"]["split_period"]
                ),
                progress=progress,
                client=client,
                b_continue=probe_continues,
            )

        # Collect binned data (of equal index period with wich we can join data)
        if bin:
            try:
                cols_save = [c for c in cols_out_h5 if c in d.columns]
                sleep(cfg["program"]["sleep_s"])
                Vne = d[cols_save].compute()  # MemoryError((1, 12400642), dtype('float64'))
                dfs_all_bins[dt_bin].append(
                    Vne if cfg["out"]['b_all_to_one_col'] else
                    Vne.rename(columns=map_to_suffixed(cols_save, pcid))
                )
                tbls[dt_bin].append(pcid)
            except Exception as e:
                lf.exception('Can not cumulate result! ')
                raise
                # todo: if low memory do it separately loading from temporary tables in chanks

        gc.collect()  # frees many memory. Helps to not crash

    # Combine data to hdf5
    ######################
    if not any(dfs_all_bins.values()):
        return

    tbl_avg_ptn = "i_bin{bin}s"

    def gen_cobined_avg(skip_for_meta: Callable[[Any], Any] = None):
        """
        Collect accumulated data ,
        skipping to save the existed up to date text data in raw DB (in that case load it).
        For use in `h5.append_through_temp_db_gen`
        :param skip_for_meta: function to send meta to the caller `h5.append_through_temp_db_gen` that
        determines wherther to reuse saved noAvg data instead of this function output to calc it
        yields df_raw_or_None, (df_raw, metacsv_rec_tblraw) where:
        - df_raw_or_None raw data or None if this data is not needed,
        - df_raw: raw data,
        - metacsv_rec_tblraw: tuple where `metacsv` is (ipid, pcid, csv) yelded by `load_from_csv_gen()`
        Call in calc phys. cycle inside `h5.append_through_temp_db_gen` that can reuse data from noAvg DB
        """


        for dt_bin, dfs in dfs_all_bins.items():
            bin = int(dt_bin.total_seconds())
            if len(dfs) <= 1:
                if not dfs:
                    lf.warning('No data found for bin = {}s', bin)
                # else all data is from one device => not need concatenate (can be already saved in DB / csv)
                continue

            # # Allow save and log each df to proc DB consequently
            # if cfg["out"]['b_all_to_one_col']
            #     for df, tbl in zip(dfs, tbls[dt_bin]):
            #         yield df, tbl

            if skip_for_meta is not None:
                # Send meta to the __caller__ `table_from_meta`, `meta_to_record` to reuse saved comb. avg
                dfs_all_log = pd.DataFrame.from_records(
                    [df.index[[0, -1]].to_list() + [tbl] for df, tbl in zip(dfs, tbls[dt_bin])],
                    columns=["Time", "DateEnd", "table"],
                    index="Time",
                    # exclude=["Time"],
                )  #.set_index('table_name')\ #.sort_index() #\
                tbl_cmb_avg = tbl_avg_ptn.format(bin=bin)
                b_skip = skip_for_meta((dfs_all_log, tbl_cmb_avg))
            else:
                b_skip = False

            # Concatenate several columns in parallel (default: add columns) or consequently to 1-probe data
            df = pd.concat(dfs, sort=True, axis=int(not cfg["out"]['b_all_to_one_col']))

            yield None if b_skip else df, dt_bin


    if cfg["out"]["db_path"]:
        # Update accumulated avg DB (`*.proc.h5`) through temporary DB
        for df, (dt_bin, rec, tbl) in h5.append_through_temp_db_gen(
            gen_cobined_avg,
            db_path=cfg["out"]["db_path"],
            # temp_db_path=cfg["out"]["temp_db_path"],  # need?
            table_from_meta=lambda rec_tbl: rec_tbl[1],
            meta_to_record=lambda rec_tbl: rec_tbl[0],
            skip_fun=(
                partial(h5.keep_recorded_file, keep_newer_records=False)
                if cfg["out"]["b_incremental_update"]
                else (lambda cur, existing: False)
            )
        ):
            dfs_all_bins[dt_bin] = df

    # Write concatenated dataframe to ascii (? with resample if b_all_to_one_col)
    for dt_bin, df in dfs_all_bins.items():
        dd_to_csv(
            (lambda x: x.resample(rule=dt_bin).first() if cfg["out"]["b_all_to_one_col"] else x)(
                dd.from_pandas(df, chunksize=500000)
            ),
            text_path=cfg["out"]["text_path"],
            text_date_format=(  # Shorter out time format if seconds fractional part not needed
                (lambda dt: dt[: -len(".%f")] if dt.endswith(".%f") and not dt_bin.microseconds else dt)(
                    cfg["out"]["text_date_format"]
                )
            ),
            text_columns=cfg["out"]["text_columns"],
            suffix=(
                (lambda bin: f"bin{int(bin)}s" if bin else "")(dt_bin.total_seconds())
                + f"@{','.join(tbl.removeprefix('i_') for tbl in tbls[dt_bin])}"
            ),
            progress=progress,
            client=client,
        )

    print('Ok.', end=' ')
    return dfs_all_bins

if __name__ == '__main__':
    main()
