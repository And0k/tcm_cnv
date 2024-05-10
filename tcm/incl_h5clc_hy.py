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
from dataclasses import asdict, dataclass, field
from datetime import date as datetime_date
from datetime import datetime, timedelta
from functools import wraps
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

# v2
from .csv_load import load_from_csv_gen
from .csv_specific_proc import parse_name
from .filters import b1spike, rep2mean
from .h5_dask_pandas import (
    cull_empty_partitions,
    dd_to_csv,
    filter_global_minmax,
    filter_local,
    h5_append_to,
    h5_load_range_by_coord,
    i_bursts_starts,
)
from .h5inclinometer_coef import dict_matrices_for_h5, h5copy_coef, rot_matrix_x, rotate_y

# v1
from .h5toh5 import h5_close, h5_dispenser_and_names_gen, h5coords, h5find_tables, h5move_tables, h5out_init
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
    Ag: List[List[float]] = field(default_factory=lambda: [[0.00173, 0, 0], [0, 0.00173, 0], [0, 0, 0.00173]])
    Cg: Annotated[list[float], 3] = field(default_factory=lambda: [10, 10, 10])
    Ah: List[List[float]] = field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Ch: Annotated[list[float], 3] = field(default_factory=lambda: [10, 10, 10])
    Rz: Optional[List[List[float]]] = field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    kVabs: List[float] = field(default_factory=lambda: [10, -10, -10, -3, 3, 70])
    # kP: Optional[List[float]] = field(default_factory=lambda: [0, 1])  # default polynom will not change input
    P_t: Optional[List[List[float]]] = None
    azimuth_shift_deg: float = 180
    g0xyz: Optional[Annotated[list[float], 3]] = None  # field(default_factory=lambda: [0, 0, 1])  #


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
    tables_log: List[str] = field(default_factory=list)
    # aggregate_period_min_to_load_proc  # switch to use processed no avg db as input when aggregate_period is >  # uncomment to not use hard coded 2s default

    # params needed to process data when loading directly from csv:
    prefix: Optional[str] = 'I*_'  # 'INKL_'
    text_type: Optional[str] = None  # corresponds to predefined text_line_regex. If it and text_line_regex is None then
    # will be guessed by file name
    text_line_regex: Optional[str] = None  # if not None then ignore text_type else will be selected based on text_type

    coefs: Optional[ConfigInCoefs_InclProc] = None  # field(default_factory=ConfigInCoefs_InclProc)
    coefs_path: Optional[str] = r'd:\WorkData\~configuration~\inclinometer\190710incl.h5'

    date_to_from_list: Optional[List[str]] = None  # alternative to set: dt_from_utc_hours = diff(array(
    # date_to_from_list, 'M8[ms]')).astype('float')/3600000
    dt_from_utc_hours = 0
    min_date: Optional[str] = None  # imput data time range minimum
    max_date: Optional[str] = None  # imput data time range maximum
    time_ranges: Optional[List[str]] = None
    coordinates: Optional[List[float]] = None  # add magnetic declination at coordinates [Lat, Lon], degrees

    time_ranges_zeroing: Optional[List[str]] = field(default_factory=list)
    azimuth_add: float = 0
    max_incl_of_fit_deg: Optional[float] = None
    calc_version: str = 'trigonometric(incl)'

    # minimum time between blocks, required in filt_data_dd() for data quality control messages:
    dt_between_bursts: Optional[float] = np.inf  # inf to not use bursts, None to auto-find and repartition
    dt_hole_warning_seconds: Optional[int] = 600


@dataclass
class ConfigOut_InclProc(cfg_d.ConfigOutSimple):
    """
    "out": parameters of output files
    #################################
    :param not_joined_db_path: If set then save processed not averaged data for each probe individually to this path. If set to True then path will be out.db_path with suffix ".proc_noAvg.h5". Table names will be the same as input data.
    :param table: table name in hdf5 store to write combined/averaged data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in auto numbered locations (see dask to_hdf())
    :param split_period: string (pandas offset: D, H, 2S, ...) to process and output in separate blocks. If saving csv for data with original sampling period is set then that csv will be splitted with by this length (for data with no bin averaging  only)
    :param aggregate_period: s, pandas offset strings  (5D, H, ...) to bin average data. This can greatly reduce sizes of output hdf5/text files. Frequenly used: None, 2s, 600s, 7200s
    :param text_path: path to save text files with processed velocity (each probe individually). No file if empty, "text_output" if ``out.aggregate_period`` is set. Relative paths are from out.db_path
    :param text_date_format: default '%Y-%m-%d %H:%M:%S.%f', format of date column in output text files. (.%f will be removed from end when aggregate_period > 1s). Can use float or string representations
    :param text_columns_list: if not empty then saved text files will contain only columns here specified
    :param b_all_to_one_col: concatenate all data in same columns in out db. both separated and joined text files will be written
    :param b_del_temp_db: default='False', temporary h5 file will be automatically deleted after operation. If false it will be remained. Delete it manually. Note: it may contain useful data for recovery

    """
    not_joined_db_path: Any = None
    raw_db_path: Any = 'auto'  # will be the parent of input.path dir for text files
    table: str = ''
    aggregate_period: Optional[str] = ''
    split_period: str = ''
    text_path: Optional[str] = 'text_output'
    text_date_format: str = '%Y-%m-%d %H:%M:%S.%f'
    text_columns: List[str] = field(default_factory=list)
    b_split_by_time_ranges: bool = False  # split averaged data by cfg['in']['time_ranges'] only if this set
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
    # Fields that specifies different values for each probe ID. May be dicts of existed in
    # ConfigIn/ConfigFilter fields:
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
    force[is_nans] = np.NaN
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
            incl_rad[bad_g] = np.NaN

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
                    Mcnts_list[i] = np.NaN + da.empty_like(HsumMinus1)
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

def incl_calc_velocity(a: dd.DataFrame,
                       filt_max: Optional[Mapping[str, float]] = None,
                       cfg_proc: Optional[Mapping[str, Any]] = None,
                       Ag: Optional[np.ndarray] = None, Cg: Optional[np.ndarray] = None,
                       Ah: Optional[np.ndarray] = None, Ch: Optional[np.ndarray] = None,
                       kVabs: Optional[np.ndarray] = None, azimuth_shift_deg: Optional[float] = 0,
                       cols_prepend: Optional[Sequence[str]] = incl_calc_velocity_cols,
                       **kwargs
                       ) -> dd.DataFrame:
    """
    Calculates velocity from raw accelerometer/magnetometer data. Replaces raw columns with velocity vector parameters.
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
    :param filt_max: dict with fields:
    - g_minus_1: mark bad points where |Gxyz| is greater, if any then its number will be logged,
    - h_minus_1: to set Vdir=0 and...
    :param cfg_proc: 'calc_version', 'max_incl_of_fit_deg'
    :param cols_prepend: parameters that will be prepended to a columns (only columns if a have only raw
    accelerometer and magnetometer fields). If None then ('Vabs', 'Vdir', 'v', 'u', 'inclination') will be used. Use
     any from these elements in needed order.
    :param kwargs: not affects calculation
    :return: dataframe with prepended columns ``cols_prepend`` and removed accelerometer and
    magnetometer raw data
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
    len_data = sum(lengths)

    if kVabs is not None and 'Ax' in a.columns:
        lf.debug('calculating V')
        try:    # lengths=True gets MemoryError   #.to_dask_array()?, dd.from_pandas?
            Hxyz = a.loc[:, ('Mx', 'My', 'Mz')].to_dask_array(lengths=lengths).T
            for i in range(3):
                _ = da.map_overlap(b1spike, Hxyz[i], max_spike=100, depth=1)
                Hxyz[i] = rep2mean_da(Hxyz[i], ~_)
            Hxyz, need_recover_mask = recover_magnetometer_x(Hxyz, Ah, Ch, filt_max['h_minus_1'], len_data)

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
                            (100 * n_one_ch_is_over_scaled / len_data).to_dict()
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
                            n_g_bigger_one = 100 * (g_other_sqear > 1).sum().compute() / len_data
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
                    bad_g_sum = 100 * bad_g_sum / len_data  # to %
                    if bad_g_sum > 1:
                        lf.warning(
                            'Acceleration is too high (>{}) in {:g}% points! => data nulled',
                            filt_max['g_minus_1'], bad_g_sum
                        )
                    incl_rad[bad] = np.NaN
            # else:
            #     bad = da.zeros_like(GsumMinus1, np.bool_)

            # lf.debug('{:.1g}Mb of data accumulated in memory '.format(dfs_all.memory_usage().sum() / (1024 * 1024)))

            # sPitch = f_pitch(Gxyz)
            # sRoll = f_roll(Gxyz)
            # Vdir = np.degrees(np.arctan2(np.tan(sRoll), np.tan(sPitch)) + fHeading(Hxyz, sPitch, sRoll))

            # Velocity absolute value

            #Vabs = da.map_blocks(incl_rad, kVabs, )  # , chunks=GsumMinus1.chunks
            Vabs = incl_rad.map_blocks(v_abs_from_incl,
                                       coefs=kVabs,
                                       calc_version=cfg_proc['calc_version'],
                                       max_incl_of_fit_deg=cfg_proc['max_incl_of_fit_deg'],
                                       dtype=np.float64, meta=np.float64([]))
            # Vabs = np.polyval(kVabs, np.where(bad, np.NaN, Gxyz))
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
            # Placing columns at first, defined in ``cols_prepend`` in their order, and keep remained in previous order
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
            pressure[:2] = np.NaN
            p[:] = pressure
            return p

        a.Pressure = p_bursts.map_partitions(calc_and_rem2first, meta=meta)
    else:
        a.Pressure = a.Pressure.map_partitions(lambda x: np.polyval(P, x), meta=meta)

    return a


def coef_zeroing_rotation(g_xyz_0, Ag_old, Cg) -> Union[np.ndarray, int]:
    """
    Zeroing rotation matrix Rz to correct old rotation matrices Ag_old, Ah_old (Ag = Rz @ Ag_old, Ah = Rz @ Ah_old)
    based on accelerometer values when inclination is zero.
    :param g_xyz_0: 3x1 values of columns 'Ax','Ay','Az' (it is practical to provide mean values of some time range)
    :param Ag_old, Cg: numpy.arrays, rotation matrix and shift for accelerometer
    :return Rz: 3x3 numpy.array , corrected rotation matrices

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
    # todo: use new rotate algorithm


def coef_rotate(*A, Z):
    return [Z @ a for a in A]


def coef_zeroing_rotation_from_data(
        df: dd.DataFrame | pd.DataFrame,
        Ag, Cg,
        time_intervals=None,
        **kwargs) -> np.ndarray:
    """
    Zeroing rotation matrix Rz to correct old rotation matrices Ag_old, Ah_old (Ag = Rz @ Ag_old, Ah = Rz @ Ah_old)
    based on accelerometer raw data from time range
    :param df: pandas or dask DataFrame with columns 'Ax', 'Ay', 'Az' - raw accelerometer data
    :param time_intervals: optional data index time range to select data
    Old coefficients:
    :param Ag:
    :param Cg:
    :param kwargs: not used
    :return: Ag, Ah - updated coefficients
    """

    if time_intervals:
        time_intervals = pd.to_datetime(time_intervals, utc=True)
        if len(time_intervals) <= 2:
            idx = slice(*time_intervals)
        else:
            # Create an iterator and use zip to pair up start and end times
            time_pairs = (lambda x=iter(time_intervals): list(zip(x, x)))()
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
    #         da.flatnonzero(b_ok), median3(x[b]), da.NaN, da.NaN)
    #
    # b = da.from_array(b_ok, chunks=chunks, meta=('Tfilt', 'f8'))
    # with ProgressBar():
    #     Tfilt = da.map_blocks(interp_after_median3(x, b), y, b).compute()

    # hangs:
    # Tfilt = dd.map_partitions(interp_after_median3, a['Temp'], da.from_array(b_ok, chunks=cfg_out['chunksize']), meta=('Tfilt', 'f8')).compute()

    # Tfilt = np.interp(da.arange(len(b_ok)), da.flatnonzero(b_ok), median3(a['Temp'][b_ok]), da.NaN,da.NaN)
    # @+node:korzh.20180524213634.8: *3* main
    # @+others
    # @-others


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


def filt_data_dd(a, dt_between_bursts=None, dt_hole_warning: Optional[np.timedelta64] = None, cfg_filter=None
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


def set_full_paths_and_h5_out_fields(cfg: MutableMapping, is_aggregating: bool) -> List[Path]:
    """
    1. Depending on input path and wheather aggregating is required switches output and input source:
    - returning input paths `in_paths`: if averaging is required and input are text files, then switches to
      default raw DB name. This allows to load previously saved data in subsequent runs, but if raw db not
      exist then raise.
    - cfg['in']['tables']: replace "incl{id}" with `pid` if loading from processed noAvg DB
    2. Sets ouput database parameters in cfg['out'] (wraps `h5out_init()`).

    :param is_aggregating:
    :param cfg: should contain
    'in' fields:
    - is_counts
    'out' fields:
    - raw_db_path
    :updates cfg:
    ['in'] fields:
    - tables
    ['out'] fields:
    - db_path
    - raw_db_path
    - not_joined_db_path
    - text_path
    - b_incremental_update = False
    :return: in_paths - list of full input paths (to use instead of cfg['in']['paths'])
    Modifies:
    cfg['in']['tables'],
    cfg['out']
    - db_path: switch to averaging DB (*proc.h5) if loading from processed noAvg DB
    - text_path
    - other fields, modified by `h5out_init()`
    """
    in_paths = cfg['in']['paths']
    if is_aggregating:
        if in_paths[0].suffix not in hdf5_suffixes:
            # Switch input paths to default raw DB name
            __, in_paths = in_paths, []
            for p in __:  # cfg['in']['paths'] bigger priority than cfg['out']['raw_db_path']
                p, proc_dir = get_full_raw_db_path(cfg['out']['raw_db_path'], path_in=p)
                in_paths.append(p)

        # If loading processed data then change input db and tables to that are in .proc_noAvg
        if not cfg['in']['is_counts']:
            _ = []
            for i, p in enumerate(in_paths):
                if '.proc_noAvg' not in p.suffixes[-2:-1]:
                    # If .proc_noAvg db was not set explicitly, then try load db with default name
                    db_path_proc_orig_fr = p.parent.with_name(p.stem).with_suffix('.proc_noAvg.h5')  # default
                    if not db_path_proc_orig_fr.is_file():
                        raise FileNotFoundError(
                            f"Not found processed data {db_path_proc_orig_fr} for "
                            f"{cfg['out']['aggregate_period']}-averaging.")
                    in_paths[i] = db_path_proc_orig_fr
                    _.append('{}/{}'.format(*db_path_proc_orig_fr.parts[-2:]))
            cfg['in']['tables'] = [tbl.replace('incl', 'i') for tbl in cfg['in']['tables']]
            lf.info(
                'Using found {} db and its {} tables as source for averaging', ', '.join(_), cfg['in']['tables']
            )

    raw_db_path, proc_dir = get_full_raw_db_path(cfg['out']['raw_db_path'], in_paths[0])

    if not cfg['out']['db_path'].is_absolute():
        out_db_path_stem = cfg['out']['db_path'].stem
        if not out_db_path_stem:
            out_db_path_stem = stem_proc_db(proc_dir)
        cfg['out']['db_path'] = (proc_dir / out_db_path_stem).with_suffix('').with_suffix(f'.proc.h5')

    if not is_aggregating:
        if cfg['out']['not_joined_db_path'] is True or (
            not cfg['out']['not_joined_db_path'] and                                        # not defined
            (not cfg['in']['is_counts'] or cfg['in']['aggregate_period_min_to_load_proc'])  # but will be needed
        ):
            cfg['out']['not_joined_db_path'] = (
                cfg['out']['db_path'].with_suffix('').with_suffix('').with_suffix('.proc_noAvg.h5')
            )
        cfg['out']['db_path'] = cfg['out']['not_joined_db_path']

    cfg_out_table = cfg['out']['tables']  # not need fill  # old: save because will need to change for h5_append()
    h5out_init(cfg['in'], cfg['out'])
    cfg['out']['tables'] = cfg_out_table
    cfg['out']['b_incremental_update'] = False  # todo: To can use You need provide info about chunks that will be loaded

    if (in_paths[0].suffix not in hdf5_suffixes) and raw_db_path:
        # Prepare to save raw data from csv to raw.h5 hdf5
        cfg['out']['raw_db_path'] = raw_db_path

    if (cfg['out']['text_path'] is not None and not
        cfg['out']['text_path'].is_absolute()):
        cfg['out']['text_path'] = proc_dir / cfg['out']['text_path']

    return in_paths


def stem_proc_db(proc_dir: Path) -> Path:
    proc_db_stem = (
        proc_dir.name if proc_dir.name[0].isdigit() else
        proc_dir.parent.name if proc_dir.parent.name[0].isdigit() or
        proc_dir.name.startswith('inclinometer') else
        proc_dir.name
    ).replace('@', '_').split('_')[0]
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
                    _.replace('@', '_').split('_')[0] if raw_subdir.is_dir() and _[0].isdigit() else None
                    )
        else:
            raw_db_name = None
        # If no "_raw" dir was found then try get db name from cruise dir: parent dir that starts from digits
        if not raw_db_name:
            raw_db_name = stem_proc_db(proc_dir)

    raw_db_path = ((parents[idx_raw_dir] if idx_raw_dir else p_parent) / raw_db_name).with_suffix('.raw.h5')
    return raw_db_path, proc_dir


def load_coefs(store, tbl: str):
    """
    Finds up to 2 levels of coefficients, collects to 1 level, accepts any paths
    :param store: hdf5 file with group "``tbl``/coef"
    :param tbl:
    :return:
    Example
    for coefs nodes:
    ['coef/G/A', 'coef/G/C', 'coef/H/A', 'coef/H/C', 'coef/H/azimuth_shift_deg', 'coef/Vabs0'])
    naming rule gives coefs this names:
    ['Ag', 'Cg', 'Ah', 'Ch', 'azimuth_shift_deg', 'kVabs'],
    """
    coefs_dict = {'dates': {}}
    node_coef = store.get_node(f'{tbl}/coef')
    if node_coef is None:
        return
    else:
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
    return coefs_dict


def load_coefs_(coefs_path, tbl: str, coefs_ovr: Optional[Mapping[str, Any]] = None):
    """Load coefs from path/tbl. Provide `coefs_ovr` to ovewrite"""
    if not coefs_path:
        return
    if (  # Skip loading if all fields that by default not None already defined by `coefs_ovr`
        coefs_ovr and all(
        coefs_ovr.get(k) for k, v in ConfigInCoefs_InclProc.__dataclass_fields__.items() if v.default
        )):
        return coefs_ovr

    with pd.HDFStore(coefs_path, mode='r') as store:
        coefs_dict = load_coefs(store, tbl)
    if coefs_dict is None:
        lf.warning(
            'Not found coefficients table "{:s}" ', tbl
        )
    if coefs_ovr:
        if OmegaConf.is_config(coefs_ovr):
            coefs_dict =  OmegaConf.to_container(coefs_ovr)
        coefs_dict.update(coefs_ovr)
    return coefs_dict

def coefs_format_for_h5(coef: Mapping[str, Any], pid: str, date: Optional[str] = None) -> Mapping[str, Any]:
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
    :return:
    """
    if coef is None:
        # making default coef matrix
        coef = ConfigInCoefs_InclProc().__dict__
        del coef['g0xyz']
        if not pid.startswith('p'):
            del coef['P_t']
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
        '//coef//pid': pid,
        '//coef//date': date or datetime.now().replace(microsecond=0).isoformat()
    }


def map_to_suffixed(names, suffix):
    """Adds tbl suffix to output columns before accumulate in cycle for different tables"""
    return {col: f"{col}_{suffix}" for col in names}


def probes_gen(cfg_in: Mapping[str, Any], cfg_out: None = None) -> Iterator[Tuple[str, Tuple[Any, ...]]]:
    """
    Yield table names with associated coefficients which are loaded from '{tbl_in}/coef' node of hdf5 file.
    :param cfg_in: dict with fields:
    - tables: tables names search pattern or sequence of table names
    - path: hdf5 file with tables which have coef group nodes.
    :param cfg_out: not used but kept for the requirement of h5_dispenser_and_names_gen() argument
    :return: iterator that returns (table_name, coefficients).
    - table_name is same as input tables names except that "incl" replaced with "i"
    - coefficients are None if path ends with 'proc_noAvg'.
     "Vabs0" coef. name are replaced with "kVabs"
    Updates cfg_in['tables'] - sets to list of found tables in store
    """

    with pd.HDFStore(cfg_in['path'], mode='r') as store:
        _ = cfg_in.get('tables'); n = len(_) if _ else 0
        tables = h5find_tables(store, _[0]) if n == 1 else _

        # message all not found coefs
        b_bad_coef = False
        coefs_dicts = {}
        for tbl_in in tables:
            if cfg_in['path'].suffixes[0].startswith('.proc'):  # coefs not need if source data is already processed
                coefs_dicts[tbl_in] = None
                continue
            coefs_dicts[tbl_in] = load_coefs(store, tbl_in)
            if coefs_dicts[tbl_in] is None:  # and cfg_in.get('skip_if_no_coef'):
                lf.warning('"{:s}" - not found coefs!', tbl_in)  #Skipping this table
                b_bad_coef = True
                continue
        if b_bad_coef:
            raise ValueError('Not all coefs found')

        for tbl_in in tables:
            if '.proc_noAvg' in cfg_in['path'].suffixes[-2:-1]:
                # Loading already processed data: not cfg_in['is_counts']
                tbl_in = tbl_in.replace('incl_', 'i').replace('incl', 'i')

            cfg_in['table'] = tbl_in
            # fields that need to be updated for h5del_obsolete() if this func used with it inside generator:
            if len(cfg_in['tables']) == 1:
                cfg_in['tables'] = [tbl_in]

            # tables names need to be updated for h5del_obsolete() if this func used with it inside generator:
            # if not cfg['out']['table']:
            col_out, pid = out_col_name(tbl_in)
            if cfg_out:
                aggregate_period = cfg_out.get('aggregate_period')
                cfg_out['table'] = f'i_bin{aggregate_period}' if aggregate_period else col_out
            yield tbl_in, col_out, pid, coefs_dicts[tbl_in]
    return


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
    :return: (grouped_cfg, cfg_in_common)
    - cfg_in_common dict: inmput parameters with removed plural fields
    - grouped_cfg: dict with dicts for each probe, i.e.:
      - keys: groups: probe identificators or '*' if no found,
      - values: dicts with:
        - keys: singular named keys obtained from found plural named arguments/in_many keys,
        - values: have value of argument/in_many values for each key
    """
    # Combine all input parameters different for different probes we need to convert to this dict, renaming
    # fields to singular form:
    cfg_many = {}
    cfg_in_common = in_many.copy()
    for k in [  # single_name_to_many_vals
        'min_date',
        'max_date',
        'time_ranges',
        'time_ranges_zeroing',
        'date_to_from',
        'bad_p_at_bursts_starts_period',
        'coordinates'
    ]:
        try:
            cfg_many[k] = cfg_in_common.pop(k)
        except KeyError:
            continue
    cfg_in_common.update(cfg_in.copy())


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
                    probe = out_col_name(probe.lower())[0]  # not a `pid` as out_col_name(pid) not always work
                    # if probe.startswith('i') and probe[1].isalpha():
                    #     probe = f'i_{probe[1:]}'  # pid_to_col_out(pid[])
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
        probes = [out_col_name(tbl)[0] for tbl in cfg_in["tables"]]  # tbl.rsplit('_', 1)[-1]
    return cfg_in_common, group_dict_vals(cfg_many, probes)


def pid_to_col_out(pid, probe_type='i'):
    """
    Column name suffix in processed DB for probe:
    {pid} for inclinometers without model else {probe_type}_{pid}.
    Instead inputing `pid` for inclinometers may put just number
    :param pid: probe id,
    :param probe_type: probe type ('i' for inclinometers)
    :return: column name (that is just pid for inclinometers)
    """
    if pid[0].isdigit():
        pid = f'{probe_type}{pid}'
        col_out = pid
    else:
        if pid[0] == 'i':
            if probe_type == 'i':
                probe_type = ''
            _ = ''
        else:
            _ = '_'
        col_out = f'{probe_type}{_}{pid}'
    return col_out


def out_col_name(tbl_group: str):
    """
    Output column name and pid from raw table / raw file stem search pattern/name / output column name
    :param tbl_group: input table
    :return: (col_out, pid)
    - col_out: output column name (= name in hydra specific for this probe config)
    - pid for input
    """

    pattern_name_parts = parse_name(tbl_group.replace('.', ''))
    if pattern_name_parts:
        pid = '{model}{number}'.format_map(pattern_name_parts)
        probe_type = pattern_name_parts['type']   # tbl_group[0]
    else:
        return '*', '*'

    col_out = pid_to_col_out(pid, probe_type)
    return col_out, pid


def cfg_copy_for_pid(cfg: Mapping[str, Any], cfg_in_common: Mapping[str, Any], cfg_in_cur: Mapping[str, Any]
    ) -> Mapping[str, Any]:
    """
    Combine fields from cfg, cfg_in_common, cfg_in_cur to processing configuration for current probe
    """
    cfg1 = {
        'in': {**cfg_in_common.copy(), **cfg_in_cur},
        'out': cfg['out'].copy(),  # actually may not need copy() here
        'filter': cfg['filter'].copy()
    }
    return cfg1

def get_pid_cfg_(
    cfg: Mapping[str, Any], cfg_in_common: Mapping[str, Any], cfg_in_for_probes: Mapping[str, Any], tbl_raw: str
    ) -> Tuple[Mapping[str, Any], str, str]:
    """
    Return processing configuration for probe with id `pid` when loading txt
    :param cfg:
    :param cfg_in_for_probes: config for current probe
    :param cfg_in_common: input config common to all probes
    :param tbl_raw: table name in raw db according to historical convension ("incl{id}")
    :return: (cfg1, col_out, pid)
    - cfg1: dict with
      - copy of `cfg` fields that can be changed (to can change without modifying `cfg`)
      - coefs
    - col_out: output column suffix in combined table parsed from tbl_raw
    - pid: pid parsed from tbl_raw
    """
    # `pid` key to get data from cfg_in_for_probes
    col_out, pid = out_col_name(tbl_raw)
    # col_out = col_out if col_out in cfg_in_for_probes else '*'
    cfg1 = cfg_copy_for_pid(cfg, cfg_in_common, cfg_in_for_probes.get(col_out, {}))

    # Output table name
    aggregate_period = cfg1['out'].pop('aggregate_period')
    cfg1['out']['table'] = f'i_bin{aggregate_period}' if aggregate_period else col_out

    # Load coefs
    cfg1['in']['coefs'] = load_coefs_(cfg['in']['coefs_path'], tbl_raw, coefs_ovr=cfg1['in']['coefs'])
    if cfg1['in']['coefs']:
        cfg1['in']['coefs'] = {
            k: np.float64(v) if (isinstance(v, list) and v is not None) else v
            for k, v in cfg1['in']['coefs'].items()
        }
    return cfg1, col_out, pid


def gen_subconfigs(
    cfg: MutableMapping[str, Any],
    fun_gen=probes_gen,
    **in_many) -> Iterator[Tuple[dd.DataFrame, Dict[str, np.array], str, int]]:
    """
    Yields parameters to process many paths, tables, dates_min and dates_max. For hdf5 data uses
    `h5_dispenser_and_names_gen(fun_gen)`, for text input - `load_csv.load_from_csv_gen()`

    :param cfg: dict with fields
    - in:
      - is_counts: if True then use raw (not averaged) input db tables (if input is *.h5)
    - out: dict with fields
      - aggregate_period
      - ...
    - ...
    - program: if this dict has field 'return_' equal to '<cfg_input_files>' then returned `d` will have only
    edge rows
    :param fun_gen: generator of (tbl, coefs)

    Other ):
    :param: in_many: fields originated from ConfigInMany_InclProc: these fields will be replaced by single valued fields like in ConfigIn_InclProc:
    - path: Union[str, Path], str/path of real path or multiple paths joined by '|' to be split in list cfg['paths'],
    - table: not one letter type prefix (incl) for data in counts else one letter (i). After is porobe identificator
     separated with "_" if not start with "i" else just 2 digits
    - min_date:
    - max_date:
    - dt_between_bursts, dt_hole_warning - optional - arguments
    and optional plural named fields - same meaning as keys of cfg['in'/'filter'] but for many probes:

    :return: Iterator(d, cfg1, tbl, col_out, pid, i_tbl_part), where:
    - d - data
    - cfg1.coefs, tbl, col_out, pid - fun_gen output:
    tbl: name of tables in coefs database (same as in .raw.h5) and, in proc_noAvg.h5, or in proc.h5
    - i_tbl_part - part number if loading many parts in same output table
    """
    # Construct config for 1 probe (cfg1)
    # Separate config for each probe from cfg['in_many'] fields
    cfg_in_common, cfg_in_for_probes = separate_cfg_in_for_probes(cfg["in"], in_many)

    b_input_is_h5 = cfg['in']['paths'][0].suffix in hdf5_suffixes
    if b_input_is_h5:
        # Loading from hdf5
        ###################

        # Output table pattern to get out tables from input tables names
        _ = cfg['out'].get('tables'); n = len(_) if _ else 0
        table_out_ptn = _[0] if n == 1 else _
        _ = cfg['out'].get('tables_log'); n = len(_) if _ else 0
        table_log_out_ptn = _[0] if n == 1 else _

        for tbl_out_fill, cfg_in_cur in cfg_in_for_probes.items():
            # if table explicitly defines input tables list then skip redundant tables from other plural
            # parameters
            if 'table' not in cfg_in_cur:
                continue
            # Output table name
            aggregate_period = cfg['out']['aggregate_period']
            # tbl_out_fill, pid = out_col_name(tbl_in_abbr)
            if isinstance(aggregate_period, str) and aggregate_period:  # all probes to one table
                cfg['out']['table'] = f'i_bin{aggregate_period}'
            elif not aggregate_period:
                if '{' in cfg_in_cur['table']:
                    # Convert out table name to input table to fill in search patten
                    if tbl_out_fill.startswith('*'):
                        tbl_out_fill = f'.{tbl_out_fill}'  # make valid regex
                else:
                    lf.warning(  # group msg
                        f'All found tables by pattern {cfg_in_cur["table"]} will be loaded to table '
                        '{tbl_out_fill} based on current config group name. Consider to use {} in table name '
                        'search pattern to fill each input group output name with input table name instead'
                    )
                if table_out_ptn:
                    cfg['out']['table'] = table_out_ptn.replace('.*', '').format(tbl_out_fill)
                if table_log_out_ptn:
                    cfg['out']['table_log'] = table_log_out_ptn.replace('.*', '').format(tbl_out_fill)
            else:
                raise ValueError('Config parameter out.aggregate_period should be str or None')
            # Input table name
            cfg_search_in = {
                'path': cfg_in_cur['path'],
                'tables': [
                    cfg_in_cur.pop('table').format(  # removing old table info. Changes cfg_in_for_probes!
                        tbl_out_fill.replace('i', 'incl')
                        if cfg['in']['is_counts'] and tbl_out_fill[0] == 'i'
                        else tbl_out_fill
                    ).replace('.*.*', '.*')]  # redundant regex
            }
            n_yields = 1
            for itbl, (tbl_in, col_out, pid, coefs_dict) in h5_dispenser_and_names_gen(  # opens output DB
                    cfg_search_in, cfg['out'], fun_gen=fun_gen, b_close_at_end=False
            ):  # also gets cfg['out']['db'] to write, updats cfg['out']['b_remove_duplicates']

                # Set current configuration: add copy of fields that can be changed (to not change initial cfg
                # each cycle we are starting from)
                try:
                    cfg1 = cfg_copy_for_pid(cfg, cfg_in_common, cfg_in_for_probes[col_out])
                except KeyError:
                    lf.warning(
                        'No data for existed configuration: {}! Skipping...', col_out)
                    continue
                cfg1['in']['table'] = tbl_in
                cfg1['in']['path'] = cfg_in_cur['path']

                # Possibility to overwrite probe with specific settings in "probes/{table}.yaml"
                try:  # Warning: Be careful especially if update 'out' config
                    cfg_hy = hydra.compose(overrides=[f"+probes={tbl_in}"])
                    if cfg_hy:
                        with open_dict(cfg_hy):
                            cfg_hy.pop('probes')
                        if cfg_hy:
                            lf.warning(
                                'Loaded YAML Hydra configuration for data table "{:s}" {}', tbl_in, cfg_hy)
                            for k, v in cfg_hy.items():
                                cfg1[k].update(v)
                except hydra.errors.MissingConfigException:
                    pass

                # ``time_ranges`` must be specified before loading so filling with dumb values where no info.
                if cfg1['in']['time_ranges'] is None and (cfg1['in']['min_date'] or cfg1['in']['max_date']):
                    cfg1['in']['time_ranges'] = [
                        cfg1['in']['min_date'] or '2000-01-01',
                        cfg1['in']['max_date'] or pd.Timestamp.now()
                    ]

                def yielding(d, msg=':', i_part=None):
                    """
                    Yield with message and changes the name of the output table from incl to i (my convention
                    for processed data storage)
                    :param d:
                    :param msg:
                    :return:
                    """
                    lf.warning(
                        '{: 2d}. {:s} from {:s}{:s}{:s}',
                        itbl, col_out, tbl_in, ': ' if i_part is None else f'.{i_part}: ', msg
                    )
                    # copy to not cumulate the coefs corrections in several cycles of generator consumer for
                    # same probe and convert to numpy arrays:
                    if coefs_dict is None:
                        pass
                    elif cfg1['in']['coefs'] is None:
                        cfg1['in']['coefs'] = coefs_dict
                    else:
                        # coefs_dict = None if coefs is None else {
                        #     k: np.float64(v) if (isinstance(v, list) and v is not None) else v
                        #     for k, v in (coefs if isinstance(coefs, dict) else OmegaConf.to_container(coefs)).items()
                        # }
                        cfg1['in']['coefs'].update(coefs_dict)
                    return d, cfg1, tbl_in.replace('incl', 'i'), col_out, pid, i_part

                with pd.HDFStore(cfg1['in']['path'], mode='r') as cfg1['in']['db']:
                    if ('.proc_noAvg' in cfg1['in']['path'].suffixes[-2:-1] and
                        not cfg['out']['b_split_by_time_ranges']):
                        # assume not need global filtering/get intervals for such input database
                        d = h5_load_range_by_coord(
                            **cfg1['in'], db_path=cfg1['in']['path'], range_coordinates=None)
                        yield yielding(d)
                    else:
                        # query and process several independent intervals
                        # Get index only and find indexes of data
                        try:
                            index_range, i0range, iq_edges = h5coords(
                                cfg1['in']['db'], tbl_in, q_time=cfg1['in']['time_ranges'] or None
                            )
                        except TypeError:  # skip empty nodes
                            # cannot create a storer if the object is not existing nor a value are passed
                            lf.warning('Skipping {} without data table found', tbl_in)
                            continue
                        n_parts = round(len(iq_edges) / 2)
                        if n_parts < 1:
                            lf.warning('Skipping {0}: no data found{1}{2}{4}{3}', tbl_in, *(
                                (' in range ', *cfg1['in']['time_ranges'], ' - ')
                                if cfg1['in']['time_ranges'] else
                                ['']*4)
                            )
                            continue
                        for i_part, iq_edges_cur in enumerate(zip(iq_edges[::2], iq_edges[1::2])):
                            n_rows_to_load = -np.subtract(*iq_edges_cur)
                            if n_rows_to_load == 0:  # empty interval - will get errors in main circle
                                continue  # ok when used config with many intervals from different databases
                            ddpart = h5_load_range_by_coord(
                                db_path=cfg1['in']['path'], **cfg1['in'], range_coordinates=iq_edges_cur
                            )
                            d, i_burst = filt_data_dd(
                                ddpart, cfg1['in'].get('dt_between_bursts'),
                                cfg1['in'].get('dt_hole_warning'),
                                cfg1['in']
                                )
                            if index_range is not None:
                                yield yielding(d, msg='{:%Y-%m-%d %H:%M:%S} – {:%m-%d %H:%M:%S}'.format(
                                    *index_range[iq_edges_cur - i0range]),
                                    i_part=i_part if n_parts > 1 else None
                                )
                            else:
                                yield yielding(d)
                            n_yields += 1
    else:
        # Load CSV data
        ###############

        if cfg['out']['aggregate_period']:
            ...
        if cfg['program']['return_'] and cfg['program']['return_'].startswith("<cfg_input_files"):
            # Yield metadata:
            # df_raw    # dataframes with edge rows
            # cfg1      # calculating parameters
            # tbl_raw   # same as pid with i replaced to "incl" (if no specific porbe model: pid starts from same letter as probe_type) or "incl_"
            # col_out   # output col name suffix: {probe_type}{_}{pid} ({pid} if no specific probe model)
            # pid       # 3-alphanumeric symbols word

            # load only 1st and last row for each probe (metadata to return)
            for itbl, pid, path_csv, df_raw in load_from_csv_gen(
                cfg_in_common, cfg_in_for_probes, cfg['program']['return_']
                ):
                # output table
                tbl_raw = (
                    cfg['out']['table'] or
                    pid.replace('i', 'incl') if pid[0] == 'i' else f'incl_{pid}'
                )
                # Configuration for current pid
                cfg1, col_out, pid = get_pid_cfg_(cfg, cfg_in_common, cfg_in_for_probes, tbl_raw)
                cfg1['in']['path'] = path_csv  # corrected raw txt
                if df_raw:
                    cfg1["time_ranges"] = [dt.isoformat() for dt in df_raw.index]
                yield df_raw, cfg1, tbl_raw, col_out, pid, None
            return

        tables_written_raw = set()
        log_raw = {}
        path_csv_prev = None
        pid_prev = None

        def load_from_csv_gen_for_h5_dispenser(cfg_in_common, cfg_out, **cfg_in_for_probes):
            """
            Wraps `load_from_csv_gen()` to use it in `h5_dispenser_and_names_gen()`
            :param cfg_out: dict, must have fields:
            - log: dict, with info about current data, must have fields for compare:
              - 'fileName' - in format as in log table to be able to find duplicates
              - 'fileChangeTime', datetime - to be able to find outdated data
            - b_incremental_update: if True then not yields previously processed files. But if file was
              changed then: 1. removes stored data and 2. yields `fun_gen(...)` result
            - tables_written: sequence of table names where to create index
            """
            yield from load_from_csv_gen(cfg_in_common, cfg_in_for_probes)

        # Can not use cfg["out"] as it prepared to save to `*.proc_noAvg.h5`, so we prepare new cfg by modifieng them:
        cfg_out_raw_db = {
            'db_path': cfg['out']['raw_db_path'],
            'tables': cfg['out']['tables']
            # - b_incremental_update: default False
            # todo: check for existence of text data in *.raw.h5 load it if it matches input files metadata
        }
        for itbl, (itbl, pid, path_csv, df_raw) in h5_dispenser_and_names_gen(  # opens output DB
            cfg_in_common,
            cfg_out_raw_db,
            fun_gen=load_from_csv_gen_for_h5_dispenser,
            b_close_at_end=False,
            **cfg_in_for_probes,
        ):  # also gets cfg_out_raw_db['db'] to write, updats cfg['out']['b_remove_duplicates']
            # also works without funcuionality of h5_dispenser_and_names_gen:
            # db_out_raw = pd.HDFStore(cfg['out']['raw_db_path']) if cfg['out']['raw_db_path'] else None

            # for itbl, pid, path_csv, df_raw in load_from_csv_gen(cfg_in_common, cfg_in_for_probes):
            if path_csv_prev != path_csv:
                path_csv_prev = path_csv
                csv_part = 0
            else:
                csv_part += 1  # next part of same csv
            # output table
            tbl_raw = cfg['out']['table'] or pid.replace('i', 'incl') if pid[0] == 'i' else f'incl_{pid}'

            # Configuration for pid: `cfg1` with copy of cfg fields that can be changed in cycle
            cfg1, col_out, pid = get_pid_cfg_(cfg, cfg_in_common, cfg_in_for_probes, tbl_raw)
            cfg1['in']['path_csv'] = path_csv

            lf.warning(
                '{: 2d}. csv {:s}{:s} loaded data processing...',
                itbl, tbl_raw, '' if csv_part is None else f'.{csv_part}'
            )

            df_raw = filter_local(df_raw, cfg1['filter'], ignore_absent={'h_minus_1', 'g_minus_1'})
            d = dd.from_pandas(df_raw, npartitions=1)
            data_len = df_raw.shape[0]

            yield d, cfg1, tbl_raw, col_out, pid, csv_part

            # Save raw data to hdf5 for future simplified loading here or needs of other programs
            if cfg_out_raw_db['db'] is not None:
                if csv_part:
                    # we will save log only when we after last part for input file i.e. no '_part'
                    log_raw['DateEnd'] = d.divisions[-1]
                    log_raw['rows'] += len(d)
                else:  # new file
                    if pid != pid_prev:  # new pid

                        log_raw = {
                            'Date0': d.divisions[0],
                            'DateEnd': None,
                            'fileName': ','.join([
                                f'{p.parent.name}/{p.stem}' for p in cfg1['in']['paths_csv']])[
                                -cfg['out']['logfield_fileName_len']:],
                            'fileChangeTime': datetime.fromtimestamp(
                                path_csv[-1].stat().st_mtime),
                            'rows': data_len
                        }
                    else:
                        cfg1['in']['paths_csv'].append(path_csv)
                tables_written_raw |= h5_append_to(
                    d,
                    tbl_raw,
                    {**cfg_out_raw_db, "db": cfg_out_raw_db["db"], "b_log_ready": True},
                    {} if csv_part else log_raw,
                    msg=f"saving raw data to {tbl_raw}",
                )

        if tables_written_raw:
            # Save info about last loaded csv info log
            # h5_append(cfg_out_mod, pd.DataFrame(), log_raw)
            tables_written_raw |= h5_append_to(
                pd.DataFrame(),
                tbl_raw,
                {**cfg_out_raw_db, "db": cfg_out_raw_db["db"], "b_log_ready": True},
                log_raw,
                msg=f"data saved {tbl_raw}",
            )
            cfg_out_raw_db["db"].close()


# -----------------------------------------------------------------------------------------------------------

@hydra.main(config_name=cs_store_name, config_path='cfg', version_base='1.3')
def main(config: ConfigType) -> Union[None, Mapping[str, Any], pd.DataFrame]:
    """
    Load data from raw csv file or hdf5 table (or group of them)
    Calculate new data or average by specified interval
    Combine this data to new table
    :param config:
    :return:
    """
    global cfg
    if config.input.path is None and config.input.paths is None:
        raise ValueError('At least one of input `path` or `paths` must be provided.')
    cfg = cfg_d.main_init(config, cs_store_name, __file__=None)
    cfg['input']['paths'] = (
        [Path(config.input['path'])] if config.input['paths'] is None
        else [Path(p) for p in config.input['paths']])
    cfg = cfg_d.main_init_input_file(cfg, cs_store_name, msg_action='Loading data from', in_file_field='path')
    lf.info('Begin {:s}(aggregate_period={:s})', this_prog_basename(__file__),
            cfg['out']['aggregate_period'] or 'None'
            )
    aggregate_period_timedelta = pd.Timedelta(pd.tseries.frequencies.to_offset(_)) if (
        _ := ('0s' if (_ := cfg['out']['aggregate_period']) == '0' else _)  # allows "0" input
    ) else None

    # Use physical values from .proc_noAvg db as input to average data > 2s (set cfg['in']['is_counts'])
    _ = cfg['in'].get('aggregate_period_min_to_load_proc', '2s')
    cfg['in']['aggregate_period_min_to_load_proc'] = _ = pd.Timedelta(  # with allowing "0" input
        pd.tseries.frequencies.to_offset(_ if (_ := ('0s' if _ == '0' else _)) else '2s'))

    if aggregate_period_timedelta:
        cfg['in']['is_counts'] = aggregate_period_timedelta <= _
        # If 'split_period' not set use custom splitting to be in memory limits
        cols_out_h5 = ['v', 'u', 'Pressure', 'Temp']  # absent here cols will be ignored

        # Restricting number of counts to 100000 in dask partition to not overflow memory
        split_for_memory = cfg["out"]["split_period"] or next(
            iter(  # timedelta value of biggest unit component + 1
                [
                    f"{t + 1}{c}"
                    for t, c in zip(tuple((100000 * aggregate_period_timedelta).components), "Dhms")
                    if t > 0
                ]
            )
        )

        # Shorter out time format if milliseconds=0, microseconds=0, nanoseconds=0
        if cfg['out']['text_date_format'].endswith('.%f') and not np.any(
            aggregate_period_timedelta.components[-3:]):
            cfg['out']['text_date_format'] = cfg['out']['text_date_format'][:-len('.%f')]
    else:
        cfg['in']['is_counts'] = True
        cols_out_h5 = ['v', 'u', 'Pressure', 'Temp', 'inclination']  # may be  cols will be ignored

        # Restricting dask partition size by time period
        split_for_memory = cfg['out']['split_period'] or pd.Timedelta(1, 'D')
        cfg['out']['aggregate_period'] = None  # 0 to None

    for b_repeat in [False, True]:
        try:
            cfg['in']['paths'] = set_full_paths_and_h5_out_fields(cfg, bool(aggregate_period_timedelta))
            b_input_is_h5 = cfg['in']['paths'][0].suffix in hdf5_suffixes
            break
        except FileNotFoundError:  # No raw db was found
            # Switch to load text data to raw db this run as direct averaging of txt (controlled by
            # aggregate_period_min_to_load_proc) is switched off
            cfg['in']['is_counts'] = True
            aggregate_period_timedelta = None
            # config.hydra.multirun = True  # possible set to run this prog second time here?

    # Set columns from incl_calc_velocity() we need to prepend: Vabs/dir columns needed only to save them in
    # txt
    if cfg['out']['text_path']:
        incl_calc_kwargs = {}  # all its cols
    else:
        _ = [c for c in cols_out_h5 if c not in ('Pressure', 'Temp')]  # ('v', 'u', 'inclination')
        incl_calc_kwargs = {'cols_prepend': _}

    # Filtering values [min/max][M] could be specified with just key M to set same value for keys Mx My Mz
    for lim in ['min', 'max']:
        if 'M' in cfg['filter'][lim]:
            for ch in ('x', 'y', 'z'):
                set_field_if_no(cfg['filter'][lim], f'M{ch}', cfg['filter'][lim]['M'])

    if cfg['program']['dask_scheduler'] == 'distributed':
        from dask.distributed import Client, progress
        # cluster = dask.distributed.LocalCluster(n_workers=2, threads_per_worker=1, memory_limit="5.5Gb")
        client = Client(processes=False)
        # navigate to http://localhost:8787/status to see the diagnostic dashboard if you have Bokeh installed
        # processes=False: avoide inter-worker communication for computations releases the GIL (numpy, da.array)  # without is error
    else:
        if cfg['program']['dask_scheduler'] == 'synchronous':
            lf.warning('using "synchronous" scheduler for debugging')
        import dask
        dask.config.set(scheduler=cfg['program']['dask_scheduler'])
        progress = None
        client = None

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
            for df, cfg1, tbl, col_out, pid, _ in gen_subconfigs(
                    cfg, fun_gen=probes_gen, **cfg['in_many']
                ):
                if dir_cfg_proc:
                    # Save calculating parameters
                    # rename incompatible to OmegaConf "in" name, remove paths as we save for each file
                    cfg1 = {"input": {**cfg1.pop("in"), "tables": [tbl], "paths": None}, **cfg1}
                    conf_ = cfg_d.omegaconf_merge_compatible(cfg1, ConfigType)
                    conf = OmegaConf.create(conf_)

                    # Save to `defaults` dir so we can use them as configs with new defaults to run later
                    while True:  # Make unique file name (can gen_subconfigs yield same pid?)
                        file_stem = file_name_pattern.format(id=pid)
                        if file_stem in file_stems:  # change time-based name
                            file_name_pattern = f"{{id}}_{datetime.now():%Y%m%d_%H%M%S}.yaml"
                            continue
                        file_stems.add(file_stem)
                        break
                    with dir_cfg_proc / file_stem.with_suffix('.yaml') as fp:
                        OmegaConf.save(conf, fp)  # or pickle.dump(conf, fp)
                        # fp.flush()
                out_dicts[str(cfg1['input']['path'])] = cfg1  # [tbl]? cfg_?
            return out_dicts

    # Data processing cycle
    dfs_all_list = []
    tbls = []
    dfs_all: Optional[pd.DataFrame] = None
    cfg['out']['tables_written'] = set()
    tables_written_not_joined = set()
    tbl_prev = pid_prev = None
    for d, cfg1, tbl, col_out, pid, i_tbl_part in gen_subconfigs(
            cfg,
            fun_gen=probes_gen,
            **cfg['in_many']
            ):
        if i_tbl_part:
            probe_continues = True
        else:
            probe_continues = (tbl == tbl_prev and pid == pid_prev)
            tbl_prev = tbl
            pid_prev = pid

        if b_input_is_h5:
            d = filter_local(d, cfg1['filter'], ignore_absent={'h_minus_1', 'g_minus_1'})  # todo: message of filtered especially if all was filt out
        # d[['Mx','My','Mz']] = d[['Mx','My','Mz']].mask(lambda x: x>=4096):
        if cfg['in']['is_counts']:  # Raw data processing
            try:  # prepare to save coefficient dates (some of them can be updated below)
                dates = cfg1['in']['coefs'].pop('dates')
            except (KeyError, AttributeError):
                dates = {}
            msg_rotated = ''
            # Zeroing
            if _ := cfg1['in']['time_ranges_zeroing']:
                _ = coef_zeroing_rotation_from_data(d, _, **cfg1['in']['coefs'])
                if _ is not None:
                    dates['Rz'] = True
                    msg_rotated = ("with new rotation to direction averaged on configured time_ranges_zeroing "
                        "interval ")
                else:
                    lf.debug('time_ranges_zeroing not in current data range')
            # Azimuth correction
            if cfg1["in"]["azimuth_add"] or cfg1["in"]["coordinates"]:
                msgs = ['(coef. {:g})'.format(cfg1['in']['coefs']['azimuth_shift_deg'].item())]
                if cfg1['in']['azimuth_add']:
                    # individual or the same correction for each table:
                    msgs.append('(azimuth_shift_deg {:g})'.format(cfg1['in']['azimuth_add']))
                    cfg1['in']['coefs']['azimuth_shift_deg'] += cfg1['in']['azimuth_add']
                if cfg1["in"]["coordinates"]:  # add magnetic declination at coordinates [Lat, Lon]
                    mag_decl = mag_dec(
                        *cfg1["in"]["coordinates"],
                        d.divisions[0],
                        depth=-1,  # depth has negligible effect if it is less than several km
                    )
                    msgs.append("(magnetic declination {:g})".format(mag_decl))
                    cfg1['in']['coefs']['azimuth_shift_deg'] += mag_decl

                lf.warning('Azimuth correction updated to {:g} = {}°',
                    cfg1['in']['coefs']['azimuth_shift_deg'].item(), ' + '.join(msgs)
                )
                dates['azimuth_shift_deg'] = True
        if aggregate_period_timedelta:
            # Binning
            if cfg['in']['is_counts']:
                lf.warning('Raw data {}-bin averaging before the physical parameters calculation',
                    cfg['out']['aggregate_period'])
            # else loading from {cfg['in']['path'].stem.removesuffix('.raw')}.proc_noAvg.h5
            if aggregate_period_timedelta > np.diff(d.divisions[0::(len(d.divisions)-1)]).item():  # ~ period > data time span
                # convert to one row dask dataframe with index = mean index
                df = d.mean().compute().to_frame().transpose()
                df.index = [d.index.compute().mean()]
                d = dd.from_pandas(df, npartitions=1)
            else:
                counts = (~d.iloc[:, 0].isna()).resample(aggregate_period_timedelta).count()
                d = d.resample(
                    aggregate_period_timedelta,
                    # closed='right' if 'Pres' in cfg['in']['path'].stem else 'left'
                    # 'right' for burst mode because the last value of interval used in wavegauges is round - not true?
                    ).mean()
                # detect number of existed values in each resampled period (using 1st column)
                d = d.where(counts > counts[counts > 0].mean()/10).dropna(how='all')  # axis=0 is default
                try:  # persist speedups calc_velocity greatly but may require too many memory
                    lf.debug('Persisting data aggregated by {:s}', cfg['out']['aggregate_period'])
                    d.persist()  # excludes missed values?
                except MemoryError:
                    lf.info('Persisting failed (not enough memory). Continue...')

            # Recalculating aggregated polar coordinates and angles that are invalid after the direct aggregating
            d = dekart2polar_df_uv(d)
            if cfg['out']['split_period']:  # for csv splitting only
                d = d.repartition(freq=cfg['out']['split_period'])

        if cfg['in']['is_counts']:
            # Apply rotation to Ag & Ah coefficients
            if cfg1['in']['coefs']:
                need_update = (_ := cfg1['in']['coefs'].get('g0xyz')) is not None
                if need_rotate := (cfg1['in']['coefs'].get('Rz') is not None) or need_update:
                    if need_update:  # get Rz from g0xyz
                        cfg1['in']['coefs']['Rz'] = coef_zeroing_rotation(
                            _[:, None], np.float64(cfg1['in']['coefs']['Ag']), cfg1['in']['coefs']['Cg']
                        )
                        dates['Rz'] = True
                        msg_rotated = 'with new rotation to user defined zero point (g0xyz) '

            if not probe_continues and cfg['out']['raw_db_path'] and not b_input_is_h5:
                lf.info(f'Saving coefficients {msg_rotated}to {tbl}')  # save to cfg['out']['raw_db_path']
                # to save manually when b_input_is_h5: stop above and execute below, see 'auto' file in cwd
                h5copy_coef(
                    None, cfg['out']['raw_db_path'], tbl,
                    dict_matrices=coefs_format_for_h5(cfg1['in']['coefs'], pid),
                    dates=dates
                )
            if not (cfg['out']['not_joined_db_path'] or cfg['out']['text_path']):
                continue
            elif cfg1['in']['coefs'] is None:
                if not probe_continues:
                    lf.warning(f'Skipping processing {tbl} because no coefficients was defined')
                continue
            # Apply loaded or calculated rotation
            if need_rotate:
                for c in ['Ag', 'Ah']:
                    cfg1['in']['coefs'][c] = cfg1['in']['coefs']['Rz'] @ cfg1['in']['coefs'][c]
            else:
                lf.info('No rotation coefficient (Rz) loaded. Suppose rotations already applied')

            # Velocity calculation
            # --------------------
            try:  # with repartition for ascii splitting (also helps to prevent MemoryError)
                d = d.repartition(freq=split_for_memory)
            except TypeError as e:
                lf.error('No data? Data length = {}, skipping {}. Continue...', len(d.index), tbl)
                continue
            d = incl_calc_velocity(
                d,
                cfg_proc=cfg1["in"],
                filt_max=cfg1["filter"]["max"],
                **cfg1["in"]["coefs"],
                **incl_calc_kwargs,
            )
            d = calc_pressure(              # [c for c in cols_out_h5 if c in d.columns] if cfg['out']['text_path']
                d, **cfg1['in']['coefs']    # no redundant cols if not need save them to txt
            )                               # **{(pb := 'bad_p_at_bursts_starts_period'): cfg['filter'][pb]},

            # Write velocity to h5 - for each probe (group) in separated table
            if cfg['out']['not_joined_db_path']:
                try:  # select real file (that replaces the files pattern if it was used (as default raw_db_path)):
                    path_in = cfg1['in']['path'] if b_input_is_h5 else cfg1['out']['raw_db_path']
                except KeyError:  # need?
                    try:
                        path_in = cfg1['in']['path_csv']
                    except KeyError:
                        path_in = cfg1['in']['path']  # should be files pattern: not real file
                log = {
                    'Date0': d.divisions[0],
                    'DateEnd': d.divisions[-1],
                    'fileName': f"{path_in.parent.name}/{path_in.stem}"[-cfg['out']['logfield_fileName_len']:],
                    'fileChangeTime': datetime.fromtimestamp(path_in.stat().st_mtime),
                    'rows': len(d)
                    }
                tables_written_not_joined |= (
                    h5_append_to(
                        d, tbl, cfg['out'], log,
                        msg=f'saving {tbl} to temporary store'
                    )
                )
        probe_continue_file_name = dd_to_csv(
            d, cfg['out']['text_path'], cfg['out']['text_date_format'], cfg['out']['text_columns'],
            cfg['out']['aggregate_period'], suffix=f'@{col_out}',
            single_file_name=probe_continue_file_name if probe_continues else not cfg['out']['split_period'],
            progress=progress, client=client, b_continue=probe_continues
        )

        # Combine data columns if we aggregate (in such case all data have index of equal period)
        if aggregate_period_timedelta:
            try:
                cols_save = [c for c in cols_out_h5 if c in d.columns]
                sleep(cfg["program"]["sleep_s"])
                Vne = d[cols_save].compute()  # MemoryError((1, 12400642), dtype('float64'))

                if not cfg['out']['b_all_to_one_col']:
                    Vne.rename(
                        columns=map_to_suffixed(cols_save, col_out),
                        inplace=True
                    )
                dfs_all_list.append(Vne)
                tbls.append(tbl)
            except Exception as e:
                lf.exception('Can not cumulate result! ')
                raise
                # todo: if low memory do it separately loading from temporary tables in chanks

        gc.collect()  # frees many memory. Helps to not crash

    # Combined data to hdf5
    #######################

    if aggregate_period_timedelta:
        if dfs_all_list:
            # Concatenate several columns by one of method:
            # - consequently, like 1-probe data or
            # - parallel (add columns): dfs_all without any changes
            dfs_all = pd.concat(
                dfs_all_list, sort=True, axis=(0 if cfg['out']['b_all_to_one_col'] or probe_continues else 1)
                )
            dfs_all_log = pd.DataFrame(
                [df.index[[0, -1]].to_list() for df in dfs_all_list], columns=['Date0', 'DateEnd']
                ).assign(table_name=tbls) #.set_index('table_name')\
                #.sort_index() #\

            for after_remove_dup_index in [False, True]:
                try:
                    cfg['out']['tables_written'] |= (
                        h5_append_to(dfs_all, cfg['out']['table'], cfg['out'], log=dfs_all_log,
                                     msg='Saving accumulated data'
                                     )
                        )
                    break
                except ValueError as e:
                    # ValueError: cannot reindex from a duplicate axis
                    if dfs_all.index.is_unique and dfs_all_log.index.is_unique:
                        lf.exception('Can not understand problem. Skipping saving to hdf5!')
                        break
                    lf.error('Removing duplicates in index')
                    dfs_all = dfs_all.loc[~dfs_all.index.duplicated(keep='last')]
        else:
            lf.warning('No data found')
    h5_close(cfg['out'])  # close temporary output store
    if tables_written_not_joined:
        try:
            failed_storages = h5move_tables(
                {**cfg['out'], 'db_path': cfg['out']['not_joined_db_path'], 'b_del_temp_db': False},
                tables_written_not_joined
            )
        except Ex_nothing_done as e:
            lf.warning('Tables {} of separate data not moved', tables_written_not_joined)
    if cfg['out']['tables_written']:
        try:
            failed_storages = h5move_tables(cfg['out'], cfg['out']['tables_written'])
        except Ex_nothing_done as e:
            lf.warning('Tables {} of combined data not moved', cfg['out']['tables_written'])

        # Write concatenated dataframe to ascii (? with resample if b_all_to_one_col)
        if dfs_all is not None and len(dfs_all_list) > 1:
            call_with_valid_kwargs(
                dd_to_csv,
                (lambda x:
                    x.resample(rule=aggregate_period_timedelta)
                    .first() if cfg['out']['b_all_to_one_col'] else x
                 )(dd.from_pandas(dfs_all, chunksize=500000)),  # .fillna(0): absent values filling with 0  ???
                **cfg['out'],
                suffix=f"@{','.join(tbls)}",  # t.format() for t in cfg['in']['tables']
                progress=progress, client=client
            )

    print('Ok.', end=' ')
    return dfs_all


if __name__ == '__main__':
    main()

r"""
    # h5index_sort(cfg['out'], out_storage_name=f"{cfg['out']['db_path'].stem}-resorted.h5", in_storages= failed_storages)
    # dd_out = dd.multi.concat(dfs_list, axis=1)

# old coefs uses:
da.degrees(da.arctan2(
(Gxyz[0, :] * Hxyz[1, :] - Gxyz[1, :] * Hxyz[0, :]) * (GsumMinus1 + 1),
Hxyz[2, :] * da.square(Gxyz[:-1, :]).sum(axis=0) - Gxyz[2, :] * ((Gxyz[:-1, :] * Hxyz[:-1, :]).sum(axis=0))))

else:
    Vdir = da.zeros_like(HsumMinus1)
    lf.warning('Bad Vdir: set all to 0 degrees')

a.drop(set(a.columns).difference(columns + [col]), axis=1)
for c, ar in zip(columns, arrays_list):
    # print(c, end=' ')
    if isinstance(ar, da.Array):
        a[c] = ar.to_dask_dataframe(index=a.index)
    else:
        a[c] = da.from_array(ar, chunks=GsumMinus1.chunks).to_dask_dataframe(index=a.index)
        #dd.from_array(ar, chunksize=int(np.ravel(GsumMinus1.chunksize)), columns=[c]).set_index(a.index) ...??? - not works

df = dd.from_dask_array(arrays, columns=columns, index=a.index)  # a.assign(dict(zip(columns, arrays?)))    #
if ('Pressure' in a.columns) or ('Temp' in a.columns):
    df.assign = df.join(a[[col]])

# Adding column of other (complex) type separatly
# why not works?: V = df['Vabs'] * da.cos(da.radians(df['Vdir'])) + 1j*da.sin(da.radians(df['Vdir']))  ().to_frame() # v + j*u # below is same in more steps
V = polar2dekart_complex(Vabs, Vdir)
V_dd = dd.from_dask_array(V, columns=['V'], index=a.index)
df = df.join(V_dd)

df = pd.DataFrame.from_records(dict(zip(columns, [Vabs, Vdir, np.degrees(incl_rad)])), columns=columns, index=tim)  # no sach method in dask



    dfs_all = pd.merge_asof(dfs_all, Vne, left_index=True, right_index=True,
                  tolerance=pd.Timedelta(cfg['out']['aggregate_period'] or '1ms'),
                            suffixes=('', ''), direction='nearest')
    dfs_all = pd.concat((Vne, how='outer')  #, rsuffix=tbl[-2:] join not works on dask
    V = df['V'].to_frame(name='V' + tbl[-2:]).compute()
if dfs_all is computed it is in memory:
mem = dfs_all.memory_usage().sum() / (1024 ** 2)
if mem > 50:
    lf.debug('{:.1g}Mb of data accumulated in memory '.format(mem))

df_to_csv(df, cfg_out, add_subdir='V,P_txt')
? h5_append_cols()
df_all = dd.merge(indiv, cm.reset_index(), on='cmte_id')


old cfg

    cfg = {  # how to load:
        'in': {
            'db_path': '/mnt/D/workData/BalticSea/181116inclinometer_Schuka/181116incl.h5', #r'd:\WorkData\BalticSea\181116inclinometer_Schuka\181116incl.h5',
            'tables': ['incl.*'],
            'chunksize': 1000000, # 'chunksize_percent': 10,  # we'll repace this with burst size if it suit
            'min_date': datetime.strptime('2018-11-16T15:19:00', '%Y-%m-%dT%H:%M:%S'),
            'max_date': datetime.strptime('2018-12-14T14:35:00', '%Y-%m-%dT%H:%M:%S')
            'split_period': '999D',  # pandas offset string (999D, H, ...) ['D' ]
            'aggregate_period': '2H',  # pandas offset string (D, 5D, H, ...)
            #'max_g_minus_1' used only to replace bad with NaN
        },
        'out': {
            'db_path': '181116incl.proc.h5',
            'table': 'V_incl',

    },
        'program': {
            'log': str(scripts_path / 'log/incl_h5clc.log'),
            'verbose': 'DEBUG'
        }
    }

    # optional external coef source:
    # cfg['out']['db_coef_path']           # defaut is same as 'db_path'
    # cfg['out']['table_coef'] = 'incl10'  # defaut is same as 'table'
"""
