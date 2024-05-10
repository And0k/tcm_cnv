"""
autocalibration of all found probes. export coefficients to hdf5.
Does not align (rotate) coefficient matrix to North / Gravity.
Need check if rotation exist?:
                # Q, L = A.diagonalize() # sympy
                # E_w, E_v = np.linalg.eig(E)
Probes data table contents: index,Ax,Ay,Az,Mx,My,Mz
"""
from dataclasses import dataclass, field
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
from scipy import linalg, stats
import dask.dataframe as dd

# from omegaconf import MISSING
import hydra

if __debug__:
    import matplotlib

    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['figure.figsize'] = (16, 7)
    try:
        matplotlib.use(
            'Qt5Agg')  # must be before importing plt (raises error after although documentation sed no effect)
    except ImportError:
        pass
    from matplotlib import pyplot as plt

    matplotlib.interactive(True)
    plt.style.use('bmh')

# import my functions:
from .cfg_dataclasses import hydra_cfg_store, ConfigInHdf5_Simple, ConfigProgram, main_init, main_init_input_file  #  ConfigProgram is used indirectly
from .h5_dask_pandas import filter_local
from .h5toh5 import h5load_ranges

from .incl_h5clc_hy import ConfigIn_InclProc, incl_calc_velocity_nodask, gen_subconfigs, probes_gen, norm_field
from .h5inclinometer_coef import channel_cols, dict_matrices_for_h5, h5copy_coef

from .utils2init import this_prog_basename, standard_error_info, my_logging
from .filters_scipy import despike
from .graphics import make_figure

lf = my_logging(__name__)  # LoggingStyleAdapter(logging.getLogger(__name__))
VERSION = '0.0.1'
hydra.output_subdir = 'cfg'

# import my functions:
drive_d = Path('D:/' if sys.platform == 'win32' else '/mnt/D')  # allows to run on both my Linux and Windows systems
import_dir = drive_d.joinpath('Work/_Python3/And0K/h5toGrid/scripts/third_party')
sys.path.append(str(Path(import_dir).parent.resolve()))
from third_party.ellipsoid_fit import ellipsoid_fit

# @dataclass hydra_conf(hydra.conf.HydraConf):
#     run: field(default_factory=lambda: defaults)dir
# hydra.conf.HydraConf.output_subdir = 'cfg'
# hydra.conf.HydraConf.run.dir = './outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}'


@dataclass
class ConfigFilterComponent:
    """
    Filter values specific to components, will overwrite common values of inherited ConfigFilter class next to this
    """
    blocks: Optional[List[int]] = field(default_factory=lambda: [21, 7])
    offsets: Optional[List[float]] = field(default_factory=lambda: [1.5, 2])
    std_smooth_sigma: Optional[float] = 4

@dataclass
class ConfigFilterChannel(ConfigFilterComponent):
    x: Optional[ConfigFilterComponent] = field(default_factory=lambda: ConfigFilterComponent(None, None, None))
    y: Optional[ConfigFilterComponent] = field(default_factory=lambda: ConfigFilterComponent(None, None, None))
    z: Optional[ConfigFilterComponent] = field(default_factory=lambda: ConfigFilterComponent(None, None, None))
    # = {c: ConfigFilterComponent(None, None, None) for c in 'xyz'}?

@dataclass
class ConfigFilter(ConfigFilterComponent):
    """
    "filter": excludes some data:

    no_works_noise: is_works() noise argument for each channel: excludes data if too small changes
    blocks: List[int] ='21, 7' despike() argument
    offsets: List[float] despike() argument
    std_smooth_sigma: float = 4, help='despike() argument
    """
    # Optional[Dict[str, float]] = field(default_factory= dict) leads to .ConfigAttributeError/ConfigKeyError: Key 'Sal' is not in struct
    min_date: Optional[Dict[str, str]] = field(default_factory=dict)
    max_date: Optional[Dict[str, str]] = field(default_factory=dict)
    A: Optional[ConfigFilterChannel] = field(default_factory=lambda: ConfigFilterChannel)
    M: Optional[ConfigFilterChannel] = field(default_factory=lambda: ConfigFilterChannel)
    no_works_noise: Dict[str, float] = field(default_factory=lambda: {'M': 10, 'A': 100})


@dataclass
class ConfigOut:
    """
    "out": all about output files:

    db_paths: hdf5 stores paths where to write resulting coef. Tables will be the same as configured for input data
    (cfg[in].tables)
    """
    # raw_db_path: Any = 'auto'  # will be the parent of input.path dir for text files
    table: str = ''         # not used but required by incl_h5clc_hy.gen_subconfigs()
    table_log: str = ''     # not used but required by incl_h5clc_hy.gen_subconfigs()
    db_paths: Optional[List[str]] = field(default_factory=list)


# @dataclass
# class DictTimeRange(ConfigInHdf5_Simple):
#     03: List[str] = field(default_factory=list)


@dataclass
class ConfigInHdf5_InclCalibr(ConfigIn_InclProc):
    """
    Same as ConfigInHdf5_Simple + specific (inclinometer calibration) data properties:
    channels: List: (, channel can be "magnetometer" or "M" for magnetometer and any else for accelerometer',
    chunksize: limit loading data in memory (default='50000')
    time_range: time range to use
    time_ranges: time range to use for each inclinometer number (consisted of digits in table name), overwrites time_range
    time_range_nord_list: time range to zeroing north. Not zeroing Nord if not used')
    time_range_nord_dict: time range to zeroing north for each inclinometer number (consisted of digits in table name)')
    """
    # path_cruise: Optional[str] = None  # mainly for compatibility with incl_load but allows db_path be relative to this

    channels: List[str] = field(default_factory=lambda: ['M', 'A'])
    chunksize: int = 50000
    time_range: List[str] = field(default_factory=list)
    time_ranges: Optional[Dict[str, Any]] = field(default_factory=dict)  # Dict[int, str] not supported in omegaconf
    time_range_nord: List[str] = field(default_factory=list)
    time_range_nord_dict: Dict[str, str] = field(default_factory=dict)  # Dict[int, str] not supported in omegaconf
    # Dict[Optional[str], Optional[List[str]]]
    
@dataclass
class ConfigProgramSt(ConfigProgram):
    step_start: int = 1
    step_end: int = 1


cs_store_name = Path(__file__).stem
cs, ConfigType = hydra_cfg_store(
    cs_store_name, {
        'input': [ConfigInHdf5_InclCalibr],  # Load the config ConfigInHdf5_InclCalibr to the config group "input"
        'out': [ConfigOut],  # Set as MISSING to require the user to specify a value on the command line.
        'filter': [ConfigFilter],
        'program': [ConfigProgramSt],
    },
    module=sys.modules[__name__]
    )


def fG(Axyz, Ag, Cg):
    return Ag @ (Axyz - Cg)


def filter_channes(
        a3d: np.ndarray, a_time=None, fig=None, fig_save_prefix=None, window_title=None,
        blocks=(21, 7), offsets=(1.5, 2), std_smooth_sigma=4,
        x: Mapping[str, Any] = None, y: Mapping[str, Any] = None, z: Mapping[str, Any] = None,
        **kwargs) -> Tuple[np.ndarray, np.ndarray, matplotlib.figure.Figure]:
    """
    Filter back and forward each column of a3d by despike()
    despike a3d - 3 channels of data and plot data and overlapped results

    :param a3d: shape = (3,len)
    :param a_time: x data to plot. If None then use range(len)
    :param fig:
    :param fig_save_prefix: save figure to this path + 'despike({ch}).png' suffix
    :param blocks: filter window width - see despike()
    :param offsets: offsets to std - see despike(). If empty then only filters NaNs.
    Note: filters too many if set some item < 3.
    :param std_smooth_sigma - see despike()
    :param x: blocks, offsets, std_smooth_sigma dict for channel x
    :param y: blocks, offsets, std_smooth_sigma dict for channel y
    :param z: blocks, offsets, std_smooth_sigma dict for channel z
    :param window_title: str
    :return: a3d[ :,b_ok], b_ok
    """
    args = locals()
    dim_length = 1   # dim_channel = 0
    blocks = np.minimum(blocks, a3d.shape[dim_length])
    b_ok = np.ones((a3d.shape[dim_length],), np.bool_)
    if fig:
        fig.axes[0].clear()
        ax = fig.axes[0]
    else:
        ax = None

    if a_time is None:
        a_time = np.arange(a3d.shape[1])

    for i, (ch, a) in enumerate(zip(('x', 'y', 'z'), a3d)):
        ax_title = f'despike({ch})'
        ax, lines = make_figure(x=a_time, y_kwrgs=((
            {'data': a, 'label': 'source', 'color': 'r', 'alpha': 1},
            )), ax_title=ax_title, ax=ax, lines='clear', window_title=window_title)
        b_nan = np.isnan(a)
        n_nans_before = b_nan.sum()
        b_ok &= ~b_nan

        if len(offsets):
            # back and forward:
            a_f = np.float64(a[b_ok][::-1])

            cfg_filter_component = {
                p: (args[p] if (ach := args[ch][p]) is None else ach) for p in ['offsets', 'blocks', 'std_smooth_sigma']
                }

            a_f, _ = despike(a_f, **cfg_filter_component, x_plot=a_time)
            a_f, _ = despike(a_f[::-1], **cfg_filter_component, x_plot=a_time[b_ok], ax=ax, label=ch)

            b_nan[b_ok] = np.isnan(a_f)
            n_nans_after = b_nan.sum()
            b_ok &= ~b_nan

            # ax, lines = make_figure(y_kwrgs=((
            #     {'data': a, 'label': 'source', 'color': 'r', 'alpha': 1},
            # )), mask_kwrgs={'data': b_ok, 'label': 'filtered', 'color': 'g', 'alpha': 0.7}, ax=ax,
            #     ax_title=f'despike({ch})', lines='clear')

            ax.legend(prop={'size': 10}, loc='upper right')
            lf.info('despike({ch:s}, offsets={offsets}, blocks={blocks}): deleted={dl:d}'.format(
                    ch=ch, dl=n_nans_after - n_nans_before, **cfg_filter_component))
        plt.show()
        if fig_save_prefix:  # dbstop
            try:
                ax.figure.savefig(fig_save_prefix + (ax_title + '.png'), dpi=300, bbox_inches="tight")
            except Exception as e:
                lf.warning(f'Can not save fig: {standard_error_info(e)}')
        # Dep_filt = rep2mean(a_f, b_ok, a_time)  # need to execute waveletSmooth on full length

    # ax.plot(np.flatnonzero(b_ok), Depth[b_ok], color='g', alpha=0.9, label=ch)
    return a3d[:, b_ok], b_ok, ax.figure


def calc_vel_flat_coef(coef_nested: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """ Convert coef_nested to format of incl_calc_velocity() args"""
    arg = {}
    for ch, coefs in coef_nested.items():
        sfx = channel_cols(ch)[1].lower()
        for key, val in coefs.items():
            arg[f"{'C' if key == 'b' else 'A'}{sfx}"] = val
    return arg


def str_range(ranges, ind):
    return "'{}'".format(', '.join(f"'{t}'" for t in ranges[ind])) if ind in ranges else ''


# for copy/paste ##########################################################
def plotting(a):
    """
    plot source
    :param a:
    :return:
    """

    plt.plot(a['Mx'])  # , a['My'], a['Mz'])

    if False:
        msg = 'Loaded ({})!'.format(a.shape)
        fig = plt.figure(msg)
        ax1 = fig.add_subplot(112)

        plt.title(msg)
        plt.plot(a['Hx'].values, color='b')
        plt.plot(a['Hy'].values, color='g')
        plt.plot(a['Hz'].values, color='r')


def axes_connect_on_move(ax, ax2):
    canvas = ax.figure.canvas

    def on_move(event):
        if event.inaxes == ax:
            ax2.view_init(elev=ax.elev, azim=ax.azim)
        elif event.inaxes == ax2:
            ax.view_init(elev=ax2.elev, azim=ax2.azim)
        else:
            return
        canvas.draw_idle()

    c1 = canvas.mpl_connect('motion_notify_event', on_move)
    return c1


def plotEllipsoid(center, radii, rotation, ax=None, plotAxes=False, cageColor='b', cageAlpha=0.2):
    """Plot an ellipsoid"""
    make_ax = ax == None
    if make_ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)

    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    if plotAxes:
        # make some purdy axes
        axes = np.array([[radii[0], 0.0, 0.0],
                         [0.0, radii[1], 0.0],
                         [0.0, 0.0, radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)

        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cageColor)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)

    if make_ax:
        plt.show()
        plt.close(fig)
        del fig


def fit_quadric_form(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
     Estimate quadric form parameters from a set of points.
    :param s: array_like
          The samples (M,N) where M=3 (x,y,z) and N=number of samples.
    :return: M, n, d : The quadric form parameters in: h.T*M*h + h.T*n + d = 0

    References
    ----------
    .. [1] Qingde Li; Griffiths, J.G., "Least squares ellipsoid specific fitting," in
    Geometric Modeling and Processing, 2004. Proceedings, vol., no., pp.335-340, 2004
    Source
    ------
    https://teslabs.com/articles/magnetometer-calibration/
    """

    # D (samples)
    D = np.array([
        s[0] ** 2,
        s[1] ** 2,
        s[2] ** 2,
        2 * s[1] * s[2],
        2 * s[0] * s[2],
        2 * s[0] * s[1],
        2 * s[0],
        2 * s[1],
        2 * s[2],
        np.ones_like(s[0])
    ])
    # S, S_11, S_12, S_21, S_22 (eq. 11)
    S = np.dot(D, D.T)
    S_11 = S[:6, :6]
    S_12 = S[:6, 6:]
    S_21 = S[6:, :6]
    S_22 = S[6:, 6:]

    # inv(C) (Eq. 8, k=4)
    c_inv = np.array(              # C = np.array([
        [[0, 0.5, 0.5, 0, 0, 0],   # [-1, 1, 1, 0, 0, 0],
         [0.5, 0, 0.5, 0, 0, 0],   # [ 1,-1, 1, 0, 0, 0],
         [0.5, 0.5, 0, 0, 0, 0],   # [ 1, 1,-1, 0, 0, 0],
         [0, 0, 0, -0.25, 0, 0],   # [ 0, 0, 0,-4, 0, 0],
         [0, 0, 0, 0, -0.25, 0],   # [ 0, 0, 0, 0,-4, 0],
         [0, 0, 0, 0, 0, -0.25]])  # [ 0, 0, 0, 0, 0,-4]])
    
    if not linalg.det(S_22):
        print('Singular matrix: no unique solution.')  # numpy.linalg.LinAlgError is going below
        
    # v_1 (eq. 15, solution)
    E = np.dot(c_inv, S_11 - np.dot(S_12, np.dot(linalg.inv(S_22), S_21)))

    E_w, E_v = np.linalg.eig(E)

    v_1 = E_v[:, np.argmax(E_w)]
    if v_1[0] < 0:
        v_1 = -v_1

    # v_2 (eq. 13, solution)
    v_2 = np.dot(np.dot(-np.linalg.inv(S_22), S_21), v_1)

    # quadric-form parameters
    M = v_1[np.array([[0, 5, 4],
                      [5, 1, 3],
                      [4, 3, 2]], np.int8)]
    n = v_2[:-1, np.newaxis]
    d = v_2[3]
    
    
    # Modified according to Robert R - todo: check:
    # I believe in your code example your M is incorrect. Based on your notation in your Quadric section you have
    # [[a f g],
    #  [f b h],
    #  [g h c]] as your M matrix. A simple check here is that in the D matrix, your 5th element
    # is your 2XY term. This term should go in the h positions as per
    # [[a h g],
    #  [h b f],
    #  [g f c]], instead you have the XY term assigned in the f positions.
    # The overall result is that your A_1 matrix will have "mirrored" column 1 and row 1.
    # Interestingly enough, this flip doesn't seem to impact the calibration significantly.
    
    return M, n, d


def calibrate(raw3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param raw3d:
    :return: (A, b): combined scale factors and combined bias
    """

    # Expected earth magnetic field intensity, default=1.
    F = np.float64(1)
    # initialize values  # b = np.zeros([3, 1])  # A_1 = np.eye(3)

    # Ellipsoid fit
    mean_Hxyz = np.mean(raw3d, 1)[:, np.newaxis]  # dfcum[['Hx', 'Hy', 'Hz']].mean()
    s = np.array(raw3d - mean_Hxyz)
    try:
        Q, n, d = fit_quadric_form(s)  # Q= A.T*A.inv, n= -2*Q*b, d= b.T*n/2 - F**2
    except np.linalg.LinAlgError:  # singular matrix
        _ = [ch for ch, vals in zip('xyz', s) if not any(np.diff(vals, 2))]
        if any(_):
            print('Same values in channel(s):', _)
            if len(_) == 1:
                print(f'Getting coefs for channels with changing values using its max and min')
                i_ch_bad = 'xyz'.index(_[0])
                i_ch_ok = [i for i in range(3) if i != i_ch_bad]
                ch_ok_min = s[i_ch_ok].min(axis=1)
                ch_ok_max = s[i_ch_ok].max(axis=1)
                center = np.mean([ch_ok_min, ch_ok_max], axis=0)
                
                # create transformation matrix
                t = np.reciprocal((ch_ok_max - ch_ok_min)/2).tolist()
                t.insert(i_ch_bad, 0)
                a2d = np.diag(t)
                b = np.zeros((3, 1))
                b[i_ch_ok, 0] = center + mean_Hxyz[i_ch_ok, 0]
                return a2d, b
                # pv = ls_ellipse(s[i_ch_ok, ])
                # center, gains, tilt, R = ellipse_params_from_poly(pv, verbose=True)
            else:
                raise
        else:
            raise
        # center, evecs, radii = ellipsoid_fit()
    
    # Calibration parameters

    Q_inv = linalg.inv(Q)
    # combined bias:           
    b = -np.dot(Q_inv, n) + mean_Hxyz
    # scale factors, soft iron, and misalignment:
    # note: some implementations of sqrtm return complex type, taking real
    a2d = np.real(F / np.sqrt(np.dot(n.T, np.dot(Q_inv, n)) - d) * linalg.sqrtm(Q))

    return a2d, b


def coef2str(a2d: np.ndarray, b: np.ndarray) -> Tuple[str, str]:
    """
    Numpy text representation of matrix a2d and vector b
    :param a2d:
    :param b:
    :return:
    """
    A1e4 = np.round(np.float64(a2d) * 1e4, 1)
    A_str = 'float64([{}])*1e-4'.format(
        ',\n'.join(
            ['[{}]'.format(','.join(str(A1e4[i, j]) for j in range(a2d.shape[1]))) for i in range(a2d.shape[0])]))
    b_str = 'float64([[{}]])'.format(','.join(str(bi) for bi in b.flat))
    return A_str, b_str


def calibrate_plot(raw3d: np.ndarray, a2d: np.ndarray, b, fig=None, window_title=None, clear=True,
                   raw3d_other=None, raw3d_other_color='r'):
    """

    :param raw3d:
    :param a2d:
    :param b:
    :param fig:
    :param window_title:
    :param clear:
    :param raw3d_other
    :return:
    """
    make_fig = fig is None
    if make_fig:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'})
    else:
        ax1, ax2 = fig.axes
        if clear:
            ax1.clear()
            ax2.clear()
    if window_title:  # not works
        man = plt.get_current_fig_manager()
        man.canvas.setWindowTitle(window_title)
    # output data:
    s = norm_field(raw3d, a2d, b)  # s[:,:]

    # Calibrated magnetic field measurements plotted on the
    # sphere manifold whose radius equals 1
    # the norm of the local Earth’s magnetic field

    # ax = axes3d(fig)
    # ax1 = fig.add_subplot(121, projection='3d')
    marker_size = 5  # 0.2
    ax1.set_title('source')

    raw3d_norm = np.linalg.norm(raw3d, axis=0)
    ax1.scatter(*raw3d, c=abs(np.mean(raw3d_norm) - raw3d_norm), marker='.', s=marker_size)  # сolor='k'
    if raw3d_other is not None:
        ax1.scatter(
            xs=raw3d_other[0, :], ys=raw3d_other[1, :], zs=raw3d_other[2, :], c=raw3d_other_color, s=4, marker='.'
            )
        # , alpha=0.1) # dfcum['Hx'], dfcum['Hy'], dfcum['Hz']
    # plot sphere
    # find the rotation matrix and radii of the axes
    try:
        U, c, rotation = linalg.svd(linalg.inv(a2d))
        b_plot_ellipsoid = True
    except np.linalg.LinAlgError:
        print('LinAlgError("singular matrix"): can not find spherical parameters. Continue...')
        b_plot_ellipsoid = False
    if b_plot_ellipsoid:
        radii = c
        plotEllipsoid(b.flatten(), radii, rotation, ax=ax1, plotAxes=True, cageColor='r', cageAlpha=0.1)

    # ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('calibrated')
    # plot points
    s_norm = np.linalg.norm(s, axis=0)
    ax2.scatter(*s, c=abs(1 - s_norm), marker='.', s=marker_size)  # , alpha=0.2  # s is markersize,

    axes_connect_on_move(ax1, ax2)
    # plot unit sphere
    center = np.zeros(3, float)
    rotation = np.diag(np.ones(3, float))
    radii = np.ones(3, float)
    plotEllipsoid(center, radii, rotation, ax=ax2, plotAxes=True)

    # if make_fig:
    #     plt.show()
    #     plt.close(fig); del fig
    #     return None
    return fig


def zeroing_azimuth(store: pd.HDFStore, tbl, time_range_nord, coefs=None, cfg_in=None,
                    filter_query='10 < inclination & inclination < 170') -> float:
    """
    Get correction of azimuth by:
    1. calculating velocity (u, v) in ``time_range_nord`` interval of tbl data using coefficients to be adjusted,
    2. filtering with ``filter_query`` and taking median,
    3. calculating direction,
    4. multiplying result by -1.
    :param time_range_nord:
    :param store: opened pandas HDFStore: interface to its objects in PyTables hdf5 store
    :param tbl: table name in store
    :param coefs: dict with fields having values of array type with sizes:
    'Ag': (3, 3), 'Cg': (3, 1), 'Ah': (3, 3), 'Ch': array(3, 1), 'azimuth_shift_deg': (1,), 'kVabs': (n,)
    :param cfg_in: dict with fields:
        - time_range_nord
        - other, needed in h5load_ranges() and optionally in incl_calc_velocity_nodask()
    :param filter_query: upply this filter query to incl_calc_velocity*() output before mean azimuth calculation
    :return: azimuth_shift_deg: degrees
    """
    lf.debug('Zeroing Nord direction')
    df = h5load_ranges(store, table=tbl, t_intervals=time_range_nord)
    if df.empty:
        lf.info('Zero calibration range out of data scope')
        return
    dfv = incl_calc_velocity_nodask(
        df, **coefs, cfg_filter=cfg_in, cfg_proc={
            'calc_version': 'trigonometric(incl)', 'max_incl_of_fit_deg': 70
            })
    dfv.query(filter_query, inplace=True)
    dfv_mean = dfv.loc[:, ['u', 'v']].median()
    # or df.apply(lambda x: [np.mean(x)], result_type='expand', raw=True)
    # df = incl_calc_velocity_nodask(dfv_mean, **calc_vel_flat_coef(coefs), cfg_in=cfg_in)

    # coefs['M']['A'] = rotate_z(coefs['M']['A'], dfv_mean.Vdir[0])
    azimuth_shift_deg = -np.degrees(np.arctan2(*dfv_mean.to_numpy()))
    lf.info('Nord azimuth shifting coef. found: {:f} degrees', azimuth_shift_deg)
    return azimuth_shift_deg


def bin_avg_3d_partial(index: np.ndarray | pd.DatetimeIndex, vec3d: np.ndarray, bins=200):
    """
    Reduce nonuniform distribution of data points by replacing where more than 1 point per bin to the bin average.
    :param index: time coord of points
    :param vec3d: 3D coordinates of points
    :param bins: number of spatial bins (see scipystats.binned_statistic_2d()) of one coordinate
    :return: (index, vec3d, raw3d_other, n_in_cell) where
    - index: original index with removed averaged points in same bins faced later,
    - raw3d_other: original points that are replaced by averaging,
    - n_in_cell: number of averaged points in bin
    - vec3d, original points if one per bin and averaged else, sorted in order of index.
    """
    rtp = xyz2spherical(vec3d)  # spherical coord (radius, theta, phi)
    # We will set u=cos(phi) to be uniformly distributed (so we have du=sin(phi)d(phi))

    # Compare required number of bins with recommended minimum to get 1 point in 2D if points positioned nealy ideal
    bins_min_recommended = int(np.sqrt(rtp.shape[1]))
    if bins is None:
        bins = bins_min_recommended
    elif bins > bins_min_recommended:
        # should be warning?
        lf.info('Bins/coordinate ({bins}) > maximum for our data size to can fill each bin ({bins_min})!',
                {'bins': bins, 'bins_min': bins_min_recommended, 'filter_same': 6})

    # Bin statistics
    bin_stats = stats.binned_statistic_2d(
        rtp[1, :], np.cos(rtp[-1, :]), rtp[0, :], bins=bins, range=[[0, 2 * np.pi], [-1, 1]])

    # Replace where more than 1 point per bin to the bin average
    i_sort = np.argsort(bin_stats.binnumber)
    vec3d = vec3d[:, i_sort]
    i_bins_sorted = bin_stats.binnumber[i_sort]
    i_bins_sorted_change = np.hstack((0, np.flatnonzero(np.diff(i_bins_sorted)) + 1, len(i_sort)))
    b_need_mean = (n_in_cell := np.diff(i_bins_sorted_change)) > 1
    intervals_avg = np.column_stack((i_bins_sorted_change[:-1], i_bins_sorted_change[1:]))
    # for display: all original points that will be replaced by averaging
    raw3d_other = vec3d[:, np.hstack([np.arange(*i_st_en) for i_st_en in intervals_avg[b_need_mean]])]
    # intact points that alone in grid cell and averaging other points in cell
    vec3d = np.column_stack([vec3d[:, slice(*i_st_en)].mean(axis=1) if b_need else vec3d[:, i_st_en[0]] for
                             i_st_en, b_need in zip(intervals_avg, b_need_mean)])
    index = index[i_sort[i_bins_sorted_change[:-1]]]  # starts of vec3d
    index = index[i_sort := np.argsort(index)]
    vec3d = vec3d[:, i_sort]
    return index, vec3d, raw3d_other, n_in_cell


def xyz2spherical(xyz):
    """
    Cartesian to spherical coordinates
    :param xyz: 3xN array
    :return: rtp (radius, theta, phi)
    """
    rtp = np.zeros_like(xyz)
    xy = xyz[0, :]**2 + xyz[1, :]**2
    rtp[0, :] = np.sqrt(xy + xyz[2, :] ** 2)
    rtp[1, :] = np.arctan2(np.sqrt(xy), xyz[2, :])  # for elevation angle defined from Z-axis down
    #rtp[1, :] = np.arctan2(xyz[2, :], np.sqrt(xy))  # for elevation angle defined from XY-plane up
    rtp[2, :] = np.arctan2(xyz[1, :], xyz[0, :])
    return rtp



cfg = {}

@hydra.main(config_name=cs_store_name, config_path='cfg', version_base='1.3')  # adds config store cs_store_name data/structure
def main(config: ConfigType) -> None:
    """
    ----------------------------
    Calculates coefficients from
    data of Pandas HDF5 store*.h5
    and saves them back
    ----------------------------
    1. Obtains command line arguments (for description see my_argparser()) that can be passed from new_arg and ini.file
    also.
    2. Loads device data of calibration in laboratory from hdf5 database (cfg['in']['db_path'])
    2. Calibrates configured by cfg['in']['channels'] channels ('accelerometer' and/or 'magnetometer'): soft iron
    3. Wrong implementation - not use cfg['in']['time_range_nord']! todo: Rotate compass using cfg['in']['time_range_nord']
    :param config: returns cfg if new_arg=='<cfg_from_args>' but it will be None if argument
     argv[1:] == '-h' or '-v' passed to this code
    argv[1] is cfgFile. It was used with cfg files:


    """
    global cfg
    cfg = main_init(config, cs_store_name, __file__=None)
    cfg = main_init_input_file(cfg, cs_store_name, in_file_field='path')
    print()
    lf.info(
        "{:s}({:s}) for channels: {} started. ",
        this_prog_basename(__file__), ', '.join(cfg['in']['tables']), cfg['in']['channels']
    )
    fig_filt = None
    fig = None
    if not cfg['in']['path'].is_absolute():
        cfg['in']['path'] = cfg['in']['path_cruise'] / str(cfg['out']['path'])

    fig_save_dir_path = cfg['in']['path'].parent
    (fig_save_dir_path / 'images-channels_calibration').mkdir(exist_ok=True)
    tbl_prev, pid_prev = None, None
    coefs = {}
    cfg['in']['is_counts'] = True
    cfg['out']['aggregate_period'] = None
    
    for d, cfg1, tbl, col_out, pid, i_tbl_part in gen_subconfigs(
            cfg,
            fun_gen=probes_gen,
            **cfg['in']
    ):
        if i_tbl_part:
            probe_continues = True
        else:
            probe_continues = (tbl == tbl_prev and pid == pid_prev)
            tbl_prev = tbl
            pid_prev = pid
        
        n_parts = round(len(cfg1['in']['time_range']) / 2)
        if n_parts > 1:
            if i_tbl_part == 0:
                d_list_save = []
            d_list_save.append(d)
            if (i_tbl_part + 1) < n_parts:
                continue
            else:
                d = dd.concat(d_list_save)
        
        # replacing changed table name back (that was prepared by gen_subconfigs() to output processed data we don't)
        tbl = tbl.replace('i', 'incl')
        d = filter_local(d, cfg['filter'], ignore_absent={'h_minus_1', 'g_minus_1'})

        a = d.compute()
        if a.empty:
            lf.error('No data for {}!!! Skipping it...', tbl)
            continue
        # iUseTime = np.searchsorted(stime, [np.array(s, 'datetime64[s]') for s in np.array(strTimeUse)])

        # Calibrate channels of 'accelerometer' or/and 'magnetometer'
        coefs[tbl] = {}
        for channel in cfg['in']['channels']:
            print(f' channel "{channel}"', end=' ')
            (col_str, coef_str) = channel_cols(channel)

            # Filtering

            # ## Filter same values is not needed to because of better suited spatial bin average later
            # b_ok = np.zeros(a.shape[0], bool)
            # for component in ['x', 'y', 'z']:
            #     b_ok |= is_works(a[col_str + component], noise=cfg['filter']['no_works_noise'][channel])
            # lf.info('Filtered not working area: {:2.1f}%', (b_ok.size - b_ok.sum())*100/b_ok.size)

            vec3d = a.loc[:, [f'{col_str}{component}' for component in ['x', 'y', 'z']]].to_numpy(float).T
            index = a.index
            fig_filt_save_prefix = f"{fig_save_dir_path / 'images-channels_calibration' / tbl}-'{channel}'"
            vec3d, b_ok, fig_filt = filter_channes(
                vec3d, index, fig_filt, fig_save_prefix=fig_filt_save_prefix, window_title=f'channel {channel}',
                # we almost not filter spikes in magnetometer by this method:
                **(cfg['filter'][col_str]
                   # if col_str == 'A' else {
                   #  **cfg['filter'], **{c: {**cfg['filter'][c], 'offsets': [7, 4]} for c in ['x', 'y', 'z']}  # 10, 5
                   #  }
                   )
                )
            # # Remove unit sphere outliers relative to standard coefficients
            # if True:
            #     A, b = [default_coef[f'//coef//{coef_str}//{ac}'] for ac in 'AC']
            #     s = np.dot(A, vec3d - b)
            #     dists_to_unit_sphere = abs(1 - np.linalg.norm(s, axis=0))
            #     hist, dists = np.histogram(dists_to_unit_sphere, bins=20, range=[0, 0.75])
            #     outliers_percents = np.cumsum(hist[::-1]) * 100 / b_ok.size
            #     outliers_percent = 15   # remove bins that contain less than ~ outliers_percent outliers
            #     # i_dist_use = np.searchsorted(outliers_percents, outliers_percent); dist_use = dists[-i_dist_use]
            #     dist_use = np.interp(outliers_percent, outliers_percents, dists[:0:-1])  # dists[-2::-1]?
            #     b_ok_new = dists_to_unit_sphere < dist_use`
            #     b_ok[b_ok] = b_ok_new
            #     # exact % of removed outliers
            #     outliers_percent = (dists_to_unit_sphere.size - b_ok_new.sum()) * 100 / dists_to_unit_sphere.size
            #     lf.info('unit sphere outliers > {} for default coef: {:2.1f}% - deleted', dist_use, outliers_percent)
            #     vec3d = vec3d[:, b_ok_new]
            index = index[b_ok]

            # Repeat calibration removing outliers of the previous increasing input data requirements
            max_outliers_percent = 10  # %
            b_ok = np.ones(vec3d.shape[1], dtype=np.bool_)
            # from too big to not too small:
            dist_check = np.linspace(*np.sqrt([0.3, dist_check_min := 0.07]), dist_check_len := 10)**2  # slower at end
            i_dc = 0
            while i_dc < dist_check_len:
                dc = dist_check[i_dc]
                # Bin average spatially
                index_avg, vec3d_avg, raw3d_other, n_in_cell = bin_avg_3d_partial(index[b_ok], vec3d[:, b_ok], bins=1000)
                # Calibrate
                A, b = calibrate(vec3d_avg)

                s = np.dot(A, vec3d[:, b_ok] - b)  # s[:,:]
                # Remove outliers > 1/sqrt(dc) from unit sphere
                b_ok_new = (dist := abs(1 - np.linalg.norm(s, axis=0))) < dc

                outliers_percent = (b_ok.size - b_ok_new.sum()) * 100 / b_ok.size
                if outliers_percent > max_outliers_percent:
                    lf.info('sphere outliers {:2.4f}: {:2.1f}% - too many -> use previous', dc, outliers_percent)
                    break
                # Jumps too far: where number of ouliers > max_outliers_percent
                # elif b_ok_new.all():  # jumping to smaller dc
                #     lf.info('Distance to unit sphere max/mean: {:.5f}/{:.5f}', dist_max := dist.max(), dist.mean())
                #     if dist_max >= dist_check_min:  # decreasing dc (as we here because dist_max < dc)
                #         i_dc = max(np.searchsorted(-dist_check, -dist_max), i_dc + 1)
                #         continue
                #     else:
                #         break
                else:
                    lf.info('sphere outliers {:2.4f}: {:2.1f}%', dc, outliers_percent)
                    b_ok[b_ok] = b_ok_new
                i_dc += 1

            # Overlay remained points on raw data (of last drawn coordinate) to get into what data is good/bad
            fig_filt.axes[0].plot(index_avg, vec3d_avg[2, :], label='distributed 3D avg,\nnear to sphere',
                                  marker='.', color='k', linewidth=0.5, markersize=2, linestyle='dashed')
            # fig_filt.axes[0].scatterplot(n_in_cell, alpha=0.5)
            fig_filt.axes[0].legend(prop={'size': 10}, loc='upper right')
            try:
                fig_filt.axes[0].figure.savefig(
                    f'{fig_filt_save_prefix}{fig_filt.axes[0].title._text}.png', dpi=300, bbox_inches="tight"
                    )
            except Exception as e:
                lf.warning(f'Can not save fig (filt): {standard_error_info(e)}')

            window_title = f"{tbl} '{channel}' channel ellipse"
            fig = calibrate_plot(vec3d_avg, A, b, fig, window_title=window_title, raw3d_other=raw3d_other)
            fig.savefig(fig_save_dir_path / 'images-channels_calibration' / (window_title + '.png'),
                        dpi=300, bbox_inches='tight')
            A_str, b_str = coef2str(A, b)
            lf.info('Calibration coefficients calculated: \nA = \n{:s}\nb = \n{:s}', A_str, b_str)
            coefs[tbl][channel] = {'A': A, 'b': b}

        # Zeroing Nord direction (should be done after zero calibration?)
        time_range_nord = cfg['in']['time_range_nord']
        if isinstance(time_range_nord, Mapping):
            time_range_nord = time_range_nord.get(pid)
        if time_range_nord:
            coefs[tbl]['M']['azimuth_shift_deg'] = zeroing_azimuth(
                cfg['in']['db'], tbl, time_range_nord, calc_vel_flat_coef(coefs[tbl]), cfg['in']
                )
        else:
            lf.info('not zeroing North')

    # Write coefs to each of output tables
    if not coefs:
        lf.warning('No probes found in {}! The end.', cfg['in']['path'])
        return
    for db_path in cfg['out']['db_paths']:
        db_path = Path(db_path)
        lf.info('Writing to {}', db_path)
        for itbl, (tbl, coef) in enumerate(coefs.items(), start=1):
            if not coef:
                continue
            # i_search = re.search('\d*$', tbl)
            # for channel in cfg['in']['channels']:
            #     (col_str, coef_str) = channel_cols(channel)
            #     dict_matrices = {f'//coef//{coef_str}//A': coefs[tbl][channel]['A'],
            #                      f'//coef//{coef_str}//C': coefs[tbl][channel]['b'],
            #                      }
            #     if channel == 'M':
            #         if coefs[tbl]['M'].get('azimuth_shift_deg'):
            #             dict_matrices[f'//coef//{coef_str}//azimuth_shift_deg'] = coefs[tbl]['M']['azimuth_shift_deg']
            #         # Coping probe number to coefficient to can manually check when copy manually
            #         if i_search:
            #             try:
            #                 dict_matrices['//coef//i'] = int(i_search.group(0))
            #             except Exception as e:
            #                 pass
            dict_matrices = dict_matrices_for_h5(coef, tbl, cfg['in']['channels'])
            h5copy_coef(None, db_path, tbl, dict_matrices=dict_matrices)
            # todo: add text info what source data (path), was used to calibrate which channels, date of calibration
    print('Ok>', end=' ')


# def main_call(
#         cmd_line_list: Optional[List[str]] = None,
#         fun: Callable[[], Any] = main
#         ) -> Dict:
#     """
#     Adds command line args, calls fun, then restores command line args
#     :param cmd_line_list: command line args of hydra commands or config options selecting/overwriting
#
#     :return: global cfg
#     """
#
#     sys_argv_save = sys.argv.copy()
#     if cmd_line_list is not None:
#         sys.argv += cmd_line_list
#
#     # hydra.conf.HydraConf.run.dir = './outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}'
#     fun()
#     sys.argv = sys_argv_save
#     return cfg



if __name__ == '__main__':
    main()
