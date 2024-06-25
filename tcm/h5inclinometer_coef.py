# @+leo-ver=5-thin
# @+node:korzh.20180529212530.11: * @file h5inclinometer_coef.py
# !/usr/bin/env python
# coding:utf-8
# @+others
# @+node:korzh.20180525044634.2: ** Declarations
"""
Save/modify coef in hdf5 data in "/coef" table of PyTables (pandas hdf5) store
"""
from datetime import datetime
import re
from typing import Iterable, Mapping, Union, Tuple
from pathlib import Path
import h5py
import numpy as np
# my
from .utils2init import FakeContextIfOpen, standard_error_info, my_logging

lf = my_logging(__name__)


def rot_matrix_x(c, s) -> np.ndarray:
    """ Rotation matrix to rotate 3D vector around x axis
    :param c, s: - cos() and sin() of rotation angle
    """
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], np.float64)


def rot_matrix_y(c, s) -> np.ndarray:
    """ Rotation matrix to rotate 3D vector around y axis
    :param c, s: - cos() and sin() of rotation angle
    """
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], np.float64)


def rot_matrix_z(c, s) -> np.ndarray:
    """ Rotation matrix to rotate 3D vector around z axis
    :param c, s: - cos() and sin() of rotation angle
    """
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], np.float64)


def rotate_x(a2d, angle_degrees=None, angle_rad=None):
    """

    :param a2d:
    :param angle_degrees: roll
    :param angle_rad: roll
    :return:
    """
    if angle_rad is None:
        angle_rad = np.radians(angle_degrees)
    out2d = rot_matrix_x(np.cos(angle_rad), np.sin(angle_rad)) @ a2d  # np.dot()
    return out2d


def rotate_y(a2d, angle_degrees=None, angle_rad=None):
    """

    :param a2d:
    :param angle_degrees: pitch because of x direction to Left
    :param angle_rad: pitch
    :return:
    """
    if angle_rad is None:
        angle_rad = np.radians(angle_degrees)
    out2d = rot_matrix_y(np.cos(angle_rad), np.sin(angle_rad)) @ a2d
    return out2d


def rotate_z(a2d, angle_degrees=None, angle_rad=None):
    """

    :param a2d:
    :param angle_degrees: yaw
    :param angle_rad: yaw
    :return:
    """
    if angle_rad is None:
        angle_rad = np.radians(angle_degrees)
    out2d = rot_matrix_z(np.cos(angle_rad), np.sin(angle_rad)) @ a2d  # np.dot()
    return out2d


def h5savecoef(h5file_dest, path, coef):
    """

    :param coef:
    :return:

    Example:
    h5_savecoef(h5file_dest, path='//incl01//coef//Vabs', coef)
    """
    if np.any(~np.isfinite(coef)):
        lf.error('NaNs in coef detected! Aborting')
    else:
        with h5py.File(h5file_dest, 'a') as h5dest:
            # or if you want to replace the dataset with some other dataset of different shape:
            # del f1['meas/frame1/data']
            try:
                h5dest.create_dataset(path, data=coef, dtype=np.float64)
                return
            except (OSError, RuntimeError) as e:
                try:
                    print(f'updating {h5file_dest}/{path}')  # .keys()
                    h5dest[path][...] = coef
                    h5dest.flush()
                    return
                except Exception as e:
                    pass  # prints error message?
                lf.exception('Can not save/update coef to hdf5 %s. There are error ', h5file_dest)


# @+node:korzh.20180525125303.1: ** h5copy_coef
def h5copy_coef(h5file_source=None, h5file_dest=None, tbl=None, tbl_source=None, tbl_dest=None,
                dict_matrices: Union[Mapping[str, np.ndarray], Iterable[str], None] = None,
                dates=None,
                ok_to_replace_group=False
                ):
    """
    Copy tbl from h5file_source to h5file_dest overwriting tbl + '/coef/H/A and '/coef/H/C' with H and C if
    provided. Skip to write `dict_matrices` elements that are None, but check that they are not exist else
    error
    :param h5file_source: name of any hdf5 file with existed coef to copy structure
    :param h5file_dest: name of hdf5 file to paste structure
    :param dict_matrices: dict of numpy arrays - to write or list of paths to coefs (to matrices) under tbl -
    to copy them. See `dict_matrices_for_h5()` to create standard paths from dict with fields Ag, Cg, ...
    :param dates: date attributes for dict_matrices keys that will be written to HDF5
    :param tbl:
    :param tbl_source:
    :param tbl_dest:
    :param ok_to_replace_group:

    # Example save H and C: 3x3 and 1x3, rotation and shift matrices
    >>> h5copy_coef(h5file_source,h5file_dest,tbl)
            dict_matrices={'//coef//H//A': H,
                           '//coef//H//C': C})
    """

    if h5file_dest is None:
        h5file_dest = h5file_source
    if h5file_source is None:
        if h5file_dest is None:
            print('skipping: output not specified')
            return
        h5file_source = h5file_dest

    if tbl_source is None:
        tbl_source = tbl
    if tbl_dest is None:
        tbl_dest = tbl

    # class File_context:
    #     """
    #     If input is string filename then acts like usual open context manager
    #     else treat input as opened file object and do nothing
    #     """
    #
    #     def __init__(self, h5file_init):
    #         self.h5file_init = h5file_init
    #
    #     def __enter__(self):
    #         if isinstance(self.h5file_init, str):
    #             self.h5file = h5py.File(self.h5file_init, 'a')
    #             return self.h5file
    #         else:
    #             self.h5file = self.h5file_init
    #
    #     def __exit__(self, exc_type, ex_value, ex_traceback):
    #         if exc_type is None and isinstance(self.h5file_init, str):
    #             self.h5file.close()
    #         return False

    def path_h5(file):
        try:
            return Path(file.filename)  # if isinstance(file, (h5py._hl.files.File, tables.file.File))
        except AttributeError:
            return Path(file)

    def save_operation(h5source=None):
        """
        Update dict_matrices in h5file_dest. h5source may be used to copy from h5source
        :param h5source: opened h5py.File, if not None copy h5file_source//tbl_source//coef to h5file_dest//tbl//coef before update
        uses global:
            h5file_dest
            tbl_dest, tbl_source
            dict_matrices
        """
        nonlocal dict_matrices
        table_written = None
        with FakeContextIfOpen(lambda f: h5py.File(f, 'a'), h5file_dest) as h5dest:
            try:  # we may generate FileExistsError error that to catch (not real error if use dict_matrices)
                if h5source is None:
                    if tbl_dest != tbl_source:
                        h5source = h5dest
                    else:
                        raise FileExistsError(f'Can not copy to itself {h5dest.filename}//{tbl_dest}')
                elif path_h5(h5dest) == path_h5(h5source) and tbl_dest == tbl_source:
                    raise FileExistsError(f'Can not copy to itself {h5dest.filename}//{tbl_dest}')

                # Copy using provided paths:
                if h5source:
                    coef_path_src = f"//{tbl_source}//coef"
                    lf.info(
                        f'copying "coef" from {path_h5(h5source)}//{tbl_source} '
                        f'to {h5dest.filename}//{tbl_dest}'
                        )
                    # Reuse previous calibration structure:
                    # import pdb; pdb.set_trace()
                    # h5source.copy('//' + tbl_source + '//coef', h5dest[tbl_dest + '//coef'])
                    try:
                        h5source.copy(coef_path_src, h5dest[tbl_dest])
                        # h5source[tbl_source].copy('', h5dest[tbl_dest], name='coef')
                    except RuntimeError as e:  # Unable to copy object (destination object already exists)
                        replace_coefs_group_on_error(h5source, h5dest, coef_path_src, e)
                    except KeyError:  # Unable to open object (object 'incl_b11' doesn't exist)"
                        lf.debug('Creating "{:s}"', tbl_source)
                        try:
                            h5dest.create_group(tbl_source)
                        except (ValueError, KeyError) as e:  # already exists
                            replace_coefs_group_on_error(h5source, h5dest, tbl_source, e)
                        else:
                            h5source.copy(coef_path_src, h5dest[tbl_dest])
                    table_written = f"{tbl_dest.replace('//', '/')}/coef"
            except FileExistsError:
                if dict_matrices is None:
                    raise

            if dict_matrices:  # not is None:
                have_values = isinstance(dict_matrices, dict)
                lf.info(f'updating {h5file_dest}/{tbl_dest}/{{}}', str(dict_matrices))  # .keys()
                if have_values:
                    # Save provided values
                    date_now_str = datetime.now().replace(microsecond=0).isoformat()
                    for rel_path in dict_matrices.keys():
                        path = f'{tbl_dest}{rel_path}'
                        data = dict_matrices[rel_path]
                        if data is None:
                            continue
                        if isinstance(dict_matrices[rel_path], (int, float)) and not (
                            isinstance(dict_matrices[rel_path], bool) and rel_path.endswith('date')):
                            data = np.atleast_1d(data)  # to can load in Veusz (can't load 0d single values)
                        try:
                            if isinstance(data, (str, bool)):
                                if rel_path.endswith('date'):
                                    if isinstance(data, bool):
                                        data = date_now_str
                                        dict_matrices[rel_path] = data
                                    dtype = 'S19'
                                else:
                                    dtype = 'S10'  # 'pid' is only currently possible rel_path str
                                try:
                                    dset = h5dest.create_dataset(path, data=data, dtype=dtype)
                                except ValueError as e:
                                    lf.info(
                                        'Can not create {} to write text "{}". Already exist? => Writing...',
                                        path, data
                                    )
                                    h5dest[path][...] = data
                            else:
                                dtype = np.float64
                                b_isnan = np.isnan(data)
                                if np.any(b_isnan):
                                    lf.warning('not writing NaNs: {}{}...', rel_path, np.flatnonzero(b_isnan))
                                    h5dest[path][~b_isnan] = data[~b_isnan]
                                else:
                                    h5dest[path][...] = data
                                dset = h5dest[path]
                        except TypeError as e:
                            lf.error(
                                'Replacing dataset "{}" TypeError: {} -> recreating...', path,
                                '\n==> '.join([a for a in e.args if isinstance(a, str)])
                            )
                            try:
                                # to replace the dataset with some other dataset of different shape
                                del h5dest[path]
                            except KeyError as e:
                                pass
                            dset = h5dest.create_dataset(path, data=data, dtype=dtype)
                        except KeyError as e:  # Unable to open object (component not found)
                            lf.debug('Creating "{}"', path)
                            dset = h5dest.create_dataset(path, data=data, dtype=dtype)
                        if dates:
                            date = isinstance(dates, bool) or dates.get(rel_path) or dates.get(
                                rel_path.rsplit('//', 1)[-1])
                            if date:
                                if isinstance(date, str):
                                    date = str(date)  # required if date is numpy.str_
                                try:
                                    dset.attrs.modify(
                                        'timestamp', date if isinstance(date, str) else date_now_str)
                                except Exception as e:
                                    dset.attrs['timestamp'] = date if isinstance(date, str) else date_now_str
                else:
                    paths = list(dict_matrices)
                    dict_matrices = {}
                    for rel_path in paths:
                        path = tbl_source + rel_path
                        try:
                            dict_matrices[path] = h5source[path][...]
                        except AttributeError as e:  # 'ellipsis' object has no attribute 'encode'
                            lf.error(
                                'Skip update coef: dict_matrices must be None or its items must point to '
                                'matrices {}', '\n==> '.join(a for a in e.args if isinstance(a, str)))
                            continue
                        h5dest[path][...] = dict_matrices[path]
                h5dest.flush()
                table_written = f"{tbl_dest.replace('//', '/')}/coef"
            else:
                dict_matrices = {}
            return table_written
            # or if you want to replace the dataset with some other dataset of different shape:
            # del f1['meas/frame1/data']
            # h5dest.create_dataset(tbl_dest + '//coef_cal//H//A', data= A  , dtype=np.float64)
            # h5dest.create_dataset(tbl_dest + '//coef_cal//H//C', data= C, dtype=np.float64)
            # h5dest[tbl_dest + '//coef//H//C'][:] = C

    def replace_coefs_group_on_error(h5source, h5dest, path, e=None):
        if ok_to_replace_group:
            lf.warning('Replacing group "{}"', path)
            del h5dest[path]
            h5source.copy(path, h5dest[tbl_dest])
        else:
            lf.error('Skip copy coef{}!', f': {standard_error_info(e)}' if e else '')

    # try:
    with FakeContextIfOpen(
            (lambda f: h5py.File(f, 'r')) if h5file_source != h5file_dest else None, h5file_source
    ) as h5source:
        _ = save_operation(h5source)
        tables_written = [_] if _ else []


    # if h5file_source != h5file_dest:
    #     with h5py.File(h5file_source, 'r') as h5source:
    #         save_operation(h5source)
    # else:
    #     save_operation()
    # except Exception as e:
    #     raise e.__class__('Error in save_operation()')

    # Confirm the changes were properly made and saved:
    b_ok = True
    if dict_matrices:
        with FakeContextIfOpen(lambda f: h5py.File(f, 'r'), h5file_dest) as h5dest:
            for rel_path, v in dict_matrices.items():
                if isinstance(v, str):
                    if h5dest[tbl_dest + rel_path][...].item().decode() != v[:19]:
                        lf.error('h5copy_coef(): value of {}{} not updated to "{}"!', tbl_dest, rel_path, v)
                        b_ok = False
                elif v is None:
                    assert (tbl_dest + rel_path) not in h5dest
                elif not np.allclose(h5dest[tbl_dest + rel_path][...], v, equal_nan=True):
                    lf.error('h5copy_coef(): coef. {}{} not updated!', tbl_dest, rel_path)
                    b_ok = False
        if b_ok:
            print('h5copy_coef() updated coefficients Ok>')
        else:
            tables_written = []
    return tables_written

def h5_rotate_coef(h5file_source, h5file_dest, tbl):
    """
    Copy tbl from h5file_source to h5file_dest overwriting tbl + '/coef/H/A with
    previous accelerometer coef rotated on Pi for new set up
    :param h5file_source: name of any hdf5 file with existed coef to copy structure
    :param h5file_dest: name of hdf5 file to paste structure
    """
    with h5py.File(h5file_source) as h5source:
        with h5py.File(h5file_dest, "a") as h5dest:
            # Reuse previous calibration structure:
            in2d = h5source[tbl + '//coef//G//A'][...]
            out2d = rotate_x(in2d, 180)
            h5dest[tbl + '//coef//G//A'][:, :] = out2d
            h5dest.flush()

    print('h5copy_coef(): coefficients updated.')


def channel_cols(channel: str) -> Tuple[str, str]:
    """
    Data columns names (col_str M/A) and coef letters (coef_str H/G) from parameter name (or its abbreviation)
    :param channel: magnetometer (M) or accelerometer (A)
    :return: (col, coef) = ('M', 'H') or ('A', 'G') letters denoting channel
    """
    str_col__coef = (
        ('M', 'H') if channel in ('magnetometer', 'M') else
        ('A', 'G')
        )
    return str_col__coef


def dict_matrices_for_h5(coefs=None, tbl=None, channels=None, msg='Saving coefficients'):
    """
    Create coefficients dict with fields of fixed size (fill with dummy values if no corresponded coefs):
    - A: for accelerometer:
      - A: A 3x3 scale and rotation matrix
      - C: U_G0 accelerometer 3x1 channels shifts
    - M: for magnetometer:
      - A: M 3x3 scale and rotation matrix
      - C: U_B0 3x1 channels shifts
      - azimuth_shift_deg: Psi0 magnetometer direction shift to the North, radians
    - Vabs0: for calculation of velocity magnitude from inclination - 6 element vector:
      - 5 coefs of trigonometric approx fun
      - its linear extrapolation start, degrees

    :param coefs: fields from: {
    'M': {'A', 'b', 'azimuth_shift_deg'},
    'A': {'A', 'b', 'azimuth_shift_deg'},
    'Vabs0'}
    :param tbl: should include probe number in name. Example: "incl_b01"
    :param channels: some from default ['M', 'A']: magnetometer, accelerometer
    :param msg: logging message. Will be appended by warning for dummy coefs created
    :return: dict_matrices
    """
    if channels is None:
        channels = ['M', 'A']

    # Fill coefs where not specified
    dummies = []
    b_have_coefs = coefs is not None
    if b_have_coefs:
        coefs = coefs.copy()  # todo: deep copy better!
    else:
        coefs = {}  # dict(zip(channels,[None]*len(channels)))

    for channel in channels:
        if not coefs.get(channel):
            coefs[channel] = (
                {'A': np.identity(3) * 0.00173, 'b': np.zeros((3, 1))} if channel == 'M' else
                              {'A': np.identity(3) * 6.103E-5, 'b': np.zeros((3, 1))}
                              )
            if b_have_coefs:
                dummies.append(channel)
        if channel == 'M':
            if not coefs['M'].get('azimuth_shift_deg'):
                coefs['M']['azimuth_shift_deg'] = 180
                if b_have_coefs and not 'M' in dummies:
                    dummies.append('azimuth_shift_deg')  # only 'azimuth_shift_deg' for M channel is dummy

    if coefs.get('Vabs0') is None:
        coefs['Vabs0'] = np.float64([10, -10, -10, -3, 3, 70])
        if b_have_coefs:
            dummies.append('Vabs0')

    if dummies or not b_have_coefs:
        lf.warning('{:s} {:s} - dummy!', msg, ','.join(dummies) if b_have_coefs else 'all')
    else:
        lf.info(msg)

    # Fill dict_matrices with coefs values
    dict_matrices = {}
    if tbl:
        # Coping probe number to coefficient to can manually check when copy manually
        i_search = re.search('\d*$', tbl)
        if i_search:
            try:
                dict_matrices['//coef//i'] = int(i_search.group(0))
            except Exception as e:
                pass
        dict_matrices['//coef//Vabs0'] = coefs['Vabs0']

    for channel in channels:
        (col_str, coef_str) = channel_cols(channel)
        dict_matrices.update(
            {f'//coef//{coef_str}//A': coefs[channel]['A'],
             f'//coef//{coef_str}//C': coefs[channel]['b'],
             })
        if channel == 'M':
            dict_matrices[f'//coef//{coef_str}//azimuth_shift_deg'] = coefs['M']['azimuth_shift_deg']

    return dict_matrices
