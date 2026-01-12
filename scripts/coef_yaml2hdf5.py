# Andrey Korzh, 12.08.2023
#
import logging
from datetime import datetime
from pathlib import Path
from ruamel.yaml import YAML
from typing import List
import numpy as np
from tcm.h5inclinometer_coef import h5copy_coef
from tcm.incl_h5clc_hy import pcid_to_raw_name, to_pcid_from_name

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


# from numpy.polynomial.polynomial import polyval2d

def poly_str2list(poly_str) -> List[float]:
    """
    Converts a polynomial string like 'A + B * u - C * t - D * u**2 + E * u * t + F * t**2' to a list of
    polynomial coefficients like [A, B, C, D, E, F]
    :param poly_str: The polynomial string containing coefficients and terms represented by letters
    :return: A list of polynomial coefficients parsed from the input string
    """
    str_splitted = poly_str.split(' ')
    coefs = [float(str_splitted[0])]  # A
    v = None
    for s in str_splitted[1:]:
        if s in ('+', '-'):
            v = s
            continue
        elif v is not None:
            coefs.append(float(''.join([v, s])))  # B, C, D, E, F
        v = None
    return coefs


def coef_yaml2hdf5(path_yaml, paths_hdf5, tbl_prefix='incl_p'):
    """
    _summary_

    :param path_yaml: _description_
    :param paths_hdf5: _description_
    :param tbl_prefix: _description_, defaults to 'incl_p' is useful for old format to construct raw table
    name
    """
    print(f'Copy {path_yaml} coefficients...')
    with path_yaml.open('r') as f:
        data = yaml.load(f)
    arg_dict = coef_prepare_from_yaml(data)

    for db in paths_hdf5:
        print(f'to {db}:')
        for pnum, arg_coef in arg_dict.items():
            try:
                tbl = f'{tbl_prefix}{int(pnum):0>2}'
            except ValueError:
                # new format with probe id instead of just number (ignoring tbl_prefix argument)
                tbl = pcid_to_raw_name(to_pcid_from_name(pnum))

            print(f'{tbl} ', end='')
            h5copy_coef(
                h5file_dest=db, tbl=tbl,
                **arg_coef,
                ok_to_replace_group=True
            )


def coef_prepare_from_yaml(data):
    """
    Prepare arguments with termocompensated pressure coef info for h5copy_coef()
    Note: to get formula from result coefficient_matrix, eval:
    `sympy.expand(polyval2d(sympy.Symbol('u'), sympy.Symbol('t'), coefficient_matrix))`
    todo: keep only max date data if multiple records for the same `pnum` (last is kept now)
    :param data: yaml loaded data
    :return: _description_
    """
    arg_dict = {}

    def construct_dict(date_str, data_coefs, rel_path):
        coefs = poly_str2list(data_coefs['poly'])
        # convert to matrix coefficient ready to use in numpy.polynomial.polynomial.polyval2d()
        coefficient_matrix = np.zeros((3, 3))
        coefficient_matrix.flat[[0, 3, 1, 6, 4, 2]] = coefs
        return {
            "dict_matrices": {rel_path: coefficient_matrix},
            "dates": {rel_path: datetime.strptime(date_str.split('. ')[0], '%y%m%d_%H%M')}
        }

    # relative group path_
    rel_path = '//coef//P_t'

    for i, (data_header, data_coefs) in enumerate(data.items()):
        try:
            date_str, pnum = data_header.split(': ')
        except ValueError:  # new format with separate 2nd date level:
            pnum = data_header
            for j, (date_str, data_coefs) in enumerate(data_coefs.items()):
                arg_dict[pnum] = construct_dict(date_str, data_coefs, rel_path)
            continue
        else:
            date_str = date_str.split('. ')[0]

        arg_dict[pnum] = construct_dict(date_str, data_coefs, rel_path)
    return arg_dict




if __name__ == '__main__':
    path_yaml = Path(
        r"D:\WorkData\_experiment\P~tc\250827@ip2\_raw\stand\vsz(txt, range=12h)\coefs_250929.yaml"
        # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\coefs_230808.yaml'
    )
    paths_hdf5 = [Path(p) for p in [
        r"D:\Cruises\BlackSea\250909_Katsiveli@i\_raw\coefs.h5"
        # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\230428P.raw.h5'
        # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\230428stand.raw.h5',

    ]]
    coef_yaml2hdf5(path_yaml, paths_hdf5, tbl_prefix="incl_p")
