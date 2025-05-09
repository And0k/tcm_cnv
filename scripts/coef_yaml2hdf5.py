# Andrey Korzh, 12.08.2023
# 
import logging
from datetime import datetime
from pathlib import Path
from ruamel.yaml import YAML
from typing import List

from h5inclinometer_coef import h5copy_coef

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

import numpy as np
# from numpy.polynomial.polynomial import polyval2d

def poly_str2list(poly_str) -> List[float]:
    """
    Converts a polynomial string like 'A + B * u - C * t - D * u**2 + E * u * t + F * t**2' to a list of polynomial coefficients like [A, B, C, D, E, F]
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
    
    with path_yaml.open('r') as f:
        data = yaml.load(f)
    
    print(f'Copy {path_yaml} coefficients...')
    for db in paths_hdf5:
        print(f'to {db}:')
        for i, (data_header, data_coefs) in enumerate(data.items()):
            date_str, pnum = data_header.split(': ')
            date_str = date_str.split('. ')[0]
            coefs = poly_str2list(data_coefs['poly'])
            
            # convert to matrix coefficient ready to use in numpy.polynomial.polynomial.polyval2d()
            coefficient_matrix = np.zeros((3, 3))
            coefficient_matrix.flat[[0, 3, 1, 6, 4, 2]] = coefs
            # to get formula eval: sympy.expand(polyval2d(sympy.Symbol('u'), sympy.Symbol('t'), coefficient_matrix))
            
            tbl = f'{tbl_prefix}{int(pnum):0>2}'
            print(f'{tbl} ', end='')
            key = '//coef//P_t'
            h5copy_coef(
                h5file_dest=db, tbl=tbl,
                dict_matrices={key: coefficient_matrix},
                dates={key: datetime.strptime(date_str.split('. ')[0], '%y%m%d_%H%M')},
                ok_to_replace_group=True
            )
    

if __name__ == '__main__':
    path_yaml = Path(
        r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\coefs_230808.yaml')
    paths_hdf5 = [Path(p) for p in [
        # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\230428P.raw.h5'
        r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\230428stand.raw.h5',
        
    ]]
    coef_yaml2hdf5(path_yaml, paths_hdf5)
