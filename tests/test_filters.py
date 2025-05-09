# -*- coding: utf-8 -*-
import pytest
import sys
from pathlib import Path
from datetime import timedelta
import logging
import numpy as np
from tcm import filters
from ..tcm.utils_time import check_time_diff

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# import imp; imp.reload(csv2h5)

# drive_d = Path('D:/' if sys.platform == 'win32' else '/mnt/D')  # allows to run on both my Linux and Windows systems:
# scripts_path = drive_d.joinpath('Work/_Python3/And0K/h5toGrid/scripts')
# sys.path.append(str(Path(scripts_path).parent.resolve()))  # os.getcwd()

# path_data = Path(r'd:\Work\_Python3\hartenergy-find_formation_names\test\200202_1st750rows.tsv')
# path_data_full = Path(r'd:\Work\_Python3\hartenergy-find_formation_names\data\HartEnergy\wells_US_all.tsv')

# from to_vaex_hdf5.h5tocsv import *


# VERSION = '0.0.1'
# l = logging.getLogger(__name__)


# @pytest.mark.skip(reason="passed")
@pytest.mark.parametrize('t', [
    np.load('tests/data/time_t.npy')  # slow changing frequency (and no gaps between frequency changes)
    ])
def test_make_linear(t: np.int64):
    """ Test make_linear(t: np.int64, freq: float, dt_big_hole: Optional[timedelta] = None)"""
    # parameters
    t_before = t.copy()
    dt_big_hole = timedelta(seconds=1.5)
    dt_accuracy_s = dt_big_hole
    freq = 5 # Hz
    # Run test
    t = filters.make_linear_with_shifts(t, freq=freq, linearize_accuracy_s=dt_accuracy_s.total_seconds())
    check_output_accuracy(t, t_before, dt_accuracy_s)

    # Run failing test
    t = t_before.copy()  # restart with original time
    i_st_hole = filters.make_linear(t, freq=freq, dt_big_hole=dt_big_hole)  # changes t
    check_output_accuracy(t, t_before, dt_accuracy_s, i_st_hole)

    # # Check with dt_big_hole < 1s
    # if dt_big_hole >= timedelta(seconds=1):
    #     dt_big_hole = timedelta(seconds=0.999)

def check_output_accuracy(t, t_before, dt_accuracy_s, i_st_hole=None):
    """Check: (t_before - t) < dt_accuracy_s"""
    bbad = np.ones_like(t, dtype=np.bool_)
    if i_st_hole:
        bbad[i_st_hole] = False
        bbad[i_st_hole[1:] - 1] = False  # last data point before hole
    bbad[bbad], dt_arr = check_time_diff(
        t[bbad].view("M8[ns]"),
        t_before[bbad].view("M8[ns]"),
        dt_warn=dt_accuracy_s,  # pd.Timedelta(minutes=2)
        msg="Time difference [{units}] in {n} points exceeds {dt_warn} after flattening:",
        max_msg_rows=20,
        return_diffs=True,
    )
    assert max(abs(dt_arr)).astype("m8[s]").item() < dt_accuracy_s
