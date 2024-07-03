from pathlib import Path
import sys
from yaml import safe_dump as yaml_safe_dump

from tcm import cfg_dataclasses as cfg_d
from tcm import incl_h5clc_hy

# Set text_path or raw data db (data that was converted from csv). Latter will be used if defined both.

# Input text data: text_path should contain {prefix:} and {number:0>3} placeholders (or pass dir only for default
# '*{prefix:}{number:0>3}*.[tT][xX][tT]') to separate data from files by device type and number (else all data will be
# joined)
# can use `prefix`, id=allowed model char/0 and digits: "[0bdp]{number}", model and number (char and digits separately)

text_path = Path(
    r"d:\WorkData\_experiment\inclinometer\_type_b\231009_tank@iB18,25-30\_raw\tank\I*{id}.TXT"
).absolute()

bins = []  # for joining binned data we should use loading from DB to not combine data after each probe
searchpath = Path(text_path).parent.parent / "cfg_proc"
config_files = [f.stem for f in (searchpath / "defaults").glob("*.yaml")]
if not config_files:  # 1st run. Create config file for each found probe determined from raw file name
    # True: #
    cfg_d.main_call(
        [
            f'in.path="{text_path}"',
            'program.return_="<cfg_input_files_list_to_yaml>"',
            "in.coefs.kVabs=[10.0, -10.0, -10.0, -3.0, 3.0, 75.0]",
            "+in_many.date_to_from={b18:[2022-06-08T20:17:34,2000-01-01T03:20:59],b26:[2022-06-08T20:17:34,2000-01-01T02:20:33],b27:[2022-06-08T20:17:34,2000-01-01T03:10:41]}",
            r'in.coefs_path="d:\WorkData\~configuration~\inclinometer\incl#b.h5"',
            #
            # here we set some coefs data different from default to prevent no-coef-error
            # and to use default coefs
        ],
        my_hydra_module=incl_h5clc_hy,
    )
else: # 2nd run. Use saved configs (that you could modify before this run)
    cfg_d.main_call(
        [
            f"out.dt_bins=[{','.join(str(b) for b in bins)}]",
            # yaml files in probes configs in "defaults" dir
            f'hydra.searchpath=["{searchpath.as_posix()}"]',
            f"+defaults@_global_={','.join(config_files)}",
            "--multirun",
        ],
        my_hydra_module=incl_h5clc_hy,
    )
