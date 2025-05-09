import pytest
from pathlib import Path
import sys
import os
# from yaml import safe_dump as yaml_safe_dump

from tcm import cfg_dataclasses as cfg_d
import tcm.incl_h5clc_hy as testing_module

text_path = Path(
    r'C:\Work\Python\AB_SIO_RAS\tcm\tests\data\inclinometer\_raw\mag_components_calibration.zip\I*{id}.TXT'
)
# path_db = Path(
#     r'C:\Work\Python\AB_SIO_RAS\h5toGrid\inclinometer\tests\data\inclinometer\210519incl.h5'
#     ).absolute()


# dir_raw = Path(path_db_raw or text_path)

# Setting hydra.searchpath to cruise specific config dir: "{path_db_raw.parent}/cfg_proc" (probes config directory)
# within inclinometer/cfg/incl_h5clc_hy.yaml - Else it will be not used and hydra will only warn
# try:
#     dir_raw = dir_raw.parents[-dir_raw.parts.index('_raw') - 1]
# except (ValueError, IndexError):
#     dir_raw = dir_raw.parent / '_raw'
# path_cfg_default = (lambda p: p.parent / 'cfg' / p.name)(Path(incl_h5clc_hy.__file__)).with_suffix('.yaml')
# with path_cfg_default.open('w') as f:
#     yaml_safe_dump({
#         'defaults': ['base_incl_h5clc_hy', '_self_'],
#         'hydra': {'searchpath': [f'file://{(dir_raw / "cfg_proc").as_posix()}'.replace('///', '//')]}
#         }, f)
#     f.flush()

bins = [600]  # s
coefs_path = Path(
    r'C:\Work\Python\AB_SIO_RAS\h5toGrid\inclinometer\tests\data\inclinometer\incl#b.h5'
    # r'd:\WorkData\~configuration~\inclinometer\190710incl_no_pid&string_date.h5'
    # r'C:\Work\Python\AB_SIO_RAS\tcm\tcm\cfg\coef\190710incl.h5'
    # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\230428tank.raw.h5'
).as_posix()


def test_tcm_proc_loading_from_raw_db():

    sys_argv_save = sys.argv.copy()
    sys.argv = [__file__]  # config dir will be relative to this dir
    path_db_raw = Path(  # for 3rd variant
        r"C:\Work\Python\AB_SIO_RAS\tcm\tests\data\inclinometer\_raw\data.raw.h5"
    )   # C:\Work\Python\AB_SIO_RAS\tcm\tests\data\inclinometer\_raw\data.raw.h5
    out_dicts = cfg_d.main_call(
        [
            f'in.path="{path_db_raw}"',
            f'in.coefs_path="{coefs_path}"',
            f"out.dt_bins=[{','.join(f'{dt}' for dt in bins)}]",
            # # # 3: yaml file in probes (best variant): configs in "defaults" dir
            # f'hydra.searchpath=["{Path(cfg_proc_path.parent.parent).as_posix()}"]',
            # f"+defaults@_global_={','.join(cfg_proc_stems)}",
            # # f"defaults@_global_=[{','.join(cfg_proc_stems)}]"  # combines all cfg_proc_stems in one job?
            # "--multirun",
        ],
        my_hydra_module=testing_module,
    )

    sys.argv = sys_argv_save


# @pytest.mark.parametrize('return_', ['<end>', '<cfg_before_cycle>'])
def test_tcm_proc_with_cmd_line_args(return_=None):
    sys_argv_save = sys.argv.copy()
    sys.argv = [__file__]  # config dir will be relative to this dir

    # db_path_in = str(path_db).replace('\\', '/')
    # device = 'incl10'
    # bins = (0, 600)

    # df = cfg_d.main_call([
    #     f'in.path="{db_path_in}"',
    #     # '++filter.time_bad_intervals=[2021-06-02T13:49, now]', # todo
    #     'in.tables=["incl.*"]',  # (','.join([f'"{d}"' for d in device]))
    #     f'out.db_path="{db_path_in}proc.h5"',
    #     # f'out.table=V_incl_bin{bin}s',
    #     'out.b_del_temp_db=True',
    #     'program.verbose=INFO',
    #     'program.dask_scheduler=synchronous',
    #     f'program.return_="{return_}"',
    #     f"out.dt_bins={','.join(f'{dt}' for dt in bins)}",
    #     '--multirun',
    #     '--config-path=tests/hydra_cfg',
    #     #'--config-dir=hydra_cfg'  # additional cfg dir
    #     ], fun=main)

    df = cfg_d.main_call(
        [
            # '--info', 'defaults',
            # Switch between text_path and db_path:
            f'in.path="{text_path}"',
            # f'out.raw_db_path="{str_raw_db_path}"'  # if need write to other raw db
            f'in.coefs_path="{coefs_path}"',
            "out.b_del_temp_db=True",
            # 'out.raw_db_path=None',
            f"out.text_path={'text_output' if bins else 'None'}",  # text_output  # None to skip save text
            "program.verbose=INFO",
            "program.dask_scheduler=threads",
            # search parameters in "inclinometers" config directory defined in hydra.searchpath that will be updated with dir_raw(in.path) / "cfg_proc"
            f"+probes=inclinometers",
            f"out.dt_bins={','.join(str(b) for b in bins)}",
            # 'in.text_type=p',  # not need if file named according standard
            # 'in.coefs.Ag=[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]',
            # 'in.coefs.Cg=[10.0,10.0,10.0]',
            # 'in.coefs.Ah=[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]',
            # 'in.coefs.Ch=[10.0,10.0,10.0]',
            # 'in.coefs.kVabs=[10.0, -10.0, -10.0, -3.0, 3.0, 70.0]',
            # 'in.coefs.azimuth_shift_deg=0.0',
            # '+in.coefs.g0xyz={[0,0,1]}',
            # 'in.skip_if_no_coef=True',
            # 'in.coefs.kP=[0.0,1.0]',
            # out.not_joined_db_path=True  # set to save in .proc_noAvg.h5 if not need filter data and save all loaded time range
            # '++filter.time_bad_intervals=[2021-06-02T13:49, now]', # todo
            # 'input.tables=["incl.*"]', # was set in probes config directory
            ## f'out.db_path={db_out}',
            # f'out.table=V_incl_bin{bin}s',
            # f'out=out0',
            # '--config-path=cfg_proc',  # Primary config module 'inclinometer.cfg_proc' not found.
            # '--config-dir=cfg_proc'  # additional cfg dir
            # 'in.min_date=2023-09-06T16:00',
            # 'in.max_date=2023-05-23T19:40',
            # 'in.date_to_from_list=[2023-10-11T13:00:00,2000-01-01T00:00:05]',
            # '+in.date_to_from_lists={i28:[2023-06-30T11:47:26,2000-01-01T01:03:07]}',  # can use if load from text
            # Note: "*" is mandatory for regex, "incl" only used in raw files, but you can average data in processed db.
            ## "in.tables=['i.*']" if probes == 'inclinometers' else  # ['incl(23|30|32).*']  # ['i.*']
            ## "in.tables=['w.*']",                                    # ['w0[2-6].*']         # ['w.*']
            # hydra / sweeper / params =
            # hydra.job.chdir=true
            'program.return_="<cfg_input_files_to_json>"',  # "<cfg_input_files>"
            "--config-path=C:/Work/Python/AB_SIO_RAS/tcm/tests/data/inclinometer/_raw/cfg_proc"
            # ../tests/data/inclinometer/_raw/hydra_cfg'  # works but it is relative to curdir,
            "--multirun",  # no effect
        ],
        fun=testing_module.main,
    )

    if return_ == '<cfg_before_cycle>':
        cfg = df
        # assert 'in' in cfg

    sys.argv = sys_argv_save


# @pytest.mark.parametrize(
# 'return_', [
#     '<end>', '<cfg_before_cycle>',
#     "<cfg_input_files_list>",
#     "<cfg_input_files_list_to_yaml>",
#     "<cfg_input_files_meta>",
#     "<cfg_input_files_meta_to_yaml>"
# ])
def test_tcm_proc_with_overriding_dict(return_=None):
    """Overwrites default config using dict in arguments and config file probes/inclinometers.yaml"""
    sys_argv_save = sys.argv.copy()
    sys.argv = [__file__]  # config dir will be relative to this dir


    out_dicts = cfg_d.main_call(
        [  # Command line parameters that will not work if paste them in overrides
            # Add "inclinometers.yaml" config from "probes" directory.
            # Note: "probes" directory must be in hydra.searchpath. Searchpath will be updated with
            # `dir_raw(in.path) / "cfg_proc"`
            "+probes=inclinometers",  # b10_20240506_172301_
            # "--config-path=../tests/hydra_cfg",
            "--multirun",  # no effect
        ],
        overrides={
            "input": {
                "path": text_path,
                "coefs_path": coefs_path,
            },
            "out": {
                "b_del_temp_db": True,
                "text_path": "text_output" if bins else None,  # text_output  # None to skip save text
                # 'raw_db_path': None,  # str_raw_db_path if need write to other raw db
                "dt_bins": bins,
            },
            "program": {
                "verbose": "INFO",
                "dask_scheduler": "threads",
                "return_": "<cfg_input_files_list_to_yaml>",
            },
        },
        my_hydra_module=testing_module,
    )

    sys.argv = sys_argv_save

def test_tcm_proc_with_overriding_cfg_file(return_=None):
    sys_argv_save = sys.argv.copy()
    sys.argv = [__file__]  # config dir will be relative to this dir
    cfg_proc_path = Path(  # for 3rd variant
        r"C:\Work\Python\AB_SIO_RAS\tcm\tests\data\inclinometer\_raw\cfg_proc\defaults\b10_20240610_093018.yaml"
    )
    cfg_proc_stems = ["b08_20240610_093018", cfg_proc_path.stem]
    # cfg_proc_path = r"d:\WorkData\_experiment\inclinometer\_type_b\231009_tank@iB18,25-30\_raw\cfg_proc"
    # not works with "quotes":
    # cfg_proc_path_relative = os.path.relpath(
    # cfg_proc_path.parent, r"C:\Work\Python\AB_SIO_RAS\tcm"
    # )

    out_dicts = cfg_d.main_call(
        [  # Command line parameters that will not work if paste them in overrides
            # Add "inclinometers.yaml" config from "probes" directory.
            # Note: "probes" directory must be in hydra.searchpath. Searchpath will be updated with
            # `dir_raw(in.path) / "cfg_proc"`
            # "--config-path=../tests/hydra_cfg",
            # "hydra.run.dir="
            # Not works:
            # 'hydra.job.override_dirname="{cfg_proc_path.parent}"',
            # f'+my_config="{cfg_proc_path.as_posix()}"',
            #
            # Works (partly):
            # # 1: yaml file must contain all parameters
            # # 1.1: absolute path yaml, yaml references defaults
            # f'in.path="{text_path}"',
            # f"--config-dir={Path(cfg_proc_path.parent).as_posix()}",
            # f"--config-name={cfg_proc_path.stem}",
            # # 1.2: relative yaml file (loads but error as not contains all parameters)
            # f"--config-dir={Path(cfg_proc_path_relative).as_posix()}",
            # f"--config-name={cfg_proc_path.stem}",
            # # 2: yaml file in probes/ must starts with "# @package _global_"
            # f'hydra.searchpath=["{Path(cfg_proc_path.parent).as_posix()}"]',
            # "+probes=b10_20240506_172301_",
            # # 3: yaml file in probes (best variant): configs in "defaults" dir
            f'hydra.searchpath=["{Path(cfg_proc_path.parent.parent).as_posix()}"]',
            f"+defaults@_global_={','.join(cfg_proc_stems)}",
            # f"defaults@_global_=[{','.join(cfg_proc_stems)}]"  # combines all cfg_proc_stems in one job?
            "--multirun",
        ],
        my_hydra_module=testing_module,
    )

    sys.argv = sys_argv_save


if __name__ == '__main__':
    test_tcm_proc_loading_from_raw_db()
    # test_tcm_proc_with_cmd_line_args()
    # test_tcm_proc_with_overriding_dict()
    # test_tcm_proc_with_overriding_cfg_file()