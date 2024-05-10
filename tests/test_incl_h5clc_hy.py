import pytest
from pathlib import Path
import sys
# from yaml import safe_dump as yaml_safe_dump

from tcm import cfg_dataclasses as cfg_d
import tcm.incl_h5clc_hy as testing_module

# @pytest.fixture()
# def autofon_dict():
#     """
#
#     3179282432 - будильники
#     :return:
#     """
#     with Path(r'data\tracker\autofon_coord_200msgs.json').open('r') as fp:
#         return json.load(fp)
#
#
# #@pytest.mark.parametrize('autofon_dict', )
# def test_autofon_df_from_dict(autofon_dict):
#     df = autofon_df_from_dict(autofon_dict,timedelta(0))
#     assert ['Lat',
#             'Lon',
#             'Speed',
#             'LGSM',
#             'HDOP',
#             'n_GPS',
#             'Temp',
#             'Course',
#             #'Height',
#             #'Acceleration'
#             ] == df.columns.to_list()
#     dtype = np.dtype
#     assert [dtype('float32'),
#             dtype('float32'),
#             dtype('float32'),
#             dtype('int8'),
#             dtype('float16'),
#             dtype('int8'),
#             dtype('int8'),
#             dtype('int8'),
#             #dtype('int8'),
#             #dtype('int8')
#             ] == df.dtypes.to_list()
#
#     assert len(df) == 200
#
#
# @pytest.mark.parametrize(
#     'file_raw_local', [
#         r'data\tracker\AB_SIO_RAS_tracker',
#         r'data\tracker\SPOT_ActivityReport.xlsx'
#         ])
# def test_file_raw_local(file_raw_local):
#     path_raw_local = Path(file_raw_local)
#     cfg_in = {'time_interval': ['2021-06-02T13:49', '2021-06-04T20:00']}  # UTC
#     time_interval = [pd.Timestamp(t, tz='utc') for t in cfg_in['time_interval']]
#
#     df = loading(
#         table='sp4',
#         path_raw_local=path_raw_local,
#         time_interval=time_interval,
#         dt_from_utc=timedelta(hours=2)
#         )
#     assert all(df.columns == ['Lat', 'Lon'])
#     assert df.shape[0] == 3

text_path = Path(
    r'd:\Work\_Python3\And0K\tcm\tests\data\inclinometer\_raw\mag_components_calibration.zip\I*{id}.TXT'
)
# path_db = Path(
#     r'D:\Work\_Python3\And0K\h5toGrid\inclinometer\tests\data\inclinometer\210519incl.h5'
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

aggr = 600  # s
coefs_path = Path(
    r'd:\Work\_Python3\And0K\h5toGrid\inclinometer\tests\data\inclinometer\incl#b.h5'
    # r'd:\WorkData\~configuration~\inclinometer\190710incl_no_pid&string_date.h5'
    # r'd:\WorkData\~configuration~\inclinometer\190710incl.h5'
    # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\230428tank.raw.h5'
).as_posix()

# @pytest.mark.parametrize('return_', ['<end>', '<cfg_before_cycle>'])
def test_call_example_with_cmd_line_args(return_=None):
    sys_argv_save = sys.argv.copy()
    sys.argv = [__file__]  # config dir will be relative to this dir

    # db_path_in = str(path_db).replace('\\', '/')
    # device = 'incl10'
    # aggregate_period_s = (0, 600)

    # df = cfg_d.main_call([
    #     f'in.path="{db_path_in}"',
    #     # '++filter.time_bad_intervals=[2021-06-02T13:49, now]', # todo
    #     'in.tables=["incl.*"]',  # (','.join([f'"{d}"' for d in device]))
    #     f'out.db_path="{db_path_in}proc.h5"',
    #     # f'out.table=V_incl_bin{aggregate_period_s}s',
    #     'out.b_del_temp_db=True',
    #     'program.verbose=INFO',
    #     'program.dask_scheduler=synchronous',
    #     f'program.return_="{return_}"',
    #     f"out.aggregate_period={','.join(f'{a}s' for a in aggregate_period_s)}",
    #     '--multirun',
    #     '--config-path=tests/hydra_cfg',
    #     #'--config-dir=hydra_cfg'  # additional cfg dir
    #     ], fun=main)

    # for aggr in aggregate_period_s[probes]:
    df = cfg_d.main_call([
        # '--info', 'defaults',

        # Switch between text_path and db_path:
        f'in.path="{text_path}"',
        # f'out.raw_db_path="{str_raw_db_path}"'  # if need write to other raw db
        f'in.coefs_path="{coefs_path}"',
        'out.b_del_temp_db=True',
        # 'out.raw_db_path=None',
        f"out.text_path={'text_output' if aggr else 'None'}",  # text_output  # None to skip save text
        'program.verbose=INFO',
        'program.dask_scheduler=threads',
        # search parameters in "inclinometers" config directory defined in hydra.searchpath that will be updated with dir_raw(in.path) / "cfg_proc"
        f'+probes=inclinometers',
        f"out.aggregate_period={aggr}s",  # {','.join(f'{a}s' for a in aggregate_period_s[probes])}",  # for multirun

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
        # f'out.table=V_incl_bin{aggregate_period_s}s',


        #f'out=out0',
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
        '--config-path=d:/Work/_Python3/And0K/tcm/tests/data/inclinometer/_raw/cfg_proc'
        #../tests/data/inclinometer/_raw/hydra_cfg'  # works but it is relative to curdir,
        '--multirun'  # no effect
        ],
        fun=testing_module.main
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
def test_call_example_with_overriding_dict(return_=None):
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
                "text_path": "text_output" if aggr else None,  # text_output  # None to skip save text
                # 'raw_db_path': None,  # str_raw_db_path if need write to other raw db
                "aggregate_period": f"{aggr}s",
                # for multirun can set to ','.join(f'{a}s' for a in aggregate_period_s[probes]),
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

def test_call_example_with_overriding_file(return_=None):
    import os


    sys_argv_save = sys.argv.copy()
    sys.argv = [__file__]  # config dir will be relative to this dir
    cfg_proc_path = Path(  # for 3rd variant
        r"d:\Work\_Python3\And0K\tcm\tests\data\inclinometer\_raw\cfg_proc\defaults\b10_20240507_194129.yaml"
    )
    cfg_proc_stems = ["b08_20240507_194129", cfg_proc_path.stem];
    # cfg_proc_path = r"d:\WorkData\_experiment\inclinometer\_type_b\231009_tank@iB18,25-30\_raw\cfg_proc"
    # not works with "quotes":
    # cfg_proc_path_relative = os.path.relpath(
    # cfg_proc_path.parent, r"d:\Work\_Python3\And0K\tcm"
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
    # test_call_example_with_cmd_line_args()
    test_call_example_with_overriding_dict()
    # test_call_example_with_overriding_file()