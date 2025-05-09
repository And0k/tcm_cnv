from pathlib import Path
# import sys
# from yaml import safe_dump as yaml_safe_dump
from itertools import dropwhile
from tcm import cfg_dataclasses as cfg_d
from tcm import incl_h5clc_hy
from datetime import datetime
# Set input path - text or raw data db (data that was converted from csv):

# If text data, then should contain {prefix:} (by default, command line argument "prefix=I*" is used) and {number:0>3} placeholders (or pass dir only for default
# '*{prefix:}{number:0>3}*.[tT][xX][tT]') to separate data from files by device type and number (else all
# data will be joined)
# can use `prefix`, id=allowed model char/0 and digits: "[0bdp]{number}", model and number (char and digits separately)

path_in = Path(  # can use {model}, {number} that can be extracted from `in.tables`
    r"D:\WorkData\BalticSea\250415_ABP60\inclinometer\_raw\С ловушкой\I_063.TXT"
    # r"D:\WorkData\_experiment\inclinometer\250505_tube\_raw\{model}*{number}.TXT"
    # r"B:\WorkData\BalticSea\240625_ABP56-incl,t-chain\inclinometer\_raw\240616.raw.h5"
    # r"D:\WorkData\BalticSea\240616_ABP56\inclinometer\_raw\_found_in_2025_04\{model}*{number}.TXT"
    # r"F:\_\copy\AB_SIO_RAS\240527_stand,tank,tube@i61-90\_raw\240613_tube@i61-68-no_good_date.rar"
    # r"D:\WorkData\BalticSea\_Pregolya,Lagoon\241115\inclinometer\_raw\20241213_Inkl.rar"
    # r"D:\WorkData\BalticSea\240616_ABP56\inclinometer\_raw\240616.raw.h5"
    # r"d:\WorkData\_experiment\inclinometer\240527_stand,tank,tube@i61-90\_raw\240604_tank@59-61,68,70,71.zip"
    # r"d:\WorkData\_experiment\inclinometer\240527_stand,tank,tube@i61-90\_raw\240920tube@"
    # r"d:\WorkData\BalticSea\240616_ABP56\inclinometer\_raw\Каньон\{model}*{number}.TXT"
    # r"d:\WorkData\BalticSea\240616_ABP56\inclinometer\_raw\Склон\{model}*{number}.TXT"
    # r"d:\WorkData\BalticSea\240827_AI68\inclinometer\_raw\P01 (Падающий инклинометр)\*\*"  # todo: allow
    # r"d:\WorkData\BalticSea\240827_AI68\inclinometer\_raw\P01 (Падающий инклинометр)"
    # r"d:\WorkData\_experiment\inclinometer\240527_stand,tank,tube@i61-90\_raw\stand"
    # r"d:\WorkData\BalticSea\240730_inclinometer\_raw\ink.zip\ink\{model}*{number}.TXT"
    # r'd:\WorkData\_experiment\inclinometer\240813_free_fall_in_tube@ip05\_raw\{model}*{number}.TXT'
    # r"d:\WorkData\BalticSea\231121_ABP54\inclinometer@i4,37,38,58-60,91\_raw\231117@i4,91.zip\{model}*{number}.TXT"
    # r"d:\WorkData\_experiment\inclinometer\_type_b\231009_tank@iB18,25-30\_raw\tank\I*{id}.TXT"
).absolute()
b_calibr = (
    "stand" in path_in.name.lower()  # switch to use dumb coefs as they should not exist yet
    or "tube" in path_in.name.lower()
)
# Requires several re-runs
#
# 1st run to get probes and their configs in yaml
# 1.a: run to process txt to raw db to use it for quick draw to get data picture and update config time_ranges with start/end of clean data
# 2nd run to process each probe and save separately to (raw if 1.a haven't run), proc.noAvg, proc.Avg db / text. (May be easier to run averaging after 1.a also because it will be possible to average all data without splits?)
# 3rd run to join data from db to proc db / text (else combined data after each probe will be overwritten)

# Todo: how to run this all at once
# Todo: how to break sequence of multirun on 1st error
# or 2nd: create common config to process all probes and 3rd: process accumulating?

# Set usage of _configs_: "*" - all, list(int) - specified probes only, [] - none (only get configs from data)
# Not works for loading from DB (loads all tables): todo find why, writes '"/.*" tables found in '
incl_type_nums = {
    "*"  # "*" 88, 89  79 60  #  61,68,70,71  "04" #  "91"
}

# Set averaging bin [s]: [] for default [0, 2, 600, 3600, 7200]
bins = [0, 2, 600, 3600, 7200]  # [] - loads to raw db only (todo: switch [] to default if loading from db)  # [0, 2]  # [1800]  # [0, 2, 600, 3600, 7200]

# Search existed configs
_ = next(dropwhile(lambda p: p.name.lower() != "_raw", path_in.parents), path_in)
cfgs_dir = _ / "cfg_proc"
device_dir = _.parent

cfg_name2pcid = lambda name: name.split("_", 1)[0]
cfgs_existed = {
    int(cfg_name2pcid(f.stem)[1:]): f.stem for f in (cfgs_dir / "defaults").glob("*.yaml")
}
if not incl_type_nums:
    cfgs = {}
else:
    if incl_type_nums == {"*"} or incl_type_nums == "*":
        cfgs = cfgs_existed
    else:
        incl_type_nums_without_config = incl_type_nums.difference(cfgs_existed)
        if incl_type_nums_without_config:
            raise FileNotFoundError(
                f"Not found config files for {len(incl_type_nums_without_config)}:"
                f"{incl_type_nums_without_config}!"
            )
        cfgs = {num: stem for num, stem in cfgs_existed.items() if num in incl_type_nums}
print(
    "== Using {}/{} probes having configs in {}".format(
        len(cfgs), len(cfgs_existed), cfgs_dir / "defaults",
    ), "====================================================="
)
# todo: check that is no duplicated probes
args_common = [] # ["hydra/hydra_logging.root.level=INFO"]

# Load date range from `info_devices.json`
if cfgs:
    try:
        print('Load date range from "info_devices.json"')
        from vsz_loader import get_path_in_parents, load_info_json
        import json  # , yaml
        from ruamel.yaml import YAML
        from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dq, PlainScalarString as plain_str
        ry = YAML(typ="safe", pure=True)
        ry.default_flow_style = False
        ry.allow_unicode=True
        ry.preserve_quotes = True
        # ry.indent(mapping=2, sequence=4, offset=2)

        _ = get_path_in_parents(device_dir, file_name="info_devices.json")
        # _ = next(device_dir.glob("info_devices.json"))
        device_info = load_info_json({"devices": [cfg_name2pcid(v) for v in cfgs.values()]}, _)
        with _.open(encoding="utf8") as f:
            device_info_loaded = json.load(f)

        print("Updating yaml configurations with loaded time ranges", end="...")
        for k, v in cfgs.items():
            print(k, end=': ')
            _ = (cfgs_dir / "defaults" / v).with_suffix(".yaml")
            cfg_cur = ry.load(_)
            # Update date_range
            # if cfg_cur["input"].get("date_range"):  # if not defined
            #     continue
            i_time_range = 6
            time_ranges_list_of_str = device_info_loaded[cfg_name2pcid(v)][i_time_range:]
            # cfg_cur["input"]["time_ranges"] = time_ranges_list_of_str
            if time_ranges_list_of_str:
                cfg_cur["input"]["time_ranges"] = [
                    datetime.fromisoformat(t).strftime("%Y-%m-%dT%H:%M:%S")  # only with seconds we get quotes!
                    for t in time_ranges_list_of_str
                ]
                ry.dump(cfg_cur, stream=_)
                print(end="set, ")
            else:
                print(end="no, ")
            # with _.open("w") as f:
            #     yaml.dump(cfg_cur, stream=f)
        print()
    except Exception as e:
        print(e)

# hydra/job_logging



if not cfgs:  # True: #
    # 1st run. Create config file for each found probe determined from raw file name
    cfg_d.main_call(
        [
            f'in.path="{path_in}"',
            "in.tables=[{}]".format(",".join(f"incl{t}" for t in incl_type_nums)),
            'program.return_="<cfg_input_files_list_to_yaml>"',
            # "+in_many.date_to_from={b18:[2022-06-08T20:17:34,2000-01-01T03:20:59],b26:[2022-06-08T20:17:34,2000-01-01T02:20:33],b27:[2022-06-08T20:17:34,2000-01-01T03:10:41]}",
            r'in.coefs_path="C:\Work\Python\AB_SIO_RAS\tcm\tcm\cfg\coef\190710incl.h5"'
            # (<- default)
            # r'in.coefs_path="F:\_\copy\AB_SIO_RAS\240527_stand,tank,tube@i61-90\_raw\240527stand.raw.h5"',
            # f'out.db_path="{path_in.parent.with_name("240604tube.h5")}"',
            # f'out.raw_db_path="{path_in.parent.with_name("240604tube.h5")}"',  # without will be based on parent dir (todo: base on db_path)
            # "+in.date_to_from=[2024-06-04T13:00:00,2000-01-01T23:00:00]",
            # "d:\WorkData\_experiment\inclinometer\240527_stand,tank,tube@i61-90\_raw\240527stand.raw.h5"
            # "d:\WorkData\~configuration~\inclinometer\incl#b.h5",
        ]
        + (
            ["in.coefs.kVabs=[10.0, -10.0, -10.0, -3.0, 3.0, 75.0]"] if b_calibr else []
            # here we have set some coefs data different from default to prevent no-coef-error
            # and to mainly use default coefs
        )
        + args_common,
        my_hydra_module=incl_h5clc_hy,
    )
# elif incl_type_nums:  # True: # need config path correction!
#     # find and process specified probes using config from cfg_prog/inclinometers
#     cfg_d.main_call(
#         [
#             f'in.path="{path_in}"',
#             "in.tables=[{}]".format(",".join("incl{}".format(cfg_name2pcid(stem)[1:]) for stem in cfgs)),
#             f"out.dt_bins=[{','.join(str(b) for b in bins)}]",
#             "+probes=inclinometers",
#             # # This probe have no coefs, so we overwrite some defaults to prevent raising error:
#             # "in.coefs.kVabs=[-7, 30, 5, -0.2, -9, 78]",
#         ],
#         my_hydra_module=incl_h5clc_hy,
#     )
else:
    if not path_in.parent.glob("*.h5") and bins:
        print("no raw db found => running without calculation this time so you can adjust time_ranges")
        bins = []

    if path_in.suffix == ".h5":
        for num, stem in cfgs.items():
            print(f"Processing #{num} using config {stem}...")
            cfg_d.main_call(
                (
                    [
                        f'in.path="{path_in}"',
                        "in.tables=[incl{}]".format(cfg_name2pcid(stem)[1:]),
                        # "out.b_overwrite_text=False",  # to faster continue after crash
                        # "b_reuse_temporary_tables=True",
                    ]
                    if path_in.suffix == ".h5"
                    else []
                )  # load from h5 / txt path of configs.
                # todo: not delete previous data from not_sorted if was error: use cycle here?
                + [
                    f"out.dt_bins=[{','.join(str(b) for b in bins)}]",
                    # yaml files in probes configs in "defaults" dir
                    f'hydra.searchpath=["{cfgs_dir.as_posix()}"]',
                    f"+defaults@_global_={stem}",
                    # "+in.date_to_from=[2024-06-04T13:00:00,2000-01-01T23:00:00]",
                    "in.coefs=null",  # not use from yaml cfg
                    "in.coefs_path=null",  # load from DB if loading from raw DB else default
                    "out.b_overwrite_text=False",
                ]
                + args_common,
                my_hydra_module=incl_h5clc_hy,
            )
    else:  # 2nd run. Use saved configs (that you could modify before this run)
        cfg_d.main_call(
            (
                [  # Not works: Uses all tables for each of multirun with 1st config only
                    f'in.path="{path_in}"',
                    # If want change some values in yaml-configs (may be better del yaml to regenerate):
                    # f'out.db_path="{path_in.with_name("240604tube.h5")}"',
                    # "out.not_joined_db_path=null",  # auto
                    # "out.raw_db_path='auto'",  # default
                    #     "in.tables=[{}]".format(
                    #         ",".join("incl{}".format(cfg_name2pcid(stem)[1:]) for stem in cfgs.values())
                    #     ),
                    #     "out.b_overwrite_text=False",  # to faster continue after crash
                    #     # "b_reuse_temporary_tables=True",
                ]
                # if path_in.suffix == ".h5"
                # else
                # []
            )  # load from h5 / txt path of configs.
            # todo: not delete previous data from not_sorted if was error: use cycle here?
            + [
                f"out.dt_bins=[{','.join(str(b) for b in bins)}]",
                # yaml files in probes configs in "defaults" dir
                f'hydra.searchpath=["{cfgs_dir.as_posix()}"]',
                f"+defaults@_global_={','.join(cfgs.values())}",
                # "+in.date_to_from=[2024-06-04T13:00:00,2000-01-01T00:00:00]",
                "in.min_date=2024-05-30T23:00:00",
                # "in.coefs=null",  # not use from yaml cfg
                # "in.coefs_path=null",  # load from DB if loading from raw DB else default
                "--multirun",
            ]
            + args_common,
            my_hydra_module=incl_h5clc_hy,
        )


# Notes on data
#
# Magnetometers filtering: i88 Hx spikes to: [-509, -506, -1, 0, 2]