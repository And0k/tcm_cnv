"""
Set path_in pointing to {model}*{number}.TXT files or already loaded DB raw.h5.
- In 1st case the program will not calc/avg data even if you configure (for this the rerun is required). Also 1st metadata collected run step will only add *.yaml configureations to {dir_cfgs}/"defaults".
- In last case it is required to put/keep *.yaml configurations only for probes, you needed to calculate.

So Note:
requires several re-runs and, if `time_ranges` is not known, calculating with temporary
setting `bins = []` to determine `time_ranges` from not averaged data!
- 1st: run to get probes and their configs in yaml
- 2nd: run to process txt to raw db (to use it for quick draw to get data picture and update config time_ranges with start/end of clean data. Alternatively draw raw txt fils directly): 2nd run to process each probe
- 3rd run to calc and save separately to proc.noAvg, proc.Avg db and text.

# Todo: 4th run join data from db to proc db / text (else combined data after each probe will be overwritten)
# Todo: how to run this all at once
# Todo: how to break sequence of multirun on 1st error
# or 2nd: create common config to process all probes and 3rd: process accumulating?
# Todo: take time_ranges from info_devices.json

# Last error (todo: check):
- coef get processing date, not date from last coef in 250909_Katsiveli@i/_raw/250909.raw.h5
- h5.append_through_temp_db_gen when temp_db.create_table_index() on 1st run only.

"""
from datetime import datetime
from pathlib import Path
# import sys
# from yaml import safe_dump as yaml_safe_dump
from itertools import dropwhile
from tcm import cfg_dataclasses as cfg_d
from tcm.csv_specific_proc import parse_name
from tcm.csv_load import pcid_from_parts
from tcm import incl_h5clc_hy
from tcm.utils2init import standard_error_info
# import re
# Set input path - text or raw data db (data that was converted from csv):

# If text data, then should contain {prefix:} (by default, command line argument "prefix=I*" is used) and {number:0>3} placeholders (or pass dir only for default
# '*{prefix:}{number:0>3}*.[tT][xX][tT]') to separate data from files by device type and number (else all
# data will be joined)
# can use `prefix`, id=allowed model char/0 and digits: "[0bdp]{number}", model and number (char and digits separately)

# Set raw txt/h5 path. Note: REMOVE *any* configs (cfg_proc/defaults/*.yaml) if you not want process them now!
# If configs exist path_in overwrites path (no effect if you just rerun) else set it to dir under "raw".
path_in = Path(  # can use {model}, {number} that can be extracted from `in.tables`
    r"B:\Cruises\BalticSea\201202_BalticSpit\inclinometers\201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6\_raw\201202\@i23,32,w4,5,6_get210623_zip"  # \w{number:0>2}*.txt
    # r"B:\Cruises\BalticSea\201202_BalticSpit\inclinometers\201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6\_raw\201202\@i30,w02_get210419_rar\w02.txt"
    # r"B:\Cruises\BalticSea\201202_BalticSpit\inclinometers\201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6\_raw\201202_@i3,5,9,10,11,15,19,28,w1_get210109_zip\*{model}*{number:0>2}*.txt"
    # r"D:\Cruises\BalticSea\251201_ABP64\inclinometers\_raw\*{model}*0{number:0>2}*.TXT"
    # r"D:\Cruises\BlackSea\250909_Katsiveli@i\_raw\250909.raw.h5"
    # r"B:\WorkData\BlackSea\250909_Katsiveli@i\_raw\*{model}*{number:0>2}*.TXT"
    # r"D:\WorkData\_experiment\P~tc\250924@ip1-3_old\_raw\stand\*{model}*{number:0>2}*.TXT"
    # r"D:\WorkData\_experiment\P~tc\250827@ip2\_raw\stand\*{model}*{number:0>2}*.TXT"
    # r"D:\Cruises\BalticSea\240616_ABP56-incl,t-chain\inclinometer\_raw\240625.raw.h5"
    # r"D:\Cruises\BalticSea\240616_ABP56-incl,t-chain\inclinometer\_raw\Каньон\{model}*{number}.TXT"
    # r"B:\WorkData\_experiment\inclinometer\250610tank@i63,64,67,68,78,86,87\_raw\{model}*{number}.TXT"
    # r"D:\WorkData\BalticSea\250415_ABP60\inclinometer\_raw\С ловушкой\I_063.TXT"
    # r"D:\Cruises\BalticSea\250415_ABP60\inclinometer\_raw\250415.raw.h5"
    # r"D:\WorkData\_experiment\inclinometer\250505_tube\_raw\{model}*{number}.TXT"
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

b_calibr = any((  # Switch to use dumb coefs as they should not exist yet
    lambda dir_lower: any(_ in dir_lower for _ in ("stand", "tube", "tank"))
)(p.lower()) for p in path_in.parts)


# Set usage of _configs_: "*" - all, list(int) - specified probes only, [] - none (only get configs from data)
# Not works for loading from DB (loads all tables): todo find why, writes '"/.*" tables found in '
# Specify device numbers to filter existed configs by set(int) or set("*") to use all
incl_type_nums = {
    "*"  # "*" 88, 89  79 60  #  61,68,70,71  "04" #  "91"
}

# Set averaging bin [s]: []: to raw db only, [0]: notAvg, [0, 2, 600, 3600, 7200]: +Avg
bins = [0]  # 0, 2, 600, 3600, 7200]  # (todo: switch [] to default if loading from db, +[1800] if such burst data)  # [0] also to output no avg text

# Search existed configs and filter them by incl_type_nums
dir_raw = next(dropwhile(lambda p: p.name.lower() != "_raw", path_in.parents), path_in)
dir_cfgs = dir_raw / "cfg_proc"
dir_device = dir_raw.parent

# cfg_name2pcid = lambda name: name.rsplit("_", 2)[0]
cfgs_existed = {
    int(parse_name(f.stem)["number"]): f.stem for f in (dir_cfgs / "defaults").glob("*.yaml")
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
        len(cfgs), len(cfgs_existed), dir_cfgs / "defaults",
    ), "====================================================="
)
# todo: check that is no duplicated probes
args_common = [] # ["hydra/hydra_logging.root.level=INFO"]

# Load date range from `info_devices.json` and update date_ranges in yaml with them
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

        _ = get_path_in_parents(dir_device, file_name="info_devices.json")
        # _ = next(dir_device.glob("info_devices.json"))
        pcids = [pcid_from_parts(**parse_name(v)).replace("_","") for v in cfgs.values()]
        device_info = load_info_json({"devices": pcids}, _)
        # with _.open(encoding="utf8") as f:
        #     device_info_loaded = json.load(f)

        time_ranges_found = {pcid: v["t"] for pcid, v in device_info.items() if "t" in v}
        if any(time_ranges_found):
            print("Updating yaml configurations with time ranges loaded from json metadata", end="...")
            for pcid, (k, v) in zip(pcids, cfgs.items()):
                print(k, end=': ')
                _ = (dir_cfgs / "defaults" / v).with_suffix(".yaml")
                cfg_cur = ry.load(_)
                time_ranges = cfg_cur["input"].get("time_ranges")
                if time_ranges:  # already set
                    print(f"Already have time ranges in {v}.yaml:", time_ranges)
                    continue
                time_ranges_list_of_str = time_ranges_found[pcid]
                if time_ranges_list_of_str:
                    cfg_cur["input"]["time_ranges"] = [  # only with seconds we get quotes!
                        datetime.fromisoformat(t).strftime("%Y-%m-%dT%H:%M:%S")
                        for t in time_ranges_list_of_str
                    ]
                    ry.dump(cfg_cur, stream=_)
                    print(end="set, ")
                else:
                    print(end="no, ")
                # with _.open("w") as f:
                #     yaml.dump(cfg_cur, stream=f)
        else:
            # todo: do opposite too
            print(
                "No time records in json metadata. Updating with time ranges from current config",
                "(todo)")  # end="...")
        print()
    except Exception as e:
        print(standard_error_info(e))

# hydra/job_logging


if not cfgs:  # True: #
    print(" --- 1st run. Create config file for each found probe determined from raw file name ---")
    cfg_d.main_call(
        [
            f'in.path="{path_in}"',
            "in.tables=[{}]".format(",".join(f"incl{t}" for t in incl_type_nums)),
            'program.return_="<cfg_input_files_list_to_yaml>"',
            # "+in_many.date_to_from={b18:[2022-06-08T20:17:34,2000-01-01T03:20:59],b26:[2022-06-08T20:17:34,2000-01-01T02:20:33],b27:[2022-06-08T20:17:34,2000-01-01T03:10:41]}",
            r'in.coefs_path="{}"'.format(
                # r"D:\Cruises\BlackSea\250909_Katsiveli@i\_raw\coefs_.h5",
                r"B:\Cruises\BalticSea\201202_BalticSpit\inclinometers\201202P1-5,I1-2@i3,5,9,10,11,15,19,23,28,30,32,33,w1-6\_raw\to_out\201202P1-5,I1-2.raw.h5"
                # r"C:\Work\Python\AB_SIO_RAS\tcm\tcm\cfg\coef\190710incl.h5",  # (<- default)
                # r"B:\WorkData\_experiment\inclinometer\240527_stand,tank59-61,68,70,71,tube@i61-90\_raw\240527stand.raw.h5"
            ),
            # f'out.db_path="{path_in.parent.with_name("240604tube.h5")}"',
            # f'out.raw_db_path="{path_in.parent.with_name("240604tube.h5")}"',  # without will be based on parent dir (todo: base on db_path)
            # "+in.date_to_from=[2024-06-04T13:00:00,2000-01-01T23:00:00]",
            # r"d:\WorkData\_experiment\inclinometer\240527_stand,tank,tube@i61-90\_raw\240527stand.raw.h5"
            # r"d:\WorkData\~configuration~\inclinometer\incl#b.h5",
        ]
        + (
            print("Using default coefs.kVabs for calibration")
            or ["in.coefs.kVabs=[10.0, -10.0, -10.0, -3.0, 3.0, 75.0]"]
            if b_calibr
            else []
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
    #             "in.tables=[{}]".format(",".join("incl{}".format(parse_name(stem)["number"]) for stem in cfgs)),
    #             f"out.dt_bins=[{','.join(str(b) for b in bins)}]",
    #             "+probes=inclinometers",
    #             # # This probe have no coefs, so we overwrite some defaults to prevent raising error:
    #             # "in.coefs.kVabs=[-7, 30, 5, -0.2, -9, 78]",
    #         ],
    #         my_hydra_module=incl_h5clc_hy,
    #     )
else:
    if not [p for p in dir_raw.glob("*.h5") if p.suffixes[-2] != ".raw_not_sorted"] and bins:
        print("--- 2nd run: No raw DB found ⇒ loading to raw DB only (don't forget adjust time_ranges after)")
        bins = []  # running without calculation/binning this time

    if path_in.suffix == ".h5":
        for num, stem in cfgs.items():
            print(f"Processing #{num} using config {stem}...")
            cfg_d.main_call(
                (   ([] if path_in.is_dir() else [f'in.path="{path_in}"']) +
                    [
                        "in.tables=[{}]".format(
                            incl_h5clc_hy.pcid_to_raw_name(pcid_from_parts(**parse_name(stem)))
                        ),
                        # "out.b_overwrite_text=False",  # to faster continue after crash
                        # "b_reuse_temporary_tables=True",
                    ]
                )  # load from h5 / txt path of configs.
                # todo: not delete previous data from not_sorted if was error: use cycle here?
                + [
                    f"out.dt_bins=[{','.join(str(b) for b in bins)}]",
                    "out.dt_bins_min_save_text=0",  # most big files!
                    # yaml files in probes configs in "defaults" dir
                    f'hydra.searchpath=["{dir_cfgs.as_posix()}"]',
                    f"+defaults@_global_={stem}",
                    # "+in.date_to_from=[2024-06-04T13:00:00,2000-01-01T23:00:00]",
                    "in.coefs=null",  # not use from yaml cfg
                    "in.coefs_path=null",  # load from DB if loading from raw DB else default
                    "out.b_overwrite_text=False",
                ]
                + args_common,
                my_hydra_module=incl_h5clc_hy,
            )
    else:
        print(" --- 2nd run. Using saved configs (that you could modify before this run) --- ")
        cfg_d.main_call(
            (
                ([] if path_in.is_dir() else [f'in.path="{path_in}"'])
                + [  # Not works: Uses all tables for each of multirun with 1st config only
                    # If want change some values in yaml-configs (may be better del yaml to regenerate):
                    # f'out.db_path="{path_in.with_name("240604tube.h5")}"',
                    # "out.not_joined_db_path=null",  # auto
                    # "out.raw_db_path='auto'",  # default
                    #     "in.tables=[{}]".format(
                    #         ",".join("incl{}".format(parse_name(stem)["number"]) for stem in cfgs.values())
                    #     ),
                    "out.b_overwrite_text=False",  # to faster continue after crash
                    #     # "b_reuse_temporary_tables=True",
                ]
                # if path_in.suffix == ".h5"
                # else
                # []
            )  # load from h5 / txt path of configs.
            # todo: not delete previous data from not_sorted if was error: use cycle here?
            + [
                f"out.dt_bins=[{','.join(str(b) for b in bins)}]",
                # "out.dt_bins_min_save_text=0",  # most big files!
                # yaml files in probes configs in "defaults" dir
                f'hydra.searchpath=["{dir_cfgs.as_posix()}"]',
                f"+defaults@_global_={','.join(cfgs.values())}",
                # "+in.date_to_from=[2024-06-04T13:00:00,2000-01-01T00:00:00]",
                # "in.min_date=2024-05-30T23:00:00",
                # "in.coefs=null",  # not use from yaml cfg
                # "in.coefs_path=null",  # load from DB if loading from raw DB else default
                "--multirun",
            ]
            + args_common,
            my_hydra_module=incl_h5clc_hy,
        )
print(f"{datetime.now():%Y-%m-%d %H:%M:%S} Ok>")

# Notes on data
#
# Magnetometers filtering: i88 Hx spikes to: [-509, -506, -1, 0, 2]