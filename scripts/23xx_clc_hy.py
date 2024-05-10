from pathlib import Path
import sys
from yaml import safe_dump as yaml_safe_dump

from tcm import cfg_dataclasses as cfg_d
from tcm import incl_h5clc_hy as incl_h5clc_hy

# Set text_path or raw data db (data that was converted from csv). Latter will be used if defined both.

# Input text data: text_path should contain {prefix:} and {number:0>3} placeholders (or pass dir only for default
# '*{prefix:}{number:0>3}*.[tT][xX][tT]') to separate data from files by device type and number (else all data will be
# joined)
# can use `prefix`, id=allowed model char/0 and digits: "[0bdp]{number}", model and number (char and digits separately)

text_path = Path(
    r'd:\WorkData\_experiment\inclinometer\231009_tank@iB18,25-30\_raw\tank\I*{id}.TXT'
    # r'd:\WorkData\BalticSea\_Pregolya,Lagoon\231208@i19,ip4,5\_raw\2023_12_08.rar\I*{id}.TXT'
    # r'd:\WorkData\_experiment\inclinometer\231229_stand@i58-60\_raw\Mag&truba.rar\Mag&truba\I*{id}.TXT'
    # r'd:\WorkData\BalticSea\231121_ABP54\inclinometer@i37,38,58-60\_raw\Inklinometrsrar\I*{id}*.TXT'
    # r'd:\WorkData\KaraSea\231110_AMK93\inclinometer\_raw\I*{id}*.TXT'
    # r'd:\WorkData\KaraSea\231110_AMK93\inclinometer\_raw\I*{id}*.TXT'
    # r'd:\WorkData\_experiment\inclinometer\231102@i52,54,57\_raw\I*{id}*.TXT'
    # r'd:\WorkData\BalticSea\_Pregolya,Lagoon\230906@i3,38,54\_raw\I*{id}*.TXT'
    # r'd:\WorkData\_experiment\inclinometer\231010_stand,tank@i52-56\_raw\tank\bad_date\I*{id}.TXT'
    # r'd:\WorkData\_experiment\inclinometer\231010_stand,tank@i52-56\_raw\tank\I*{id}_truba.TXT'
    # r'd:\WorkData\_experiment\inclinometer\231010_stand,tank@i52-56\_raw\stand\bad_date\I*{id}.TXT'
    # r'd:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\inclinometer\_raw\куликово 09.23\I*{id}.TXT'
    # r'd:\WorkData\BalticSea\230507_ABP53\inclinometer@i3,4,15,19,37,38;ib27-30,ip6\_raw\_raw.zip\INKL_P06.TXT'
    # r'd:\WorkData\BalticSea\230616_Kulikovo@inclinometer\_raw\INKL_P01.TXT'
    # r'd:\WorkData\BalticSea\230616_Kulikovo@i3,4,19,37,38,p1-3,5,6\_raw\*'  # treated as folders
    # r'd:\WorkData\BalticSea\230616_Kulikovo@inclinometer\_raw\230616_Куликово\_bad_time\I_003.TXT'
    # r'd:\WorkData\_experiment\inclinometer\230727_stand@i3,4\_raw\*\I*_{id}.TXT'
    # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\_magn05Hz\INKL_P05.TXT'
    # r'd:\WorkData\BalticSea\221105_ASV54\inclinometer\_raw\I*_{id}.TXT'
    # r'd:\WorkData\_experiment\inclinometer\230614_tank@i3,4,15,19,28,33,37,38;В27-30\_raw\230623_tank@i28,33\I_{id}.TXT'  #I_{id}.TXT
    # r'd:\WorkData\BalticSea\230507_ABP53\inclinometer\_raw\@i_{id}.TXT'  # @i_003_needed_part.TXT
    # r'd:\WorkData\_experiment\inclinometer\230614_tank@i3,4,15,19,37,38;В27-30\_raw\_raw_tank\I_0{number}*.TXT'
    # r'd:\WorkData\_experiment\inclinometer\230614_tank@i3,4,15,19,37,38;В27-30\_raw\_raw_tank\I_B{number}*.TXT'
    # I_{model}{number}*.TXT I_0{number}*.TXT'
    # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\_P'
    # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\_tank\INKL_P06.TXT'
).absolute()

path_db_raw = Path(
    # r'd:\WorkData\KaraSea\231110_AMK93\inclinometer\231110.proc.h5'
    # r'd:\WorkData\BalticSea\_Pregolya,Lagoon\231208@i19,ip4,5\_raw\231208.raw.h5'
    # r'd:\WorkData\BalticSea\231121_ABP54\inclinometer@i37,38,58-60\_raw\231121.raw.h5'
    # r'd:\WorkData\KaraSea\231110_AMK93\inclinometer\_raw\231110.raw.h5'
    # r'D:\WorkData\BalticSea\_Pregolya,Lagoon\230906@i3,38,54\_raw\230906.raw.h5'
    # r'd:\WorkData\BalticSea\230825_Kulikovo@ADCP,ADV,i,tr\inclinometer\_raw\230825.raw.h5'
    # r'd:\WorkData\BalticSea\230616_Kulikovo@inclinometer\_raw\230616.raw.h5'
    # r'd:\WorkData\BalticSea\230507_ABP53\inclinometer@i3,4,15,19,37,38;ib27-30,ip6\_raw\230507.raw.h5'
)  # and None  # path_db_raw = None
if path_db_raw == Path():
    path_db_raw = None
dir_raw = Path(path_db_raw or text_path)

# Setting hydra.searchpath to cruise specific config dir: "{path_db_raw.parent}/cfg_proc" (probes config directory)
# within inclinometer/cfg/incl_h5clc_hy.yaml - Else it will be not used and hydra will only warn
try:
    dir_raw = dir_raw.parents[-dir_raw.parts.index('_raw') - 1]
except (ValueError, IndexError):
    dir_raw = dir_raw.parent / '_raw'
path_cfg_default = (lambda p: p.parent / 'cfg' / p.name)(Path(incl_h5clc_hy.__file__)).with_suffix('.yaml')
with path_cfg_default.open('w') as f:
    yaml_safe_dump({
        'defaults': ['base_incl_h5clc_hy', '_self_'],
        'hydra': {'searchpath': [f'file://{(dir_raw / "cfg_proc").as_posix()}'.replace('///', '//')]}
        }, f)
    f.flush()

str_raw_db_path = 'auto'  # '230428stand.raw.h5'  # '230428tank.raw.h5'  #

db_out = None  # '"{}"'.format((path_db_raw.parent.parent / f'{path_db_raw.stem}_proc23,32;30.h5').replace('\\', '/'))

aggregate_period_s = {  # 2,  will be joined for multirun. Default: [0, 2, 300, 600, 1800, 7200]: [300, 1800] is for burst mode
    # use [0] if input is not all probes, then remove input tables restriction (in .yaml) and average with other bins
    # (not needed for 2s-averages  as it channel's data from *.raw.h5 DB)
    'inclinometers': [0, 600, 3600, 7200],  # 0, 2,  # [3600*24],  # [0, 2, 600, 3600, 7200],   # [0, 2, 600, 1800, 7200]
    'wavegauges': [0, 2, 300, 3600],    # 0, [0, 2, 300, 3600]   #[0],
    }

# todo: Change config dir and hydra output dir. will be relative to this dir. to raw data dir.
# sys.argv = ["c:/temp/cfg_proc"]  #[str(path_db_raw.parent / 'cfg_proc')]  # path of yaml config for hydra (main_call() uses sys.argv[0] to add it)
split_avg_out_by_time_ranges = False   # True  # Run only after common .proc_noAvg.h5 saved (i.e. with aggregate_period_s=0)

sys_argv_save = sys.argv.copy()
for probes in 'inclinometers'.split():  # 'inclinometers wavegauges', 'inclinometers_tau600T1800'
    if False:  # not False - save by time_ranges
        if split_avg_out_by_time_ranges:
            db_out = str(path_db_raw.parent.with_name(
                '_'.join([
                    path_db_raw.parent.parent.name.replace('-inclinometer', ''),
                    'ranges'  # st.replace(":", "")
                ])
            ).with_suffix(".proc.h5")).replace('\\', '/')
            print('Saving to', db_out)
        cfg_d.main_call([
            # f'in.min_date=2021-11-11T{st}:00',
            # f'in.max_date=2021-11-11T{en}:00',
            # f'in.db_path="{str_raw_db_path}"',  # default: "auto"
            f'out.db_path="{db_out}"',
            'out.b_del_temp_db=True',
            'program.verbose=INFO',
            'program.dask_scheduler=threads',
            f'+probes={probes}',  # see ``probes`` config directory
            f"out.aggregate_period={','.join(f'{a}s' for a in aggregate_period_s[probes])}",  # for multirun
            ] + ([
                #  f"out.aggregate_period={','.join(f'{a}s' for a in aggregate_period_s[probes] if a !=0)}",
                'out.b_split_by_time_ranges=True',  # flag to split by time_ranges (defined in cfg_proc\probes\inclinometers.yaml)
                ] if split_avg_out_by_time_ranges else [
                ]) +
            (
                ["in.tables=['i.*']"] if probes == 'inclinometers' else  # ['incl(23|30|32).*']  # ['i.*']
                ["in.tables=['w.*']"]                                    # ['w0[2-6].*']         # ['w.*']
            ) + ['--multirun'], fun=incl_h5clc_hy.main)
    else:
        # cfg_d.main_call(['--info', 'defaults'], fun=incl_h5clc_hy.main)
        coefs_path = Path(
            # r'd:\Work\_Python3\And0K\h5toGrid\inclinometer\tests\data\inclinometer\incl#b.h5'
            # r'd:\WorkData\~configuration~\inclinometer\190710incl_no_pid&string_date.h5'
            r'd:\WorkData\~configuration~\inclinometer\190710incl.h5'
            # r'd:\WorkData\_experiment\inclinometer\230428_stand,tank,pres@ip1-6\_raw\230428tank.raw.h5'
        ).as_posix()
        for aggr in aggregate_period_s[probes]:
            df = cfg_d.main_call([
                    # '--info', 'defaults',

                    # Switch between text_path and db_path:
                    f'in.path="{path_db_raw or text_path}"'
                ] + ([
                    f'out.raw_db_path="{str_raw_db_path}"'
                ] if text_path else []) + [

                # 'in.text_type=p',  # not need if file named according standard
                # 'in.coefs.Ag=[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]',
                # 'in.coefs.Cg=[10.0,10.0,10.0]',
                # 'in.coefs.Ah=[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]',
                # 'in.coefs.Ch=[10.0,10.0,10.0]',
                # 'in.coefs.kVabs=[10.0, -10.0, -10.0, -3.0, 3.0, 70.0]',
                # 'in.coefs.azimuth_shift_deg=0.0',
                # '+in.coefs.g0xyz={[0,0,1]}',

                f'in.coefs_path="{coefs_path}"',
                # 'in.skip_if_no_coef=True',
                # 'in.coefs.kP=[0.0,1.0]',
                # out.not_joined_db_path=True  # set to save in .proc_noAvg.h5 if not need filter data and save all loaded time range

                # '++filter.time_bad_intervals=[2021-06-02T13:49, now]', # todo
                # 'input.tables=["incl.*"]', # was set in probes config directory
                ## f'out.db_path={db_out}',
                # f'out.table=V_incl_bin{aggregate_period_s}s',
                'out.b_del_temp_db=True',
                # 'out.raw_db_path=None',
                f"out.text_path={'text_output' if aggr else 'None'}",  # text_output  # None to skip save text
                'program.verbose=INFO',
                'program.dask_scheduler=threads',
                # set additional parameters in probes config directory defined in hydra.searchpath earlier
                f'+probes={probes}',
                #f'out=out0',
                # '--config-path=cfg_proc',  # Primary config module 'inclinometer.cfg_proc' not found.
                # '--config-dir=cfg_proc'  # additional cfg dir
                'in.min_date=2023-09-06T16:00',
                # 'in.max_date=2023-05-23T19:40',
                # 'in.date_to_from_list=[2023-10-11T13:00:00,2000-01-01T00:00:05]',
                # '+in.date_to_from_lists={i28:[2023-06-30T11:47:26,2000-01-01T01:03:07]}',  # can use if load from text
                # Note: "*" is mandatory for regex, "incl" only used in raw files, but you can average data in processed db.
                ## "in.tables=['i.*']" if probes == 'inclinometers' else  # ['incl(23|30|32).*']  # ['i.*']
                ## "in.tables=['w.*']",                                    # ['w0[2-6].*']         # ['w.*']
                # hydra / sweeper / params =
                # hydra.job.chdir=true
                f"out.aggregate_period={aggr}s"  # {','.join(f'{a}s' for a in aggregate_period_s[probes])}",  # for multirun
                # '--multirun'
                ],
                fun=incl_h5clc_hy.main
            )


#{Ag: [[1,0,0],[0,1,0],[0,1,0]], Cg: [10,10,10], Ah: [[1,0,0],[0,1,0],[0,1,0]], Ch: [10,10,10], kVabs: [10, -10, -10, -3, 3, 70], azimuth_shift_deg: 0}

sys.argv = sys_argv_save
