from pathlib import Path
import numpy as np
from tcm import cfg_dataclasses as cfg_d
from tcm import incl_h5spectrum
from tcm.utils2init import st

# Run steps (inclusive):
st.start = 3  # 1
st.end = 1  # 3
st.go = True  # False #

# path_in = r"D:\WorkData\BalticSea\240616_ABP56\t-chain\240625@TCm1,2.csv"

# (  # cfg_d.main_call
#     [
#         f"out.dt_bins=[{','.join(str(b) for b in bins)}]",
#         # yaml files in probes configs in "defaults" dir
#         f'hydra.searchpath=["{search_path.as_posix()}"]',
#         f"+defaults@_global_={','.join(config_files)}",
#         "+in.date_to_from=[None,None,2024-06-04T13:00:00,2000-01-01T23:00:00]",
#         "in.min_date=2024-05-30T23:00:00",
#         "--multirun",
#     ],
#     my_hydra_module=incl_h5clc_hy,
# )


db_path_in = Path(r"C:\Work\_\t-chain\240625isolines(t)@TCm1,2.h5")



# Calculate spectrograms.
if st(3):  # Can be done at any time after step 1

    def raise_ni():
        raise NotImplementedError(
            "Can not proc probes having different fs in one run: you need to do it separately"
        )

    # prefix = prefix.replace("incl", "i")
    cfg_path = r"C:\Work\Python\AB_SIO_RAS\tcm\tcm\cfg\t_chain_PSD.yaml"
    # r"C:\Work\Python\AB_SIO_RAS\tcm\tcm\cfg\t_chain_PSD.yaml"
    # c:\Work\Python\AB_SIO_RAS\h5toGrid\inclinometer\scripts\cfg\incl_h5spectrum.yaml
    # Path(incl_h5clc.__file__).parent / "cfg" / f"incl_h5spectrum{db_path.stem}.yaml",
    # if no such file all settings are below
    args = [
        f"--db_path={db_path_in}",
        # C:\Work\_\t-chain\240625isolines(t)@TCm1,2.h5
        # "--tables_list", ",".join(
        #     [f"z_t{f'{t:g}'.replace('.','p')}" for t in np.arange(4.5, 7.5+0.001, 0.5)]
        #     # 'z_t4p5', 'z_t5', 'z_t5p5', 'z_t6', 'z_t6p5', 'z_t7', 'z_t7p5',
        # ),
        # '--aggregate_period', f'{aggregate_period_s}S' if aggregate_period_s else '',
        # f"--min_date={datetime64_str(min_date[0])}",
        # f"--max_date={datetime64_str(max_date[0])}",  # '2019-09-09T16:31:00',  #17:00:00
        # '--max_dict=M[xyz]:4096',  # use if db_path is not ends with .proc_noAvg.h5 i.e. need calc velocity
        # f"--out.db_path={db_path.with_name(db_path.stem.replace("incl", prefix))}_proc_psd.h5',
        # f'--table=psd{aggregate_period_s if aggregate_period_s else ""}',
        "--fs_float=0.016666666666666666",  # f'{fs(probes[0], prefix)}',
        # (lambda x: x == x[0])(np.vectorize(fs)(probes, prefix))).all() else raise_ni()
        #
        # '--time_range_zeroing_list', '2019-08-26T04:00:00, 2019-08-26T05:00:00'
        # '--verbose', 'DEBUG',
        # '--chunksize', '20000',
        # "--split_period=2H",
        "--b_interact=0",
    ]

    # single spectrum
    args += [
        "--split_period=",
        "--dt_interval_minutes=None",  # burst mode
        f"--fmin={1/(1000*3600)}",
        f"--fmax={1/(3600)}",
        f"--out.db_path={db_path_in.with_name('PSD_avg_all.h5')}",
    ]

    if False:
        if "w" in prefix:
            args += [
                "--split_period=1H",
                "--dt_interval_minutes=10",  # burst mode
                "--fmin=0.0001",
                "--fmax=4",
            ]
        else:
            # burst mode
            args += [
                "--split_period=30T",
                "--dt_interval_minutes=2",  # 1 + delta of inaccuracy
                "--fmin=0.03334",  # >= 1/(2*dt_interval_minutes)
                "--fmax=1",  # <= fs/2
            ]
        if False:
            args += [
                "--split_period=2H",
                "--fmin=0.0004",  # 0.0004
                "--fmax=1.05",
            ]

    incl_h5spectrum.main([cfg_path] + [a_splitted for a in args for a_splitted in a.split("=", 1)])
