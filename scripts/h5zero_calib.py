from pathlib import Path
import pandas as pd
from datetime import datetime


from tcm.incl_h5clc_hy import get_coefs, coef_prepare, coefs_format_for_h5, gen_raw_from_h5
from tcm.utils2init import (
    init_logging,
    LoggingStyleAdapter,
    call_with_valid_kwargs,
    dir_create_if_need,
    set_field_if_no,
    this_prog_basename,
    my_logging
)
from tcm.h5inclinometer_coef import h5copy_coef

l = init_logging("", level_console="INFO")  # root logger
# logger=Path(__file__).stem - logging only this file, also not works for __name__ at all
lf = LoggingStyleAdapter(l)

tbl_in = "incl89"
path_in = r"D:\WorkData\BalticSea\240616_ABP56\inclinometer\_raw\240616.raw.h5"  # Path to the database


cfg_in_for_probes = {
    tbl_in.replace("incl", "i", 1): {
        "table": "incl89",  # Input table name or pattern.
        "path": Path(path_in),  # Path to the database
        "time_ranges": ["2024-06-30T00:00:00", "2024-06-30T00:10:00"],  # Time intervals for filtering data.
        # 'min_date': # Minimum date for filtering.
        # 'max_date': # Maximum date for filtering.
    }
}

cfg_in_common = {

}
cfg_in = {
    "coefs_path": path_in
}
cfg_out = {
    "tables": [tbl_in],  # Pattern for output table names.
    # 'tables_log': Pattern for log table names.
    "raw_db_path": Path(path_in),  # (str, optional): Path to the raw database.
    # 'not_joined_db_path' (str, optional): Path to the non-joined database.
}


cfg_in["coefs"] = get_coefs([cfg_in["coefs_path"]], tbl_in, coefs_ovr=None)


for df_raw, ((i1pid, pcid, path_csv), rec_raw, tbl_raw) in gen_raw_from_h5(
        cfg_in_for_probes, cfg_in_common, cfg_out, b_raw=True
    ):
    time_range_loaded = getattr(df_raw, 'index' if isinstance(df_raw, pd.DataFrame) else 'divisions')
    print("Loaded time range:", time_range_loaded)

    # Calibration
    coefs, coef_zeroing_matrix, dates, msg_coefs = coef_prepare(
        cfg_in["coefs"],
        time_ranges_zeroing=time_range_loaded,
        azimuth_add=0,
        coordinates=None,
        data=df_raw,
        data_date=time_range_loaded[0]
    )

    # print(coefs)
    coef_updated = {k: coefs[k] for k, v in dates.items() if v == True}

    lf.info(f'Updating coefficients {msg_coefs}in "{cfg_out['tables']}":\n{coef_updated}')
    # coef_dict # coefs_format_for_h5({k: coefs[k] for k, v in dates.items() if v == True}, pcid)
    save_args = {
        "dict_matrices": {
            **{f"//coef//{k}": v for k, v in coef_updated.items()},
            "//coef//pid": pcid,
            "//coef//date": datetime.now().replace(microsecond=0).isoformat(),
        },
        "h5file_dest": cfg_out["raw_db_path"],
        "tbl": cfg_out["tables"][0],
        "dates": dates,
    }

table_written = h5copy_coef(**save_args)
print("Ok")

"""
0.9999298289085138	3.802745989822893e-05	0.011846341751894165
3.802745989822893e-05	0.9999793919735936	-0.006419827258914228
-0.011846341751894165	0.006419827258914228	0.9999092208821074




        :param coefs: Dictionary of ndarrays, original loaded coefficients.
        Required fields:
        - ``azimuth_shift_deg``: ndarray, original azimuth shift degrees.
        - ``dates``: Dictionary, dates of the coefficients.
    :param time_ranges_zeroing: List of tuples, each containing start and end time strings for zeroing.
    :param azimuth_add: Float, azimuth addition value.
    :param coordinates: [Lat, Lon]: coordinates to get declination shift in addition to ``azimuth_add``
        Required fields:
        - ``latitude``: Float, latitude value.
        - ``longitude``: Float, longitude value.
    :param data: Dictionary containing data for coefficient zeroing.
    :param data_date: String, date of the data.
"""