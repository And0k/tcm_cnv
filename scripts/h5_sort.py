# Andrey Korzh, 20.10.2023
#
import logging
from pathlib import Path
from tcm import h5

def h5_sort(path_db_in, tables=None, path_db_out=None):
    failed_storages = h5.move_tables({
        'temp_db_path': Path(path_db_in),
        'db_path':      path_db_out or Path(path_db_in).with_name(f'{Path(path_db_in).stem}_sorted.h5.'),
        'b_del_temp_db': False
        # 'tables_log': []
    },
        tbl_names=[tables] if isinstance(tables, str) else tables,
        # do not sort (suppose data already sorted) - if we not set ``arguments`` to default None or "fast":
        # arguments=[
        #     '--chunkshape=auto', '--propindexes', '--checkCSI',
        #     '--verbose', '--complevel=9',
        #     '--overwrite-nodes']  # , '--complib=blosc' - lib not installed
    )
    if failed_storages:
        print('Failed:', failed_storages)
    else:
        print('Ok>')


if __name__ == '__main__':
    path_db_in = Path(r'd:\WorkData\_experiment\inclinometer\231010_stand,tank@i52-56\_raw\231011tank.raw.h5.')
    tables = 'incl53'
    h5_sort(path_db_in, tables)
