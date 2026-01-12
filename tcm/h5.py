#!/usr/bin/env python
# coding:utf-8
"""

Author:  Andrey Korzh <ao.korzh@gmail.com>
"""

import logging
from pathlib import Path, PurePath
from contextlib import nullcontext
import re
import sys  # from sys import argv
import warnings
import itertools
from os import path as os_path, getcwd as os_getcwd, chdir as os_chdir, remove as os_remove
from datetime import timedelta, datetime
from functools import partial, wraps
from time import sleep
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    List,
    Set,
    Callable,
)
import numpy as np
import pandas as pd
from tables import NaturalNameWarning
from tables.exceptions import HDF5ExtError, ClosedFileError, NodeError
from tables.scripts.ptrepack import main as ptrepack

warnings.catch_warnings()
warnings.simplefilter("ignore", category=NaturalNameWarning)
# warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
# my
from .filters import inearestsorted, inearestsorted_around
from .utils2init import (
    set_field_if_no,
    dir_create_if_need,
    getDirBaseOut,
    Ex_nothing_done,
    standard_error_info,
    LoggingStyleAdapter,
    ExitStatus
)
from .utils_time import check_time_diff, minInterval, multiindex_timeindex, multiindex_replace, timezone_view

pd.set_option("io.hdf.default_format", "table")
# pd.pandas.set_option('display.max_columns', None)  # for better debug display
lf = LoggingStyleAdapter(logging.getLogger(__name__))


if __debug__:
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        lf.warning("matplotlib not installed, but may be needed for display data")

def main():
    """
    Purpose: execute query from command line and returns data to stdout
    (or writes to new hdf5 - to do) from PyTables hdf5 file
    Created: 30.07.2013

    """
    import argparse

    parser = argparse.ArgumentParser(description="Save part of source file.")
    parser.add_argument("Input", nargs=2, type=str, help="Source file full path, Node name")
    parser.add_argument("-columns", nargs="*", type=str)
    parser.add_argument("-where", nargs="*", type=str)
    parser.add_argument("-chunkDays", type=float)
    # to do:logging
    parser.add_argument("-saveTo", type=str, help="Save result (default: not save, index of first )")

    args = parser.parse_args()

    def proc_h5toh5(args):
        """
        args: argparse.Namespace, must have attributes:
            Input[0] - name of hdf5 PyTables file
            Input[1] - name of table in this file
            where    - query for data in this table
            args may have fields:
            columns  - name of columns to return
        :return: numpy array of int, indexes of data satisfied query
        """
        fileInF = args.Input[0]
        strProbe = args.Input[1]
        str_where = args.where
        with pd.get_store(fileInF, mode="r") as store:  # error if open with fileInF
            try:
                if not args.chunkDays:
                    args.chunkDays = 1
                if str_where:  # s=str_where[0]
                    Term = []
                    b_wait = False
                    for s in str_where:
                        if b_wait:
                            if s[-1] == '"':
                                b_wait = False
                                Term[-1] += f" {s[:-1]}"
                            else:
                                Term[-1] += f" {s}"
                        elif s[0] == '"':
                            b_wait = True
                            Term.append(s[1:])
                        else:
                            Term.append(s)
                    # Term= [pd.Term(s[1:-1]) if s[-1]==s[0]=='"' else pd.Term(s) for s in str_where]
                    # Term= pd.Term(str_where)
                    if args.columns:
                        df = store.select(strProbe, Term, columns=args.columns)
                    else:
                        df = store.select(strProbe, Term)
                    df = df.index
                    # start=0,  stop=10)
                    coord = store.select_as_coordinates(strProbe, Term).values[[0, -1]]
                else:
                    df = store.select_column(strProbe, "index")
                    coord = [0, df.shape[0]]
            except:
                if str_where:
                    #  this  is  in-memory  version  of  this  type  of  selection
                    df = store.select(strProbe)
                    coord = [0, df.shape[0]]
                    df = df[eval(str_where)]
                    df = df.index
                else:
                    df = store.select_column(strProbe, "index")
                    coord = [0, df.shape[0]]
        # df= store.get(strProbe)
        # store.close()
        if df.shape[0] > 0:
            tGrid = np.arange(
                df[0].date(),
                df[df.shape[0] - 1].date() + pd.Timedelta(days=1),
                pd.Timedelta(days=args.chunkDays),
                dtype="datetime64[D]",
            ).astype("datetime64[ns]")
            iOut = np.hstack([coord[0] + np.searchsorted(df.values, tGrid), coord[1]])
            if coord[0] == 0 and iOut[0] != 0:
                iOut = np.hstack([0, iOut])

        else:
            iOut = 0
        return iOut

    return proc_h5toh5(args)


if __name__ == "__main__":
    # sys.stdout.write('hallo\n')
    sys.stdout.write(str(main()))


def unzip_if_need(lst_of_lsts: Iterable[Union[Iterable[str], str]]) -> Iterator[str]:
    if isinstance(lst_of_lsts, set):
        lst_of_lsts = sorted(lst_of_lsts)
    for lsts in lst_of_lsts:
        if isinstance(lsts, str):
            yield lsts
        else:
            yield from lsts


def unzip_if_need_enumerated(lst_of_lsts: Iterable[Union[Iterable[str], str]]) -> Iterator[Tuple[int, str]]:
    """
    Enumerate each group of elements from 0. If element is not a group (str) just yield it with index 0
    :param lst_of_lsts:
    :return:
    """
    if isinstance(lst_of_lsts, set):
        lst_of_lsts = sorted(lst_of_lsts)
    for lsts in lst_of_lsts:
        if isinstance(lsts, str):
            yield (0, lsts)
        else:
            yield from enumerate(lsts)


def get_store_and_print_table(file_or_handle, strProbe):
    import pprint

    with (
            pd.HDFStore(file_or_handle, mode="r") if isinstance(file_or_handle, (str, PurePath)) else
            nullcontext(file_or_handle)
        ) as store:
        try:
            pprint.pprint(store.get_storer(strProbe).group.table)
        except AttributeError as e:
            print("Error", standard_error_info(e))
            print("Checking all root members:")
            nodes = store.root.__members__
            for n in nodes:
                print("  ", n)
                try:
                    pprint.pprint(store.get_storer(n))
                except Exception as e:
                    print(n, "error!", standard_error_info(e))
    # return store


def find_tables(store: pd.HDFStore, pattern_tables: str, parent_name=None) -> List[str]:
    """
    Get list of tables in hdf5 store node
    :param store: pandas hdf5 store
    :param pattern_tables: str, substring to search paths or regex if with '*'
    :param parent_name: str, substring to search parent paths or regex if with '*'
    :return: list of paths
    For example: w05* finds all tables started from w0
    """
    if parent_name is None:
        if "/" in pattern_tables:
            parent_name, pattern_tables = pattern_tables.rsplit("/", 1)
            if not parent_name:
                parent_name = "/"
        else:
            parent_name = "/"

    if "*" in parent_name:
        regex_parent = re.compile(parent_name)
        parent_names = {tbl for tbl in store.root.__members__ if regex_parent.match(tbl)}
    else:
        parent_names = [parent_name]  # store.get_storer(parent_name)

    if "*" in pattern_tables:
        regex_parent = re.compile(pattern_tables)
        regex_tables = lambda tbl: regex_parent.match(tbl)
    else:
        regex_tables = lambda tbl: pattern_tables in tbl
    tables = []

    for parent_name in parent_names:
        node = store.get_node(parent_name)
        if not node:
            continue
        for tbl in node.__members__:  # (store.get_storer(n).pathname for n in nodes):
            if tbl in ("table", "_i_table"):
                continue
            if regex_tables(tbl):
                tables.append(f"{parent_name}/{tbl}" if parent_name != "/" else tbl)
    lf.info(
        '{:d} "{:s}/{:s}" table{:s} found in {:s}',
        l := len(tables),
        parent_name if parent_name != "/" else "",
        pattern_tables,
        "s" if l != 1 else "",
        Path(store.filename).name,
    )
    tables.sort()
    return tables


query_range_pattern_default = "index>='{}' & index<='{}'"


def sel_index_and_istart(
    store: pd.HDFStore,
    tbl_name: str,
    query_range_lims: Optional[Iterable[Any]] = None,
    query_range_pattern: str = query_range_pattern_default,
    to_edge: Optional[timedelta] = None,
) -> Tuple[pd.Index, int]:
    """
    Get index and index[0] counter from start of stored table (called index coordinate in pandas)
    satisfying ``query_range_lims``
    :param store:
    :param tbl_name:
    :param query_range_lims: values to print in query_range_pattern
    :param query_range_pattern:
    :param to_edge: timedelta
    :return: (empty columns dataframe with index[range_query], coordinate index of range_query[0] in table)
    """
    if query_range_lims is None:  # select all
        df0 = store.select(tbl_name, columns=[])
        i_start = 0
    else:
        # Select reduced range and its starting db-index counter number (coordinate)
        # (Tell me, if you know, how to do this with only one query, please)
        if to_edge:
            query_range_lims = [pd.Timestamp(lim) for lim in query_range_lims]
            query_range_lims[0] -= to_edge
            if len(query_range_lims) > 1:
                query_range_lims[-1] += to_edge
        qstr = query_range_pattern.format(*query_range_lims)
        lf.info(f"query {tbl_name}: {qstr}... ")
        df0 = store.select(tbl_name, where=qstr, columns=[])
        try:
            i_start = store.select_as_coordinates(tbl_name, qstr)[0]
        except IndexError:
            i_start = 0
    return df0, i_start


def sel_interpolate(
    i_queried: str, store: pd.HDFStore, tbl_name: str, columns=None, time_points=None, method="linear"
):
    """

    :param i_queried:
    :param store:
    :param tbl_name:
    :param columns:
    :param time_points:
    :param method: see pandas interpolate. Most likely only 'linear' is relevant for 2 closest points
    :return: pandas Dataframe with out_cols columns
    """
    lf.info("time interpolating...")
    df = store.select(tbl_name, where=i_queried, columns=columns)
    if not (isinstance(time_points, pd.DatetimeIndex) or isinstance(time_points, pd.Timestamp)):
        t = pd.DatetimeIndex(time_points, tz=df.index.tz)  # to_datetime(t).tz_localize(tzinfo)
    else:
        t = time_points.tz_localize(df.index.tzinfo)
    # if not drop duplicates loc[t] will return many (all) rows having same index t, so we do it:
    new_index = df.index.union(t).drop_duplicates()

    # pd.Index(timezone_view(time_points, dt_from_utc=df.index.tzinfo._utcoffset))
    # except TypeError as e:  # if Cannot join tz-naive with tz-aware DatetimeIndex
    #     new_index = timezone_view(df.index, dt_from_utc=0) | pd.Index(timezone_view(time_points, dt_from_utc=0))

    df_interp_s = df.reindex(new_index).interpolate(
        method=method,
    )  # why not works fill_value=new_index[[0,-1]]?
    df_interp = df_interp_s.loc[t, :]
    return df_interp


def coords(
    store: pd.HDFStore,
    tbl_name: str,
    q_time: Optional[Sequence[Any]] = None,
    query_range_lims: Optional[Sequence[Any]] = None,
    query_range_pattern: Optional[str] = None,
    to_edge=None
) -> Tuple[Union[pd.Index, None], int, Union[List[int], np.ndarray]]:
    """
    Get table's index for ``q_time`` edges / ``query_range_lims`` and coordinates indexes of ``q_time`` in
    ``store`` table

    :param store:
    :param tbl_name: table name in ``store``
    :param q_time: optional, points. If strings - converts them to 'M8[ns]'.
    :param query_range_lims: optional, needed interval. If None, then use 1st and last of q_time.
    :param query_range_pattern:
    :param to_edge: if not None, then extend query_range_lims at both sides. Example: pd.Timedelta(minutes=10)
    :return (df0range.index, i0range, i_queried):
    - df0range.index: index[query_range_lims[0]:(query_range_lims[-1] + to_edge)];
    - i0range: starting index of returned df0range.index in store table;
    - i_queried: coordinates indexes of q_time in store table. It is a list only if 1st element is zero.
    If both q_time and query_range_lims are None then returns df0range=None, i_queried=[0, ``number of rows in table``]).
    We not return i_queried = [0, np.iinfo(np.intp).max] because trying loading bigger number of rows will raise
    ValueError in dask.read_hdf(stop=bigger).
    """
    if query_range_lims is None:
        if q_time is None:
            return None, 0, [0, store.get_storer(tbl_name).nrows]
        else:
            if isinstance(q_time[0], str):
                q_time = np.array(q_time, "M8[ns]")
            # needed interval from q_time edges: should be min and max of q_time
            # uses "index" value 2 times (if pattern not defined then it will be default):
            b_have_max_lim = (not query_range_pattern) or query_range_pattern.count("index") > 1

            # Make `query_range_lims` length compatible with `query_range_pattern`
            q_time_len = len(q_time)
            if b_have_max_lim:
                query_range_lims = list(q_time)*2 if q_time_len == 1 else q_time[:: (q_time_len - 1)]
            else:
                query_range_lims = q_time[:1]
    else:
        b_have_max_lim = len(query_range_lims)%2 == 0
    if not query_range_pattern:
        # Make `query_range_pattern` compatible with `query_range_lims` length
        query_range_pattern = query_range_pattern_default if b_have_max_lim else "index>='{}'"

    df0range, i0range = sel_index_and_istart(
        store,
        tbl_name,
        query_range_lims,
        query_range_pattern,
        to_edge=to_edge,
        # msg_add='with padding to edges'
    )
    if q_time is None:
        return df0range.index, 0, [0, df0range.index.size]

    i_queried = inearestsorted(df0range.index.values, np.array(q_time, df0range.index.dtype.str))
    return (
        df0range.index,
        i0range,
        (i_queried if b_have_max_lim else np.append(i_queried, df0range.index.size - 1)) + i0range
    )


def load_points(
    store: pd.HDFStore,
    tbl_name: str,
    columns: Optional[Sequence[Union[str, int]]] = None,
    time_points: Optional[Union[np.ndarray, pd.Series, Sequence[int]]] = None,
    dt_check_tolerance=pd.Timedelta(seconds=1),
    query_range_lims: Optional[Union[np.ndarray, pd.Series, Sequence[int]]] = None,
    query_range_pattern: str = query_range_pattern_default,
    interpolate: str = "time",
    to_edge: Optional[timedelta] = timedelta(),
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Get hdf5 data with index near the time points or between time ranges, or/and in specified query range
    :param store: pandas hdf5 store
    :param tbl_name: table having sorted index   # tbl_name in store.root.members__
    :param columns: a list of columns that if not None, will limit the return columns
    :param time_points: numpy.array(dtype='datetime64[ns]') to return rows with closest index. If None then
    uses time_ranges
    :param dt_check_tolerance: pd.Timedelta, display warning if found index far from requested values
    :param query_range_lims: initial data range limits to query data, 10 minutes margin will be added for
    time_points/time_ranges. Note: useful to reduce size of intermediate loading index even if
    time_points/time_ranges is used
    :param query_range_pattern: format pattern for query with query_range_lims
    :param interpolate: "method" arg of pandas.Series.interpolate. If not interpolate, then return
    closest points
    :return: (df, bbad):
        df - table of found points, bbad - boolean array returned by other_filters.check_time_diff() or
        df - dataframe of query_range_lims if no time_ranges nor time_points

    Note: `query_range_pattern` will be used only if `query_range_lims` specified
    My use:
    select(
        store, cfg['in']['table_nav'], ['Lat', 'Lon', 'DepEcho'], dfL.index,
        query_range_lims=(dfL.index[0], dfL['DateEnd'][-1]),
        query_range_pattern=cfg['process']['dt_search_nav_tolerance']
    )
    """
    if (time_points is not None) and len(time_points) and any(time_points):
        try:
            if isinstance(time_points, list):
                if isinstance(time_points[0], datetime):
                    time_points = np.array([int(t.timestamp()) for t in time_points], "M8[s]")
                else:
                    time_points = np.array(time_points)  # more check/conversion needed!
            elif not np.issubdtype(time_points.dtype, np.datetime64):  # '<M8[ns]'
                time_points = time_points.values
        except TypeError:  # not numpy 'datetime64[ns]'
            time_points = time_points.values

    input_data = None
    while True:
        df_index_range, i0range, i_queried = coords(
            store, tbl_name, time_points, query_range_lims, query_range_pattern, to_edge=to_edge
        )
        if len(i_queried):
            break
        input_data = input(  # todo recover from file name
            f'No data for specified interval in "{tbl_name}"! Input {columns} for {time_points}\n>'
        ).strip(" [];,.")  # remove some symbols at edges that seems logically good but not needed/supported
        if not input_data:
            lf.info(
                f"No input, hmmm... - Then trying select nearest time in ±1H-increased searching range"
            )
            if to_edge is None:
                to_edge = timedelta()
            to_edge += timedelta(hours=1)
            continue

        try:
            df = pd.DataFrame.from_records(
                [
                    [float(word) for word in row.replace(",", " ").split()]
                    for row in input_data.split(";")
                ],
                columns=columns,
                index=time_points,
            )
            lf.info(f"Input accepted: {df}")
            dt = np.zeros_like(time_points, dtype="timedelta64[ns]")
        except ValueError as e:
            raise IndexError("Error with input data")

    bbad, dt = check_time_diff(
        t_queried=time_points,
        t_found=df_index_range[i_queried - i0range].values,
        dt_warn=dt_check_tolerance,
        return_diffs=True,
    )
    if any(bbad) and interpolate:
        i_queried = inearestsorted_around(df_index_range.values, time_points) + i0range
        df = sel_interpolate(
            i_queried, store, tbl_name, columns=columns, time_points=time_points, method=interpolate
        )
    else:
        df = store.select(tbl_name, where=i_queried, columns=columns)
        if input_data == '':  # user expects that we loaded data from DB from increased searching range
            lf.info(f"Data found: {df}")
    return df, dt


def load_range(
    store: pd.HDFStore,
    tbl_name: str,
    columns: Optional[Sequence[Union[str, int]]] = None,
    query_range_lims: Optional[Union[pd.Series, Sequence[Any]]] = None,
    query_range_pattern: str = query_range_pattern_default,
) -> pd.DataFrame:
    """
    Get hdf5 data with index near the time points or between time ranges, or/and in specified query range
    :param store: pandas hdf5 store
    :param tbl_name: table having sorted index   # tbl_name in store.root.members__
    :param columns: a list of columns that if not None, will limit the return columns
    :param query_range_lims: data range limits to query data
    :param query_range_pattern: format pattern for query with query_range_lims, will be used only if query_range_lims
    specified
    :return: df - table of found range
    """

    df = store.select(
        tbl_name,
        where=query_range_pattern.format(*query_range_lims) if query_range_pattern else None,
        columns=columns,
    )
    return df


def load_ranges(
    store: pd.HDFStore, table: str, t_intervals=None, query_range_pattern=query_range_pattern_default
) -> pd.DataFrame:
    """
    Load data
    :param t_intervals: an even sequence of datetimes or strings convertible to index type values. Each pair
    defines edges of data that will be concatenated. 1st and last must be min and max values in sequence.
    :param table:
    :return:
    """

    n = len(t_intervals) if t_intervals is not None else 0
    if n > 2:
        query_range_pattern = "|".join(
            f"({query_range_pattern.format(*query_range_lims)})"
            for query_range_lims in ((lambda x=iter(t_intervals): zip(x, x))())
        )
    elif n < 2:
        query_range_pattern = None
        t_intervals = []
    df = load_range(
        store, table, query_range_lims=t_intervals[0 :: (n - 1)], query_range_pattern=query_range_pattern
    )
    return df


def append_data(df: pd.DataFrame, tbl_name: str, cfg_out: Mapping[str, Any], **kwargs):
    df.to_hdf(
        cfg_out["db"],
        key=tbl_name,
        append=True,
        data_columns=cfg_out.get("data_columns", True),
        format="table",
        index=False,
        dropna=cfg_out.get("dropna", not cfg_out.get("b_insert_separator")),
        **kwargs,
    )
    return tbl_name


def append_log(df: pd.DataFrame, tbl_name: str, cfg_out: Mapping[str, Any]) -> str:
    """
    Append a pandas DataFrame to an HDF5 store log table with specified configuration settings.

    :param df: metadata DataFrame to append to the log table.
    :param tbl_name: Name of the log table.
    :param cfg_out: Configuration dictionary with the following fields:
    - db: HDF5 file handle to the store.
    - nfiles: Expected number of rows in the log table, used to optimize file size.
    - logfield_fileName_len: Optional int or dictionary specifying the maximum field length for string columns. The keys are column names and the values are the lengths.
    :return: Name of the log table to which the DataFrame was appended.
    """
    str_field_len = cfg_out.get("logfield_fileName_len", {})
    if str_field_len:
        pass  # str_field_len = {'values': logfield_fileName_len}
    else:
        try:  #
            m = cfg_out["db"].get_storer(tbl_name).table
            strcolnames = m._strcolnames
            str_field_len = {col: m.coldtypes[col].itemsize for col in strcolnames}
        except:
            pass

    cfg_out["db"].append(
        tbl_name,
        df,
        data_columns=True,
        expectedrows=cfg_out.get("nfiles", 1),
        index=False,
        min_itemsize=str_field_len,
    )
    return tbl_name


def replace_bad_db(temp_db_path: Path, db_path: Optional[Path] = None):
    """
    Make copy of bad temp_db_path and replace it with copy of db_path or delete if failed
    :param temp_db_path:
    :param db_path:
    :return:
    """
    from shutil import copyfile

    temp_db_path_copy = temp_db_path.with_suffix(".bad_db.h5")
    try:
        temp_db_path.replace(temp_db_path_copy)
    except:
        lf.exception(f"replace to {temp_db_path_copy} failed. Trying copyfile method")
        copyfile(
            temp_db_path, temp_db_path_copy
        )  # I want out['temp_db_path'].rename(temp_db_path_copy) but get PermissionError

    if db_path is None and temp_db_path.stem.endswith("_not_sorted"):
        db_path = temp_db_path.with_name(temp_db_path.name.replace("_not_sorted", "", 1))
    if db_path is None:
        return temp_db_path_copy

    try:
        copyfile(db_path, temp_db_path)  # I want temp_db_path.unlink() but get PermissionError
    except FileNotFoundError:
        try:
            temp_db_path.unlink()
        except PermissionError:
            open(temp_db_path, "a").close()
            try:
                temp_db_path.unlink()
            except PermissionError:
                copyfile(db_path, temp_db_path)
    return temp_db_path_copy


def remove(db: pd.HDFStore, node: Optional[str] = None, query: Optional[str] = None):
    """
    Removes table or rows from table if query is not None, skips if not(node) or no such node in currently open db.
    :param db: pandas hdf5 store
    :param node: str, table name
    :param query:  str, ``where`` argument of ``db.remove`` method
    :modifies db: if reopening here (dew to pandas bug?)
    Note: Raises HDF5ExtError on KeyError if no such node in db.filename and it is have not opened
    """

    try:
        was = node and (node in db)
        if was:
            mdg_query = f" {query} from" if query is not None else ""
            db.remove(node, where=query)
            lf.info("table {} removed", node)
    except (KeyError, HDF5ExtError) as e:
        lf.info("Trouble when removing{} {}. Solve pandas bug by reopen store.", mdg_query, node)
        sleep(1)
        db.close()
        # db_filename = db.filename
        # db = None
        sleep(1)
        db.open(mode="r+")  # reading and writing, file must exist
        try:
            db.remove(node, where=query)
            return True
        except KeyError:
            raise HDF5ExtError('Can not remove{} table "{}"'.format(mdg_query, node))
    return was


def remove_tables(db: pd.HDFStore, tables: Iterable[str], tables_log: Iterable[str], temp_db_path=None):
    """
    Removes (tables + tables_log) from db in sorted order with not trying to delete deleted children.
    Retries on error, flushes operation
    :param db: pandas hdf5 store
    tables names:
    :param tables: list of str
    :param tables_log: list of str
    :param temp_db_path: path of db. Used to reopen db if removing from `db` is not succeed
    :return: db
    """
    tbl_prev = "?"  # Warning side effect: ignores 1st table if its name starts with '?'
    for tbl in sorted(tables + tables_log):
        if len(tbl_prev) < len(tbl) and tbl.startswith(tbl_prev) and tbl[len(tbl_prev)] == "/":
            continue  # parent of this nested have deleted on previous iteration
        for i in range(1, 4):  # for retry, may be not need
            try:
                remove(db, tbl)
                tbl_prev = tbl
                break
            except ClosedFileError as e:  # file is not open
                lf.error("waiting {:d} (/3) because of error: {:s}", i, str(e))
                sleep(i)
            # except HDF5ExtError as e:
            #     break  # nothing to remove
        else:
            lf.error("failed => Reopening...")
            if temp_db_path:
                db = pd.HDFStore(temp_db_path)
            else:
                db.open(mode="r+")
            remove(db, tbl)
    db.flush(fsync=True)
    return db


# ----------------------------------------------------------------------
class ReplaceTableKeepingChilds:
    """
    Saves childs (before You delete tbl_parent)
    #for find_tables(store, '', parent_name=tbl_parent)

    cfg_out must have field: 'db' - handle of opened store
    """

    def __init__(
        self,
        dfs: Union[pd.DataFrame, List[pd.DataFrame]],
        tbl_parent: str,
        cfg_out: Mapping[str, Any],
        write_fun: Optional[Callable[[pd.DataFrame, str, Dict[str, Any]], None]] = None,
    ):
        self.cfg_out = cfg_out
        self.tbl_parent = tbl_parent[1:] if tbl_parent.startswith("/") else tbl_parent
        self.dfs = [dfs] if isinstance(dfs, pd.DataFrame) else dfs
        self.write_fun = write_fun
        self.temp_group = "to_copy_back"

    def __enter__(self):
        self.childs = []
        try:
            parent_group = self.cfg_out["db"].get_storer(self.tbl_parent).group
            nodes = parent_group.__members__
            self.childs = [f"/{self.tbl_parent}/{g}" for g in nodes if (g != "table") and (g != "_i_table")]
            if self.childs:
                lf.info("found {} children of {}. Copying...", len(self.childs), self.tbl_parent)
                for i, tbl in enumerate(self.childs):
                    try:
                        self.cfg_out["db"]._handle.move_node(
                            tbl, newparent=f"/{self.temp_group}", createparents=True, overwrite=True
                        )
                    except HDF5ExtError:
                        if i == 0:  # try another temp node
                            self.temp_group = "to_copy_back2"
                            self.cfg_out["db"]._handle.move_node(
                                tbl, newparent=f"/{self.temp_group}", createparents=True, overwrite=True
                            )
                        else:
                            raise

                self.cfg_out["db"].flush(fsync=True)

        except AttributeError:
            pass  # print(tbl_parent + ' has no childs')
        # Make index to be UTC

        # remove parent table that must be written back in "with" block
        try:
            remove(self.cfg_out["db"], self.tbl_parent)
        except KeyError:
            print("was removed?")

        return self.childs

    def __exit__(self, exc_type, ex_value, ex_traceback):
        # write parent table

        if len(self.dfs):
            if self.write_fun is None:

                def write_fun(df, tbl, cfg):
                    return df.to_hdf(
                        cfg["db"], tbl, format="table", data_columns=True, append=False, index=False
                    )

                self.write_fun = write_fun

            for df in self.dfs:
                self.write_fun(df, self.tbl_parent, self.cfg_out)
            self.cfg_out["db"].create_table_index(self.tbl_parent, columns=["index"], kind="full")

        # write childs back
        self.cfg_out["db"].flush(fsync=True)
        if exc_type is None:
            for tbl in self.childs:
                self.cfg_out["db"]._handle.move_node(
                    tbl.replace(self.tbl_parent, self.temp_group, 1),
                    newparent=f"/{self.tbl_parent}",
                    createparents=True,
                    overwrite=True,
                )
        # cfg_out['db'].move('/'.join(tbl.replace(tbl_parent, self.temp_group), tbl))
        # cfg_out['db'][tbl] = df # need to_hdf(format=table)
        return False


# ----------------------------------------------------------------------
def remove_duplicates(cfg, cfg_table_keys: Iterable[Union[Iterable[str], str]]) -> Set[str]:
    """
    Remove duplicates inplace
    :param cfg: dict with keys specified in cfg_table_keys
    :param cfg_table_keys: list, in which 'tables_log' means that cfg['tables_log'] is a log table. Alternatively group tables in subsequences such that log tables names is after data table in each subsequence (cfg[cfg_table_keys[group]])
    :return dup_tbl_set: tables that still have duplicates
    """

    # load data frames from store to memory removing duplicates
    dfs = {}
    dup_tbl_set = set()  # will remove tables if will found duplicates
    for cfgListName in cfg_table_keys:
        for tbl in unzip_if_need(cfg[cfgListName]):
            if tbl in cfg["db"]:
                ind_series = cfg["db"].select_column(tbl, "index")
                # dfs[tbl].index.is_monotonic_increasing? .is_unique()?
                b_dup = ind_series.duplicated(keep="last")
                if b_dup.any():
                    i_dup = b_dup[b_dup].index
                    lf.info(
                        "deleting {} duplicates in {} (first at {}){}",
                        len(i_dup),
                        tbl,
                        ind_series[i_dup[0]],
                        ""
                        if i_dup.size < 50 or Path(cfg["db"].filename).stem.endswith("not_sorted")
                        else ". Note: store size will not shrinked!",
                    )  # if it is in temp db to copy from then it is ok
                    try:
                        cfg["db"].remove(tbl, where=i_dup)  # may be very long.
                        # todo: if many to delete try remove_duplicates_by_loading()
                    except:
                        lf.exception("can not delete duplicates")
                        dup_tbl_set.add(tbl)
    return dup_tbl_set


def remove_duplicates_by_loading(cfg, cfg_table_keys: Iterable[Union[Iterable[str], str]]) -> Set[str]:
    """
    Remove duplicates by coping tables to memory, keep last. todo: merge fields
    :param cfg: dict with keys:
        keys specified by cfg_table_keys
        chunksize - for data table
        logfield_fileName_len, nfiles - for log table
    :param cfg_table_keys: list, in which 'tables_log' means that cfg['tables_log'] is a log table. Alternatively group tables in subsequences such that log tables names is after data table in each subsequence (cfg[cfg_table_keys[group]])
    :return dup_tbl_set: tables that still have duplicates
    """
    cfg["db"].flush(fsync=True)
    # not worked without fsync=True (loading will give only part of data), but this worked:
    # cfg["db"].close()
    # cfg["db"].open("r+")

    # load data frames from store to memory removing duplicates
    dfs = {}
    dup_tbl_set = set()  # remove tables if we will find duplicates
    for cfgListName in cfg_table_keys:
        for tbl in unzip_if_need(cfg[cfgListName]):
            if tbl in cfg["db"]:
                dfs[tbl] = cfg["db"][tbl]
                # dfs[tbl].index.is_monotonic_increasing? .is_unique()?
                b_dup = dfs[tbl].index.duplicated(keep="last")
                if np.any(b_dup):
                    dup_tbl_set.add(tbl)
                    lf.info(
                        "{} duplicates in {} (first at {})",
                        sum(b_dup),
                        tbl,
                        dfs[tbl].index[np.flatnonzero(b_dup)[0]],
                    )
                    dfs[tbl] = dfs[tbl][~b_dup]

    # update data frames in store
    if len(dup_tbl_set):
        lf.info("Removed duplicates. ")
        for cfgListName in cfg_table_keys:
            for i_in_group, tbl in unzip_if_need_enumerated(cfg[cfgListName]):
                if tbl in dup_tbl_set:
                    try:
                        with ReplaceTableKeepingChilds(
                            [dfs[tbl]],
                            tbl,
                            cfg,
                            append_log if (cfgListName == "tables_log" or i_in_group > 0) else append_data,
                        ):
                            pass
                            # cfg['db'].append(tbl, dfs[tbl], data_columns=True, index=False, **(
                            #     {'expectedrows': cfg['nfiles'],
                            #      'min_itemsize': {'values': cfg['logfield_fileName_len']}
                            #      } if (cfgListName == 'tables_log' or i_in_group > 0) else
                            #     {'chunksize': cfg['chunksize']
                            #      }
                            #     ))

                            dup_tbl_set.discard(tbl)
                    except Exception as e:
                        lf.exception("Table {:s} not recorded because of error when removing duplicates", tbl)
                        # cfg['db'][tbl].drop_duplicates(keep='last', inplace=True) #returns None
    else:
        lf.info("Not need remove duplicates. ")
    return dup_tbl_set


def create_indexes(cfg_out, cfg_table_keys):
    """
    Create full indexes. That is mandatory before using pandas `ptrepack` in our `move_tables()`
    :param cfg_out: must hav fields
    - 'db': handle of opened HDF5Store
    - fields specified in :param cfg_table_keys where values are table names that need index. Special field name:
        - 'tables_log': means that cfg_out['tables_log'] is a log table
    - 'index_level2_cols': second level for Multiindex (only 2 level supported, 1st is always named 'index')
    :param cfg_table_keys: list of cfg_out field names having set of (tuples) names of tables that need index: instead of using 'tables_log' for log tables the set can contain subsequences where log tables names fields will be after data table in each subsequence
    :return:
    """
    lf.debug("Creating index")
    for cfgListName in cfg_table_keys:
        for i_in_group, tbl in unzip_if_need_enumerated(cfg_out[cfgListName]):
            if not tbl:
                continue
            try:
                if i_in_group == 0 or cfgListName != "tables_log":  # not nested (log) table
                    navp_all_index, level2_index = multiindex_timeindex(cfg_out["db"][tbl].index)
                else:
                    level2_index = None
                columns = ["index"] if level2_index is None else ["index", cfg_out["index_level2_cols"]]
                cfg_out["db"].create_table_index(tbl, columns=columns, kind="full")  # ,optlevel=9
            # except KeyError:
            #     pass  # 'No object named ... in the file'
            except Exception as e:
                lf.warning('Index in table "{}" not created - error: {}', tbl, standard_error_info(e))
            # except TypeError:
            #     print('can not create index for table "{}"'.format(tbl))


def close(cfg_out: Mapping[str, Any]) -> None:
    """
    Closes cfg_out['db'] store, removes duplicates (if needed) and creates indexes
    :param cfg_out: dict, with optional fields:
    - b_remove_duplicates: True - to remove duplicates
    - tables_written: cfg_out fields where stored table names (to create indexes), default ('tables', 'tables_log').
    These fields must be in cfg_out.
    :return: None
    """
    try:
        print("")
        cfg_table_keys = ["tables_written"] if ("tables_written" in cfg_out) else ("tables", "tables_log")
        if cfg_out["b_remove_duplicates"]:
            tbl_dups = remove_duplicates_by_loading(cfg_out, cfg_table_keys=cfg_table_keys)
            # or remove_duplicates() but it can take very long time
        create_indexes(cfg_out, cfg_table_keys)
    except Exception as e:
        lf.exception("\nError of adding data to temporary store: ")

        import traceback
        import code
        from sys import exc_info as sys_exc_info

        tb = sys_exc_info()[2]  # type, value,
        traceback.print_exc()
        last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
        frame = last_frame().tb_frame
        ns = dict(frame.f_globals)
        ns.update(frame.f_locals)
        code.interact(local=ns)
    finally:
        try:
            if cfg_out["db"] is None:
                return
            cfg_out["db"].close()
        except HDF5ExtError:
            lf.exception(f"Error closing: {cfg_out['db']}")
        if cfg_out["db"].is_open:
            print("Wait store closing...")
            sleep(2)
            if cfg_out["db"].is_open:
                cfg_out["db_is_bad"] = True  # failed closing

        cfg_out["db"] = None
        return


def sort_pack(
    path: str,
    out_name: str,
    table_node: str,
    arguments: Optional[Sequence[str]] = None,
    addargs: Optional[Sequence[str]] = None,
    b_remove: Optional[bool] = False,
    col_sort: Optional[str] = "index",
):
    """
    Compress and save table (with sorting by index) from hdf5 file at `path` to hdf5 with `out_name` using
    ``ptrepack`` utility. Output hdf5 file can contain other tables.
    :param path: - hdf5 source full file name
    :param out_name: base file name + ext of cumulative hdf5 store only. Name should be different to that in
    source path
    :param table_node: node name in hdf5 source file
    :param arguments: list, 'fast' or None. None is equal to ['--chunkshape=auto',
        '--complevel=9', '--complib=zlib',f'--sortby={col_sort}', '--overwrite-nodes']
    :param addargs: list, extend arguments with more parameters
    :param b_remove: hdf5 source file will be deleted after operation!
    :param col_sort:
    :return: full path of cumulative hdf5 store
    Note: ``ptrepack`` not closes hdf5 source if not finds data!
    """

    h5dir, h5source = os_path.split(path)
    h5out_path = os_path.join(h5dir, out_name)
    if not table_node:
        return h5out_path
    print(f"sort&pack({table_node}) to {out_name}")
    path_prev = os_getcwd()
    argv_prev = sys.argv.copy()
    os_chdir(h5dir)  # os.getcwd()
    try:  # Extend selected default set of arguments
        if arguments == "fast":  # bRemove= False, bFast= False,
            arguments = ["--chunkshape=auto", "--overwrite-nodes", "--dont-regenerate-old-indexes"]
            if addargs:
                arguments.extend(addargs)
        else:
            if arguments is None:
                arguments = [
                    "--checkCSI",
                    "--chunkshape=auto",
                    "--propindexes",
                    "--complevel=9",
                    "--complib=zlib",
                    f"--sortby={col_sort}",
                    "--overwrite-nodes",
                ]
            if addargs:
                arguments.extend(addargs)
        # arguments + [sourcefile:sourcegroup, destfile:destgroup]
        sys.argv[1:] = arguments + [f"{h5}:/{table_node}" for h5 in (h5source, out_name)]
        # --complib=blosc --checkCSI=True

        ptrepack()
        # then copy children  # '--non-recursive' requires sys.argv[1:] = arguments + [f'{h5source}:/{table_node}/table', f'{out_name}:/{table_node}/table'] i.e. can not copy indexes
        # with pd.HDFStore(h5out_path, 'a') as store_out, pd.HDFStore(path, 'r') as store_in:
        #     parent_group_in = store_in.get_storer(f'/{table_node}').group
        #     parent_group_out = store_out.get_storer(f'/{table_node}').group
        #     parent_group_in._f_copy_children(parent_group_out, overwrite=True)
        # nodes = parent_group.__members__
        # childs = [f'/{table_node}/{g}' for g in nodes if (g != 'table') and (g != '_i_table')]
        # if childs:
        #     lf.info('found {} children of {}. Copying...', len(childs), table_node)
        #     for i, tbl in enumerate(childs):
        #         store_in._handle.move_node(tbl,
        #                                              newparent=f'/{self.temp_group}',
        #                                              createparents=True,
        #                                              overwrite=True)

    except Exception as e:
        tbl_cur = "ptrepack failed!"
        try:
            if f"--sortby={col_sort}" in arguments:
                # check that requirement of fool index is recursively satisfied
                with pd.HDFStore(path) as store:
                    print("Trying again:\n\t1. Creating index for all childs not having one...")
                    nodes = store.get_node(table_node).__members__
                    for n in nodes:
                        tbl_cur = table_node if n == "table" else f"{table_node}/{n}"
                        try:
                            store_tbl = store.get_storer(tbl_cur)
                        except AttributeError:
                            raise HDF5ExtError(f"Table {tbl_cur} error!")
                        if "index" not in store_tbl.group.table.colindexes:
                            print(tbl_cur, end=" - was no indexes, ")
                            try:
                                store.create_table_index(tbl_cur, columns=["index"], kind="full")
                                print("\n")
                            except Exception as ee:  # store.get_storer(tbl_cur).group AttributeError: 'UnImplemented' object has no attribute 'description'
                                print(
                                    "Error:",
                                    standard_error_info(ee),
                                    f'- creating index on table "{tbl_cur}" is not success.',
                                )
                            store.flush(fsync=True)
                        else:
                            print(
                                tbl_cur,
                                store.get_storer(tbl_cur).group.table.colindexes,
                                end=" - was index. ",
                            )
                    # store.get_storer('df').table.reindex_dirty() ?
                print("\n\t2. Restart...")
                ptrepack()
                print("\n\t- Ok")
        except HDF5ExtError:
            raise
        except Exception as ee:  # store.get_storer(tbl_cur).group AttributeError: 'UnImplemented' object has no attribute 'description'
            print("Error:", standard_error_info(ee), f"- no success.")
            # try without --propindexes yet?
            raise e
        except:
            print("some error")
    finally:
        os_chdir(path_prev)
        sys.argv = argv_prev

    if b_remove:
        try:
            os_remove(path)
        except:
            print(f'can\'t remove temporary file "{path}"')
    return h5out_path


def move_tables(
    cfg_out, tbl_names: Union[Sequence[str], Sequence[Sequence[str]], None] = None, **kwargs
) -> Dict[str, str]:
    """
    Copy pytables tables `tbl_names` from one store to another using `ptrepack` utility. If fail to store
    in specified location then creates new store and tries to save there.
    :param cfg_out: dict - must have fields:
    - temp_db_path: source of not sorted tables, if None tries to use `cfg_out['db'].filename` if ends with
    '_not_sorted.h5'
    - db_path: pathlib.Path, full path name (extension ".h5" will be added if absent) of hdf store to put
    - tables, tables_log: Sequence[str], if tbl_names not specified
    - b_del_temp_db: bool, remove source store after move tables. If False (default) then deletes nothing
    - addargs: ptrepack params, they will be added to the defaults specified in sort_pack()
    :param tbl_names: list of strings or list of lists (or tuples) of strings. List of lists is useful to keep
    order of operation: put nested tables last.
    :param kwargs: ptrepack params
    Note: ``ptrepack`` not closes hdf5 source if it not finds data!
    Note: Not need specify childs (tables_log) if '--non-recursive' not in kwargs
        Strings are names of hdf5 tables to copy
    :return: Empty dict if all success else if we have errors - Dict [tbl: HDF5store file name] of locations
    of last tried savings for each table
    """
    failed_storages = {}
    if tbl_names is None:  # copy all cfg_out tables
        tbl_names = cfg_out["tables"].copy()
        if cfg_out.get("tables_log"):
            tbl_names += cfg_out["tables_log"]
        tables_top_level = cfg_out["tables"] if len(cfg_out["tables"]) else cfg_out["tables_log"]
    else:
        tables_top_level = []
        tbl_prev = "?"  # Warning side effect: ignores 1st table if its name starts with '?'
        for i, tbl in unzip_if_need_enumerated(tbl_names):
            if i == 0 and not tbl.startswith(f"{tbl_prev}/"):
                tables_top_level.append(tbl)
                tbl_prev = tbl
    tables = list(unzip_if_need(tbl_names))
    if tables:
        lf.info("moving tables {:s} to {:s}:", ", ".join(tables), cfg_out["db_path"].name)
    else:
        raise Ex_nothing_done("no tables to move")
    try:
        temp_db_path = cfg_out["temp_db_path"]
    except KeyError:
        try:
            temp_db_path = cfg_out['db'].filename
            if temp_db_path.endswith('_not_sorted.h5'):  # ok, default temp_db suffix
                temp_db_path = Path(temp_db_path)
            else:
                raise KeyError("temp_db_path")
        except (KeyError, AttributeError):
            raise KeyError("temp_db_path")

    ptrepack_add_args = cfg_out.get("addargs", [])
    if "--overwrite" in ptrepack_add_args:
        if len(tables_top_level) > 1:
            lf.error(
                'in "--overwrite" mode with move many tables: will remove each previous result after each '
                'table and only last table wins!'
            )
    elif "--overwrite-nodes" not in ptrepack_add_args:  # default: False
        # sort_pack can not remove/update dest table, and even corrupt it if existed, so we do:
        try:
            with pd.HDFStore(cfg_out["db_path"]) as store:
                # remove(store, tbl)
                for tbl in tables_top_level:
                    remove(store, tbl)

        except HDF5ExtError:
            file_bad = Path(cfg_out["db_path"])
            file_bad_keeping = file_bad.with_suffix(".bad.h5")
            lf.exception(
                'Bad output file - can not use!!! Renaming to "{:s}". Delete it if not useful',
                str(file_bad_keeping),
            )
            file_bad.replace(file_bad_keeping)
            lf.warning(
                "Renamed: old data (if any) will not be in {:s}!!! Writing current data...", str(file_bad)
            )

    recover_indexes(temp_db_path, tables, cfg_out, failed_storages)

    tables_all_or_top = tables if "--non-recursive" in ptrepack_add_args else tables_top_level
    if any(tables_all_or_top):
        # remove source store only after last table has copied
        b_when_remove_store = [False] * len(tables_all_or_top)
        b_when_remove_store[-1] = cfg_out.get("b_del_temp_db")
        i = 0
        for tbl, b_remove in zip(tables_all_or_top, b_when_remove_store):
            # if i == 2:  # no more deletions - to not delete parent table before writing child
            #     try:
            #         ptrepack_add_args.remove('--overwrite')
            #     except ValueError:
            #         pass
            try:
                sort_pack(
                    temp_db_path,
                    cfg_out["db_path"].with_suffix(".h5").name,
                    tbl,
                    addargs=ptrepack_add_args,
                    b_remove=b_remove,
                    **kwargs,
                )
                sleep(2)
            except Exception as e:
                lf.error(
                    'Error: "{}"\nwhen write table "{}" from {} to {}',
                    e,
                    tbl,
                    temp_db_path,
                    cfg_out["db_path"],
                )
    else:
        raise Ex_nothing_done(f"Not valid table names: {tbl_names}!")
    return failed_storages

def recover_indexes(
    temp_db_path, tables, cfg_out: MutableMapping[str, Any], failed_storages: MutableMapping[str, str]
):
    with pd.HDFStore(temp_db_path) as store_in:  # pd.HDFStore(cfg_out['db_path']) as store,
        for tbl in tables:
            try:
                _ = store_in.get_storer(tbl).group.table.colindexes
            except KeyError as e:
                failed_storages[tbl] = temp_db_path.name
                lf.error("move_tables({:s}) failed: {}, continue...", tbl, e)
                continue
            if "index" not in _:
                print(tbl, end=" - was no indexes, creating.")
                try:
                    try:
                        store_in.create_table_index(tbl, columns=["index"], kind="full")
                    except (
                        NodeError
                    ):  # NodeError: group ``/tr2/log/_i_table`` already has a child node named ``index``
                        store_in.remove(f"{tbl}/_i_table")
                        store_in.create_table_index(tbl, columns=["index"], kind="full")
                    continue
                except (HDF5ExtError, NodeError):
                    lf.error("move_tables({:s}): failed to create indexes", tbl)
                    failed_storages[tbl] = temp_db_path.name
                    if cfg_out.get("recreate_index_tables_set"):
                        cfg_out["recreate_index_tables_set"].add(tbl)
                    else:
                        cfg_out["recreate_index_tables_set"] = {tbl}
            if cfg_out.get("recreate_index_tables_set") and tbl in cfg_out["recreate_index_tables_set"]:
                print(
                    tbl,
                    end=" - was indexes, but recreating by loading, saving with no index then add index: ",
                )

                df = store_in[tbl].sort_index()  # ptrepack not always sorts all data!
                cfg = {"db": store_in}
                with ReplaceTableKeepingChilds([df], tbl, cfg):
                    pass

                    # cfg['db'].append(
                    #     tbl, df, data_columns=True, index=False, **(
                    #         {'expectedrows': cfg['nfiles'],
                    #          'min_itemsize': {'values': cfg['logfield_fileName_len']}} if (
                    #                 cfgListName == 'tables_log' or i_in_group > 0) else
                    #         {'chunksize': cfg['chunksize']})
                    #     )

    # storage_basenames = {}
    #         if False:  # not helps?
    #             storage_basename = os_path.splitext(cfg_out['db_base'])[0] + "-" + tbl.replace('/', '-') + '.h5'
    #             lf.info('so start write to {}', storage_basename)
    #             try:
    #                 sort_pack(cfg_out['temp_db_path'], storage_basename, tbl, addargs=cfg_out.get('addargs'), **kwargs)
    #                 sleep(4)
    #             except Exception as e:
    #                 storage_basename = cfg_out['db_base'] + '-other_place.h5'
    #                 lf.error('Error: "{}"\nwhen write {} to original place! So start write to {}', e, tbl,
    #                                                                                                    storage_basename)
    #                 try:
    #                     sort_pack(cfg_out['temp_db_path'], storage_basename, tbl, addargs=cfg_out.get('addargs'), **kwargs)
    #                     sleep(8)
    #                 except:
    #                     lf.error(tbl + ': no success')
    #             storage_basenames[tbl] = storage_basename
    # if storage_basenames == {}:
    #     storage_basenames = None


def index_sort(
    cfg_out,
    out_storage_name=None,
    in_storages: Optional[Mapping[str, str]] = None,
    tables: Optional[Iterable[Union[str, Tuple[str]]]] = None,
) -> None:
    """
    Checks if tables in store have sorted index and if not then sort it by loading, sorting and saving data.
    :param cfg_out: dict - must have fields:
        'db_path': store where tables will be checked
        'temp_db_path': source of not sorted tables for move_tables() if index is not monotonic
        'base': base name (extension ".h5" will be added if absent) of hdf store to put
        'tables' and 'tables_log': tables to check monotonousness and if they are sorted, used if :param tables: not specified only
        'dt_from_utc'
        'b_remove_duplicates': if True then deletes duplicates by loading data in memory
    :param out_storage_name:
    :param in_storages: Dict [tbl: HDF5store file name] to use its values instead cfg_out['db_path']
    :param tables: iterable of table names
    :return:
    """
    lf.info("Checking that indexes are sorted:")
    if out_storage_name is None:
        out_storage_name = cfg_out["storage"]
    set_field_if_no(cfg_out, "dt_from_utc", 0)

    if not in_storages:
        in_storages = cfg_out["db_path"]
    else:
        in_storages = list(in_storages.values())
        if len(in_storages) > 1 and any(in_storages[0] != storage for storage in in_storages[1:]):
            lf.warning("Not implemented for result stored in multiple locations. Check only first")

        in_storages = cfg_out["db_path"].with_name(in_storages[0])

    if tables is None:
        tables = cfg_out.get("tables", []) + cfg_out.get("tables_log", [])
    with pd.HDFStore(in_storages) as store:
        # store= pd.HDFStore(cfg_out['db_path'])
        # b_need_save = False
        dup_tbl_set = set()
        nonm_tbl_set = set()
        for tbl in unzip_if_need(tables):
            if tbl not in store:
                lf.warning("{} not in {}", tbl, in_storages)
                continue
            try:
                df = store[tbl]
                if df is None:
                    lf.warning("None table {} in {}", tbl, store.filename)
                    continue
            except TypeError as e:
                lf.exception("Can not access table {:s}", tbl)
                continue
            # store.close()
            if df.index.is_monotonic_increasing:
                if df.index.is_unique:
                    lf.info(f"{tbl} - sorted")
                else:
                    lf.warning(f"{tbl} - sorted, but have duplicates")

                    # experimental
                    if cfg_out["b_remove_duplicates"]:
                        lf.warning(f"{tbl} - removing duplicates - experimental!")
                        dup_tbl_set.update(
                            remove_duplicates(
                                {
                                    **cfg_out,
                                    "db": store,
                                    "tables": [t for t in cfg_out["tables"] if t],
                                    "tables_log": [t for t in cfg_out["tables_log"] if t],
                                },
                                cfg_table_keys=["tables", "tables_log"],
                            )
                        )
                    else:
                        dup_tbl_set.add(tbl)
                continue
            else:  # only printing messages about what the problem with sorting by trying it
                nonm_tbl_set.add(tbl)  # b_need_save = True
                lf.warning(f"{tbl} - not sorted!")
                print(repr(store.get_storer(tbl).group.table))

                df_index, itm = multiindex_timeindex(df.index)
                if __debug__:
                    plt.figure(
                        f'Not sorted index that we are sorting {"on" if itm is None else "before"} saving...'
                    )
                    plt.plot(np.arange(df_index.size), df_index.values)  # np.diff(df.index)
                    plt.show()

                if itm is not None:
                    lf.warning("sorting multiindex...")
                    df = df.sort_index()  # inplace=True
                    if df.index.is_monotonic_increasing:
                        if df.index.is_unique:
                            lf.warning("Ok")
                        else:
                            dup_tbl_set.add(tbl)
                            lf.warning("Ok, but have duplicates")
                        continue
                    else:
                        print("Failure!")
                else:
                    try:
                        df = df.sort_index()
                        with ReplaceTableKeepingChilds(
                            [df],
                            tbl,
                            {**cfg_out, "db": store},
                            append_log if ("log" in tbl) else append_data,
                        ):
                            nonm_tbl_set.discard(tbl)
                        lf.warning("Saved sorted in memory - ok.")
                    except Exception as e:
                        lf.exception(
                            "Error sorting Table {:s} in memory. Will try sort by ptrepack in move_tables() to temp_db_path and back",
                            tbl,
                        )
                        # #lf.warning('skipped of sorting ')
        if dup_tbl_set:
            lf.warning(
                "To drop duplicates from {} restart with [out][b_remove_duplicates] = True", dup_tbl_set
            )
            nonm_tbl_set -= dup_tbl_set
        else:
            lf.info("no duplicates...")
        if nonm_tbl_set:
            lf.warning(
                "{} have no duplicates but nonmonotonic. Forcing update index before move and sort...",
                nonm_tbl_set,
            )
            if nonm_tbl_set:
                # as this fun is intended to check move_tables stranges, repeat it with forcing update index
                if not cfg_out[
                    "temp_db_path"
                ].is_file():  # may be was deleted because of cfg_out['b_del_temp_db']
                    # create temporary db with copy of table
                    move_tables(
                        {"temp_db_path": cfg_out["db_path"], "db_path": cfg_out["temp_db_path"]},
                        tbl_names=list(nonm_tbl_set),
                    )

                move_tables({**cfg_out, "recreate_index_tables_set": nonm_tbl_set}, tbl_names=tables)
        else:
            lf.info(f'{"other" if dup_tbl_set else "all"} tables monotonic.{"" if dup_tbl_set else " Ok>"}')

        # if b_need_save:
        #     # out to store
        #     cfg_out['db_path'], cfg_out['temp_db_path'] = cfg_out['temp_db_path'], cfg_out['temp_db_path']
        #     move_tables(cfg_out, tbl_names=tables)
        #     cfg_out['db_path'], cfg_out['temp_db_path'] = cfg_out['temp_db_path'], cfg_out['temp_db_path']
        #     move_tables(cfg_out, tbl_names=tables)

        # store = pd.HDFStore(cfg_out['temp_db_path'])
        # store.create_table_index(tbl, columns=['index'], kind='full')
        # store.create_table_index(cfg_out['tables_log'][0], columns=['index'], kind='full') #tbl+r'/logFiles'
        # append(store, df, log, cfg_out, cfg_out['dt_from_utc'])
        # store.close()
        # sort_pack(cfg_out['temp_db_path'], out_storage_name, tbl) #, ['--overwrite-nodes=true']


def rem_rows(db, tbl_names, qstr, qstr_log):
    """

    :param tbl_names: list of strings or list of lists (or tuples) of strings: nested tables last.
    :param db:
    :param qstr, qstr_log:
    :return: n_rows - rows number removed
    """
    print("removing obsolete stored data rows:", end=" ")
    tbl = ""
    tbl_log = ""
    sum_rows = 0
    try:
        for i_in_group, tbl in unzip_if_need_enumerated(tbl_names):
            if not tbl:
                continue
            if i_in_group == 0:  # i_in_group == 0 if not nested (i.e. no log) table
                tbl_parent = tbl
            else:
                tbl = tbl.format(tbl_parent)
            q, msg_t_type = (qstr, "table, ") if i_in_group == 0 else (qstr_log, "log.")
            try:
                n_rows = db.remove(tbl, where=q)
                print(f"{n_rows} in {msg_t_type}", end="")
                sum_rows += n_rows
            except KeyError:  # No object named {table_name} in the file
                pass  # nothing to delete
    except (HDF5ExtError, NotImplementedError) as e:
        lf.exception(
            "Can not delete obsolete rows, so removing full tables {:s} & {:s} and filling with all currently found data",
            tbl,
            tbl_log,
        )
        try:
            try:
                if tbl_log:
                    db.remove(tbl_log)  # useful if it is not a child
            except HDF5ExtError:  # else it may be deleted with deleting parent even on error
                if tbl:
                    db.remove(tbl)
                else:
                    raise
            else:
                if tbl:
                    db.remove(tbl)
            return 0
        except HDF5ExtError:
            db.close()
            replace_bad_db(db.filename)

    return sum_rows


def rem_last_rows(db, tbl_names, df_logs: List[pd.DataFrame], t_start=None):
    """
    Remove rows by `rem_rows` then replace back removed 1st log row updated for remaining data index
    :param db:
    :param tbl_names:
    :param df_logs: list of logs DataFrames of length >= (number of items in tbl_names) - len(tbl_names)
    :param t_start: datetime or None. If None then do nothing
    :return:
    """
    if t_start is None:
        return
    rem_rows(db, tbl_names, qstr="index>='{}'".format(t_start), qstr_log="DateEnd>='{}'".format(t_start))
    #
    i_group = -1
    for i_in_group, tbl in unzip_if_need_enumerated(tbl_names):
        if i_in_group == 0:  # skip not nested (i.e. no log) table at start of each group
            tbl_parent = tbl
            i_group += 1
            continue
        else:
            if not tbl:
                continue
            tbl = tbl_parent.format(tbl)

        df_log_cur = df_logs[i_group]
        df_log_cur["DateEnd"] = t_start.tz_convert(df_log_cur["DateEnd"][0].tz)
        df_log_cur["rows"] = -1  # todo: calc real value, keep negative sign to mark updated row
        append_log(df_log_cur, tbl, {"db": db, "nfiles": None})


def del_obsolete(
    cfg_out: MutableMapping[str, Any],
    log: Mapping[str, Any],
    df_log: pd.DataFrame,
    field_to_del_older_records=None,
) -> Tuple[bool, bool]:
    """
    Check that current file has been processed and it is up to date
    Removes all data (!) from the store table and log table where time indices of existed data >= `t_start`,
    where `t_start` - time index of current data (log['index']) or nearest log record index (df_log.index[0])
    (see :param:field_to_del_older_records description)

    Also removes duplicates in the table if found duplicate records in the log
    :param cfg_out: dict, must have field
        'db' - handle of opened store
        'b_reuse_temporary_tables' - for message
        'tables_log', 'tables' - metadata and data tables where we check and deleting
    :param log: current data dict, must have fields needed for compare:
        'index' - log record corresponded starting data index
        'fileName' - in format as in log table to be able to find duplicates
        'fileChangeTime', datetime - to be able to find outdated data
    :param df_log: log record data - loaded from store before updating
    :param field_to_del_older_records: str or (default) None.
    - If None then:
        - checks duplicates among log records with 'fileName' equal to log['fileName'] and deleting all of
    them and data older than its index (with newest 'fileChangeTime' if have duplicates, and sets b_stored_dups=True).
        - sets b_stored_newer=True and deletes nothing if 'fileChangeTime' newer than that of current data
    - Else if not "index" then del. log records with field_to_del_all_older_records >= current file have, and
      data having time > its 1st index.
    - If "index" then del. log records with "DateEnd" > current file time data start, but last record wrote
      back with changed "DateEnd" to be consistent with last data time.

    :return: (b_stored_newer, b_stored_dups):
        - b_stored_newer: Have recorded data that was changed after current file was changed (log['fileChangeTime'])
        - b_stored_dups: Duplicate entries in df_log detected
    """
    b_stored_newer = False  # not detected yet
    b_stored_dups = False  # not detected yet
    if cfg_out["tables"] is None or df_log is None:
        return b_stored_newer, b_stored_dups
    if field_to_del_older_records:
        # priority to the new data
        if field_to_del_older_records == "index":
            # - deletes data after df_log.index[0]
            # - updates db/tbl_log.DateEnd to be consisted to remaining data.
            t_start = log["index"]
            if t_start is None:
                lf.info("delete all previous data in {} and {}", cfg_out["tables"], cfg_out["tables_log"])
                try:
                    t_start = df_log.index[0]
                except IndexError:
                    t_start = None  # do nothing
                # better?:
                # remove_tables(db, tables, tables_log, temp_db_path=None)
                # return b_stored_newer, b_stored_dups
            df_log_cur = df_log
        else:  # not tested
            b_cur = df_log[field_to_del_older_records] >= log[field_to_del_older_records]
            df_log_cur = df_log[b_cur]
            if not df_log_cur.empty:
                t_start = df_log_cur.index[0]
        # removing
        rem_last_rows(cfg_out["db"], zip(cfg_out["tables"], cfg_out["tables_log"]), [df_log_cur], t_start)
    else:
        # stored data will be replaced only if have same fileName with 'fileChangeTime' older than new
        df_log_cur = df_log[df_log["fileName"] == log["fileName"]]
        n_log_rows = len(df_log_cur)
        if n_log_rows:
            if n_log_rows > 1:
                b_stored_dups = True
                print(
                    'Multiple entries in log for same file ("{}") detected. Will be check for dups'.format(
                        log["fileName"]
                    )
                )
                cfg_out["b_remove_duplicates"] = True
                if cfg_out["b_reuse_temporary_tables"]:
                    print("Consider set [out].b_reuse_temporary_tables=0,[in].b_incremental_update=0")
                print("Continuing...")
                imax = df_log_cur["fileChangeTime"].argmax()  # np.argmax([r.to_pydatetime() for r in ])
                df_log_cur = df_log_cur.iloc[[imax]]  # [np.arange(len(df_log_cur)) != imax]
            else:  # no duplicates:
                imax = 0  # have only this index
                n_log_rows = 0  # not need delete data

            # to return info about newer stored data:
            last_file_change_time = df_log_cur["fileChangeTime"].dt.to_pydatetime()
            if any(last_file_change_time >= log["fileChangeTime"]):
                b_stored_newer = True  # can skip current file, because we have newer data records
                print(">", end="")
            else:
                # delete all next records of current file
                if n_log_rows and not rem_rows(
                    cfg_out["db"],
                    zip(cfg_out["tables"], cfg_out["tables_log"]),
                    qstr="index>='{}'".format(i0 := df_log_cur.index[0]),
                    qstr_log="fileName=='{}'".format(df_log_cur.loc[i0, "fileName"]),
                ):
                    b_stored_newer = False  # deleted
                    b_stored_dups = False  # deleted
    return b_stored_newer, b_stored_dups


# Functions to iterate rows of db log instead of files in dir


def time_range_query(min_time=None, max_time=None, **kwargs) -> str:
    """
    Query Time for pandas.Dataframe
    :param min_time:
    :param max_time:
    :return:
    """
    if min_time:
        range_query = f"index>='{min_time}' & index<='{max_time}'" if max_time else f"index>='{min_time}'"
    elif max_time:
        range_query = f"index<='{max_time}'"
    else:
        range_query = None
    return range_query


def log_rows_gen(
    db_path: Union[str, Path, None] = None,
    table_log: str = "log",
    min_time: Optional[datetime] = None,
    max_time: Optional[datetime] = None,
    range_query: Optional[Sequence[datetime]] = None,
    db: Optional[pd.HDFStore] = None,
    **kwargs,
) -> Iterator[Dict[str, Any]]:
    """
    Dicts from each hdf5 log row
    :param db_path: name of hdf5 pandas store where is log table, used only for message if it is set and db is set
    :param db: handle of already open pandas hdf5 store
    :param table_log: name of log table - table with columns for intervals:
    - `index` - starts, pd.DatetimeIndex
    - `DateEnd` - ends, pd.Datetime
    :param min_time:
    :param max_time: allows limit the range of table_log rows, not used if range_query is set
    :param range_query: query str to limit the range of table_log rows to load
        Example table_log name: cfg_in['table_log'] ='/CTD_SST_48M/logRuns'
    :param kwargs: not used
    Yields dicts where keys: col names, values: current row values of tbl_intervals = cfg_in['table_log']
    """
    if range_query is None:
        range_query = time_range_query(min_time, max_time)
    with nullcontext(db) if db else pd.HDFStore(db_path, mode="r") as db:
        print(f'loading from "{db_path if db_path else db.filename}": ', end="")
        for n, rp in enumerate(db.select(table_log, where=range_query).itertuples()):
            r = dict(zip(rp._fields, rp))
            yield r  # r.Index, r.DateEnd


def log_names_gen(
    cfg_in: Mapping, f_row_to_name=lambda r: "{Index:%y%m%d_%H%M}-{DateEnd:%H%M}".format_map(r)
) -> Iterator[Any]:
    """
    Generates outputs of f_row_to_name function which receives dicts from each hdf5 log row (see log_rows_gen)
    :param cfg_in: keyword arguments for log_rows_gen()
    :param f_row_to_name: function(dict) where dict have fields from hdf5 log row
    :return: iterator, by default - of strings, suitable to name files by start-end date/time

    :Modifies cfg_in: adds/replaces field cfg_in['log_row'] = log_rows_gen(cfg_in) result before each yielding
    Replacing for veuszPropagate.ge_names() to use tables instead files
    """
    for row in log_rows_gen(**cfg_in):
        cfg_in["log_row"] = row
        yield f_row_to_name(row)


def merge_two_runs(df_log: pd.DataFrame, irow_to: int, irow_from: Optional[int] = None):
    """
    Merge 2 runs: copy metadata about profile end (columns with that ends with 'en') to row `irow_to` from `irow_from` and then delete it
    :param df_log: DataFrame to be modified (metadata table)
    :param irow_to: row index where to replace metadata columns that ends with 'en' and increase 'rows' and 'rows_filtered' columns by data from row `irow_from`
    :param irow_from: index of row that will be deleted after copying its data
    :return: None
    """
    if irow_from is None:
        irow_from = irow_to + 1
    df_merging = df_log.iloc[[irow_to, irow_from], :]
    k = input(f"{df_merging} rows selected (from, to). merge ? [y/n]:\n")
    if k.lower() != "y":
        print("done nothing")
        return
    cols_en = ["DateEnd"] + [col for col in df_log.columns if col.endswith("en")]
    ind_to, ind_from = df_merging.index
    df_log.loc[ind_to, cols_en] = df_log.loc[ind_from, cols_en]
    cols_sum = ["rows", "rows_filtered"]
    df_log.loc[ind_to, cols_sum] += df_log.loc[ind_from, cols_sum]
    df_log.drop(ind_from, inplace=True)
    print("ok, 10 nearest rows became:", df_log.iloc[(irow_from - 5) : (irow_to + 5), :])


def names_gen(cfg_out: Mapping[str, Any], paths, check_have_new_data=True, **kwargs) -> Iterator[Path]:
    """
    Yields Paths from cfg_in['paths'] items
    :updates: cfg_out['log'] fields 'fileName' and 'fileChangeTime'

    :param paths: iterator - returns full file names
    :param cfg_out: dict, with fields needed for dispenser_and_names_gen() and print info:
        - log: current file info with fields that should be updated before each yield:
            - Date0, DateEnd, rows: if no Date0, then prints "file not processed"
    :param check_have_new_data: bool, if False then do not check Date0 presence and print "file not processed"
    :param kwargs: not used
    """
    set_field_if_no(cfg_out, "log", {})
    for name_full in paths:
        pname = Path(name_full)

        cfg_out["log"]["fileName"] = f"{pname.parent.name}/{pname.stem}"[
            -cfg_out.get("logfield_fileName_len", 255):]
        cfg_out["log"]["fileChangeTime"] = datetime.fromtimestamp(pname.stat().st_mtime)

        try:
            yield pname  # Traceback error line pointing here is wrong
        except GeneratorExit:
            print("Something wrong?")
            return

        # Log to logfile
        if cfg_out["log"].get("Date0"):
            strLog = "{fileName}:\t{Date0:%d.%m.%Y %H:%M:%S}-{DateEnd:%d.%m %H:%M:%S%z}\t{rows}rows".format(
                **cfg_out["log"]
            )  # \t{Lat}\t{Lon}\t{strOldVal}->\t{mag}
            lf.info(strLog)
        elif check_have_new_data:
            strLog = "file not processed"
            lf.info(strLog)


def out_init(
    cfg_out: MutableMapping[str, Any],
    path=None,
    db_path=None,
    cfgFile=None,
    tables=[],
    table=None,
    raw_dir_words=("raw", "_raw", "source", "_source", "WorkData", "workData"),
    logfield_fileName_len=255,
    chunksize=None,
    b_incremental_update=False,
    b_remove_duplicates=False,
    b_reuse_temporary_tables=False,
    nfiles=1,
    **kwargs
) -> None:
    """
    Init output DB (hdf5 data store) information in `cfg_out` if it is not exist_
    :param cfg_out: configuration dict, with optional fields. If `cfg_out['tables']` is None then function
    returns (does nothing)
    Input configuration:
    :param path: if no 'db_path' in cfg_out, or it is not absolute
    :param cfgFile: if no `cfg_out['b_insert_separator']` defined or determine the table name is failed - to extract from cfgFile name
    :param raw_dir_words: (optional) - see getDirBaseOut()
    :param nfiles: (optional)
    :param b_incremental_update: (optional) to copy it to cfg_out
    :param kwargs: not used

    Sets or updates fields of `cfg_out`:
    % paths %
    - db_path: absolute path of hdf5 store with suffix ".h5"
    - temp_db_path: temporary hdf5 file name
    - tables, tables_log: tables names of data and log (metadata) - based on `raw_dir_words` and other params.
    Note: The function not sets `table` field, but uses it if it is defined to set `tables` field, else if
    `table` argument not defined then sets `tables` field from `db_path` or `cfgFile` name
    % other %
    - nfiles: use it somewhere to set store.append() 'expectedrows' argument
    - b_incremental_update: default False, copied from `b_incremental_update` argument
    - chunksize: default None
    - logfield_fileName_len: default 255
    - b_remove_duplicates: default False
    - b_reuse_temporary_tables: default False

    :return: None
    """
    if "tables" in cfg_out and cfg_out["tables"] is None:
        return
    set_field_if_no(cfg_out, "logfield_fileName_len", logfield_fileName_len)
    set_field_if_no(cfg_out, "chunksize", chunksize)
    set_field_if_no(cfg_out, "b_incremental_update", b_incremental_update)
    set_field_if_no(cfg_out, "b_remove_duplicates", b_remove_duplicates)
    set_field_if_no(cfg_out, "b_reuse_temporary_tables", b_reuse_temporary_tables)
    set_field_if_no(cfg_out, "nfiles", nfiles)

    if cfg_out.get("b_insert_separator") is None:
        if cfgFile:
            cfg_file = PurePath(cfgFile).stem
            cfg_out["b_insert_separator"] = "_ctd_" in cfg_file.lower()
        # else:
        #     cfg_out['b_insert_separator'] = False

    # Automatic db file path
    if not (cfg_out.get("db_path") and cfg_out["db_path"].is_absolute()):
        path_in = Path(path or db_path).parent
        cfg_out["db_path"] = path_in / (
            f"{path_in.stem}_out" if not cfg_out.get("db_path") else cfg_out["db_path"]
        )
    dir_create_if_need(cfg_out["db_path"].parent)
    cfg_out["db_path"] = cfg_out["db_path"].with_suffix(".h5")

    # temporary db file path
    set_field_if_no(
        cfg_out, "temp_db_path", cfg_out["db_path"].with_name(f"{cfg_out['db_path'].stem}_not_sorted.h5")
    )

    # Tables
    if "tables" in cfg_out and cfg_out["tables"]:
        set_field_if_no(
            cfg_out, "tables_log", [((f"{tbl}/logFiles") if tbl else "") for tbl in cfg_out["tables"]]
        )
    elif cfg_out.get("table"):
        cfg_out["tables"] = [cfg_out["table"]]
        set_field_if_no(cfg_out, "tables_log", [f"{cfg_out['table']}/logFiles"])
    else:
        # auto table name
        if not table:
            if tables and len(tables) == 1:
                cfg_out["tables"] = tables
            else:
                _, _, table = getDirBaseOut(
                    cfg_out["db_path"],
                    raw_dir_words
                )
                if not table and cfgFile:
                    table = Path(cfgFile).stem
                    lf.warning(
                        "Can not dertermine table_name from file structure. "
                        'Set [tables] in ini! Now use table_name "{:s}"',
                        table,
                    )
                if not table:
                    return
                cfg_out["tables"] = [table]
        set_field_if_no(cfg_out, "tables_log", [f"{table}/logFiles"])


def copy_to_temp_db(
    db_in: Optional[pd.HDFStore],
    db_path: Optional[str],
    temp_db_name: Optional[str],
    tables: Sequence[str],
    tables_log: Sequence[str],
    b_incremental_update: bool = True,
) -> bool:
    """
    Copies `tables` from the original HDF5 store to a temporary store with indexing its sorted index. Checks if incremental update if possible
    :param db_in: A pre-existing HDFStore object or None. If None, 'db_path' is used to open the store.
    :param db_path: The file path to the original HDF5 store.
    :param temp_db_name: The file name of the temporary HDF5 store.
    :param tables: A list of table names to be copied.
    :param tables_log: Log table names to be copied alongside `tables`.
    :param b_incremental_update: Flag to determine if incremental update is possible.
    :returns: A boolean indicating whether an incremental update is possible after the operation.

    :raises HDF5ExtError: If there are issues with specific tables conforming to expected HDF5 format.
    :raises RuntimeError: If there's a failure during the copy operation that might be resolved by indexing.

    Side Effects:
    - Prints informational and error messages.
    - Modifies the temporary HDF5 store by writing new tables to it or updating existing ones.
    """
    lf.info("Copying previous store data to temporary one:")
    tbl = "is absent"
    try:
        with nullcontext(db_in) if db_in else pd.HDFStore(db_path, mode="r") as db_in:
            tbl_prev = "?"  # Warning side effect: ignores 1st table if its name starts with '?'
            for tbl in sorted(tables + tables_log):
                if (  # parent of this nested have moved on previous iteration
                    (len(tbl_prev) < len(tbl) and tbl.startswith(tbl_prev) and tbl[len(tbl_prev)] == "/")
                    or not tbl
                ):
                    continue
                try:  # Check output store
                    if tbl in db_in:  # avoid harmful sortAndPack errors
                        sort_pack(db_path, temp_db_name, tbl, arguments="fast")
                    else:
                        lf.info(f"Table {tbl} does not exist")
                        continue
                        # raise HDF5ExtError(f'Table {tbl} does not exist')
                except HDF5ExtError as e:
                    if tbl in db_in.root._v_children:  # allows compare to not pandas
                        lf.warning("Node exists but store does not conform to Pandas.")
                        get_store_and_print_table(db_in, tbl)
                    raise e  # exclude next processing
                except RuntimeError as e:
                    lf.error("Failed copy from output store (RuntimeError). Trying to add full index.")
                    nodes = db_in.get_node(tbl)._v_children.keys()  # sorted(, key=number_key)

                    # Reopen for modifcation
                    db_in.close()
                    db_in.open()  # = pd.HDFStore(db_in.filename)
                    for n in nodes:
                        tbl_cur = tbl if n == "table" else f"{tbl}/{n}"
                        lf.info(tbl_cur, end=", ")
                        db_in.create_table_index(tbl_cur, columns=["index"], kind="full")
                        # db_in.flush(fsync=True)
                    lf.error("Trying again")
                    db_in.close()
                    sort_pack(db_path, temp_db_name, tbl)
                    db_in.open("r")

                tbl_prev = tbl

    except HDF5ExtError as e:
        lf.warning(e.args[0])  # print('processing all source data... - no table with previous data')
        b_incremental_update = False
    except Exception as e:
        lf.error(''.join([
            f"Copying previous data (table {tbl}) to the temporary store failed ",
            "=> incremental update not possible: " if b_incremental_update else "",
            '\n==> '.join([s for s in e.args if isinstance(s, str)])
        ]))
        b_incremental_update = False
    else:
        if b_incremental_update:
            lf.info("Will append data only from new files.")
    return b_incremental_update


def temp_open(
    db_path: Optional[Path] = None,
    temp_db_path: Optional[Path] = None,
    tables: Optional[Sequence[str]] = None,
    tables_log: Optional[Sequence[str]] = (),
    db: Optional[pd.HDFStore] = None,
    b_incremental_update: bool = False,
    b_reuse_temporary_tables: bool = False,
    b_overwrite: bool = False,
    db_in=None,
    **kwargs,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.HDFStore], bool]:
    """
    1. Loads existed log records attached to pandas HDF5 store table for incremental appending data to store
    (do not reloads existed data).
    2. Opens a temporary HDF5 store and copies data from an existing target store for further processing.
    This allows you first efficiently load data without indexing and sorting and then do it once
    copying data back to original HDF5 store using `ptrepack` pandas utility.

    :param db_path: Path to the original HDF5 store. If None, the function may return early.
    :param temp_db_path: Path to the temporary HDF5 store.
    :param tables: Sequence of table names to be processed. If None, the function returns early.
    :param tables_log: A list of log tables. If not specified then `b_incremental_update` of 'tables' skipped.
    :param b_incremental_update: A flag indicating that existed data must be kept if metadata matches new
        data.
    :param b_reuse_temporary_tables: If True, do not copy existing tables from 'db_path' to 'temp_db_path'.
    :param b_overwrite: When True, existing data in the `tables` will be removed.
    :param db: An optional pre-existing HDFStore object: reuses it if `kwargs['b_allow_use_opened_temp_db']`.
        else raises FileExistsError If `db` is already open.
    :param db_in: Optional; Handle of an already opened original store. If None and `tables` is also None the
        function may return early.
    :param kwargs: Additional optional parameters that may include:
    - 'b_allow_use_opened_temp_db': Allows the reuse of an already opened temporary database.
    - Other parameters from `out_init()` config if 'temp_db_path' is None. These parameters can override
    existing ones from the function arguments:
        - path: if no 'db_path' in cfg_out, or it is not absolute
        - cfgFile - if no `cfg_out['b_insert_separator']` defined or determine the table name is failed - to
        extract from cfgFile name

    :return: A tuple (df_log, db, b_incremental_update) where:
    - df_log: DataFrame of the log from the store if 'b_incremental_update' is True, otherwise None.
    - db: a handle to the opened pandas HDF5 store at 'temp_db_path'.
    - b_incremental_update: flag changes to False if data / store is not valid or `b_overwrite` is True.
    """

    df_log = None
    if db:
        lf.warning("DB already used{}: handle detected!", " and opened" if db.is_open else "")
        if not kwargs.get("b_allow_use_opened_temp_db"):
            raise FileExistsError(f"{db.filename} must be closed")  # may be useful close temporary?
    if tables is None or (db_in or db_path) is None:
        return None, None, False  # skipping open, may be need if not need write

    if temp_db_path is None:
        cfg_out = {"tables": tables}
        out_init(cfg_out, b_incremental_update=b_incremental_update, db_path=db_path, **kwargs)
        temp_db_path = cfg_out["temp_db_path"]

    print("saving to", temp_db_path / ",".join(tables).strip("/"), end=":\n")

    # If table name in tables_log has placeholder then fill it
    tables_log = (
        [t.format(tables[i if i < len(tables) else 0]) for i, t in enumerate(tables_log)]
        if tables
        else list(tables_log)  # if have tuple allows concatenate below
    )

    try:
        if not b_reuse_temporary_tables:
            # Remove existed `tables` to write
            try:  # open temporary output file
                if temp_db_path.is_file():
                    with nullcontext(db) if db else pd.HDFStore(temp_db_path, mode="r") as db:
                        tables_in_root = [
                            t
                            for tbl in tables
                            for t in find_tables(db, tbl.format(".*").replace(".*.*", ".*"))
                            if tbl
                        ]
                        remove_tables(db, tables_in_root, tables_log)
            except IOError as e:
                print(e)

            if not b_overwrite:
                # Copying previous store data to temporary one
                b_incremental_update = copy_to_temp_db(
                    db_in, db_path, temp_db_path.name, tables, tables_log, b_incremental_update
                )

        if (db is None) or not db.is_open:
            # Open temporary output file to return
            db = open_trying(temp_db_path)

        if not b_overwrite:
            for tbl_log in tables_log:
                if tbl_log and (tbl_log in db):
                    try:
                        df_log = db[tbl_log]
                        break  # only one log table is supported (using 1st found in tables_log)
                    except AttributeError:  # pytables.py: 'NoneType' object has no attribute 'startswith'
                        # - no/bad table log
                        b_overwrite = True
                        lf.exception(
                            "Bad log: {}, removing... {}",
                            tbl_log,
                            "Switching off incremental update" if b_incremental_update else "",
                        )
                        remove(db, tbl_log)
                    except UnicodeDecodeError:
                        lf.exception(
                            "Bad log: {}. Suggestion: remove non ASCII symbols from log table manually!",
                            tbl_log,
                        )
                        raise
            else:
                df_log = None

        if b_overwrite:
            df_log = None
            b_incremental_update = False  # new start, fill table(s) again

    except (HDF5ExtError, FileModeError) as e:
        if (db is not None) and db.is_open:
            db.close()
            db = None
        print("Can not use old temporary output file. Deleting it...")
        for k in range(10):
            try:
                os_remove(temp_db_path)
            except PermissionError:
                print(end=".")
                sleep(1)
            except FileNotFoundError:
                print(end=" - was not exist")
        if os_path.exists(temp_db_path):
            p_name, p_ext = os_path.splitext(temp_db_path)
            temp_db_path = f"{p_name}-{p_ext}"
            print('Can not remove temporary db! => Use another temporary db: "{}"'.format(temp_db_path))
        sleep(1)
        for k in range(10):
            try:
                db = pd.HDFStore(temp_db_path)
            except HDF5ExtError:
                print(end=".")
                sleep(1)
        b_incremental_update = False
    except Exception as e:
        lf.exception("Can not open temporary hdf5 store")
    return df_log, db, b_incremental_update

def open_trying(db_path, change_name = False, **kwargs):
    """
    :param change_name: after failed retrying of opening same path try new path
    :param kwargs: mode {'a', 'w', 'r', 'r+'}, complevel {0-9}, ...: see pd.HDFStore
    """
    for attempt in range(2):
        try:
            db = pd.HDFStore(db_path, **kwargs)
            # db.flush(fsync=True)
            break
        except (IOError, ValueError) as e:  # ValueError: The file '.h5' is already opened.
            print(e)
        except HDF5ExtError as e:
            if db_path.is_file():
                print(
                    f"Can not use old temporary output file {db_path.name} ({e}). Trying to delete...",
                    end="",
                )
                try:
                    os_remove(db_path)
                    print(" - deleted")
                except PermissionError:
                    print(" - failed")
                    continue
            else:
                print(f"Can not use old temporary output file name: even not exist ({e})!")
            # lf.exception(f"can not open {db_path.name}")
    else:
        if change_name:
            for r in range(1, 100):
                db_path_new = db_path.parent / f"{db_path.stem}_{r}{db_path.suffix}"
                if not db_path_new.is_file():
                    print(f"changing temp db name to {db_path_new.name}")
                    break
        db = pd.HDFStore(db_path_new, **kwargs)
    return db


def dispenser_and_names_gen(
    fun_gen: names_gen,  # usually Callable[[Mapping[str, Any], Mapping[str, Any], Any], Iterator[Any]]
    cfg_out: Optional[MutableMapping[str, Any]] = None,
    b_close_at_end: Optional[bool] = True,
    fun_update_cfg_out=None,
    args=[],
    **kwargs,
) -> Iterator[Tuple[int, Any]]:
    """
    Warning: deprecated in favor of `append_through_temp_db_gen()`

    Prepares HDF5 store to insert/update data and yields fun_gen(...) outputs:
        - Opens DB for writing (see temp_open() requirements)
        - Finds data labels by fun_gen(): default are file names and their modification date
        - Removes outdated labels in log table and data part in data table they points to
        - Generates fun_gen() output (if b_incremental_update, only data labels which is absent in DB (to upload new/updated data))
        - Tide up DB: creates index, closes DB.
    This function supports storing data in HDF5 used in h5toGrid: dataframe's child 'table' node always contain adjacent
    "log" node. "log" dataframe labels parent dataframe's data segments and allows to check it for existence and
    relevance.
    :param fun_gen: function with arguments `(args, **kwargs)`, that
        - generates data labels, default are file's ``Path``s,
    :param fun_update_cfg_out: function with arguments `(cfg_out, fun_gen's output)`, that should
        update`cfg_out['log']` fields needed to store and find data:
    'fileName' - by current label,
    'fileChangeTime'.
    They named historically, in principle, you can use any unique identifier composed of this two fields.

    :param cfg_out: dict, must have fields:
        - log: dict, with info about current data, must have fields for compare:
            - 'fileName' - in format as in log table to be able to find duplicates
            - 'fileChangeTime', datetime - to be able to find outdated data
        - tables_log
        - b_incremental_update: if True then not yields previously processed files. But if file was changed
          then: 1. removes stored data and 2. yields `fun_gen(...)` result
        - tables_written: sequence of table names where to create index
        - temp_db_path
        ... - see `temp_open()` parameters that is can be obtained by calling `out_init(cfg_out, **cfg_in)`

    :param b_close_at_end: if True (default) then closes store after generator exhausted
    :param args: positional parameters of `fun_gen()`
    :param kwargs: keyword parameters of `fun_gen()`
    :return: Iterator that returns (i1, pname):
        - i1: index (starting with 1) of `fun_gen` generated data label (maybe file)
        - pname: fun_gen output (may be path name)
        Skips (i1, pname) for existed labels that also has same stored data label (file) modification date
    :updates:
        - cfg_out['db'],
        - cfg_out['b_remove_duplicates'] and
        - that what `fun_gen()` do
    """
    # copy data to temporary HDF5 store and open it
    # **{{} in cfg_out} db=db
    df_log_old, cfg_out["db"], cfg_out["b_incremental_update"] = temp_open(
        **cfg_out, b_allow_use_opened_temp_db=not b_close_at_end
    )
    try:
        for i1, gen_out in enumerate(fun_gen(cfg_out, *args, **kwargs), start=1):
            if fun_update_cfg_out:
                fun_update_cfg_out(cfg_out, gen_out)
            # if current file is newer than its stored data then remove data and yield its info to process
            # again
            if cfg_out["b_incremental_update"]:
                b_stored_newer, b_stored_dups = del_obsolete(
                    cfg_out, cfg_out["log"], df_log_old, cfg_out.get("field_to_del_older_records")
                )
                if b_stored_newer:
                    continue  # not need process: current file already loaded
                if b_stored_dups:
                    cfg_out["b_remove_duplicates"] = True  # normally no duplicates but we set if detect

            yield i1, gen_out

    except Exception as e:
        lf.exception("\nError preparing data:")
        sys.exit(ExitStatus.failure)
    finally:
        if b_close_at_end:
            close(cfg_out)


qstr_range_pattern = "index>='{}' & index<='{}'"


# def q_interval2coord(
#         db_path,
#         table,
#         t_interval: Optional[Sequence[Union[str, pd.Timestamp]]] = None,
#         time_range: Optional[Sequence[Union[str, pd.Timestamp]]] = None) -> pd.Index:
#     """
#     Edge coordinates of index range query
#     As it is nearly a part of h5.h5select() may be depreciated? See Note
#     :param: db_path, str
#     :param: table, str
#     :param: t_interval: array or list with strings convertable to pandas.Timestamp
#     :param: time_range: same as t_interval (but must be flat numpy array)
#     :return: ``qstr_range_pattern`` edge coordinates
#     Note: can use instead:
#     >>> from hdf5_pandas.h5 import load_points
#     ... with pd.HDFStore(db_path, mode='r') as store:
#     ...     df, bbad = load_points(store,table,columns=None,query_range_lims=time_range)

#     """

#     if not t_interval:
#         t_interval = time_range
#     if not (isinstance(t_interval, list) and isinstance(t_interval[0], str)):
#         t_interval = np.array(t_interval).ravel()

#     qstr = qstr_range_pattern.format(*t_interval)
#     with pd.HDFStore(db_path, mode='r') as store:
#         lf.debug("loading range from {:s}/{:s}: {:s} ", db_path, table, qstr)
#         try:
#             ind_all = store.select_as_coordinates(table, qstr)
#         except Exception as e:
#             lf.debug("- not loaded: {:s}", e)
#             raise
#         if len(ind_all):
#             ind = ind_all[[0, -1]]  # .values
#         else:
#             ind = []
#         lf.debug('- gets {}', ind)
#     return ind


# def q_intervals_indexes_gen(
#         db_path,
#         table: str,
#         t_prev_interval_start: pd.Timestamp,
#         t_intervals_start: Iterable[pd.Timestamp],
#         i_range: Optional[Sequence[Union[str, pd.Timestamp]]] = None) -> Iterator[pd.Index]:
#     """
#     Yields (`start`, `end`) coordinate pares (0 based indexes) of hdf5 store table index, whith `start` found
#     as next nearest to each `t_intervals_start` element
#     :param db_path
#     :param table, str (see q_interval2coord)
#     :param t_prev_interval_start: first index value
#     :param t_intervals_start:
#     :param i_range: Sequence, 1st and last element will limit the range of returned result
#     :return: Iterator[pd.Index] of lower and upper int limits (adjacent intervals)
#     """

#     for t_interval_start in t_intervals_start:
#         # load_interval
#         start_end = q_interval2coord(
#             db_path, table, [t_prev_interval_start.isoformat(), t_interval_start.isoformat()]
#         )
#         if len(start_end):
#             if i_range is not None:  # skip intervals that not in index range
#                 start_end = minInterval([start_end], [i_range], start_end[-1])[0]
#                 if not len(start_end):
#                     if 0 < i_range[-1] < start_end[0]:
#                         raise Ex_nothing_done
#                     continue
#             yield start_end
#         else:  # no data
#             print('-', end='')
#         t_prev_interval_start = t_interval_start


def q_ranges_gen(cfg_in: Mapping[str, Any], df_intervals: pd.DataFrame):
    """
    Loading intervals using ranges dataframe (defined by Index and DateEnd column - like in h5toGrid hdf5 log
    tables)
    :param df_intervals: dataframe, with:
        index - pd.DatetimeIndex for starts of intervals
        DateEnd - pd.Datetime col for ends of intervals
    :param cfg_in: dict, with fields:
        db_path, str
        table, str
    Exsmple:
    >>> df_intervals = pd.DataFrame({'DateEnd': pd.DatetimeIndex([2,4,6])}, index=pd.DatetimeIndex([1,3,5]))
    ... a = q_ranges_gen(df_intervals, cfg['out'])
    """
    with pd.HDFStore(cfg_in['db_path'], mode='r') as store:
        print("loading from {db_path}: ".format_map(cfg_in), end='')
        # Query table tblD by intervals from table tblL
        # dfL = store[tblL]
        # dfL.index= dfL.index + dtAdd
        df = pd.DataFrame()
        for n, r in enumerate(df_intervals.itertuples()):  # if n == 3][0]  # dfL.iloc[3], r['Index']= dfL.index[3]
            qstr = qstr_range_pattern.format(r.Index, r.DateEnd)  #
            df = store.select(cfg_in['table'], qstr)  # or dd.query?
            print(qstr)
            yield df


def is_dask_dataframe(obj):
    cl = obj.__class__
    return cl.__name__ == "DataFrame" and "dask" in cl.__module__


def append_on_inconsistent_index(cfg_out, tbl_parent, df, df_append_fun, e, msg_func):
    """
    Align types: Make index to be UTC
    :param cfg_out:
    :param tbl_parent:
    :param df:
    :param df_append_fun:
    :param e:
    :param msg_func:
    :return:
    """

    if tbl_parent is None:
        tbl_parent = cfg_out["table"]

    error_info_list = [s for s in e.args if isinstance(s, str)]
    msg = msg_func + " Error: " + e.__class__ + "\n==> ".join(error_info_list)
    if not error_info_list:
        lf.error(msg)
        raise e
    b_correct_time = False
    b_correct_str = False
    b_correct_cols = False
    str_check = "invalid info for [index] for [tz]"
    if error_info_list[0].startswith(str_check) or error_info_list[0] == "Not consistent index":
        if error_info_list[0] == "Not consistent index":
            msg += "Not consistent index detected"
        lf.error(msg + "Not consistent index time zone? Changing index to standard UTC")
        b_correct_time = True
    elif error_info_list[0].startswith("Trying to store a string with len"):
        b_correct_str = True
        lf.error(msg + error_info_list[0])  # ?
    elif error_info_list[0].startswith("cannot match existing table structure"):
        b_correct_cols = True
        lf.error(f"{msg} => Adding columns...")
        # raise e #?
    elif error_info_list[0].startswith(
        "invalid combination of [values_axes] on appending data"
    ) or error_info_list[0].startswith("invalid combination of [non_index_axes] on appending data"):
        # old pandas version has word "combinate" insted of "combination"!
        b_correct_cols = True
        lf.error(f"{msg} => Adding columns/convering type...")
    else:  # Can only append to Tables - need resave?
        lf.error(f"{msg} => Can not handle this error!")
        raise e

    df_cor = cfg_out["db"][tbl_parent]
    b_df_cor_changed = False

    def align_columns(df, df_ref, columns=None):
        """

        :param df: changing dataframe. Will update implicitly!
        :param df_ref: reference dataframe
        :param columns:
        :return: updated df
        """
        if columns is None:
            columns = df.columns
        df = df.reindex(df_ref.columns, axis="columns", copy=False)
        for col, typ in df_ref[columns].dtypes.items():
            fill_value = np.array(
                0 if np.issubdtype(typ, np.integer) else np.nan if np.issubdtype(typ, np.floating) else "",
                dtype=typ,
            )
            df[col] = fill_value
        return df

    if b_correct_time:
        # change stored to UTC
        df_cor.index = pd.DatetimeIndex(df_cor.index.tz_convert(tz="UTC"))
        b_df_cor_changed = True

    elif b_correct_cols:
        new_cols = list(set(df.columns).difference(df_cor.columns))
        if new_cols:
            df_cor = align_columns(df_cor, df, columns=new_cols)
            b_df_cor_changed = True
            # df_cor = df_cor.reindex(columns=df.columns, copy=False)
        # add columns to df same as in store
        new_cols = list(set(df_cor.columns).difference(df.columns))
        if new_cols:
            if is_dask_dataframe(df):
                df = df.compute()
            df = align_columns(df, df_cor, columns=new_cols)

    elif b_correct_str:
        # error because our string longer => we need to increase store's limit
        b_df_cor_changed = True

    for col, dtype in zip(df_cor.columns, df_cor.dtypes):
        d = df_cor[col]
        if dtype != df[col].dtype:
            if b_correct_time and isinstance(
                d[0], pd.Timestamp
            ):  # is it possible that time types are different?
                try:
                    df_cor[col] = d.dt.tz_convert(tz=df[col].dt.tz)
                    b_df_cor_changed = True
                except {
                    AttributeError,
                    ValueError,
                }:  # AttributeError: Can only use .dt accessor with datetimelike values
                    pass
            elif b_correct_str:
                # todo:
                pass
            else:
                try:
                    dtype_max = np.result_type(df_cor[col].dtype, df[col].dtype)
                    if df[col].dtype != dtype_max:
                        df[col] = df[col].astype(dtype_max)
                    if df_cor[col].dtype != dtype_max:
                        df_cor[col] = df_cor[col].astype(dtype_max)
                        b_df_cor_changed = True
                except e:
                    lf.exception('Col "{:s}" have not numpy dtype?', col)
                    df_cor[col] = df_cor[col].astype(df[col].dtype)
                    b_df_cor_changed = True
                # pd.api.types.infer_dtype(df_cor.loc[df_cor.index[0], col], df.loc[df.index[0], col])
        elif b_correct_time and isinstance(d[0], pd.Timestamp):
            try:
                if d.dt.tz != df[col].dt.tz:
                    df_cor[col] = d.dt.tz_convert(tz=df[col].dt.tz)
                    b_df_cor_changed = True
            except (
                AttributeError,
                ValueError,
            ):  # AttributeError: Can only use .dt accessor with datetimelike values
                pass  # TypeError: Cannot convert tz-naive timestamps, use tz_localize to localize

    if b_df_cor_changed:
        # Update all cfg_out['db'] store data
        try:
            with ReplaceTableKeepingChilds([df_cor, df], tbl_parent, cfg_out, df_append_fun):
                pass
            return tbl_parent
        except Exception as e:
            lf.error(
                "{:s} Can not write to store. May be data corrupted. {:s}", msg_func, standard_error_info(e)
            )
            raise e
        except HDF5ExtError as e:
            lf.exception(e)
            raise e
    else:
        # Append corrected data to cfg_out['db'] store
        try:
            return df_append_fun(df, tbl_parent, cfg_out)
        except Exception as e:
            lf.error(
                "{:s} Can not write to store. May be data corrupted. {:s}", msg_func, standard_error_info(e)
            )
            raise e
        except HDF5ExtError as e:
            lf.exception(e)
            raise e


"""     store.get_storer(tbl_parent).group.__members__
        if tblD == cfg_out['table_log']:
            try:
                df.to_hdf(store, tbl_parent, append=True,
                            data_columns=True)  # , compute=False
                # store.append(tbl_parent, df, data_columns=True, index=False,
                #              chunksize=cfg_out['chunksize'])

            except ValueError as e:

        store.append(tbl_parent,
                        df_cor.append(df, verify_integrity=True),
                        data_columns=True, index=False,
                        chunksize=cfg_out['chunksize'])

        childs[tblD] = store[cfg_out['table_log']]

        dfLog = store[cfg_out['table_log']] if cfg_out['table_log'] in store  else None# copy before delete

        # Make index to be UTC
        df_cor.index = pd.to_datetime(store[tbl_parent].index, utc=True)
        store.remove(tbl_parent)
        store.append(tbl_parent,
                     df_cor.append(df, verify_integrity=True),
                     data_columns=True, index=False,
                     chunksize=cfg_out['chunksize'])
        if dfLog: # have removed only if it is a child
            store.remove(cfg_out['table_log'])
            store.append(cfg_out['table_log'], dfLog, data_columns=True, expectedrows=cfg_out['nfiles'], index=False, min_itemsize={'values': cfg_out['logfield_fileName_len']})  # append

"""


def add_log(
    log: Union[pd.DataFrame, MutableMapping, None], cfg_out: Dict[str, Any],
    tim=None, df=None, log_dt_from_utc=0
) -> str:
    """
    Append rows (metadata) to a log-table in an HDF5 store (create if necessary) with log information.

    :param cfg_out: Configuration dictionary with fields:
    - b_log_ready: If False or not present, updates log['Date0'] and log['DateEnd'] using df or tim.
    - db: Handle to the opened HDF5 store.
    - table_log: Optional string specifying the path of the log table. If no then we will try to get it from
    'tables_log, 'table' or 'tables' as order of precedence.
    - tables_log: Optional list of strings where the first element is the path of the log table, that is only
    used.
    - table: Optional string used to construct the path of the log table by adding '/log'.
    - tables: Optional list of strings used to construct the path of the log table by adding '/log' to the
    first element
    - logfield_fileName_len: Optional, specifies the fixed length of the string format for the 'fileName'
    column in the HDF5 table.
    :param df: DataFrame used to get log['Date0'] and log['DateEnd'] as start and end dates if
    cfg_out['b_log_ready'] is not set.
    :param log: Mutable mapping or DataFrame to be updated with log information. If None, an empty mapping is
    used.
    :param tim: Timestamp used to set 'Date0' and 'DateEnd' if they are not provided or if
    cfg_out['b_log_ready'] is not set.
    :param log_dt_from_utc: timedelta used to convert timestamps to a different timezone view.
    :return: table log name.
    Updates The log is updated with 'Date0' and 'DateEnd' if necessary and then passed to append_log().
    Raises:
        ValueError: If neither 'table_log' nor 'tables_log' is provided in cfg_out.
        IndexError: If there is no data in df to compute time limits.
        Exception: If there are other issues with creating or updating the log DataFrame.

    This function ensures that the log table is updated with the correct 'Date0', 'DateEnd' values based on the provided data and configuration, sets "DateProc" current date for all rows. It handles the creation of the log DataFrame if it is not already in the correct format and appends it to the specified log table in the HDF5 store.
    """
    if cfg_out.get("b_log_ready") and ((isinstance(log, Mapping) and not log) or log.empty):
        return

    # Get log table name
    table_log = (  # searching definition with more priority to more exsplicit fields
        cfg_out.get("table_log") or (
            cfg_out.get("tables_log") or
                [(cfg_out.get("table") or cfg_out["tables"][0]) + "/log"]
            )[0])
    set_field_if_no(cfg_out, "logfield_fileName_len", 255)

    if (not cfg_out.get("b_log_ready")) or (log.get("DateEnd") is None):
        if log is None:
            log = {}
        try:
            t_lims = (tim if tim is not None else df.index.compute() if is_dask_dataframe(df) else df.index)[
                [0, -1]
            ]
        except IndexError:
            lf.debug("no data")
            return
        # todo: correct if log is DataFrame (will be possible SettingWithCopyWarning)
        log["Date0"], log["DateEnd"] = timezone_view(t_lims, log_dt_from_utc)

    if isinstance(log, pd.DataFrame):
        if "DateProc" in log.columns:
            log.loc[log["DateProc"].isna(), "DateProc"] = datetime.now()
        else:
            log.loc[:, "DateProc"] = datetime.now()
    else:
        # dfLog = pd.DataFrame.from_dict(log, np.dtype(np.unicode_, cfg_out['logfield_fileName_len']))
        log["DateProc"] = datetime.now()
        try:
            log = pd.DataFrame(log).set_index("Date0")
        except ValueError as e:  # , Exception
            log = pd.DataFrame.from_records(
                log,
                exclude=["Date0"],
                index=log["Date0"] if isinstance(log["Date0"], pd.DatetimeIndex) else [log["Date0"]],
            )  # index='Date0' not work for dict

    try:
        return append_log(log, table_log, cfg_out)
    except ValueError as e:
        return append_on_inconsistent_index(cfg_out, table_log, log, append_log, e, "append log")
    except ClosedFileError as e:
        lf.warning("Check code: On reopen store update store variable")


def append_dummy_row(
    df: Union[pd.DataFrame,], freq=None, tim: Optional[Sequence[Any]] = None
) -> Union[pd.DataFrame,]:
    """
    Add row of NaN with index value that will between one of last data and one of next data start
    :param df: pandas dataframe, dask.dataframe supported only if tim is not None
    :param freq: frequency to calc index. If logically equal to False, then will be calculated using tim
    :param tim: sequence having in last elements time of 2 last rows
    :return: appended dataframe
    """
    if tim is not None:
        try:
            dindex = pd.Timedelta(seconds=0.5 / freq) if freq else np.abs(tim[-1] - tim[-2]) / 2
        except IndexError:  # only one element => we think they are seldom so use 1s
            dindex = pd.Timedelta(seconds=1)
        ind_new = [tim[-1] + dindex]
    else:
        df_index, itm = multiindex_timeindex(df.index)
        try:
            dindex = pd.Timedelta(seconds=0.5 / freq) if freq else np.abs(df_index[-1] - df_index[-2]) / 2
        except (IndexError, NotImplementedError):
            # only one element => we think they are seldom so use 1s or NotImplemented in Dask
            dindex = pd.Timedelta(seconds=1)
        ind_new = multiindex_replace(df.index[-1:], df_index[-1:] + dindex, itm)

    dict_dummy = {}
    tip0 = None
    same_types = (
        True  # tries prevent fall down to object type (which is bad handled by pandas.pytables) if possible
    )
    for name, field in df.dtypes.items():
        typ = field.type
        dict_dummy[name] = (
            typ(0) if np.issubdtype(typ, np.integer) else np.nan if np.issubdtype(typ, np.floating) else ""
        )
        if same_types:
            if typ != tip0:
                if tip0 is None:
                    tip0 = typ
                else:
                    same_types = False

    df_dummy = pd.DataFrame(
        dict_dummy, columns=df.columns.values, index=ind_new, dtype=tip0 if same_types else None
    ).rename_axis("Time")

    if is_dask_dataframe(df):
        # Get `dask.concat` without importing if dask is used
        dask_dataframe_module = type(df).__module__
        concat_func = getattr(sys.modules[dask_dataframe_module], "concat")

        return concat_func(
            [df, df_dummy], axis=0, interleave_partitions=True
        )  # buggish dask not always can append
    else:
        return pd.concat([df, df_dummy])  # df.append(df_dummy)

    # np.array([np.int32(0) if np.issubdtype(field.type, int) else
    #           np.nan if np.issubdtype(field.type, float) else
    #           [] for field in df.dtypes.values]).view(
    #     dtype=np.dtype({'names': df.columns.values, 'formats': df.dtypes.values})))

    # insert separator # 0 (can not use np.nan in int) [tim[-1].to_pydatetime() + pd.Timedelta(seconds = 0.5/cfg['in']['fs'])]
    #   df_dummy= pd.DataFrame(0, columns=cfg_out['names'], index= (pd.NaT,))
    #   df_dummy= pd.DataFrame(np.full(1, np.nan, dtype= df.dtype), index= (pd.NaT,))
    # used for insert separator lines


def append(
    cfg_out: Mapping[str, Any],
    df: Union[pd.DataFrame,],
    log: MutableMapping[str, Any],
    log_dt_from_utc=pd.Timedelta(0),
    tim: Optional[pd.DatetimeIndex] = None,
):
    """
    Append dataframe to Store:
    - df to cfg_out['table'] ``table`` node of opened cfg_out['db'] store and
    - child table with 1 row - metadata including 'index' and 'DateEnd' (which is calculated as first and last
    elements of df.index)

    :param df: pandas or dask dataframe to append. If dask then log_dt_from_utc must be None (not assign log metadata here)
    :param log: dict which will be appended to child tables (having name of cfg_out['tables_log'] value)
    :param cfg_out: dict with fields:
        db: opened hdf5 store in write mode
        table: name of table to update (or tables: list, then used only 1st element). if not none tables[0] is ignored
        table_log: name of child table (or tables_log: list, then used only 1st element). if not none tables_log[0] is ignored
        tables: None - to return with done nothing!
                list of str - to assign cfg_out['table'] = cfg_out['tables'][0]
        tables_log: list of str - to assign cfg_out['table_log'] = cfg_out['tables_log'][0]
        b_insert_separator: (optional), freq (optional)
        data_columns: optional, list of column names to write.
        chunksize: may be None but then must be chunksize_percent to calcW ake Up:
            chunksize = len(df) * chunksize_percent / 100
    :param log_dt_from_utc: 0 or pd.Timedelta - to correct start and end time: index and DateEnd.
        Note: if log_dt_from_utc is None then start and end time: 'Date0' and 'DateEnd' fields of log must be filled right already
    :param tim: df time index
    :return: None
    :updates:
        log:
            'Date0' and 'DateEnd'
        cfg_out: only if not defined already:
            cfg_out['table_log'] = cfg_out['tables_log'][0]
            table_log
            tables_written set addition (or creation) with tuple `(table, table_log)`
    """
    table = None
    df_len = len(df.index) if tim is None else len(tim)  # use computed values if possible for faster dask
    if df_len:  # dask.dataframe.empty is not implemented
        if cfg_out.get("b_insert_separator"):
            # Add separation row of NaN
            msg_func = f"{df_len}rows+1dummy"
            cfg_out.setdefault("fs")
            df = append_dummy_row(df, cfg_out["fs"], tim)
            df_len += 1
        else:
            msg_func = f"{df_len}rows"

        # Save to store
        # check/set tables names
        if "tables" in cfg_out:
            if cfg_out["tables"] is None:
                lf.info("selected({:s})... ", msg_func)
                return
            set_field_if_no(cfg_out, "table", cfg_out["tables"][0])

        lf.info("append({:s})... ", msg_func)
        set_field_if_no(cfg_out, "nfiles", 1)

        if "chunksize" in cfg_out and cfg_out["chunksize"] is None:
            if "chunksize_percent" in cfg_out:  # based on first file
                cfg_out["chunksize"] = int(df_len * cfg_out["chunksize_percent"] / 1000) * 10
                if cfg_out["chunksize"] < 10000:
                    cfg_out["chunksize"] = 10000
            else:
                cfg_out["chunksize"] = 10000

                if df_len <= 10000 and is_dask_dataframe(df):
                    df = df.compute()  # dask not writes "all NaN" rows
        # Append data
        try:
            table = append_data(df, cfg_out["table"], cfg_out)
        except ValueError as e:
            table = append_on_inconsistent_index(
                cfg_out, cfg_out["table"], df, append_data, e, msg_func
            )
        except TypeError as e:  # (, AttributeError)?
            if is_dask_dataframe(df):
                last_nan_row = df.loc[df.index.compute()[-1]].compute()
                # df.compute().query("index >= Timestamp('{}')".format(df.index.compute()[-1].tz_convert(None))) ??? works
                # df.query("index > Timestamp('{}')".format(t_end.tz_convert(None)), meta) #df.query(f"index > {t_end}").compute()
                if all(last_nan_row.isna()):
                    lf.exception(f"{msg_func}: dask not writes separator? Repeating using pandas")
                    table = append_data(
                        last_nan_row,
                        cfg_out["table"],
                        cfg_out,
                        min_itemsize={
                            c: 1
                            for c in (
                                cfg_out["data_columns"]
                                if cfg_out.get("data_columns", True) is not True
                                else df.columns
                            )
                        },
                    )
                    # sometimes pandas/dask get bug (thinks int is a str?): When I add row of NaNs it tries to find ``min_itemsize`` and obtain NaN (for float too, why?) this lead to error
                else:
                    lf.exception(msg_func)
            else:
                lf.error("{:s}: Can not write to store. {:s}", msg_func, standard_error_info(e))
                raise e
        except Exception as e:
            lf.error("{:s}: Can not write to store. {:s}", msg_func, standard_error_info(e))
            raise e
    # Append log rows
    # run even if df is empty because may be writing the log is needed only
    table_log = add_log(log, cfg_out, tim, df, log_dt_from_utc)
    if table_log:
        _t = (table, table_log) if table else (table_log,)
    else:
        if table:
            _t = (table,)
        return
    if "tables_written" in cfg_out:
        cfg_out["tables_written"].add(_t)
    else:
        cfg_out["tables_written"] = {_t}


def append_to(
    dfs: Union[pd.DataFrame,],
    tbl: str,
    cfg_out: Mapping[str, Any],
    log: Optional[Mapping[str, Any]] = None,
    msg: Optional[str] = None,
):
    """
    Append data to opened cfg_out['db'] by append() without modifying cfg_out['tables_written']
    :return: modified (as in `append()` function of this module) copy of cfg_out['tables_written']
    """
    if cfg_out["db"] is None:
        return set()
    if dfs is not None:
        if msg:
            lf.info(msg)
        # try:  # tbl was removed by temp_open() if b_overwrite is True:
        #     if remove(cfg_out['db'], tbl):
        #         lf.info('Writing to new table {}/{}', Path(cfg_out['db'].filename).name, tbl)
        # except Exception as e:  # no such table?
        #     pass
        cfg_out_mod = {**cfg_out, "table": tbl, "table_log": f"{tbl}/logFiles", "tables_written": set()}
        try:
            del cfg_out_mod["tables"]
        except KeyError:
            pass
        append(cfg_out_mod, dfs, {} if log is None else log)
        # dfs_all.to_hdf(cfg_out['db_path'], tbl, append=True, format='table', compute=True)
        return cfg_out_mod["tables_written"]
    else:
        print("No data.", end=" ")
        return set()


##############################################################################################################
# Refactored versions of previous functions

def empty_table(path_or_buf, table_name):
    """Simple and safe method to empty table with remove()."""
    with (
        pd.HDFStore(path_or_buf, mode="a")
        if isinstance(path_or_buf, (str, PurePath))
        else nullcontext(path_or_buf)
    ) as store:
        if table_name not in store:
            return

        # Read only index (no data columns loaded - very fast)
        idx_df = store.select(table_name, columns=[])

        if len(idx_df) > 0:
            # Get actual index bounds
            min_val = idx_df.index.min()

            # Remove all rows between min and max (inclusive)
            lf.warning(f"Removing data from {table_name}")
            store.remove(table_name, where=f"~(index < {repr(min_val)})")  # removes NaNs too



def check_obsolete(
    metadata: Mapping[str, Any],
    df_log: pd.DataFrame,
    cfg_out: MutableMapping[str, Any],
    field_to_del_older_records=None,
) -> Tuple[bool, bool]:
    """
    Skips current file it has been processed and stored data is up to date
    Excludes all data (!) from the store table and log table where time indices of existed data >= `t_start`,
    where `t_start` - time index of current data (log['index']) or nearest log record index (df_log.index[0])
    (see :param:field_to_del_older_records description)

    :param metadata: current metadata dict, must have fields needed for compare:
        'index' - log record corresponded starting data index
        'fileName' - in format as in log table to be able to find duplicates
        'fileChangeTime', datetime - to be able to find outdated data
    :param df_log: log record data - loaded from store before updating
    :param field_to_del_older_records: str or (default) None.
    - If None then:
        - checks duplicates among log records with 'fileName' equal to log['fileName'] and deleting all of
    them and data older than its index (with newest 'fileChangeTime').
        - sets b_stored_newer=True and deletes nothing if 'fileChangeTime' newer than that of current data
    - Else if not "index" then del. log records with field_to_del_all_older_records >= current file have, and
      data having time > its 1st index.
    - If "index" then del. log records with "DateEnd" > current file time data start, but last record wrote
      back with changed "DateEnd" to be consistent with last data time.

    :return: b_stored_newer: Have recorded data that was changed after current file was changed (log['fileChangeTime'])
    """
    b_stored_newer = False  # not detected yet
    if df_log is None:
        return b_stored_newer
    if field_to_del_older_records:
        # priority to the new data
        if field_to_del_older_records == "index":
            # - deletes data after df_log.index[0]
            # - updates db/tbl_log.DateEnd to be consisted to remaining data.
            t_start = metadata["index"]
            if t_start is None:
                try:
                    t_start = df_log.index[0]
                except IndexError:
                    t_start = None  # do nothing
            df_log_cur = df_log
        else:  # not tested
            b_cur = df_log[field_to_del_older_records] >= metadata[field_to_del_older_records]
            df_log_cur = df_log[b_cur]
            if not df_log_cur.empty:
                t_start = df_log_cur.index[0]
        # removing
    else:
        # stored data will be replaced only if have same fileName with 'fileChangeTime' older than new
        b_cur = (df_log["fileName"] == metadata["fileName"]) & (df_log["fileChangeTime"] < metadata["fileChangeTime"])
        df_log_cur = df_log[b_cur]
        n_log_rows = len(df_log_cur)
        if n_log_rows:
            # check if need skip data

            # to return info about newer stored data:
            last_file_change_time = df_log_cur["fileChangeTime"].dt.to_pydatetime()
            if any(last_file_change_time >= metadata["fileChangeTime"]):
                b_stored_newer = True  # can skip current file, because we have newer data records
                print(">", end="")
            else:
                # delete all next records of current file
                if not rem_rows(
                    cfg_out["db"],
                    zip(cfg_out["tables"], cfg_out["tables_log"]),
                    qstr="index>='{}'".format(i0 := df_log_cur.index[0]),
                    qstr_log="fileName=='{}'".format(df_log_cur.loc[i0, "fileName"]),
                ):
                    b_stored_newer = False  # deleted
    return b_stored_newer


def file_name_and_time_to_record(file_path: Path, logfield_fileName_len: int = 255):
    return {
        "fileName": f"{file_path.parent.name}/{file_path.stem}"[-logfield_fileName_len:],
        "fileChangeTime": datetime.fromtimestamp(file_path.stat().st_mtime)
        }

def keep_recorded_file(
    cur: Mapping[str, Any], existing: pd.DataFrame, keep_newer_records: bool = True, time_range=None, can_return_series=False
) -> bool:
    """
    Determines whether `cur` is same or older comparing to `existing` record. Used to skip current file
    processing if it is so, by default, not newer than its existing metadata log record or no existing record.
    :param cur: dict or DataFrame (1st row will be used), current log record
    :param existing: existing log records
    :param keep_newer_records: if False then skip only if current file time is same as recorded. False Useful
    if file time is random
    :param time_range: tuple of 2 datetime-like or None. If not None then skip only if matched records are in
    range between 1st and last element of time_range
    :return: True if we need to skip
    """
    if existing.empty:
        return False
    try:
        if isinstance(cur, pd.DataFrame):
            row = cur.iloc[0]
            cur = {**row.to_dict(), "index": row.name}
        b_file_rows = (
            (cur["fileName"] == existing["fileName"])
            & (
                (cur["fileChangeTime"] <= existing["fileChangeTime"])
                if keep_newer_records
                else (cur["fileChangeTime"] == existing["fileChangeTime"])
            )
        )
        b_skip = b_file_rows.any()
        if b_skip and time_range is not None:
            b_skip = existing[b_file_rows].index[0] >= pd.to_datetime(time_range[0], utc=True)
            if b_skip and len(time_range) > 1:
                b_skip = existing.loc[b_file_rows, "DateEnd"] <= pd.to_datetime(time_range[-1], utc=True)
            if not can_return_series:
                b_skip = b_skip.all()
        return b_skip
    except Exception as e:
        lf.exception("keep_recorded_file")
        raise e
    except KeyError:
        return False
    return b_skip


def read_db_log(db, table_log):
    """Read the existing metadata from the HDF5 file."""
    if db and table_log in db:
        return db[table_log]
    else:
        return pd.DataFrame(
            columns=["fileName", "fileChangeTime"], index=pd.DatetimeIndex([], tz="UTC", name='Time')
        )


def update_log_record_with_dates(log, df):
    start_date, end_date = (
        df.divisions[:: len(df.divisions) - 1]
        if is_dask_dataframe(df)
        else df.index[[0, -1]].to_list()
    )
    if isinstance(log, pd.DataFrame):
        if "DateProc" not in log.columns:
            log["DateProc"] = datetime.now()
        if log.shape[0] == 1:

            # If log index is not datetime, then replace it to current data.index[0]
            if not pd.api.types.is_datetime64_any_dtype(log.index):
                log.set_index([start_date])

            # Add row or replace values
            log.loc[start_date, ["DateEnd", "DateProc"]] = [end_date, datetime.now()]
        elif any(log.index == start_date):
            # If log.index contains data.index[0] then do not add row

            # if log.DateEnd not contains end_date then replace its last DateEnd
            if not any(log.DateEnd == end_date):
                log.loc[log.index[-1], ["DateEnd", "DateProc"]] = [
                    end_date, datetime.now()]
        else:
            log.loc[start_date, ["DateEnd", "DateProc"]] = [end_date, datetime.now()]
    else:
        log["index"] = start_date
        log["DateEnd"] = end_date
        log["DateProc"] = datetime.now()


def older_gen(
        db: pd.HDFStore, table: str, max_time: str|pd.Timestamp, chunk_size: int = 500000,
        skip_duplicates: bool = False) -> Iterator[pd.DataFrame]:
    """
    Yield df from a specified table in an HDF5 database that have rows index older than a specified start
    time.
    :param db: The HDF5 database from which to read.
    :param table: The name of the table within the database to query.
    :param max_time: The time threshold before which data should be retrieved. Rows with an index
        older than `max_time` will be yielded.
    :param chunk_size: optional number of rows to read into memory at a time if a MemoryError occurs
        (default is 500000).
    :param skip_duplicates: optional. Whether to skip duplicate indexes equal to `max_time`. If True,
        then additionally search duplicate index matches `max_time` and skip data till the last dup.
    :Yields: A DataFrame containing a chunk of rows from the table with an index older than `max_time`,
        potentially without duplicates.
    """
    if not db:
        return
    sel_args = {'key': table}

    # We will copy data excluding `max_time` if new data starting with `max_time` expected
    b_stop_before_max_time = bool(max_time)
    if max_time:
        sel_args["where"] = f"index<'{max_time}'"
        # if isinstance(max_time, pd.Timestamp):
        #     sel_args["where"] = ["index < '{:%Y-%m-%dT%H:%M:%S}'".format(
        #         max_time.tz_localize(tz=None))]

    if skip_duplicates:
        # Find the last coordinate of the specified index, taking into account possible duplicates
        # retrieving all coordinates with the exact match of the specified datetime or last index
        if not max_time:
            total_rows = db.get_storer(table).nrows
            # Select only the last row by providing start=total_rows-1 and stop=total_rows
            last_row = db.select(**sel_args, start=total_rows-1)  # , stop=total_rows anyway
            if not isinstance(last_row.index, pd.DatetimeIndex):
                err_msg = f"{Path(db.filename).name}/{table} index is not a DatetimeIndex!"
                lf.error(err_msg)
                raise ValueError(err_msg)
            # Get the index of output df last row (1st of found indexes to skip duplicates)
            max_time = last_row.index[0].tz_localize(tz=None)
        exact_match_coords = db.select_as_coordinates(table, where=f"index='{max_time}'")

        len_match = len(exact_match_coords)
        if len_match:
            if b_stop_before_max_time:
                # if `max_time` then gives same effect as `where` argument of `select` but may make it faster
                sel_args['stop'] = exact_match_coords[-1]

            if len_match > 1:
                # +1 excludes duplicate equal to `max_time`:
                sel_args['start'] = exact_match_coords[-2] + 1

                lf.warning(
                    "Skipped {} old data rows till duplicated start {} found in db",
                    sel_args['start'],  # + 1 was useful for message too for counting from 1
                    max_time,
                )
    try:
        # Attempt to read the entire old data into memory and yield it if successful
        old_data = db.select(**sel_args)
        if old_data.empty:
            return
        yield old_data
    except MemoryError:
        # If a MemoryError occurs, read and yield the old data in chunks
        old_data_iter = db.select(**sel_args, chunksize=chunk_size)
        for chunk in old_data_iter:
            yield chunk
    except ValueError:
        if not isinstance(db[table].index, pd.DatetimeIndex):
            err_msg = f"{Path(db.filename).name}/{table} index is not a DatetimeIndex!"
            lf.exception(err_msg)
            raise ValueError(err_msg)


def after_cycle_fun_default(
        db, table, table_log, logs_all: List[Tuple[pd.DataFrame, Mapping]], **kwargs
    ) -> List[str]:
    """
    Default `after_cycle_fun()` for `append_through_temp_db_gen()`. Saves log record to DB calling `add_log()`
    :param db_path: path of target DB
    :param table_log: log table name, can be name pattern depending on "table" if it is in the `kwargs`
    :param logs_all: log data records to save. If not DataFrame, must be dict with "index" field
    :param **kwargs: optional parameters for `move_tables()`:
    - "temp_db_path"
    - "recreate_index_tables_set"
    - "b_del_temp_db"
    - "addargs"
    :return tables: table log names list saved [table_log]
    """
    cfg_out_db = {'db': db, 'table_log': table_log, 'b_log_ready': True, **kwargs}
    b_have_df = isinstance(logs_all[0], pd.DataFrame)  # `append_through_temp_db_gen` inserts log_record[0]
    if b_have_df:
        log_df0, *logs_all = logs_all
    if logs_all:
        log_df_all = pd.DataFrame.from_records(logs_all, index=["index"])
        if b_have_df:
            log_df = pd.concat([log_df0, log_df_all])
        else:
            log_df = log_df_all
    elif b_have_df:
        log_df = log_df0
    else:
        return []
    log_df.index.name = "Time"
    return [add_log(log_df, {**cfg_out_db, "b_log_ready": True})]


def append_through_temp_db_gen(
    data_gen: Callable[[Any], Iterable[Any]],
    db: Optional[pd.HDFStore] = None,
    db_path: Optional[Path] = None,
    temp_db: Optional[pd.HDFStore] = None,
    temp_db_path: Optional[Tuple[str, Path]] = None,
    in_cycle_fun: Callable[[pd.HDFStore, str, Any], None] = (
        lambda db, table, data, **kwargs: data.to_hdf(
            db,
            key=table,
            append=True,
            data_columns=True,
            format="table",
            index=False,
            dropna=True,
        )  # db.append(table, data, data_columns=True)
    ),
    after_cycle_fun: Callable[
        [pd.HDFStore, str, str, Sequence[Any], int], Tuple[str]
    ] = after_cycle_fun_default,
    skip_fun: Callable[[Any, pd.DataFrame], bool] = keep_recorded_file,
    table_from_meta: Optional[Callable[[Any], Tuple[str, str]]] = None,
    record_from_meta: Optional[Callable[[Any], Mapping[str, Any]]] = None,
    table: Optional[str] = None,
    table_log: str = "{}/logFiles",
    chunksize: int = 500000,
) -> Iterator[Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]]:
    """
    The generator yields data and metadata while simultaneously appending them to a temporary HDF5 store.
    It starts by transferring existing relevant data from a source database to the temporary store preventing
    generation or transferring of duplicate data: as it consumes new data chunk from the `data_gen` generator,
    it uses supplied metadata to discern and skip over data that was already saved. For this `data.index`
    must be of datetime type and `skip_fun` provided.
    After updateing the target table, function appends accumulated log records extracted from metadata and `data.index` to the corresponding table log in the temporary store.

    :param data_gen: A generator function that yields `(data, metadata)`:
        `data` can be pandas or dask dataframe or list with dataframe in 1st element. This dataframe used here
        to update `log_record` with 1st and last index values.
        `data_gen` takes a single argument: `skip_for_meta` - function we constuct here from other arguments:
        `skip_fun`, `record_from_meta` and `table_from_meta`. `data_gen` should evaluate`skip_for_meta(meta)`:
        - if `skip_fun` returns True then you should skip current cycle of getting data and on 1st cycle of
        new table name yield:
            - `(None, metadata)` to indicate us to load all data from db without limiting its last time, or
            - `(data, me- if it returns Truetadata)` to indicate us to load data from db till `data.index[0]`
            time.
            For loaded data from `db` we call `in_cycle_fun` and yield it. For second variant we will
            call `in_cycle_fun` for `data` and yield it too.
        - if `skip_fun` returns False on 1st cycle of new table then existed data will be skipped and we will
        call `in_cycle_fun` for your (new) `data` and yield it.
        Note: `metadata` is not used here but re-yielded (beeng appended with `record` and `table`). Contraly,
        `meta` here is an argument of `record_from_meta` (and my be `table_from_meta`) you provide, which will
        be called inside `skip_for_meta`.
        Yield [] by `data_gen` if you want skip writing to `table` entarely and delete all exited data
        Yield any not falsy and not DataFrame data by `data_gen` to not yield it out without saving
    :param db: An open HDFStore object for the source database. If None, `db_path` must be provided.
    :param db_path: The file path to the source HDF5 file. Used to open the store if `db` is not provided.
    :param temp_db: An open HDFStore object for the destination temporary database. If None, `temp_db_path`
        must be used.
    :param temp_db_path: The file path to the destination temporary HDF5 file. Used to open the store if
        `temp_db` is not provided or else db_path with appending f"{db_path.stem}_not_sorted.h5" name.
    :param in_cycle_fun: A function called for each chunk of data. It should take a `temp_db` HDFStore, a
        `table` name, a `data` chunk, and perform an action such as **appending the data chunk to the table**.
    :param after_cycle_fun: A function called after all data chunks have been processed. It should take
        `temp_db`, `table`, `table_log`, a list of accumulated log records (list containinb `existing` log
        DataFrame and appended dicts from `record_from_meta()` outputs), and perform an action such as
        appending records to the log table in `temp_db` and return list of table names. These returned tables
        will be moved back to `db` in addition to `table` after processing of all tables and closing `db`.
        Child tables may be not needed to return if I've made that they are moved automatically (todo: check)
    :param record_from_meta: A function that returns a log record `log_record` from current `meta`
        (`skip_for_meta` argument). This record must have fields requred by `skip_fun` to compare to existed
        log. `log_record` can be Mapping or DataFrame, here it will be updated with "index" and "DateEnd"
        fields from 1st and last `data` rows of `data_gen` output and will be saved to hdf5 log by `add_log`
    :param skip_fun: A function(`cur`, `existing`) that evaluates a chunk of current log record `cur` against
        the `existing` log records that will be loaded from `table_log` of source `db`. It determines whether
        the corresponding data chunk should be skipped (returns True, or False otherwise). If keep default
        then meta should have "fileName" and "fileChangeTime" fields (see `keep_recorded_file()`).
    :param table_from_meta: An optional function(`meta`) which returns `table` name (dynamically determines
        it for each chunk of metadata processed by `data_gen`)
    :param table: The name of the table in the source `db` that will be copied and appended in target
        `temp_db`. Required if `table_from_meta` is not provided.
    :param table_log: Required, the name of the logging table in the `db`. Used to get existing log and to
        append old records and records extracted from metadata to `temp_db`. Can be {}-pattern to fill with
        `table`
    :param chunksize: The size of chunks to use when reading source data to prevent MemoryErrors on large
        datasets.

    Yields (data, (metadata, log_record, table)):
    - metadata: from data_gen(). For copied old data it will be the same as for 1st new data
    - log_record: all log records corresponded to copied old data part in 1st chunk and empty list for its
    next chunks, `record_from_meta(meta)` outputs for new data
    - table: table name where data has been copied / appended
    """
    if db is None:
        if db_path is None:  # Not docomented option!
            lf.warning("No db or db_path set! Trying yield from data_gen without setting log record")
            for i_gen, (data, metadata) in enumerate(data_gen(skip_for_meta=None)):
                log_cur = {}  # record_from_meta(meta)
                yield data, (metadata, log_cur, table)
            # yield from data_gen(skip_for_meta=None)  # old
            return
        elif db_path.is_file():
            db = pd.HDFStore(db_path, mode="r")  # what if ?
        else:
            def existing_log(*args, **kwargs):
                return pd.DataFrame()
            db = nullcontext(db)
    tbl_written = set()
    with db as db:  # not nullcontext(db) to autoclose
        # Initialize the data generator with the skip function
        if table_from_meta:
            # This variables will be captured by the nested function
            existing_log = pd.DataFrame()  # argument for `skip_fun`
            table_prev = None  # current table tracker
            table_log_pattern = table_log
            b_new_table = True
            # info from current meta to append to hdf5 log:
            # - index: data start time to trim existed log and data,
            # "fileName", "fileChangeTime" fields to use in `skip_fun`
            log_cur = {}

            def get_existing_log_and_update_nonlocals(meta):
                """Callback to update `existing_log`, `table`, `table_log`, `b_new_table` in outer context."""
                nonlocal existing_log, table_prev, table, table_log, b_new_table
                table = table_from_meta(meta)
                b_new_table = table_prev != table or existing_log is None
                if b_new_table:
                    table_log = table_log_pattern.format(table)
                    existing_log = read_db_log(db, table_log)
                    table_prev = table  # Update the current table tracker
                return existing_log

            def get_existing_along_skip(meta):
                """
                `skip_fun` callback modification to update nonlocals defined here and in
                `get_existing_log_and_update_nonlocals()` in outer context
                """
                nonlocal log_cur
                log_cur = record_from_meta(meta)
                return skip_fun(
                    cur=log_cur, existing=get_existing_log_and_update_nonlocals(meta)
                )
        else:
            existing_log = read_db_log(db, table_log)

            def get_existing_along_skip(meta):
                nonlocal log_cur, b_new_table, table_prev

                log_cur = record_from_meta(meta)
                b_skip = skip_fun(cur=log_cur, existing=existing_log)

                # Set `b_new_table` to False after second chunk of new data
                if not b_skip:
                    if table_prev:
                        b_new_table = False
                    table_prev = True

                return b_skip

        appending_generator = data_gen(skip_for_meta=get_existing_along_skip)

        # Execute data generator in context of opened temporary DB were we copy old data and can append new
        logs_all = []
        if not (temp_db or temp_db_path):
            if not db_path:
                db_path = Path(db.filename)
            temp_db_path = db_path.with_name(f"{db_path.stem}_not_sorted.h5")

        # Preventing error Unable to open/create file '.../proc_noAvg_not_sorted.h5'
        b_need_close = temp_db is None
        if b_need_close:
            try:
                temp_db = open_trying(temp_db_path, change_name=True, mode="w", complevel=1)
            except:
                lf.exception(f"Can not open temp DB {temp_db_path.name}")

        # with nullcontext(temp_db) if temp_db is not None else  as temp_db:

        for i_gen, (data, metadata) in enumerate(appending_generator):
            # callback in `get_existing_along_skip()` provides `log_cur`, ..., and when table is changed
            # if `table_from_meta` is provided updates variables `table`, `table_log` and `existing_log`

            b_suprressed_saving = not (isinstance(data, pd.DataFrame) or is_dask_dataframe(data))
            if b_suprressed_saving and data:  # need just yield data
                yield data, (metadata, pd.DataFrame(), table)
                continue
            if b_new_table:
                # Initialise the log records for new table
                logs_all = []
                if data is None:
                    start_date = None  # not limit older_gen by last time to get all old data
                    # if new data skipped, loading old data will be used for all `existing_log` records
                    log_old = existing_log
                elif not any(data):  # data suppressed -> removing data by writing empty table
                    log_old = log_cur = pd.DataFrame()  # for checking the empty returning True
                    # data = pd.DataFrame()
                    # in_cycle_fun(temp_db, table, data_old)
                    # yield data, (metadata, log_cur, table)
                    # continue
                else:
                    # append log record about new data and limit old data last time with 1st new data date
                    update_log_record_with_dates(log_cur, df=data[0] if isinstance(data, list) else data)
                    start_date = log_cur["index"]
                    # Part of log records of existed data we want to keep and yield first
                    log_old = (
                        existing_log[existing_log.index <= start_date]
                        if not existing_log.empty
                        else existing_log
                    )
                if not log_old.empty:  # if same new table continues then we've emptied it already
                    # Process and yield records older than 1st from data_gen()

                    # Drop duplicates in old log index (hopefully such records can't made by this func.)
                    b_time_repeats = log_old.index.duplicated(keep="first")
                    b_time_repeats_any = b_time_repeats.any()
                    if b_time_repeats_any:
                        lf.warning(
                            "Duplicates in old log index: {} dropping",
                            log_old[b_time_repeats])
                        log_old = log_old[~b_time_repeats]
                    existing_log = log_old  # to can continue compare with new data

                    # Copy relevant data from the original db to the temp_db (by default) and yield it
                    for data_old in older_gen(db, table, start_date, chunksize, b_time_repeats_any):
                        lf.info("{}/{} << old data {}", Path(temp_db.filename).stem, table, data_old.shape)
                        in_cycle_fun(temp_db, table, data_old)
                        yield data_old, (metadata, log_old, table)
                        # Collect the old logs, empty their further output (yielded all in 1st chunk)
                        if not logs_all:
                            logs_all, log_old = [log_old], log_old.iloc[:0]
                    # existing_log = existing_log.iloc[:0]  # emptying
            elif data is not None and any(data):
                # append log record about new data
                update_log_record_with_dates(log_cur, df=data[0] if isinstance(data, list) else data)

            if data is not None:
                # Append the current data chunk to the temp_db

                if b_suprressed_saving:
                    lf.info("Clear {}/{}, have new data", Path(temp_db.filename).stem, table)
                    empty_table(temp_db, table) # delete all data, processing supprssed
                else:
                    lf.info("{}/{} << new data", Path(temp_db.filename).stem, table)
                    in_cycle_fun(temp_db, table, data)
                yield data, (metadata, log_cur, table)
                if b_suprressed_saving:
                    continue
                # Collect log records for the current table
                logs_all.append(log_cur)

                # Rewrite table only if have new data (data is None means need leave old data as is)
                # Append metadata log records for table and collect written tables (they can change each iter)
                if logs_all:
                    tbl_written.add(table)
                    tbl_written.update(after_cycle_fun(temp_db, table, table_log, logs_all))

        # Make ready for ptrepack (our wrapping would do the same adding indexes but on error with msg)
        if tbl_written:
            for tbl in tbl_written:
                temp_db.create_table_index(tbl, columns=["index"], kind="full")  # ,optlevel=9
            temp_db.flush(fsync=True)  # Ensure all data is written to disk

    if tbl_written or b_need_close:
        temp_db.close()  # can not move from opened DB
        # wait?

        # Replace all data in DB tables with data written to temporary DB tables
        if tbl_written:
            try:
                failed_storages = move_tables(
                    {"db_path": db_path, "temp_db_path": Path(temp_db.filename), "b_del_temp_db": True},
                    tbl_names=tbl_written,
                )
                return failed_storages
            except Ex_nothing_done:
                lf.warning("Tables {} of combined data not moved", tbl_written)
            except Exception:
                lf.exception(f"Can not move tables from temporary DB {Path(temp_db.filename).name}!")