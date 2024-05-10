#!/usr/bin/env python
# coding:utf-8
"""

Author:  Andrey Korzh <ao.korzh@gmail.com>
"""

import logging
from pathlib import Path, PurePath
import re
import sys  # from sys import argv
import warnings
from os import path as os_path, getcwd as os_getcwd, chdir as os_chdir, remove as os_remove
from datetime import timedelta, datetime
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

if __debug__:
    from matplotlib import pyplot as plt
warnings.catch_warnings()
warnings.simplefilter("ignore", category=NaturalNameWarning)
# warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
# my
from .filters import inearestsorted, inearestsorted_around
from .utils2init import (
    set_field_if_no,
    dir_create_if_need,
    getDirBaseOut,
    FakeContextIfOpen,
    Ex_nothing_done,
    standard_error_info,
    LoggingStyleAdapter,
    ExitStatus,
)
from .utils_time import multiindex_timeindex, check_time_diff

pd.set_option("io.hdf.default_format", "table")
lf = LoggingStyleAdapter(logging.getLogger(__name__))


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
            # store= pd.HDFStore(fileInF, mode='r')
            try:
                if not args.chunkDays:
                    args.chunkDays = 1
                if str_where:  # s=str_where[0]
                    Term = []
                    bWate = False
                    for s in str_where:
                        if bWate:
                            if s[-1] == '"':
                                bWate = False
                                Term[-1] += f" {s[:-1]}"
                            else:
                                Term[-1] += f" {s}"
                        elif s[0] == '"':
                            bWate = True
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
    for lsts in lst_of_lsts:
        if isinstance(lsts, str):
            yield lsts
        else:
            yield from lsts


def unzip_if_need_enumerated(lst_of_lsts: Iterable[Union[Iterable[str], str]]) -> Iterator[Tuple[int, str]]:
    """
    Enumerate each group of elements from 0. If element is not a group (str) just yeild it with index 0
    :param lst_of_lsts:
    :return:
    """
    for lsts in lst_of_lsts:
        if isinstance(lsts, str):
            yield (0, lsts)
        else:
            yield from enumerate(lsts)


def getstore_and_print_table(fname, strProbe):
    import pprint

    with FakeContextIfOpen(lambda f: pd.HDFStore(f, mode="r"), fname) as store:
        # if isinstance(fname, str):
        #     store = pd.HDFStore(fname, mode='r')
        # elif isinstance(fname, pd.HDFStore):
        #     store = fname
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


def h5find_tables(store: pd.HDFStore, pattern_tables: str, parent_name=None) -> List[str]:
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


def h5sort_pack(
    h5source_fullpath: str,
    h5out_name: str,
    table_node: str,
    arguments: Optional[Sequence[str]] = None,
    addargs: Optional[Sequence[str]] = None,
    b_remove: Optional[bool] = False,
    col_sort: Optional[str] = "index",
):
    """
    Compress and save table (with sorting by index) from h5_source_fullpath to h5_cumulative using ``ptprepack`` utility
    :param h5source_fullpath: - full file name
    :param h5out_name: base file name + ext of cumulative hdf5 store only
                         (other than that in h5_source_fullpath)
    :param table_node: node name in h5_source_fullpath file
    :param arguments: list, 'fast' or None. None is equal to ['--chunkshape=auto',
        '--complevel=9', '--complib=zlib',f'--sortby={col_sort}', '--overwrite-nodes']
    :param addargs: list, extend arguments with more parameters
    :param b_remove: file h5_source_fullpath will be deleted after operation!
    :param col_sort:
    :return: full path of cumulative hdf5 store
    Note: ``ptprepack`` not closes hdf5 source if not finds data!
    """

    h5dir, h5source = os_path.split(h5source_fullpath)
    h5out_path = os_path.join(h5dir, h5out_name)
    if not table_node:
        return h5out_path
    print(f"sort&pack({table_node}) to {h5out_name}")
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
        sys.argv[1:] = arguments + [f"{h5}:/{table_node}" for h5 in (h5source, h5out_name)]
        # --complib=blosc --checkCSI=True

        ptrepack()
        # then copy children  # '--non-recursive' requires sys.argv[1:] = arguments + [f'{h5source}:/{table_node}/table', f'{h5out_name}:/{table_node}/table'] i.e. can not copy indexes
        # with pd.HDFStore(h5out_path, 'a') as store_out, pd.HDFStore(h5source_fullpath, 'r') as store_in:
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
                with pd.HDFStore(h5source_fullpath) as store:
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
                            store.flush()
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
            os_remove(h5source_fullpath)
        except:
            print(f'can\'t remove temporary file "{h5source_fullpath}"')
    return h5out_path


query_range_pattern_default = "index>='{}' & index<='{}'"


def h5sel_index_and_istart(
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
    :return: (empty columns dataframe with index[query_range], coordinate index of query_range[0] in table)
    Tell me, if you know, how to do this with only one query, please
    """
    if query_range_lims is None:  # select all
        df0 = store.select(tbl_name, columns=[])
        i_start = 0
    else:  # select reduced range
        if to_edge:
            query_range_lims = [pd.Timestamp(lim) for lim in query_range_lims]
            query_range_lims[0] -= to_edge
            query_range_lims[-1] += to_edge
        qstr = query_range_pattern.format(*query_range_lims)
        lf.info(f"query:\n{qstr}... ")
        df0 = store.select(tbl_name, where=qstr, columns=[])
        try:
            i_start = store.select_as_coordinates(tbl_name, qstr)[0]
        except IndexError:
            i_start = 0
    return df0, i_start


def h5sel_interpolate(
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

    # pd.Index(timzone_view(time_points, dt_from_utc=df.index.tzinfo._utcoffset))
    # except TypeError as e:  # if Cannot join tz-naive with tz-aware DatetimeIndex
    #     new_index = timzone_view(df.index, dt_from_utc=0) | pd.Index(timzone_view(time_points, dt_from_utc=0))

    df_interp_s = df.reindex(new_index).interpolate(
        method=method,
    )  # why not works fill_value=new_index[[0,-1]]?
    df_interp = df_interp_s.loc[t, :]
    return df_interp


def h5coords(
    store: pd.HDFStore,
    tbl_name: str,
    q_time: Optional[Sequence[Any]] = None,
    query_range_lims: Optional[Sequence[Any]] = None,
    query_range_pattern: str = query_range_pattern_default,
) -> Tuple[Union[pd.Index, None], int, Union[List[int], np.ndarray]]:
    """
    Get table's index for ``q_time`` edges/``query_range_lims`` and coordinates indexes of ``q_time`` in ``store`` table

    :param store:
    :param tbl_name: table name in ``store``
    :param q_time: optional, points. If strings - converts them to 'M8[ns]'.
    :param query_range_lims: optional, needed interval. If None, then use 1st and last of q_time.
    :param query_range_pattern:
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
            query_range_lims = q_time[:: (q_time_len - 1)] if (q_time_len := len(q_time)) > 1 else q_time

    df0range, i0range = h5sel_index_and_istart(
        store,
        tbl_name,
        query_range_lims,
        query_range_pattern,
        to_edge=pd.Timedelta(minutes=10),
        # msg_add='with padding to edges'
    )
    if q_time is None:
        return df0range.index, 0, [0, df0range.index.size]

    i_queried = inearestsorted(df0range.index.values, np.array(q_time, df0range.index.dtype.str))
    return df0range.index, i0range, i_queried + i0range


def h5load_points(
    store: pd.HDFStore,
    tbl_name: str,
    columns: Optional[Sequence[Union[str, int]]] = None,
    time_points: Optional[Union[np.ndarray, pd.Series, Sequence[int]]] = None,
    dt_check_tolerance=pd.Timedelta(seconds=1),
    query_range_lims: Optional[Union[np.ndarray, pd.Series, Sequence[int]]] = None,
    query_range_pattern: str = query_range_pattern_default,
    interpolate="time",
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
    :param interpolate: str: "method" arg of pandas.Series.interpolate. If not interpolate, then return
    closest points
    :return: (df, bbad):
        df - table of found points, bbad - boolean array returned by other_filters.check_time_diff() or
        df - dataframe of query_range_lims if no time_ranges nor time_points

    Note: query_range_pattern will be used only if query_range_lims specified
    My use:
    h5select(store, cfg['in']['table_nav'], ['Lat', 'Lon', 'DepEcho'], dfL.index, query_range_lims=(dfL.index[0],
             dfL['DateEnd'][-1]), query_range_pattern=cfg['process']['dt_search_nav_tolerance'])
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

    df_index_range, i0range, i_queried = h5coords(
        store, tbl_name, time_points, query_range_lims, query_range_pattern
    )
    # if time_ranges:  # fill indexes inside intervals
    #     i_queried = np.hstack(np.arange(*se) for se in zip(i_queried[:-1], i_queried[1:] + 1))
    if not any(i_queried):
        # raise IndexError()
        input_data = input(
            f"No data for specified interval in {tbl_name}.\nInput {columns} for {time_points}?"
        )  # todo recover from file name
        try:
            df = pd.DataFrame.from_records(
                [[float(word) for word in row.replace(",", " ").split()] for row in input_data.split(";")],
                columns=columns,
                index=time_points,
            )
            lf.info(f"Ok input: {df}")
            dt = np.zeros_like(time_points, dtype="timedelta64[ns]")
        except ValueError as e:
            raise IndexError("Error with input data")
    else:
        bbad, dt = check_time_diff(
            t_queried=time_points,
            t_found=df_index_range[i_queried - i0range].values,
            dt_warn=dt_check_tolerance,
            return_diffs=True,
        )
        if any(bbad) and interpolate:
            i_queried = inearestsorted_around(df_index_range.values, time_points) + i0range
            df = h5sel_interpolate(
                i_queried, store, tbl_name, columns=columns, time_points=time_points, method=interpolate
            )
        else:
            df = store.select(tbl_name, where=i_queried, columns=columns)

    return df, dt


def h5load_range(
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


def h5load_ranges(
    store: pd.HDFStore, table: str, t_intervals=None, query_range_pattern=query_range_pattern_default
) -> pd.DataFrame:
    """
    Load data
    :param t_intervals: even sequence of datetimes or strings convertible to index type values. Each pair defines edges
    of data that will be concatenated. 1st and last must be min and max values in sequence.
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
    df = h5load_range(
        store, table, query_range_lims=t_intervals[0 :: (n - 1)], query_range_pattern=query_range_pattern
    )
    return df


def df_data_append_fun(df: pd.DataFrame, tbl_name: str, cfg_out: Mapping[str, Any], **kwargs):
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


def df_log_append_fun(df: pd.DataFrame, tbl_name: str, cfg_out: Mapping[str, Any]):
    """
    Append pandas HDF5 store with cfg_out settings for log table
    :param df:
    :param tbl_name:
    :param cfg_out: fields:
        'db'
        'nfiles',
        'logfield_fileName_len': Optional int specifies length for 'values' col. If no then no restrictions
    :return: tbl_name
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
        expectedrows=cfg_out["nfiles"],
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
        db_path = temp_db_path.with_name(temp_db_path.name.replace("_not_sorted", ""))
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


def h5remove(db: pd.HDFStore, node: Optional[str] = None, query: Optional[str] = None):
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


def h5remove_tables(db: pd.HDFStore, tables: Iterable[str], tables_log: Iterable[str], temp_db_path=None):
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
                h5remove(db, tbl)
                tbl_prev = tbl
                break
            except ClosedFileError as e:  # file is not open
                lf.error("waiting {:d} (/3) because of error: {:s}", i, str(e))
                sleep(i)
            # except HDF5ExtError as e:
            #     break  # nothing to remove
        else:
            lf.error("not successed => Reopening...")
            if temp_db_path:
                db = pd.HDFStore(temp_db_path)
            else:
                db.open(mode="r+")
            h5remove(db, tbl)
    db.flush()
    return db


# ----------------------------------------------------------------------
class ReplaceTableKeepingChilds:
    """
    Saves childs (before You delete tbl_parent)
    #for h5find_tables(store, '', parent_name=tbl_parent)

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

                self.cfg_out["db"].flush()  # .flush(fsync=True

        except AttributeError:
            pass  # print(tbl_parent + ' has no childs')
        # Make index to be UTC

        # remove parent table that must be written back in "with" block
        try:
            h5remove(self.cfg_out["db"], self.tbl_parent)
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
        self.cfg_out["db"].flush()
        if exc_type is None:
            for tbl in self.childs:
                self.cfg_out["db"]._handle.move_node(
                    tbl.replace(self.tbl_parent, self.temp_group),
                    newparent=f"/{self.tbl_parent}",
                    createparents=True,
                    overwrite=True,
                )
        # cfg_out['db'].move('/'.join(tbl.replace(tbl_parent,self.temp_group), tbl))
        # cfg_out['db'][tbl] = df # need to_hdf(format=table)
        return False


# ----------------------------------------------------------------------
def h5remove_duplicates(cfg, cfg_table_keys: Iterable[Union[Iterable[str], str]]) -> Set[str]:
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
                        # todo: if many to delete try h5remove_duplicates_by_loading()
                    except:
                        lf.exception("can not delete duplicates")
                        dup_tbl_set.add(tbl)
    return dup_tbl_set


def h5remove_duplicates_by_loading(cfg, cfg_table_keys: Iterable[Union[Iterable[str], str]]) -> Set[str]:
    """
    Remove duplicates by coping tables to memory, keep last. todo: merge fields
    :param cfg: dict with keys:
        keys specified by cfg_table_keys
        chunksize - for data table
        logfield_fileName_len, nfiles - for log table
    :param cfg_table_keys: list, in which 'tables_log' means that cfg['tables_log'] is a log table. Alternatively group tables in subsequences such that log tables names is after data table in each subsequence (cfg[cfg_table_keys[group]])
    :return dup_tbl_set: tables that still have duplicates
    """
    cfg["db"].flush()  # not works (loading will give only part of data), but this works:
    cfg["db"].close()
    cfg["db"].open("r+")

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
                            df_log_append_fun
                            if (cfgListName == "tables_log" or i_in_group > 0)
                            else df_data_append_fun,
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
    Create full indexes. That is mandatory before using ptprepack in h5move_tables() below
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
                if i_in_group == 0 or cfgListName != "tables_log":  # not nested (i.e. log) table
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


def h5move_tables(
    cfg_out, tbl_names: Union[Sequence[str], Sequence[Sequence[str]], None] = None, **kwargs
) -> Dict[str, str]:
    """
    Copy tables tbl_names from one store to another using ptrepack. If fail to store
    in specified location then creates new store and tries to save there.
    :param cfg_out: dict - must have fields:
      - temp_db_path: source of not sorted tables
      - db_path: pathlib.Path, full path name (extension ".h5" will be added if absent) of hdf store to put
      - tables, tables_log: Sequence[str], if tbl_names not specified
      - b_del_temp_db: bool, remove source store after move tables. If False (default) then deletes nothing
      - addargs: ptprepack params, they will be added to the defaults specified in h5sort_pack()
    :param tbl_names: list of strings or list of lists (or tuples) of strings. List of lists is useful to keep order of
     operation: put nested tables last.
    :param kwargs: ptprepack params
    Note: ``ptprepack`` not closes hdf5 source if it not finds data!
    Note: Not need specify childs (tables_log) if '--non-recursive' not in kwargs
        Strings are names of hdf5 tables to copy
    :return: Empty dict if all success else if we have errors - Dict [tbl: HDF5store file name] of locations of last tried
     savings for each table
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

    ptrepack_add_args = cfg_out.get("addargs", [])
    if "--overwrite" in ptrepack_add_args:
        if len(tables_top_level) > 1:
            lf.error(
                'in "--overwrite" mode with move many tables: will remove each previous result after each table \
            and only last table wins!'
            )
    elif "--overwrite-nodes" not in ptrepack_add_args:  # default: False
        # h5sort_pack can not remove/update dest table, and even corrupt it if existed, so we do:
        try:
            with pd.HDFStore(cfg_out["db_path"]) as store:
                # h5remove(store, tbl)
                for tbl in tables_top_level:
                    h5remove(store, tbl)

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

    with pd.HDFStore(cfg_out["temp_db_path"]) as store_in:  # pd.HDFStore(cfg_out['db_path']) as store,
        for tbl in tables:
            try:
                _ = store_in.get_storer(tbl).group.table.colindexes
            except KeyError as e:
                failed_storages[tbl] = cfg_out["temp_db_path"].name
                lf.error("h5move_tables({:s}) failed: {}, continue...", tbl, e)
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
                    lf.error("h5move_tables({:s}): failed to create indexes", tbl)
                    failed_storages[tbl] = cfg_out["temp_db_path"].name
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
                h5sort_pack(
                    cfg_out["temp_db_path"],
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
                    cfg_out["temp_db_path"],
                    cfg_out["db_path"],
                )
    else:
        raise Ex_nothing_done(f"Not valid table names: {tbl_names}!")
    return failed_storages
    # storage_basenames = {}
    #         if False:  # not helps?
    #             storage_basename = os_path.splitext(cfg_out['db_base'])[0] + "-" + tbl.replace('/', '-') + '.h5'
    #             lf.info('so start write to {}', storage_basename)
    #             try:
    #                 h5sort_pack(cfg_out['temp_db_path'], storage_basename, tbl, addargs=cfg_out.get('addargs'), **kwargs)
    #                 sleep(4)
    #             except Exception as e:
    #                 storage_basename = cfg_out['db_base'] + '-other_place.h5'
    #                 lf.error('Error: "{}"\nwhen write {} to original place! So start write to {}', e, tbl,
    #                                                                                                    storage_basename)
    #                 try:
    #                     h5sort_pack(cfg_out['temp_db_path'], storage_basename, tbl, addargs=cfg_out.get('addargs'), **kwargs)
    #                     sleep(8)
    #                 except:
    #                     lf.error(tbl + ': no success')
    #             storage_basenames[tbl] = storage_basename
    # if storage_basenames == {}:
    #     storage_basenames = None


def h5index_sort(
    cfg_out,
    out_storage_name=None,
    in_storages: Optional[Mapping[str, str]] = None,
    tables: Optional[Iterable[Union[str, Tuple[str]]]] = None,
) -> None:
    """
    Checks if tables in store have sorted index and if not then sort it by loading, sorting and saving data.
    :param cfg_out: dict - must have fields:
        'db_path': store where tables will be checked
        'temp_db_path': source of not sorted tables for h5move_tables() if index is not monotonic
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
                            h5remove_duplicates(
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
                            df_log_append_fun if ("log" in tbl) else df_data_append_fun,
                        ):
                            nonm_tbl_set.discard(tbl)
                        lf.warning("Saved sorted in memory - ok.")
                    except Exception as e:
                        lf.exception(
                            "Error sorting Table {:s} in memory. Will try sort by ptrepack in h5move_tables() to temp_db_path and back",
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
                # as this fun is intended to check h5move_tables stranges, repeat it with forcing update index
                if not cfg_out[
                    "temp_db_path"
                ].is_file():  # may be was deleted because of cfg_out['b_del_temp_db']
                    # create temporary db with copy of table
                    h5move_tables(
                        {"temp_db_path": cfg_out["db_path"], "db_path": cfg_out["temp_db_path"]},
                        tbl_names=list(nonm_tbl_set),
                    )

                h5move_tables({**cfg_out, "recreate_index_tables_set": nonm_tbl_set}, tbl_names=tables)
        else:
            lf.info(f'{"other" if dup_tbl_set else "all"} tables monotonic.{"" if dup_tbl_set else " Ok>"}')

        # if b_need_save:
        #     # out to store
        #     cfg_out['db_path'], cfg_out['temp_db_path'] = cfg_out['temp_db_path'], cfg_out['temp_db_path']
        #     h5move_tables(cfg_out, tbl_names=tables)
        #     cfg_out['db_path'], cfg_out['temp_db_path'] = cfg_out['temp_db_path'], cfg_out['temp_db_path']
        #     h5move_tables(cfg_out, tbl_names=tables)

        # store = pd.HDFStore(cfg_out['temp_db_path'])
        # store.create_table_index(tbl, columns=['index'], kind='full')
        # store.create_table_index(cfg_out['tables_log'][0], columns=['index'], kind='full') #tbl+r'/logFiles'
        # h5_append(store, df, log, cfg_out, cfg_out['dt_from_utc'])
        # store.close()
        # h5sort_pack(cfg_out['temp_db_path'], out_storage_name, tbl) #, ['--overwrite-nodes=true']


def h5_rem_rows(db, tbl_names, qstr, qstr_log):
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


def h5_rem_last_rows(db, tbl_names, df_logs: List[pd.DataFrame], t_start=None):
    """

    :param db:
    :param tbl_names:
    :param df_logs: list of logs DataFrames of length >= (number of items in tbl_names) - len(tbl_names)
    :param t_start: datetime or None. If None then do nothing
    :return:
    """
    if t_start is None:
        return
    h5_rem_rows(db, tbl_names, qstr="index>='{}'".format(t_start), qstr_log="DateEnd>='{}'".format(t_start))
    # place back removed 1st log row updated for remaining data index
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
        df_log_append_fun(df_log_cur, tbl, {"db": db, "nfiles": None})


def h5del_obsolete(
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
     - Else if not "index" then del. log records with field_to_del_all_older_records >= current file have, and data
      having time > its 1st index.
     - If "index" then del. log records with "DateEnd" > current file time data start, but last record wrote back
      with changed "DateEnd" to be consistent with last data time.

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
                # h5remove_tables(db, tables, tables_log, temp_db_path=None)
                # return b_stored_newer, b_stored_dups
            df_log_cur = df_log
        else:  # not tested
            b_cur = df_log[field_to_del_older_records] >= log[field_to_del_older_records]
            df_log_cur = df_log[b_cur]
            if not df_log_cur.empty:
                t_start = df_log_cur.index[0]
        # removing
        h5_rem_last_rows(cfg_out["db"], zip(cfg_out["tables"], cfg_out["tables_log"]), [df_log_cur], t_start)
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
                if n_log_rows and not h5_rem_rows(
                    cfg_out["db"],
                    zip(cfg_out["tables"], cfg_out["tables_log"]),
                    qstr="index>='{}'".format(i0 := df_log_cur.index[0]),
                    qstr_log="fileName=='{}'".format(df_log_cur.loc[i0, "fileName"]),
                ):
                    b_stored_newer = False  # deleted
                    b_stored_dups = False  # deleted
    return b_stored_newer, b_stored_dups


# Functions to iterate rows of db log instead of files in dir


def query_time_range(min_time=None, max_time=None, **kwargs) -> str:
    """
    Query Time for pandas.Dataframe
    :param min_time:
    :param max_time:
    :return:
    """
    if min_time:
        query_range = f"index>='{min_time}' & index<='{max_time}'" if max_time else f"index>='{min_time}'"
    elif max_time:
        query_range = f"index<='{max_time}'"
    else:
        query_range = None
    return query_range


def h5log_rows_gen(
    db_path: Union[str, Path, None] = None,
    table_log: str = "log",
    min_time=None,
    max_time=None,
    query_range=None,
    db: Optional[pd.HDFStore] = None,
    **kwargs,
) -> Iterator[Dict[str, Any]]:
    """
    Dicts from each h5 log row
    :param db_path: name of hdf5 pandas store where is log table, used only for message if it is set and db is set
    :param db: handle of already open pandas hdf5 store
    :param table_log: name of log table - table with columns for intervals:
      - index - starts, pd.DatetimeIndex
      - DateEnd - ends, pd.Datetime
    :param min_time, max_time: datetime, optional, allows limit the range of table_log rows, not used if query_range is set
    :param query_range: query str to limit the range of table_log rows to load
        Example table_log name: cfg_in['table_log'] ='/CTD_SST_48M/logRuns'
    :param kwargs: not used
    Yields dicts where keys: col names, values: current row values of tbl_intervals = cfg_in['table_log']
    """
    if query_range is None:
        query_range = query_time_range(min_time, max_time)
    with FakeContextIfOpen(lambda f: pd.HDFStore(f, mode="r"), file=db_path, opened_file_object=db) as store:
        # with pd.HDFStore(db_path, mode='r') as store:
        if db_path:
            print(f'loading from "{db_path}": ', end="")
        for n, rp in enumerate(store.select(table_log, where=query_range).itertuples()):
            r = dict(zip(rp._fields, rp))
            yield r  # r.Index, r.DateEnd


def h5log_names_gen(
    cfg_in: Mapping, f_row_to_name=lambda r: "{Index:%y%m%d_%H%M}-{DateEnd:%H%M}".format_map(r)
) -> Iterator[Any]:
    """
    Generates outputs of f_row_to_name function which receives dicts from each h5 log row (see h5log_rows_gen)
    :param cfg_in: keyword arguments for h5log_rows_gen()
    :param f_row_to_name: function(dict) where dict have fields from h5 log row
    :return: iterator, by default - of strings, suitable to name files by start-end date/time

    :Modifies cfg_in: adds/replaces field cfg_in['log_row'] = h5log_rows_gen(cfg_in) result before each yielding
    Replacing for veuszPropagate.ge_names() to use tables instead files
    """
    for row in h5log_rows_gen(**cfg_in):
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


def h5_close(cfg_out: Mapping[str, Any]) -> None:
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
            tbl_dups = h5remove_duplicates_by_loading(cfg_out, cfg_table_keys=cfg_table_keys)
            # or h5remove_duplicates() but it can take very long time
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


def h5_names_gen(cfg_in, cfg_out: Mapping[str, Any], check_have_new_data=True, **kwargs) -> Iterator[Path]:
    """
    Yields Paths from cfg_in['paths'] items
    :updates: cfg_out['log'] fields 'fileName' and 'fileChangeTime'

    :param cfg_in: dict, must have fields:
        - paths: iterator - returns full file names
    :param cfg_out: dict, with fields needed for h5_dispenser_and_names_gen() and print info:
        - log: current file info with fields that should be updated before each yield:
            - Date0, DateEnd, rows: if no Date0, then prints "file not processed"
    :param check_have_new_data: bool, if False then do not check Date0 presence and print "file not processed"

    """
    set_field_if_no(cfg_out, "log", {})
    for name_full in cfg_in["paths"]:
        pname = Path(name_full)

        cfg_out["log"]["fileName"] = f"{pname.parent.name}/{pname.stem}"[-cfg_out["logfield_fileName_len"] :]
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


def h5out_init(cfg_in: Mapping[str, Any], cfg_out: MutableMapping[str, Any]) -> None:
    """
    Init cfg_out database (hdf5 data store) information in cfg_out _if it is not exist_
    :param: cfg_in - configuration dicts, with fields:
    - path: if no 'db_path' in cfg_out, or it is not absolute
    - cfgFile - if no `cfg_out['b_insert_separator']` defined or determine the table name is failed - to extract from cfgFile name
    - raw_dir_words: (optional), default: ['source', 'WorkData', 'workData'] - see getDirBaseOut()
    - nfiles: (optional)
    - b_incremental_update: (optional) to copy it to cfg_out
    :param: cfg_out - configuration dict, with optional fields. If `cfg_out['tables']` is None then function
    returns (does nothing)

    Sets or updates fields of `cfg_out`:
    % paths %
    - db_path: absolute path of hdf5 store with suffix ".h5"
    - tables, tables_log: tables names of data and log (metadata) - based on cfg_in and cfg_in['raw_dir_words']
    - temp_db_path: temporary h5 file name
    % other %
    - nfiles: default 1, copied from cfg_in - I use it somewhere to set store.append() 'expectedrows' argument
    - b_incremental_update: default False, copied from cfg_in
    - chunksize: default None
    - logfield_fileName_len: default 255
    - b_remove_duplicates: default False
    - b_reuse_temporary_tables: default False

    :return: None
    """
    if "tables" in cfg_out and cfg_out["tables"] is None:
        return
    set_field_if_no(cfg_out, "logfield_fileName_len", 255)
    set_field_if_no(cfg_out, "chunksize")
    set_field_if_no(
        cfg_out,
        "b_incremental_update",
        cfg_in["b_incremental_update"] if "b_incremental_update" in cfg_in else False,
    )
    set_field_if_no(cfg_out, "b_remove_duplicates", False)
    set_field_if_no(cfg_out, "b_reuse_temporary_tables", False)
    if cfg_out.get("b_insert_separator") is None:
        if "cfgFile" in cfg_in:
            cfg_file = PurePath(cfg_in["cfgFile"]).stem
            cfg_out["b_insert_separator"] = "_ctd_" in cfg_file.lower()
        # else:
        #     cfg_out['b_insert_separator'] = False

    # Automatic db file and tables names
    if not (cfg_out.get("db_path") and cfg_out["db_path"].is_absolute()):
        path_in = Path(cfg_in.get("path" if "path" in cfg_in else "db_path")).parent
        cfg_out["db_path"] = path_in / (
            f"{path_in.stem}_out" if not cfg_out.get("db_path") else cfg_out["db_path"]
        )
    dir_create_if_need(cfg_out["db_path"].parent)
    cfg_out["db_path"] = cfg_out["db_path"].with_suffix(".h5")

    # temporary file path
    set_field_if_no(
        cfg_out, "temp_db_path", cfg_out["db_path"].with_name(f"{cfg_out['db_path'].stem}_not_sorted.h5")
    )

    set_field_if_no(cfg_out, "nfiles", cfg_in.get("nfiles", 1))

    if "tables" in cfg_out and cfg_out["tables"]:
        set_field_if_no(
            cfg_out, "tables_log", [((f"{tab}/logFiles") if tab else "") for tab in cfg_out["tables"]]
        )
    elif "table" in cfg_out and cfg_out["table"]:
        cfg_out["tables"] = [cfg_out["table"]]
        set_field_if_no(cfg_out, "tables_log", [f"{cfg_out['table']}/logFiles"])
    else:
        table_auto = cfg_in.get("table")
        if not table_auto:
            if cfg_in.get("tables") and len(cfg_in.get("tables")) == 1:
                cfg_out["tables"] = cfg_in["tables"]
            else:
                if not table_auto:
                    _, _, table_auto = getDirBaseOut(
                        cfg_out["db_path"],
                        cfg_in.get("raw_dir_words")
                        or ["raw", "_raw", "source", "_source", "WorkData", "workData"],
                    )
                if not table_auto:
                    table_auto = Path(cfg_in["cfgFile"]).stem
                    lf.warning(
                        "Can not dertermine table_name from file structure. "
                        'Set [tables] in ini! Now use table_name "{:s}"',
                        table_auto,
                    )
                cfg_out["tables"] = [table_auto]
        set_field_if_no(cfg_out, "tables_log", [f"{table_auto}/logFiles"])


def h5temp_open(
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
    Checks and generates some names used for saving data to pandas *.h5 files with log table.
    Opens temporary HDF5 store, by default, copies existed target store `db_path` data to it (to work with
    later).
    Temporary HDF5 store needed because of using ptprepack to create index and sort all data at last step
    is faster than support indexing during data appending.

    Parameters are fields that is set when called `h5out_init(cfg_in, cfg_out)`:
    :param: db_path: If None and `tables` is None then the function returns without doing anything
    :param: temp_db_path: path to temporary HDF5 store
    :param: tables, tables_log - if `tables` is None then returns (does nothing), else opens HDF5 store and
    tries to work with ``tables_log``
    :param: b_incremental_update:
    :param: b_reuse_temporary_tables: bool, default False - not copy existed tables from `db_path` to
    `temp_db_path` initially
    :param: b_overwrite: remove all existed data in tables where going to write
    :param: db_in: handle of opened db_path (if opened) else None. If None and `tables` is None then the
    function returns without doing anything
    :param: kwargs: optionals parameters:
    - `h5out_init` `cfg_in` input parameter fields, used if `temp_db_path` is None to get it and other input
    parameters. That will ovewrite parameters of this function:
        - path: if no 'db_path' in cfg_out, or it is not absolute
        - cfgFile - if no `cfg_out['b_insert_separator']` defined or determine the table name is failed - to extract from cfgFile name

    - b_allow_use_opened_temp_db:
    :return: (df_log, db, b_incremental_update)
        - df_log: dataframe of log from store if `b_incremental_update` is True else None.
        - db: pandas HDF5 store - opened `temp_db_path`
        - b_incremental_update: flag remains same as it was inputted or changed to False if can not skip or b_overwrite is True
    Modifies (creates): db - handle of opened pandas HDF5 store
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
        h5out_init({"b_incremental_update": b_incremental_update, "db_path": db_path, **kwargs}, cfg_out)
        temp_db_path = cfg_out["temp_db_path"]

    print("saving to", temp_db_path / ",".join(tables).strip("/"), end=":\n")

    if tables:  # if table name in tables_log has placeholder then fill it
        tables_log = [t.format(tables[i if i < len(tables) else 0]) for i, t in enumerate(tables_log)]

    # Side effect code removed:
    #  itbl = 0
    # for i, t in enumerate(tables_log):
    #     if '{}' in t:
    #         tables_log[i] = t.format(tables[itbl])
    #     else:
    #         itbl += 1

    try:
        try:  # open temporary output file
            if temp_db_path.is_file():
                if not b_reuse_temporary_tables:  # Remove existed tables to write
                    with FakeContextIfOpen(pd.HDFStore, temp_db_path, db) as db1:
                        # pd.HDFStore(temp_db_path) as db1:
                        tables_in_root = [
                            t
                            for tbl in tables
                            for t in h5find_tables(db1, tbl.format(".*").replace(".*.*", ".*"))
                            if tbl
                        ]
                        h5remove_tables(db1, tables_in_root, tables_log)
        except IOError as e:
            print(e)

        if not b_overwrite:
            if not b_reuse_temporary_tables:
                # Copying previous store data to temporary one
                lf.info("Copying previous store data to temporary one:")
                tbl = "is absent"
                try:
                    # Open output store (as input for this function)
                    with FakeContextIfOpen(
                        lambda path: pd.HDFStore(path, mode="r"), db_in or db_path
                    ) as db_in:
                        tbl_prev = "?"  # Warning side effect: ignores 1st table if its name starts with '?'
                        for tbl in sorted(tables + tables_log):
                            if (  # parent of this nested have moved on previous iteration
                                (
                                    len(tbl_prev) < len(tbl)
                                    and tbl.startswith(tbl_prev)
                                    and tbl[len(tbl_prev)] == "/"
                                )
                                or not tbl
                            ):
                                continue
                            try:  # Check output store
                                if tbl in db_in:  # avoid harmful sortAndPack errors
                                    h5sort_pack(db_path, temp_db_path.name, tbl, arguments="fast")
                                    # , addargs=['--verbose']  may be copynode?
                                else:
                                    lf.info(f"Table {tbl} does not exist")
                                    continue
                                    # raise HDF5ExtError(f'Table {tbl} does not exist')
                            except HDF5ExtError as e:
                                if tbl in db_in.root.__members__:
                                    print("Node exist but store is not conforms Pandas")
                                    getstore_and_print_table(db_in, tbl)
                                raise e  # exclude next processing
                            except RuntimeError as e:
                                lf.error(
                                    "Failed copy from output store (RuntimeError). "
                                    "May be need first to add full index to original store? Trying: "
                                )
                                nodes = db_in.get_node(tbl).__members__  # sorted(, key=number_key)
                                # reopen for modifcation
                                db_in.close()
                                db_in.open("r")  # = pd.HDFStore(db_in.filename)
                                for n in nodes:
                                    tbl_cur = tbl if n == "table" else f"{tbl}/{n}"
                                    lf.info(tbl_cur, end=", ")
                                    db_in.create_table_index(tbl_cur, columns=["index"], kind="full")
                                # db_in.flush()
                                lf.error("Trying again")
                                if b_we_close := ((db_in is not None) and db_in.is_open):
                                    db_in.close()
                                h5sort_pack(db_path, temp_db_path.name, tbl)
                                if b_we_close:
                                    db_in.open("r")

                            tbl_prev = tbl

                except HDF5ExtError as e:
                    lf.warning(
                        e.args[0]
                    )  # print('processing all source data... - no table with previous data')
                    b_incremental_update = False
                except Exception as e:
                    print(
                        "processing all source data... - no previous data (output table {}): {}".format(
                            tbl, "\n==> ".join([s for s in e.args if isinstance(s, str)])
                        )
                    )
                    b_incremental_update = False
                else:
                    if b_incremental_update:
                        lf.info("Will append data only from new files.")

        if (db is None) or not db.is_open:
            # Open temporary output file to return
            for attempt in range(2):
                try:
                    db = pd.HDFStore(temp_db_path)
                    # db.flush()
                    break
                except IOError as e:
                    print(e)
                except HDF5ExtError as e:  #
                    print("can not use old temporary output file. Deleting it...")
                    os_remove(temp_db_path)
                    # raise(e)

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
                        h5remove(db, tbl_log)
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

    except HDF5ExtError as e:
        if db:
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


def h5_dispenser_and_names_gen(
    cfg_in: Mapping[str, Any],
    cfg_out: Optional[MutableMapping[str, Any]] = None,
    fun_gen: Callable[[Mapping[str, Any], Mapping[str, Any], Any], Iterator[Any]] = h5_names_gen,
    b_close_at_end: Optional[bool] = True,
    **kwargs,
) -> Iterator[Tuple[int, Any]]:
    """
    Prepares HDF5 store to insert/update data and yields fun_gen(...) outputs:
        - Opens DB for writing (see h5temp_open() requirements)
        - Finds data labels by fun_gen(): default are file names and their modification date
        - Removes outdated labels in log table and data part in data table they points to
        - Generates fun_gen() output (if b_incremental_update, only data labels which is absent in DB (to upload new/updated data))
        - Tide up DB: creates index, closes DB.
    This function supports storing data in HDF5 used in h5toGrid: dataframe's child 'table' node always contain adjacent
    "log" node. "log" dataframe labels parent dataframe's data segments and allows to check it for existence and
    relevance.

    :param cfg_in: dict, must have fields:
        - fields used in your `fun_gen(cfg_in, cfg_out)`
    :param cfg_out: dict, must have fields:
        - log: dict, with info about current data, must have fields for compare:
            - 'fileName' - in format as in log table to be able to find duplicates
            - 'fileChangeTime', datetime - to be able to find outdated data
        - b_incremental_update: if True then not yields previously processed files. But if file was changed
          then: 1. removes stored data and 2. yields `fun_gen(...)` result
        - tables_written: sequence of table names where to create index
        - temp_db_path
        ... - see `h5temp_open()` parameters that is can be obtained by calling `h5out_init(cfg_in, cfg_out)`
    :param fun_gen: function with arguments `(cfg_in, cfg_out, **kwargs)`, that
        - generates data labels, default are file's ``Path``s,
        - updates `cfg_out['log']` fields 'fileName' (by current label) and 'fileChangeTime' needed to store and find
        data. They named historically, in principle, you can use any unique identifier composed of this two fields.
    :param b_close_at_end: if True (default) then closes store after generator exhausted
    :param kwargs: 3rd and other parameters of `fun_gen()`
    :return: Iterator that returns (i1, pname):
        - i1: index (starting with 1) of fun_gen generated data label (maybe file)
        - pname: fun_gen output (may be path name)
        Skips (i1, pname) for existed labels that also has same stored data label (file) modification date
    :updates:
        - cfg_out['db'],
        - cfg_out['b_remove_duplicates'] and
        - that what `fun_gen()` do
    """
    # copy data to temporary HDF5 store and open it
    df_log_old, cfg_out["db"], cfg_out["b_incremental_update"] = h5temp_open(
        **cfg_out, db_in=cfg_in.get("db"), b_allow_use_opened_temp_db=not b_close_at_end
    )
    try:
        for i1, gen_out in enumerate(fun_gen(cfg_in, cfg_out, **kwargs), start=1):
            # if current file is newer than its stored data then remove data and yield its info to process
            # again
            if cfg_out["b_incremental_update"]:
                b_stored_newer, b_stored_dups = h5del_obsolete(
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
            h5_close(cfg_out)
