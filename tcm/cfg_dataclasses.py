# !/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Help manage hydra (OmegaConf) custom configurations
  Created: 14.09.2020
  Modified: 20.09.2024
"""

import os
import sys

from dataclasses import dataclass, Field, field, fields, is_dataclass, make_dataclass, MISSING as dataclasses_missing
from datetime import date as datetime_date

# from collections.abc import Mapping
from datetime import datetime, timedelta
from types import ModuleType
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from hydra.errors import HydraException
from omegaconf import (
    MISSING,  # Do not confuse with dataclass.MISSING
    MissingMandatoryValue,
    OmegaConf,
    open_dict,
)

from .utils2init import (
    Ex_nothing_done,
    LoggingStyleAdapter,
    ini2dict,
    init_file_names,
    standard_error_info,
    this_prog_basename,
)

lf = LoggingStyleAdapter(__name__)

@dataclass
class ConfigInCsv:
    """
    "in": all about input files
    Constructor arguments:
    :param path: path to source file(s) to parse. Use patterns in Unix shell style
    :param b_search_in_subdirs: search in subdirectories, used if mask or only dir in path (not full path)
    :param exclude_dirs_endswith_list: exclude dirs which ends with this strings. This and next option especially useful when search recursively in many dirs
    :param exclude_files_endswith_list: exclude files which ends with this strings
    :param b_incremental_update: exclude processing of files with same name and which time change is not bigger than recorded in database (only prints ">" if detected). If finds updated version of same file then deletes all data which corresponds old file and after it before processing of next files
    :param dt_from_utc_seconds: add this correction to loading datetime data. Can use other suffixes instead of "seconds"
    :param dt_from_utc_hours: add this correction to loading datetime data. Can use other suffixes instead of "hours"
    :param fs_float: sampling frequency, uses this value to calculate intermediate time values between time changed values (if same time is assigned to consecutive data)
    :param fs_old_method_float: sampling frequency, same as ``fs_float``, but courses the program to use other method. If smaller than mean data frequency then part of data can be deleted!(?)
    :param header: comma separated list matched to input data columns to name variables. Can contain type suffix i.e.
     (float) - which is default, (text) - also to convert by specific converter, or (time) - for ISO format only
    :param cols_load_list: comma separated list of names from header to be saved in hdf5 store. Do not use "/" char, or type suffixes like in ``header`` for them. Default - all columns
    :param cols_not_save_list: comma separated list of names from header to not be saved in hdf5 store
    :param skiprows_integer: skip rows from top. Use 1 to skip one line of header
    :param on_bad_lines: {{'error', 'warn', 'skip'}}, default 'error'. May be better to set "comment" argument to tune format?
    :param delimiter_chars: parameter of pandas.read_csv()
    :param max_text_width: maximum length of text fields (specified by "(text)" in header) for dtype in numpy loadtxt
    :param chunksize_percent_float: percent of 1st file length to set up hdf5 store table chunk size
    :param blocksize_int: bytes, chunk size for loading and processing csv
    :param corr_time_mode: if there is time that is not increased then modify time values trying to affect small number of values. This is different from sorting rows which is performed at last step after the checking table in database
    :param fun_date_from_filename: function(file_stem: str, century: Optional[str]=None) -> Any[compartible to input of pandas.to_datetime()]: to get date from filename to time column in it.

    :param csv_specific_param_dict: not default parameters for function in csv_specific_proc.py used to load data
    """
    path: Any = 'hdf5_alt'
    b_search_in_subdirs: bool = False
    exclude_dirs_endswith: Tuple[str] = ('toDel', '-', 'bad', 'test', 'TEST')
    exclude_files_endswith: Tuple[str] = ('coef.txt', '-.txt', 'test.txt')
    b_incremental_update: bool = True
    dt_from_utc_hours = 0

    skiprows = 1
    on_bad_lines = 'error'
    max_text_width = 1000
    blocksize_int = 20000000
    corr_time_mode = True
    dir: Optional[str] = MISSING
    ext: Optional[str] = MISSING
    filemask: Optional[str] = MISSING

    paths: Optional[List[Any]] = field(default_factory=list)
    nfiles: Optional[int] = 0  # field(default=MISSING, init=False)
    raw_dir_words: Optional[List[str]] = field(default_factory= lambda: ['raw', 'source', 'WorkData', 'workData'])



@dataclass
class ConfigInHdf5_Simple:
    """
    "in": all about input files:
    :param db_path: path to pytables hdf5 store to load data. May use patterns in Unix shell style
             default='.'
    :param tables_list: table name in hdf5 store to read data. If not specified then will be generated on base of path of input files
    :param tables_log: table name in hdf5 store to read data intervals. If not specified then will be "{}/logFiles" where {} will be replaced by current data table name
    :param dt_from_utc_hours: add this correction to loading datetime data. Can use other suffixes instead of "hours",
            default='0'
    :param b_incremental_update: exclude processing of files with same name and which time change is not bigger than recorded in database (only prints ">" if detected). If finds updated version of same file then deletes all data which corresponds old file and after it before processing of next files.
            default='True'
    """
    db_path: Optional[str] = None  # if None load from
    tables: List[str] = field(default_factory=lambda: ['.*'])  # field(default_factory=list)
    tables_log: List[str] = field(default_factory=list)
    dt_from_utc_hours = 0
    b_incremental_update: bool = True


@dataclass
class ConfigInHdf5(ConfigInHdf5_Simple):
    """
    Same as ConfigInHdf5_Simple + specific (CTD and navigation) data properties:
    :param table_nav: table name in hdf5 store to add data from it to log table when in "find runs" mode. Use empty string to not add
            default='navigation'
    :param b_temp_on_its90: When calc CTD parameters treat Temp have red on ITS-90 scale. (i.e. same as "temp90"),
            default='True'

    """

    query: Optional[str] = None
    table_nav: Optional[str] = 'navigation'
    b_temp_on_its90: bool = True

    # path_coef: Optional[str] = MISSING  # path to file with coefficients for processing of Neil Brown CTD data


@dataclass
class ConfigOutSimple:
    """
    "out": all about output files:

    :param db_path: hdf5 store file path
    :param tables: tables names in hdf5 store to write data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in autonumbered locations (see dask to_hdf())
    :param b_insert_separator: insert NaNs row in table after each file data end
    :param b_reuse_temporary_tables: Warning! Set True only if temporary storage already have good data! If True and b_incremental_update= True then not replace temporary storage with current storage before adding data to the temporary storage
    :param b_remove_duplicates: Set True if you see warnings about
    :param chunksize: limit loading data in memory
    """
    db_path: Any = ''
    tables: List[str] = field(default_factory=list)
    tables_log: List[str] = field(default_factory=lambda: ['{}/log'])
    b_reuse_temporary_tables: bool = False
    b_remove_duplicates: bool = False
    b_incremental_update: bool = True  # todo: link to ConfigIn
    temp_db_path: Any = None
    b_overwrite: Optional[bool] = False
    db: Optional[Any] = None  # False?
    logfield_fileName_len: Optional[int] = 255
    chunksize: Optional[int] = None
    nfiles: Optional[int] = None


@dataclass
class ConfigOut(ConfigOutSimple):
    """
    "out": all about output files:

    :param table: table name in hdf5 store to write data. If not specified then will be generated on base of path of input files. Note: "*" is used to write blocks in autonumbered locations (see dask to_hdf())
    :param b_insert_separator: insert NaNs row in table after each file data end
    """
    table: str = 'navigation'
    tables_log: List[str] = field(default_factory=list)  # overwritten parent
    b_insert_separator: bool = True


@dataclass
class ConfigOutCsv:
    cols: Optional[Dict[str, str]] = field(default_factory=dict)
    cols_log: Optional[Dict[str, str]] = field(default_factory=dict)  # Dict[str, str] =
    text_path: Optional[str] = None
    text_date_format: Optional[str] = None
    text_float_format: Optional[str] = None
    file_name_fun: str = ''
    file_name_fun_log: str = ''
    col_station_fun: str = ''
    sep: str = '\t'

ParamsCTD = make_dataclass('ParamsCTD', [  # 'Pres, Temp90, Cond, Sal, O2, O2ppm, Lat, Lon, SA, sigma0, depth, soundV'
    (p, Optional[float], MISSING) for p in 'Pres Temp Cond Sal SigmaTh O2sat O2ppm soundV'.split()
    ])

ParamsNav = make_dataclass('ParamsNav', [
    (p, Optional[float], MISSING) for p in 'Lat Lon DepEcho'.split()
    ] + [('date', Optional[Any], MISSING)])

@dataclass
class ConfigFilterCTD:
    """
    "filter": filter all data based on min/max of parameters:

    :param min_dict: List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is below ``value``'). To filter time use ``date`` key
    :param max_dict: List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is above ``value``'). To filter time use ``date`` key
    :param b_bad_cols_in_file_name: find string "<Separator>no_<col1>[,<col2>]..." in file name. Here <Separator> is one of -_()[, and set all values of col1[, col2] to NaN
    """
    #Optional[Dict[str, float]] = field(default_factory= dict) leads to .ConfigAttributeError/ConfigKeyError: Key 'Sal' is not in struct
    min: Optional[ParamsCTD] = field(default_factory=ParamsCTD)
    max: Optional[ParamsCTD] = field(default_factory=ParamsCTD)
    b_bad_cols_in_file_name: bool = False
    corr_time_mode: Any = 'delete_inversions'  # , 'False',  'correct', 'sort_rows'

@dataclass
class ConfigFilterNav:
    """
    "filter": filter all data based on min/max of parameters:

    :param min_dict: List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is below ``value``'). To filter time use ``date`` key
    :param max_dict: List with items in  "key:value" format. Sets to NaN data of ``key`` columns if it is above ``value``'). To filter time use ``date`` key
    :param b_bad_cols_in_file_name: find string "<Separator>no_<col1>[,<col2>]..." in file name. Here <Separator> is one of -_()[, and set all values of col1[, col2] to NaN
    """
    #Optional[Dict[str, float]] = field(default_factory= dict) leads to .ConfigAttributeError/ConfigKeyError: Key 'Sal' is not in struct
    min: Optional[ParamsNav] = field(default_factory=ParamsNav)
    max: Optional[ParamsNav] = field(default_factory=ParamsNav)
    b_bad_cols_in_file_name: bool = False
    corr_time_mode: Any = 'delete_inversions'

@dataclass
class ConfigProgram:
    """

    "program": program behavior:

    :param return_: one_of('<cfg_from_args>', '<gen_names_and_log>', '<end>')
        <cfg_from_args>: returns cfg based on input args only and exit,
        <gen_names_and_log>: execute init_input_cols() and also returns fun_proc_loaded function... - see main()
    :param log_,
    :param verbose_: one_of('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'),
    """

    return_: str = '<end>'
    b_interact: bool = False
    log: str = ''
    verbose: str = 'INFO'


def camel2snake(text, sep='_'):
    """
    CamelCase to snake_case: inserts delimiter only if upper letter going after lower.
    Note: many characters have no upper or lower case.
    :param text: string to reformat  or any text
    :param sep: delimiter, default: '_'
    :return:
    """
    text_lower = text.lower()
    b_add_delimiter = False  # initial state: no need to add delimiter before 1st character


    def need_delimiter_check_saving_state(char, char_low):
        """
        :param char_low: `char` in low case to speedup detecting that `char` is not in low case
        """
        nonlocal b_add_delimiter
        add_delimiter_now_possible, b_add_delimiter = b_add_delimiter, char.islower()
        return add_delimiter_now_possible and char != char_low

    return ''.join(
        f'{sep}{l}' if need_delimiter_check_saving_state(c, l) else l for c, l in zip(text, text_lower))


def timedelta_to_iso8601(timedelta_obj):
    total_seconds = int(timedelta_obj.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"PT{hours}H{minutes}M{seconds}S"


def to_omegaconf_compatible_types(value):
    if value is None or isinstance(value, (int, float, str, bool)):
        return value    # No conversion needed for supported types
    elif isinstance(value, dict):
        # Recursively convert sub-dictionaries
        return {k: to_omegaconf_compatible_types(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        # Recursively convert elements in lists and tuples
        return [to_omegaconf_compatible_types(item) for item in value]
    elif isinstance(value, (datetime_date, datetime)):
        # Convert datetime objects to strings
        return value.isoformat()
    elif isinstance(value, timedelta):
        return timedelta_to_iso8601(value)
        # pd.tseries.frequencies.to_offset(value).freqstr
    elif str(type(value)).find("numpy.ndarray") != -1:  # isinstance(value, ndarray):
        return value.tolist() if value.size > 1 else value.item()
    else:
        # Convert unsupported types to strings
        # (or raise TypeError if you want to be strict)
        return str(value)


def is_optional_type(field_type) -> bool:
    """Check if a field is of type Optional (Union[X, None])
    alternative: str(field_type).startswith("typing.Optional")
    """
    return get_origin(field_type) is Union and type(None) in get_args(field_type)


def prune_config(mapping: Mapping[str, Any], schema: Type) -> Dict[str, Any]:
    """

    :param mapping: dict to prune to structured configuration schema
    :param schema: structured configuration schema
    :return: pruned dict
    """
    if not is_dataclass(schema):
        return mapping

    # Get type hints for all the fields in the structured schema
    type_hints = get_type_hints(schema)

    pruned = {}
    for field_info in fields(schema):
        field_name = field_info.name
        field_type = type_hints[field_name]
        is_optional = is_optional_type(field_type)

        # Ignore fields that are not in mapping
        if field_name not in mapping:  # and is_optional
            continue

        if field_name in mapping:
            if is_dataclass(field_type) or (is_optional and is_dataclass(field_type.__args__[0])):
                # If the field is optional, get the actual dataclass type from __args__
                sub_structure = field_type.__args__[0] if is_optional else field_type
                pruned[field_name] = prune_config(mapping[field_name], sub_structure)
            else:
                # If the field is not a dataclass, just copy the value
                pruned[field_name] = mapping[field_name]
        # elif not is_optional:
        #     # If a non-optional field is missing, set it to None (or you can raise an error)
        #     pruned[field_name] = None

    return pruned


def get_generic_and_actual_type(type_hint):
    """
    Unwraps Annotated and Optional types until it founds generic type or when there's no origin.
    :return: the first generic type found (else None) along with its immediate type argument
    """
    # Track the base generic type
    generic_type = None

    # Iterate until there are no more type args left
    while True:
        # Get the origin and args of the current type hint
        generic_type = get_origin(type_hint)
        if generic_type is None:
            break  # No generics, exit the loop

        type_args = get_args(type_hint)
        if generic_type == Union:
            # Special handling for Union (Optional): Find the first argument that is not NoneType
            type_hint = next((arg for arg in type_args if arg != type(None)), None)
            continue
        if generic_type == Annotated:
            type_hint = type_args[0]
            continue
        type_hint = get_args(type_hint)
        break
    return generic_type, type_hint


def to_omegaconf_compatible_type(value, field_type, default_value=None):
    """
    Convert value to OmegaConf compatible type and handle optional nested structured configs.
    This function can handle the following types:
    - Optional types (defined as Union[T, None])
    - Dataclasses (recursively converts their fields)
    - Annotated types (extracts the base type)
    - Collections including list, tuple, and dict (recursively converts their elements)
    - Common types such as int, float, str, and bool
    - Numpy arrays and numpy scalar types (converted to Python lists / scalars reducing dimension if need)
    - Date, datetime, and timedelta: converted to ISO format strings or timedelta to int seconds
    If the value can't be converted to the specified field type, raises a TypeError.
    If the converted value is equal to the default value, returns None.
    :param value: The value to be converted.
    :param field_type: The target type to convert the value to.
    :param default_value: The default value of the field. Used to determine whether to exclude the field.
    :return: The converted value.
    """

    if value is None:
        return None

    origin_type, field_type = get_generic_and_actual_type(field_type)

    if origin_type:
        if origin_type in (list, tuple, dict):
            if isinstance(value, dict):
                # include elements that are differs from the default and None
                converted_dict = {}
                _, value_type = field_type
                for k, v in value.items():
                    # k = convert_value(k, key_type, schema)
                    v = to_omegaconf_compatible_type(
                        v, value_type, default_value.get(k) if
                        default_value not in (dataclasses_missing, None) else None
                        )  # For dict, convert each value.
                    if v is None:
                        continue
                    converted_dict[k] = v
                return converted_dict

            else:  # for list, tuple types we use its base type  # convert each element.
                if hasattr(value, 'dtype'):
                    value = value.tolist()
                return [to_omegaconf_compatible_type(item, field_type[0]) for item in value]

    # If the field type is a dataclass, we recurse
    if is_dataclass(field_type):
        return to_omegaconf_merge_compatible(value, field_type)

    # squeeze array element of bigger dimensions to schema
    if field_type in (int, float, str, bool):
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        elif isinstance(value, timedelta):
            return int(value.total_seconds())

    if hasattr(value, "dtype"):
        value = value.item()

    # If the value is already the correct type, we return it as-is
    if field_type is not Any and isinstance(value, field_type):
        return value

    # Only conversion to strings are not handled yet, so `field_type` must be compatible to str:
    if field_type not in (Any, str):
        raise TypeError(f"Can not convert {value} to {field_type}")

    if isinstance(value, (datetime_date, datetime)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return timedelta_to_iso8601(value)
    return str(value)


def get_field_default(fld: Field):
    return (
        fld.default
        if fld.default is not dataclasses_missing
        else None
        if fld.default_factory is dataclasses_missing
        else fld.default_factory()
    )


def to_omegaconf_merge_compatible(unstructured: Dict[str, Any], schema) -> Dict[str, Any]:
    """
    Converts an unstructured dictionary to one that is compatible with OmegaConf creation and merging.
    Excludes None fields (keeps None elements in lists) and fields equal to default values in schema
    and properly handles optional nested structured configs and common numpy types.
    :param unstructured: The dictionary to be converted and pruned.
    :param schema: The dataclass schema to conform to.
    :return: A dictionary compatible with OmegaConf.
    Usage:
    conf = to_omegaconf_merge_compatible(my_not_structured_dict, MyDataClassSchema)
    conf_full = OmegaConf.merge(OmegaConf.create(conf), existing_conf)
    """
    if not is_dataclass(schema):
        raise TypeError("Provided schema is not a dataclass.")
    schema_fields = {fld.name: (fld.type, get_field_default(fld)) for fld in fields(schema)}
    converted = {}
    for key, value in unstructured.items():
        if key in schema_fields:
            if value is not None:
                converted_value = to_omegaconf_compatible_type(value, *schema_fields[key])
                if converted_value != schema_fields[key][1]:
                    converted[key] = converted_value
        else:
            lf.warning('Field "{}" is not in {}: removed', key, schema.__name__)
    return converted


def hydra_cfg_store(
        cs_store_name: str,
        cs_store_group_options: Mapping[str, Sequence[Any]],
        module=sys.modules[__name__]  # hdf5_alt.cfg_dataclasses
        ) -> Tuple[ConfigStore, object]:
    """
    Registering Structured config with defaults specified by dataclasses in ConfigStore
    :param cs_store_name: part of output config name: will be prefixed with 'base_'
    :param cs_store_group_options:
    - keys: config group names
    - values: list of str, - group option names used for:
        - Yaml config files stems
        - Dataclasses to use, - finds names constructed nearly like `Config{capwords(name)}` - must exist in
        `module`.
        - setting 1st values item as default option for a group, name them 'base_{group_name}' where
        group_name is current cs_store_group_options key.
    :param module: module where to search Dataclasses names, default: current module
    :return: (cs, Config)
    cs: ConfigStore
    Config: configuration dataclass
    """

    cs = ConfigStore.instance()
    defaults_list = []  # to save here 1st cs_store_group_options value
    # Registering groups schemas
    for group, classes in cs_store_group_options.items():
        group_option = None
        for cl in classes:
            if isinstance(cl, str):
                name = cl
                class_name = ''.join(
                    ['Config'] + [(s.title() if s.islower() else s) or '_' for s in name.split('_')]
                )
                try:
                    cl = getattr(module, class_name)
                except Exception as err:
                    raise KeyError(
                        f'"{class_name}" class constructed from provided group option {cl} is not found'
                    ) from err
            else:
                class_name = cl.__name__
                name = camel2snake(class_name)

            if group_option is None:  # 1st cs_store_group_options value
                name = f'base_{group}'
                defaults_list.append({group: name})
            try:
                cs.store(group=group, name=name, node=cl)
            except Exception as err:
                raise TypeError(
                    f'Error init group {group} by option "{name}" defined as class {class_name}'
                ) from err

    # Registering all groups
    # Config class (type of result configuration) with assigning defaults to 1st cs_store_group_options value
    Config = make_dataclass(
        "Config",
        [("defaults", List[Any], field(default_factory=lambda: defaults_list))]
        + [(group, typ[0], MISSING) for group, typ in cs_store_group_options.items()],
    )
    cs.store(name=f'base_{cs_store_name}', node=Config)
    cs.get_type("input")
    return cs, Config


def main_init_input_file(cfg_t, cs_store_name, in_file_field='db_path', **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    - finds input files paths
    - renames cfg['input'] to cfg['in'] and fills its field 'cfgFile' with ``cs_store_name``

    :param cfg_t:
    :param cs_store_name:
    :param in_file_field:
    :param kwargs: parameters that will be passed to utils2init.init_file_names()
    :return:
    Note: intended to use after main_init()
    """
    cfg_in = cfg_t.pop('input')
    cfg_in['cfgFile'] = cs_store_name
    try:
        n_paths = len(cfg_in['paths']) if cfg_in.get('paths') else None
        # with omegaconf.open_dict(cfg_in):
        if n_paths:
            lf.info(
                f"Loading data from configured {n_paths} paths"
                if n_paths > 1 else
                f"Loading data from {cfg_in['paths'][0]}"
            )
        else:
            cfg_in['paths'], cfg_in['nfiles'], cfg_in['path'] = init_file_names(
                **{**cfg_in, 'path': cfg_in[in_file_field]},
                b_interact=cfg_t['program']['b_interact'],
                **kwargs)
    except Ex_nothing_done as e:
        print(e.message)
        cfg_t['in'] = cfg_in
        return cfg_t
    except FileNotFoundError as e:
        print('Initialization error:', standard_error_info(e), 'Calling arguments:', sys.argv)
        raise

    cfg_t['in'] = cfg_in
    return cfg_t


def main_init(cfg: Mapping[str, Any], cs_store_name, __file__=None, ) -> Mapping[str, Any]:
    """
    - prints parameters
    - prints message that program (__file__ or cs_store_name) started
    - converts cfg parameters to types according to its prefixes/suffixes names: see utils2init.ini2dict()

    :param cfg: usually DictConfig[str, DictConfig[str, Any]]
    :param cs_store_name:
    :param __file__:
    :return:
    """

    # global lf
    # if cfg.search_path is not None:
    #     override_path = hydra.utils.to_absolute_path(cfg.search_path)
    #     override_conf = OmegaConf.load(override_path)
    #     cfg = OmegaConf.merge(cfg, override_conf)

    print("Working directory : {}".format(os.getcwd()))
    # print not empty / not False values # todo: print only if config changed instead
    try:
        print(OmegaConf.to_yaml({
            k0: ({k1: v1 for k1, v1 in v0.items() if v1}
                if hasattr(v0, 'items') else v0) for k0, v0 in cfg.items()
            }))
    except MissingMandatoryValue as e:
        lf.error(standard_error_info(e))
        raise Ex_nothing_done()

    # cfg = cfg_from_args(argparser_files(), **kwargs)
    if not cfg.program.return_:
        print('Can not initialise: there must be non empty program.return_ value')
        return cfg
    elif cfg.program.return_ == '<cfg_from_args>':  # to help testing
        return cfg

    hydra.verbose = 1 if cfg.program.verbose == 'DEBUG' else 0  # made compatible to my old cfg

    print('\n' + this_prog_basename(__file__) if __file__ else cs_store_name, end=' started. ')
    try:
        cfg_t = ini2dict(cfg)  # fields named with type prefixes/suffixes are converted
    except MissingMandatoryValue as e:
        lf.error(standard_error_info(e))
        raise Ex_nothing_done()
    except Exception:
        lf.exception('startup error')

    # OmegaConf.update(cfg, "in", cfg.input, merge=False)  # error
    # to allow non primitive types (cfg.out['db']) and special words field names ('in'):
    # cfg = omegaconf.OmegaConf.to_container(cfg)
    return cfg_t


def main_call(
        cmd_line_list: Optional[List[str]] = None,
        fun: Optional[Callable[[], Any]] = None,
        my_hydra_module: Optional[ModuleType] = None,
        overrides: Mapping['str', Any] = {}
        ) -> Dict:
    """
    Adds command line args, calls fun, then restores command line args.
    Replaces shortcut "in." in command line args to "input." (a Python reserved word "in" can not be variable)
    :param cmd_line_list: command line args of hydra commands or config options selecting/overwriting we will
    add to original command line after its 1st argument.
    :param fun: function that uses command line args, usually called ``main``
    :param my_hydra_module: module with hydra config, that contains `cs_store_name`, `ConfigType`, stores
    config in "cfg" dir. If ``fun`` is not specified then its ``main`` function will be called instead.
    :param overrides: mapping of hydra commands or config options selecting/overwriting. Applies after
    ``cmd_line_list``.
    :return: `fun()` output
    """
    argv0, *argv_from1 = sys.argv

    # Modify command line to add/change hydra settings

    # if argv0.endswith('.py'):  #?
    cmd_line_list_upd = [argv0]

    # Change config search path (if it is not modified in command line, else get it) and set hydra output dirs
    # inside
    config_path = path = None
    if cmd_line_list is not None:
        for cmd in cmd_line_list:
            if cmd.startswith(("--config-path", "--config-dir", "hydra.searchpath")):
                key, path = cmd.split('=')
                if key == "hydra.searchpath":
                    config_path = path[1:-1]  # strip brackets
                config_path = Path(config_path.strip('"'))
                break
        if config_path is None:
            # Add raw data dir path "cfg_proc" subdir to hydra config search path

            # Searching in command line arguments ending with "path" else in `overrides[in/input]` keys else argv0
            path = [cmd.split("path=") for cmd in cmd_line_list if "path=" in cmd]
            if path:
                path = Path(path[0][-1].strip('"'))
            elif overrides:
                path = [
                    v for k, v in overrides.get("in", overrides.get("input")).items() if k.endswith("path")
                ]
                if path:
                    path = Path(path[0].strip('"'))
            if not path:
                # Else add current path
                path = Path(argv0).parent  # old '"file://{}"'.format(sys.argv[0].replace("\\", "/"))
            if path:
                parents = path.parents
                try:
                    dir_raw = parents[-path.parts.index('_raw') - 1]
                except (IndexError, ValueError):
                    if not path.is_dir():
                        for path in parents:
                            if path.is_dir():
                                break
                    dir_raw = path / '_raw'
                config_path = dir_raw / "cfg_proc"

            # # Saving config_path to hydra main config - old way while absolute path in cmd haven't worked
            # py_file = Path(argv0)
            # path_cfg_default = (lambda p: p.parent / 'cfg' / p.name)(py_file).with_suffix('.yaml')
            # from yaml import safe_dump as yaml_safe_dump
            # with path_cfg_default.open('w') as f:
            #     yaml_safe_dump(
            #         {
            #             "defaults": [f"base_{py_file.stem}", "_self_"],
            #             "hydra": {"searchpath": [f"file://{config_path.as_posix()}".replace("///", "//")]},
            #         },
            #         f
            #     )
            #     f.flush()
            if config_path:
                cmd_line_list_upd += [f'hydra.searchpath=["{config_path.as_posix().strip(chr(34))}"]']
                # (strip user added quotation marks to ensure only one keep)

    # todo: check colorlog is installed before this:
    hydra_dir = (config_path / "log") if config_path else 'outputs'
    hydra_time = '${now:%Y%m%d_%H%M}'
    cmd_line_list_upd += [
        "hydra/job_logging=colorlog",
        "hydra/hydra_logging=colorlog",
        "hydra.job_logging.formatters.colorlog.format='%(cyan)s%(asctime)s %(blue)s%(funcName)10s"
        "%(cyan)s: %(log_color)s%(message)s%(reset)s\t'",  # \\< not supported
        f"hydra.run.dir='{hydra_dir}/{hydra_time}'",
        f"hydra.sweep.dir='{hydra_dir}'",
        "hydra.sweep.subdir='{}-{}'".format(
            hydra_time, "${hydra.job.num}"  # job.override_dirname?
        ),
        # "hydra.job_logging.formatters.colorlog.style='\\{'" not supported
        #'[%(blue)s%(name)s%(reset)s %(log_color)s%(levelname)s%(reset)s] - %(message)s'
        # ]
        #           - %(message)s'
        # %(asctime)s %(log_color)s[%(name)12s:%(lineno)3s'
        # ' %(funcName)18s ]\t%(levelname)-.6s  %(message)s'
    ]

    if cmd_line_list is not None:
        # Not brake diagnostic commands
        if cmd_line_list[0].startswith('--info'):
            for i, el in enumerate(cmd_line_list):
                if '=' in el:
                    break
                cmd_line_list_upd.append(el)
            sys_argv_save, sys.argv = sys.argv, cmd_line_list_upd
            fun()
            sys.argv = sys_argv_save
            return {}
        # Expand custom abbreviation of config group
        for c in cmd_line_list:
            if c.startswith(('in.', '+in.', '++in.')):
                c = c.replace('in.', 'input.', 1)
            cmd_line_list_upd.append(c)
    cmd_line_list_upd += sys.argv[1:]  # (+ ['--config-dir', config_path]) only relative and not works
    sys_argv_save, sys.argv = sys.argv, cmd_line_list_upd

    # hydra.conf.HydraConf.run.dir = './outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}'
    try:
        if my_hydra_module:

            @hydra.main(config_name=my_hydra_module.cs_store_name, config_path='cfg', version_base='1.3')
            def main_fun(config: my_hydra_module.ConfigType):
                with open_dict(config):
                    # config.update(overrides)  # , force_add=True)
                    conf = OmegaConf.unsafe_merge(config, overrides)
                # cfg = OmegaConf.create(overrides)
                try:
                    return (fun or my_hydra_module.main)(conf)
                except Exception as e: # Exit at 1st err even if `multirun` opt
                    # Set debug point here for postmortem debug
                    lf.exception("Error. Exiting the entire process")
                    sys.exit(1)  # non-zero status indicates failure.

            out = main_fun()
        else:
            out = fun()
    except HydraException as e:
        lf.exception('Check calling arguments!')
        val0 = None
        print('Command line arguments:')
        for arg in sys.argv:
            print(arg)
            if config_path and arg.startswith("+defaults@_global"):
                # We will try to load 1st config to check
                key, vals = arg.split('=', 1)
                val0 = vals.split(",", 1)[0]
                if e.args[0].startswith("In 'defaults/"):

                    yaml_with_err_name = e.args[0].split("'", 2)[1].split("/", 1)[1]
                    if yaml_with_err_name in vals.split(","):
                        print(
                            "Trying erroneous config to check",
                            "(1st)" if yaml_with_err_name[0] == val0 else ""
                        )
                        val0 = yaml_with_err_name
                else:
                    print('Trying load 1st config to check')
                # argv_no_yaml = sys.argv.copy()
                # argv_no_yaml.remove(arg)

        if val0:
            cfg_path0 = (config_path / 'defaults' / val0).with_suffix(".yaml")
            try:
                cfg_check = OmegaConf.load(cfg_path0)
            except FileNotFoundError:
                print(f"Sorry, config {cfg_path0} not found. Can not check")
            else:
                conf = to_omegaconf_merge_compatible(cfg_check, my_hydra_module.ConfigType)
                from pprint import pprint
                print("Try prune", yaml_with_err_name or "", "config to this:")
                pprint(conf)
        return
    sys.argv = sys_argv_save
    return out


# Example: override specific fields in the log config from your own config.
#
# your_config.yaml
#
# # compose the colorlog job_logging config
# defaults:
#   - hydra/job_logging: colorlog
#
# # override the formatting.
# hydra:
#   job_logging:
#     formatters:
#       colorlog:
#         format: '[%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s'



# defaults = [dict([item]) for item in {
#     'input': 'nmea_files',  # Load the config "nmea_files" from the config group "input"
#     'out': 'hdf5_vaex_files',  # Set as MISSING to require the user to specify a value on the command line.
#     'filter': 'filter',
#     'program': 'program',
#     #'search_path': 'empty.yml' not works
#      }.items()]


# @dataclass
# class Config:
#
#     # this is unfortunately verbose due to @dataclass limitations
#     defaults: List[Any] = field(default_factory=lambda: defaults)
#
#     # Hydra will populate this field based on the defaults list
#     input: Any = MISSING
#     out: Any = MISSING
#     filter: Any = MISSING
#     program: Any = MISSING
#
#     # Note the lack of defaults list here.
#     # In this example it comes from config.yaml
#     # search_path: str = MISSING  # not helps without defaults

# cs = ConfigStore.instance()
# cs.store(group='in', name='nmea_files', node=ConfigIn)
# cs.store(group='out', name='hdf5_vaex_files', node=ConfigOut)
# cs.store(group='filter', name='filter', node=ConfigFilterCTD)
# cs.store(group='program', name='program', node=ConfigProgram)
# # Registering the Config class with the name 'config'.
# cs.store(name='cfg', node=Config)


# hydra.conf.HydraConf.hydra_logging = 'colorlog'  # if installed ("pip install hydra_colorlog --upgrade")
# hydra.conf.HydraConf.job_logging = 'colorlog'
