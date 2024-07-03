#!/usr/bin/env python3
# coding:utf-8
"""
  Author:  Andrey Korzh <ao.korzh@gmail.com>
  Purpose: Update inclinometer magnetometer, accelerometer coef and Vabs coef in hdf5 tables using Veusz data
  Created: 01.03.2019

Load coefs from Veusz fitting of force(inclination) to velocity data
Save them to hdf5 coef table
"""

import logging
import re
import sys
from pathlib import Path
from datetime import datetime
import h5py
import numpy as np
import pandas as pd
from numpy.distutils.misc_util import is_sequence

# my
# sys.path.append(str(Path(__file__).parent.parent.resolve()))
import veuszPropagate
from tcm.utils2init import cfg_from_args, this_prog_basename, init_file_names, init_logging, Ex_nothing_done, dir_from_cfg,\
    set_field_if_no, path_on_drive_d, my_logging, LoggingStyleAdapter
from tcm.h5inclinometer_coef import h5copy_coef, channel_cols

lf = my_logging(__name__)


def atoi(text):
    return int(text) if text.isdigit() else text


def digits_first(text):
    return (lambda tdtd: (atoi(tdtd[1]), tdtd[0]))(re.split('(\d+)', text))


def main(new_arg=None, veusze=None):
    """
    For each veusz file according to in.pattern_path (see Veusz_propogate) and corresponded table name obtained using
    regex out.re_match_tbl_from_vsz_name
    1. gets data from Veusz file:
    - Rcor
    - coef_list[] = SETTING(in.widget)['d', 'c', 'b', 'a']
    - coef_list[] += DATA(in.data_for_coef)
    2. gets data (A and G) under table's group from out.path hdf5 file:
    - ./coef/H/A
    - ./coef/G/A
    3. Saves :
    - coef_list to ./coef/Vabs0
    4. Updates:
    - ./coef/H/A with Rcor @ A
    - ./coef/G/A with Rcor @ G

    Note: if vsz data source have 'Ag_old_inv' variable then not invert coef. Else invert to use in vsz which not invert coefs
    :param new_arg:
    :return:
    """
    global lf
    p = veuszPropagate.my_argparser()
    p_groups = {g.title: g for g in p._action_groups if
                g.title.split(' ')[-1] != 'arguments'}  # skips special argparse groups
    p_groups['in'].add(
        '--channels_list',
        help='channels needed zero calibration: "magnetometer" or "M" for magnetometer and any else for accelerometer, use "M, A" for both, empty to skip '
        )
    p_groups['in'].add(
        '--widget',
        help='path to Veusz widget property which contains coefficients. For example "/fitV(force)/grid1/graph/fit1/values"'
        )
    p_groups['in'].set_defaults(data_yield_prefix='fit')

    p_groups['in'].add(
        '--data_for_coef', default='max_incl_of_fit_t',
        help='Veusz data to use as coef. If used with widget then this data is appended to data from widget'
        )

    p_groups['out'].add(
        '--out.path',
        help='path to db where write coef')
    p_groups['out'].add(
        '--re_match_tbl_from_vsz_name', default=r'[^_@\d]+_?\d+',
        help='regex to extract hdf5 table name from to Veusz file name'
        # ? why not simply specify table name?
        )
    p_groups['out'].add(
        '--re_sub_tbl_from_vsz_name', default=r'^[\d_]*',
        help='regex to what need to be replaced in name matched by re_match_tbl* by to_sub_tbl_from_vsz_name (see below)'
        # ? why not simply specify table name?
        )
    p_groups['out'].add(
        '--to_sub_tbl_from_vsz_name', default='',
        help='string to replace result of re_sub_tbl_from_vsz_name.'
        )
    # todo:  "b_update_existed" arg will be used here for exported images. Check whether False works or prevent open vsz

    cfg = cfg_from_args(p, new_arg)

    if not Path(cfg['program']['log']).is_absolute():
        cfg['program']['log'] = str(
            Path(__file__).parent.joinpath(cfg['program']['log']))  # lf.root.handlers[0].baseFilename
    if not cfg:
        return
    if new_arg == '<return_cfg>':  # to help testing
        return cfg
    lf = my_logging(__name__, logger=init_logging('', cfg['program']['log'], cfg['program']['verbose']))
    veuszPropagate.lf = lf
    print('\n' + this_prog_basename(__file__), 'started', end=' ')
    if cfg['out']['b_images_only']:
        print('in images only mode.')
    try:
        print('Output pattern ')
        # Using cfg['out'] to store pattern information
        if not Path(cfg['in']['pattern_path']).is_absolute():
            cfg['in']['pattern_path'] = str(cfg['in']['path'].parent.joinpath(cfg['in']['pattern_path']))
        set_field_if_no(cfg['out'], 'path', cfg['in']['pattern_path'])
        cfg['out']['paths'], cfg['out']['nfiles'], cfg['out']['path'] = init_file_names(
            **cfg['out'], b_interact=cfg['program']['b_interact'], msg_action='We will update')
    except Ex_nothing_done as e:
        print(e.message, ' - no pattern')
        return  # or raise FileNotFoundError?
    try:
        cfg['in']['paths'], cfg['in']['nfiles'], cfg['in']['path'] = init_file_names(
            **cfg['in'], b_interact=False, msg_action='Loading data from'
        )
    except Ex_nothing_done as e:
        print(e.message)
        return  # or raise FileNotFoundError?
    if not cfg['out']['export_dir']:
        cfg['out']['export_dir'] = Path(cfg['out']['path']).parent
    if cfg['program']['before_next'] and 'restore_config' in cfg['program']['before_next']:
        cfg['in_saved'] = cfg['in'].copy()
    # cfg['loop'] = asyncio.get_event_loop()
    # cfg['export_timeout_s'] = 600
    cfg['out']['export_dir'] = dir_from_cfg(cfg['out']['path'].parent, cfg['out']['export_dir'])

    veuszPropagate.load_vsz = veuszPropagate.load_vsz_closure(
        cfg['program']['veusz_path'],
        b_execute_vsz=cfg['program']['b_execute_vsz']
        )
    gen_veusz_and_logs = veuszPropagate.load_to_veusz(veuszPropagate.ge_names(cfg), cfg, veusze)

    names_get = ['tofit_inclins', 'tofit_Vext_m__s']  # ['Inclination_mean_use1', 'logVext1_m__s']  # \, 'Inclination_mean_use2', 'logVext2_m__s'
    names_get_fits = ['fit']  # , 'fit2'
    vsz_data = {n: [] for n in (names_get + names_get_fits + ['Rcor'])}

    # prepare collecting all coef in text also
    names_get_txt_results = ['fit1result']  # , 'fit2result'
    txt_results = {n: {} for n in names_get_txt_results}

    for i_file, (veusze, log) in enumerate(gen_veusz_and_logs):
        if not veusze:
            continue
        table = log['out_name']
        print(f'{i_file}. Table "{table}"', end='')
        if cfg['out']['re_match_tbl_from_vsz_name']:
            table = cfg['out']['re_match_tbl_from_vsz_name'].findall(table)[0]
            print(f' -> "{table}"', end='')
        if cfg['out']['re_sub_tbl_from_vsz_name']:
            table = cfg['out']['re_sub_tbl_from_vsz_name'].sub(
               cfg['out']['to_sub_tbl_from_vsz_name'], table)
            print(f' -> "{table}"', end='')
        print()
        for n in names_get + ['Rcor']:
            try:
                vsz_data[n].append(veusze.GetData(n)[0])
            except KeyError:
                lf.warning('not found data in vsz: {}!', n)
        for n in [cfg['in']['data_for_coef']]:
            try:
                vsz_data[n] = list(veusze.GetData(n)[0])
            except KeyError:
                lf.warning('not found data in vsz: {}!', n)
        
        # Coefficients
        coef_matices_for_h5 = {}

        # Velocity coefficients to save into //{table}//coef//Vabs{i} where i - fit number enumerated from 0
        b_have_fits = False
        for k, v in vsz_data.items():
            if k == 'Rcor':
                continue
            for v_el in v:
                if is_sequence(v_el):
                    if b_have_fits := np.any(v_el):
                        break
                    continue
                if b_have_fits := bool(v_el):
                    break
            if b_have_fits:
                break
        if b_have_fits:
            for i, name_out in enumerate(names_get_fits):  # ['fit1', 'fit2']
                coef = veusze.Get(
                    cfg['in']['widget'])  # veusze.Root['fitV(inclination)']['grid1']['graph'][name_out].values.val
                if 'a' in coef:
                    coef_list = [coef[k] for k in ['d', 'c', 'b', 'a'] if k in coef]
                else:
                    coef_list = [coef[k] for k in sorted(coef.keys(), key=digits_first)]
                if cfg['in']['data_for_coef']:
                    coef_list += vsz_data[cfg['in']['data_for_coef']]

                vsz_data[name_out].append(coef_list)
                coef_matices_for_h5.update({
                    f'//coef//Vabs{i}': coef_list,
                    f'//coef//date': np.float64(
                        [np.nan, np.datetime64(datetime.now()).astype(np.int64)])
                })
                # h5savecoef(cfg['out']['path'], path=f'//{table}//coef//Vabs{i}', coef=coef_list)
                txt_results[names_get_txt_results[i]][table] = str(coef)

        # Zeroing matrix - calculated in Veusz by rotation on old0pitch and old0roll
        # Note: zeroing calc. in Veusz should be activated by specifying "USEcalibr0V_..." in Veusz Custom definitions

        try:
            pr = [veusze.GetData('old0pitch_deg')[0][0], veusze.GetData('old0roll_deg')[0][0]]
        except KeyError:
            try:
                pr = np.rad2deg([veusze.GetData('old0pitch')[0][0], veusze.GetData('old0roll')[0][0]])
            except KeyError:
                pr = ['~old0pitch_not_found~', '~old0roll_not_found~']
        if np.any(Rcor := vsz_data['Rcor'][i_file]):
            if False:  # old
                with h5py.File(cfg['out']['path'], 'a') as h5:
                    lf.info(
                        'Modifying coef. applying loaded zero calibration matrix of peach = {} and roll = {} degrees'.format(
                            *pr))
                    for channel in cfg['in']['channels']:  # if need rotate coefs
                        
                        (col_str, coef_str) = channel_cols(channel)
    
                        # h5savecoef(cfg['out']['path'], path=f'//{table}//coef//Vabs{i}', coef=coef_list), dict_matrices={'//coef//' + coef_str + '//A': coefs[tbl][channel]['A'], '//coef//' + coef_str + '//C': coefs[tbl][channel]['b']})
    
                        # Currently, used inclinometers have electronics rotated on 180deg. Before we inserted additional
                        # rotation operation in Veusz by inverting A_old. Now we want include this information in database coef only.
                        try:  # Checking that A_old_inv exist
                            A_old_inv = veusze.GetData('Ag_old_inv')
                            is_old_used = True  # Rcor is not accounted for electronic is rotated.
                        except KeyError:
                            is_old_used = False  # Rcor is accounted for rotated electronic.
    
                        if is_old_used:  # The rotation is done in vsz (A_old in vsz is inverted) so need rotate it back to
                            # use in vsz without such invertion
    
                            # Rotate on 180 deg (note: this is not inversion)
                            A_old_inv = h5[f'//{table}//coef//{coef_str}//A'][...]
                            A_old = np.dot(A_old_inv, [[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # adds 180 deg to roll
                        else:
                            A_old = h5[f'//{table}//coef//{coef_str}//A'][...]
                        # A_old now accounts for rotated electronic
    
                        A = Rcor @ A_old
                        h5copy_coef(None, h5, table,
                                    dict_matrices={f'//coef//{coef_str}//A': A})
            else:  # new
                lf.info('Saving loaded zeroing matrix (previous peach = {} and roll = {} degrees)'.format(*pr))
                coef_matices_for_h5[f'//coef//Rz'] = Rcor

        with h5py.File(cfg['out']['path'], 'a') as h5:
            h5copy_coef(
                None, cfg['out']['path'], table,
                dict_matrices={**coef_matices_for_h5, f'//coef//date': True},
                dates={k: True for k in coef_matices_for_h5.keys()}   # f'//coef//Rz': True,  f'//coef//Vabs{i}': True},
            )

        # veusze.Root['fitV(inclination)']['grid1']['graph2'][name_out].function.val
        print(vsz_data)
        veuszPropagate.export_images(
            veusze, cfg['out'], f"_{log['out_name']}",
            b_skip_if_exists=not cfg['out']['b_update_existed']
        )
        # vsz_data = veusz_data(veusze, cfg['in']['data_yield_prefix'])
        # # caller do some processing of data and gives new cfg:
        # cfgin_update = yield(vsz_data, log)  # to test run veusze.Save('-.vsz')
        # cfg['in'].update(cfgin_update)  # only update of cfg.in.add_custom_expressions is tested
        # if cfg['in']['add_custom']:
        #     for n, e in zip(cfg['in']['add_custom'], cfg['in']['add_custom_expressions']):
        #         veusze.AddCustom('definition', n, e, mode='replace')
        # #cor_savings.send((veusze, log))
        #
        #
        #
        #

    # veusze.Save(str(path_vsz_save), mode='hdf5')  # veusze.Save(str(path_vsz_save)) saves time with bad resolution
    print(f'Ok')
    if b_have_fits:
        print(txt_results)
        for n in names_get:
            pd.DataFrame.from_dict(
                dict(zip(list(txt_results['fit1result'].keys()), vsz_data[n]))
                ).to_csv(
                    Path(cfg['out']['path']).with_name(f'average_for_fitting-{n}.txt'), sep='\t',
                    header=txt_results['fit1result'].keys, mode='a'
                )
    return {**vsz_data, 'veusze': veusze}


if __name__ == '__main__':
    # Example call

    cfg_out_db_path = path_on_drive_d(
        r'd:\WorkData\_experiment\inclinometer\190711_tank\190711incl.h5'
        # r'/mnt/D/workData/_experiment/_2019/inclinometer/190704_tank_ex2/190704incl.h5'
        )
    # r'd:\WorkData\_experiment\_2019\inclinometer\190704\190704incl.h5'
    # r'd:\workData\BalticSea\190713_ABP45\inclinometer\190816incl.h5'
    # 190711incl.h5   cfg['out']['db_path']

    cfg_in = {
        'path': cfg_out_db_path.with_name('190711incl12.vsz'),  # '190704incl[0-9][0-9].vsz'
        # r'd:\WorkData\_experiment\_2019\inclinometer\190704\190704incl[0-9][0-9].vsz',
        # r'd:\WorkData\_experiment\_2019\inclinometer\190711_tank\190711incl[0-9][0-9].vsz',
        # r'd:\WorkData\_experiment\_2018\inclinometer\181004_KTI\incl09.vsz'
        'widget': '/fitV(force)/grid1/graph/fit1/values'
        }
    # , 'out': {    'db_path': }

    # if not cfg_out_db_path.is_absolute():
    #     cfg_out_db_path = Path(cfg_in['path']).parent / cfg_out_db_path
    # d:\workData\_experiment\_2018\inclinometer\180731_KTI\*.vsz

    main([str(Path(veuszPropagate.__file__).with_name('veuszPropagate.ini')),
          '--data_yield_prefix', 'Inclination',
          '--path', str(cfg_in['path']),
          '--pattern_path', str(cfg_in['path']),
          '--out.path', str(cfg_out_db_path),
          '--channels_list', 'M,A',

          '--b_update_existed', 'True',  # to not skip.
          '--export_pages_int_list', '',
          ])
