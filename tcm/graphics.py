#!/usr/bin/env python
# coding:utf-8
"""
  Plotting

  Author:  Andrey Korzh --<ao.korzh@gmail.com>

"""
import logging
from time import sleep
from typing import Any, List, Callable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import msgpack
from pathlib import PurePath
# from graphics import plot_prepare_input, move_figure, make_figure, interactive_deleter

backup_user_input_dir: PurePath = None
backup_user_input_prefix: str = ''

# graphics/interactivity
if True:  # __debug__:
    import matplotlib

    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['figure.figsize'] = (16, 7)
    try:
        matplotlib.use(
            'Qt5Agg')  # must be before importing plt (rases error after although documentation sed no effect)
    except ImportError:
        pass
    from matplotlib import pyplot as plt

    matplotlib.interactive(True)
    plt.style.use('bmh')

if __name__ == '__main__':
    l = None
else:
    l = logging.getLogger(__name__)
    # level_console = 'INFO'
    # level_file = None
    # # set up logging to console
    # console = logging.StreamHandler()
    # console.setLevel(level_console if level_console else 'INFO' if level_file != 'DEBUG' else 'DEBUG')  # logging.WARN
    # # set a format which is simpler for console use
    # formatter = logging.Formatter('%(message)s')  # %(name)-12s: %(levelname)-8s ...
    # console.setFormatter(formatter)
    # l.addHandler(console)


# from collections(!) import Sequence
class MutableTuple(Sequence):
    """Abstract Base Class for objects that work like mutable
    namedtuples. Subclass and define your named fields with
    __slots__ and away you go.
    """
    __slots__ = ()

    def __init__(self, *args):
        for slot, arg in zip(self.__slots__, args):
            setattr(self, slot, arg)

    def __repr__(self):
        return type(self).__name__ + repr(tuple(self))

    # more direct __iter__ than Sequence's
    def __iter__(self):
        for name in self.__slots__:
            yield getattr(self, name)

    # Sequence requires __getitem__ & __len__:
    def __getitem__(self, index):
        return getattr(self, self.__slots__[index])

    def __len__(self):
        return len(self.__slots__)


# class Range(MutableTuple):
#     __slots__ = 'start', 'end'

# plt_select = namedtuple(
#     typename='selected',
#     field_names='x_range_arr x_range y_range_arr finish',
#     defaults=(
#
#     ))
class Plt_select(MutableTuple):
    __slots__ = 'x_range_arr', 'x_range', 'y_range_arr', 'finish', 'reset'


plt_select = Plt_select(
    (plt_selected_x_range_arr := np.empty((2,), '<f8')),
    plt_selected_x_range_arr.copy(),
    plt_selected_x_range_arr.copy(),
    # plt_selected_x_range_arr.view(dtype=(np.record, [('start', '<f8'), ('end', '<f8')]))[0],
    False, False,
    )


def plot_prepare_input(ax,
                       callback: Optional[Callable[[Any, Any], None]] = None,
                       mask: np.ndarray = None,
                       x: Optional[np.ndarray] = None,
                       ys: Sequence[np.ndarray] = None,
                       lines: Sequence[Any] = None):
    """
    initiate RectangleSelector on ax to change plt_select.* variables accordingly
    :param ax: axis
    :param callback: if not None then may be 'fill mask' to change kwargs or provide your callable(eclick, erelease).
    Next arguments not used if callback is None.
    :param mask: mask to fill by False in selected range
    :param x: array with same length as mask. If not sorted then last must be > 1st to detect it
    :param ys: sequence of arrays with same length as mask to fill by np.nan in selected range
    :param lines: list of plot.line of length len(data - 1): data[0] is x data and next are
         y1, y2... data of same length as x data
    :return:
    """
    global plt_select
    print('Select bad x data region (click -> release), Q - close, R - remove mask (restore data) in selection')
    if x is None:

        def selected_st_en():
            return np.int64(plt_select.x_range_arr)
    else:
        if np.subtract(*x[[-1, 0]]) < 0:
            sorter_x = x.argsort()  # arg for searchsorted?
            x_sorted = x[sorter_x]

            def selected_st_en():
                t = sorter_x[np.searchsorted(x_sorted, plt_select.x_range_arr[::-1])] + 1
                return sorted(t)
        else:

            def selected_st_en():
                return np.searchsorted(x, plt_select.x_range_arr)

    def toggle_selector(event):
        """
        Do a mouseclick somewhere, move the mouse to some destination, release
        the button.  This class gives click- and release-events and also draws
        a line or a box from the click-point to the actual mouse position
        (within the same axes) until the button is released.  Within the
        method 'self.ignore()' it is checked whether the button from eventpress
        and eventrelease are the same.

        """
        global plt_select
        print('Pressed:', event.key, 'when selector is', 'active' if toggle_selector.RS.active else 'not active')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
            plt_select.finish = True
        if event.key in ['R', 'r'] and toggle_selector.RS.active:
            # mask[:] = True
            # print(' Mask cleared!')
            # toggle_selector.RS.set_active(False)
            plt_select.reset = True
            # for i in range(-1, -len(ys) - 1, -1):  # from the end
            #     data = ys[i].copy()
            #     lines[i].set_ydata(data)
            #
            # ax.figure.canvas.draw()
            
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)

    def ranges_select_callback(eclick, erelease):
        """eclick and erelease are the press and release events"""
        plt_select.x_range_arr[:] = (eclick.xdata, erelease.xdata)
        plt_select.y_range_arr[:] = (eclick.ydata, erelease.ydata)
        l.info("selected x and y ranges: (%3.2f, %3.2f), (%3.2f, %3.2f)",
               *plt_select.x_range_arr, *plt_select.y_range_arr)
        # print(" The button you used were: %s %s" % (eclick.button, erelease.button))

    if callback is None:
        callback = ranges_select_callback
    elif callback == 'fill mask':
        # mask = kwargs.get('mask', np.ones_like(ys, dtype=np.bool_))

        def fill_mask_and_nan_data_callback(eclick, erelease):
            """
            uses outer scope variables:
            - ys
            - lines
            - mask: sets its elements to False in selected regions
            :param eclick:
            :param erelease:
            :return:
            """
            ranges_select_callback(eclick, erelease)
            
            sl = slice(*(selected_st_en()))
            print(selected_st_en())
            if plt_select.reset:
                plt_select.reset = False
                mask[sl] = True
                print(' Mask in range cleared!')  # lines are recovered below
            else:
                mask[sl] = False  # np.diff(plt_select.x_range_arr) > 0
            for i in range(-1, -len(ys) - 1, -1):  # from the end
                data = ys[i].copy()  # lines[i].get_ydata()
                data[~mask] = np.nan
                # if i > 0: print(np.sum(np.isnan(lines[i]._y)))
                lines[i].set_ydata(data)
            ax.figure.canvas.draw()

        callback = fill_mask_and_nan_data_callback

    plt_select.finish = False
    plt_select.x_range_arr[:] = 0
    toggle_selector.RS = matplotlib.widgets.RectangleSelector(  # drawtype is 'box' or 'line' or 'none'
        ax, callback, drawtype='box', useblit=True,
        button=[1, 3],  # don't use middle button
        minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    plt.connect('key_press_event', toggle_selector)


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


def make_figure(x: Optional[Sequence] = None,
                y_kwrgs: Tuple[Mapping[str, Any], ...] = ({'': []},),
                mask_kwrgs: Optional[Mapping[str, Any]] = None,
                ax: Optional[matplotlib.axes.Axes] = None,
                ax_title: Optional[str] = None, ax_invert=False,
                lines: Union[List[matplotlib.lines.Line2D], str, None] = None,
                position=None,
                clear=None,
                window_title: Optional[str] = None
                ) -> Tuple[matplotlib.axes.Axes, matplotlib.lines.Line2D]:
    """
    Clear or create new axis with lines

    :param x: x data argument of matplotlib.pyplot.plot(), same for all lines plots
    :param y_kwrgs: tuple having in each element dict with fields:
        'data' (requiered) - y data for line i
         ony other matplotlib.pyplot.plot() params to plot line i (optional)
    :param mask_kwrgs: dict with argumetnts to plt.plot(x[mask_kwrgs['data']], y[-1][mask_kwrgs['data']], **mask0kwrgs)
        where:
        - 'data' field: mask
        - mask0kwrgs is mask_kwrgs without 'data'
    :param ax:
    :param ax_title:
    :param ax_invert:
    :param lines:
        - if list then used to shift colors by its length, will be appended with objects returned by ax.plot(
        x, y_kwrgs[i]['data'], ...) for i = 0..len(y_kwrgs)
        - if str 'clear' then clear axes before plot
    :param window_title:
    :param clear:
    :param position:
    :return: ax, lines

    """
    if clear:
        ax_clear = True
        n_prev_lines = 0
    elif isinstance(lines, str):
        ax_clear = True
        if (lines is None) or ax_clear:
            if ax_clear and lines != 'clear':
                l.warning(
                    'wrong lines string %s! must be "clear" or not string! Continuing like it was "clear"...', lines
                )
        n_prev_lines = 0
    else:
        ax_clear = False
        n_prev_lines = 0 if lines is None else len(lines)
    lines = []
    if x is None:
        x = np.arange(len(y_kwrgs[0]['data']))
    try:
        if (ax is None) or (not ax_clear) or (not plt.fignum_exists(ax.figure.number)):
            _, ax = plt.subplots()

        else:
            ax.clear()

        if ax_invert:
            ax.yaxis.set_inverted(True)

        if ax_title is not None:
            ax.set_title(ax_title)

        if window_title:
            man = plt.get_current_fig_manager()
            man.canvas.setWindowTitle(window_title)

        # lines will be returned allowing to change them
        for i, (y_kwrg, color, alpha) in enumerate(zip(
                y_kwrgs, ['r', 'c', 'g', 'm', 'b', 'y'][n_prev_lines:], [0.3] + [0.5] * (5 - n_prev_lines))):
            y_data = (plot_kwrgs := dict(y_kwrg)).pop('data')  # removes 'data' from copy of y_kwrg to use as plot kwargs
            lines += ax.plot(x, y_data, **{'color': color, 'alpha': alpha, **plot_kwrgs})

        if mask_kwrgs is not None and mask_kwrgs.get('data') is not None:
            # add last line with already applied mask
            mask_data = (plot_kwrgs := dict(mask_kwrgs)).pop('data')  # removes 'data' from copy of mask_kwrgs to use as plot kwrgs
            _ = y_data.copy()
            _[~mask_data] = np.nan
            lines += ax.plot(x, _, **{'color': 'r', 'label': 'masked initial', **plot_kwrgs})

        ax.legend(prop={'size': 10}, loc='upper right')
        if position is not None:
            move_figure(ax.figure, *position)
    except Exception as e:
        l.exception('make_figure error')
        return ax, lines
    return ax, lines


def interactive_deleter(x: Optional[Sequence] = None,
                        y_kwrgs: Tuple[Mapping[str, Any], ...] = ({'': []},),
                        mask_kwrgs: Optional[Mapping[str, Any]] = None,
                        ax: Optional[matplotlib.axes.Axes] = None,
                        stop: Union[str, bool, None] = True,
                        **kwargs) -> np.ndarray:
    """
    Modifies mask graphically.
    Note: If evolute in debug mode set argument stop=False else your program hangs!
    :param x:
    :param y_kwrgs:
    :param mask_kwrgs: dict with argumtents to plt.plot(x[mask_kwrgs['data']], y[-1][mask_kwrgs['data']], **mask0kwrgs)
        where:
            mask_kwrgs['data']: mask - will be interactively modified
            mask0kwrgs is mask_kwrgs without 'data'
    :param ax:
    :param stop: display figure again until user press "Q"
    :param kwargs: any other that in make_figure() (see make_figure()). Among them
    - lines: if is not None, the ax must have this lines
     and y_kwrgs must have this number of dicts with 'data' field
    :return: mask

    interactive_deleter(x=bot_edge_path_dist,
        y_kwrgs=(('source', bot_edge_path_Pres), ('depth', bot_edge_Dep)),
        mask_label='closest to bottom', mask=max_spike_ok['soft'],
        title='Bottom edge of CTD path filtering')


    todo:
        # left -> right : delete
        # right -> left : recover
    """

    b_x_is_simple_range = (x is None)
    if b_x_is_simple_range:
        x = np.arange(len(y_kwrgs[0]['data']))
    if mask_kwrgs is None:
        mask_kwrgs = {'data': np.ones_like(x, dtype=np.bool_), 'label': 'masked'}
    elif mask_kwrgs.get('data') is None:
        mask_kwrgs['data'] = np.ones_like(x, dtype=np.bool_)
    print('interactive_deleter "%s" start: %d data values masked already' % (
        kwargs.get('ax_title', ''), mask_kwrgs['data'].sum()))

    lines = kwargs.get('lines')
    if (lines is None) or (len(y_kwrgs) > len(lines)):
        ax, lines = make_figure(x, y_kwrgs, mask_kwrgs, ax=ax, **kwargs)
    else:
        ax.legend(prop={'size': 10}, loc='upper right')  # refresh legend

    # load previous user mask:
    # Read msgpack file
    if backup_user_input_dir and (ax_title := kwargs.get('ax_title')):
        file_backup = (backup_user_input_dir / f"{backup_user_input_prefix}{ax_title.replace('. ', '_')}").with_suffix('.packb')
        try:
            with open(file_backup, "rb") as h_backup:
                data_loaded = msgpack.unpackb(h_backup.read())
            print(f'Assigning previous user edited mask: set {sum(data_loaded)}/{len(data_loaded)}...')
            mask_kwrgs['data'][:] = np.array(data_loaded)
        except FileNotFoundError:
            pass
        except Exception:
            l.exception('Error load backed up user input')
    else:
        file_backup = None

    plot_prepare_input(ax, callback='fill mask',
                       mask=mask_kwrgs['data'],
                       lines=lines,
                       x=None if b_x_is_simple_range else x,
                       ys=[y['data'] for y in y_kwrgs])
    if stop:  # dbstop to make stop if noninteruct
        f_number = ax.figure.number
        plt.show(block=True)  # set False if block=True - hangs. Allows select bad regions (pycharm: not stops if dbstops before)
        while (not plt_select.finish) and plt.fignum_exists(f_number):  # or get_fignums().
            # input()
            sleep(1)
            # plt.draw()
            # dbstop
            # if not np.any(plt_select.x_range_arr):
            #     break
            # else:
            #    bDepth[slice(*np.int64(plt_select.x_range_arr))] = np.diff(plt_select.x_range_arr) > 0

            # plt.show()  # stop to allow select bad regions
            # if np.any(plt_select.x_range_arr):
            #     # Do many times if need on same figure:
            #     l.warning('Deleting selected region (%f, %f)', *plt_select.x_range_arr)
            #     del_slice = slice(*np.searchsorted(x, plt_select.x_range_arr)) if np.diff(plt_select.x_range_arr) else \
            #         np.searchsorted(x, plt_select.x_range.start)
            #     mask[del_slice] = False
            # else:
            #     break
        print('interactive_deleter "%s" end: %d mask points to NaN data' % (
            kwargs.get('ax_title', ''), mask_kwrgs['data'].sum()))

        if file_backup:
            # save user mask:
            # Write msgpack file
            with open(file_backup, "wb") as outfile:
                packed = msgpack.packb(mask_kwrgs['data'].tolist())
                outfile.write(packed)


    return mask_kwrgs['data']

#
# #from collections(!) import Sequence
# class MutableTuple(Sequence):
#     """Abstract Base Class for objects that work like mutable
#     namedtuples. Subclass and define your named fields with
#     __slots__ and away you go.
#     """
#     __slots__ = ()
#     def __init__(self, *args):
#         for slot, arg in zip(self.__slots__, args):
#             setattr(self, slot, arg)
#     def __repr__(self):
#         return type(self).__name__ + repr(tuple(self))
#     # more direct __iter__ than Sequence's
#     def __iter__(self):
#         for name in self.__slots__:
#             yield getattr(self, name)
#     # Sequence requires __getitem__ & __len__:
#     def __getitem__(self, index):
#         return getattr(self, self.__slots__[index])
#     def __len__(self):
#         return len(self.__slots__)
#
# class Range(MutableTuple):
#     __slots__ = 'start', 'end'
