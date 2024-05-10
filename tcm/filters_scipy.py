from typing import Tuple, Optional

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .filters import rolling_window

#@njit: gaussian_filter1d() not supported
def despike(x: np.ndarray,
            offsets: Tuple[int, ...] = (2, 20),
            blocks: int = 10,
            ax=None, label='', std_smooth_sigma=None, x_plot=None
            ) -> Tuple[np.ndarray, Optional[matplotlib.axes.Axes]]:
    r"""
    Compute the statistics of the x ($\mu$ and $\sigma$) and marks (but do not exclude yet) x
    that deviates more than $n1 \times \sigma$ from the mean,
    Based on [https://ocefpaf.github.io/python4oceanographers/blog/2013/05/20/spikes/]
    :param x: flat numpy array (that need to filter)
    :param offsets: offsets to std. First offsets should be bigger to delete big spikes first and then filter be
    sensitive to more subtle errors
    :param blocks: filter window width
    :param ax: if not None then plots source and x averaged(blocks) on provided ax
    :param label: for debug: if not None then allow plots. If bool(label) result will be plotted with label legend
    :param std_smooth_sigma: gaussian smooth parameter, if not None std will be smoothed before multiply to offset and
    compare to |data - <data>|.
    :param x_plot: for debug: x data to plot y
    :return y: x with spikes replaced by NaNs
    """
    if not len(offsets):
        return x, ax
    offsets_blocks = np.broadcast(offsets, blocks)
    # instead of using NaNs because of numpy warnings on compare below
    y = np.ma.fix_invalid(x, copy=True)  # suppose the default fill value is big enough to be filtered by masked_where() below.  x.copy()
    len_x = len(x)
    std = np.empty((len_x,), np.float64)
    mean = np.empty((len_x,), np.float64)

    if __debug__:
        n_filtered = []
        if ax is not None:
            colors = ['m', 'b', 'k']
            if x_plot is None:
                x_plot = np.arange(len_x)

    for i, (offset, block) in enumerate(offsets_blocks):
        start = block // 2
        end = len_x - block + start + 1
        sl = slice(start, end)
        # recompute the mean and std without the flagged values from previous pass
        # now removing the flagged y.
        roll = np.ma.array(rolling_window(y.data, block)) if y.mask.size == 1 else (
            np.ma.array(rolling_window(y.data, block), mask=rolling_window(y.mask, block)))
        # 2nd row need because use of subok=True in .as_strided() not useful: not preserves mask (numpy surprise)
        # 1st need because y.mask == False if no masked values but rolling_window needs array

        # roll = np.ma.masked_invalid(roll, copy=False)
        roll.std(axis=1, out=std[sl])
        roll.mean(axis=1, out=mean[sl])
        std[:start] = std[start]
        mean[:start] = mean[start]
        std[end:] = std[end - 1]
        mean[end:] = mean[end - 1]
        assert std[sl].shape[0] == roll.shape[0]
        if std_smooth_sigma:
            std = gaussian_filter1d(std, std_smooth_sigma)

        y = np.ma.masked_where(np.abs(y - mean) > offset * std, y, copy=False)

        if __debug__:
            n_filtered.append(y.mask.sum())
            if ax is not None:
                ax.plot(x_plot, mean, color=colors[i % 3], alpha=0.3,
                        label='{}_mean({})'.format(label if label is not None else '', block))
    y = np.ma.filled(y, fill_value=np.NaN)
    if __debug__:
        print('despike(offsets=', offsets, ', blocks=', blocks, ') deletes', n_filtered, ' points')
        if ax is not None:
            ax.plot(x_plot, y, color='g', label=f'despike{blocks}{offsets}({label})', linewidth=0.5)
            ax.set_xlim(x_plot[[0, -1]])

    return y, ax
