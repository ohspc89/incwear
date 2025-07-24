"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2025/7/16)

Plotting segments

© 2023-2025 Infant Neuromotor Control Laboratory. All rights reserved.
"""
import numpy as np
import matplotlib.pyplot as plt

LABEL_MAP = {
        'L': ('lmag', 'laccth', 'lnaccth'),
        'R': ('rmag', 'raccth', 'rnaccth'),
        'U': ('umag', 'accth', 'naccth'),
        }


def plot_segment(fs, accmags_dict, velmags_dict, thresholds, time_passed,
                 duration=20, side='L', movmat=None, title=None, show=True,
                 savepath=None):
    """
    Plot movement segments for manual verification or visualization.

    Parameters
    ----------
    fs : int
        Sampling frequency.
    accmags : numpy.ndarray
        Acceleration magnitudes.
    velmags : numpy.ndarray
        Velocity magnitudes.
    thresholds : dict
        Dictionary with keys: 'accth', 'naccth'.
    time_passed : float
        Time offset from the beginning in seconds.
    duration : int
        Duration of the segment in seconds (default=20).
    side : str
        'L', 'R' (used only for labeling).
    movmat : numpy.ndarray or None
        Optional movement index matrix (start, mid, end).
    title : str or None
        Optional title.
    show : bool
        Whether to display the figure.
    savepath : str or None
        Optional path to save the figure when show is False.

    Returns
    -------
    None
    """
    if side not in 'LR':
        raise ValueError("Invalid side. Must be 'L', 'R'")

    # Jan 31, 23 / WHY did I do this? (checking rowidx is None)
    # Feb 09, 23 / I think this can be remove
    # if self.info.rowidx is not None:
        # new_t = self.info.record_times[0]\
        #         + timedelta(seconds=time_passed)
        # end_t = self.info.record_times[1]
    pth = thresholds[labels[1]]
    nth = thresholds[labels[2]]

    startidx = int(time_passed * fs)
    endidx = startidx + int(duration * fs) + 1

    _, ax = plt.subplots(1)
    handles = []
    accline, = ax.plot(accmags[startidx:endidx], marker='o',
                       c='pink', label='acceleration')
    handles.append(accline)
    pthline = ax.axhline(y=pth, c='k', linestyle='dotted',
                         label='positive threshold')
    handles.append(pthline)
    nthline = ax.axhline(y=nth, c='k', linestyle='dashed',
                         label='negative threshold')
    handles.append(nthline)

    ax.axhline(y=0, c='r')  # baseline
    # If Ax6 or Active, convert from 1 deg/s to 0.017453 rad/s
    # rad_convert = 0.017453 if self._name in ['Ax6', 'Ax6Single'] else 1
    velline, = ax.plot(velmags[startidx:endidx], c='deepskyblue',
                       linestyle='dashdot', label='angular velocity')
    handles.append(velline)

    if movmat is not None:
        mov_st = np.where(movmat[:, 0] >= startidx)[0]
        mov_fi = np.where(movmat[:, 2] <= endidx)[0]

        if all((mov_st.size, mov_fi.size)):
            if mov_st[0] == mov_fi[-1]:
                mov_lens = movmat[mov_st[0], 2] - movmat[mov_st[0], 0]
            else:
                fi2 = mov_fi[-1] + 1
                mov_lens = movmat[mov_st[0]:fi2, 2] - movmat[mov_st[0]:fi2, 0]

            if np.isscalar(mov_lens) or np.ndim(mov_lens) == 0:
                if mov_lens <= 0:
                    hull = []
                else:
                    hull = [np.arange(movmat[mov_st[0], 0],
                                      movmat[mov_st[0], 0] + mov_lens + 1)]
            else:
                hull = [np.arange(x, x + l + 1) for x, l in
                        zip(movmat[mov_st[0]:fi2, 0], mov_lens) if l > 0]

            if len(hull) > 0:
                hl, = ax.plot(hull[0] - startidx, accmags[hull[0]], c='g',
                              linewidth=2, label='movement')
                handles.append(hl)
                prevend = (hull[0] - startidx)[-1]

                for j in range(1, len(hull)):
                    new_xs = hull[j] - startidx
                    ax.plot(new_xs, accmags[hull[j]], c='g', linewidth=2)
                    if new_xs[0] == prevend:
                        ax.scatter(x=new_xs[0], y=accmags[hull[j]][0],
                                   color='red', marker='s')
                    prevend = new_xs[-1]

    ax.legend(handles=handles)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Acc. magnitude (m/s^2)")
    ax.set_title(title or f"Movement Segment ({side})")

    ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, pos: f"{x / fs:.1f}")
    )

    default_tiff = f'Seg-{int(time_passed)}_Dur_{int(duration)}-{side}.tiff'
    if not show:
        outpath = savepath or default_tiff
        plt.savefig(outpath, dpi=600, format='tiff',
                    pil_kwargs={"compression": "tiff_lzw"})
    else:
        plt.show()
