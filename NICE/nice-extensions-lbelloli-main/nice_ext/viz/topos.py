# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import matplotlib.pyplot as plt
import numpy as np
import mne

from ..equipments import get_roi, prepare_layout
from .utils import _map_marker_to_text


def plot_markers_topos(markers, reductions, picks=None, outlines='head',
                       units=None, same_scale=False, ignore_non_scalp=False,
                       fig_kwargs=None, sns_kwargs=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()

    n_axes = len(markers)
    fig = None
    if fig_kwargs is None:
        fig_kwargs = dict(figsize=(3 * n_axes if n_axes > 1 else 4, 4))
    if units is None:
        units = [r''] * n_axes
    fig, axes = plt.subplots(1, n_axes, **fig_kwargs)
    if n_axes == 1:
        axes = [axes]
    scalp_roi = get_roi(config=outlines, roi_name='scalp')

    non_scalp = get_roi(config=outlines, roi_name='nonscalp')
    ch_info = markers[list(markers.keys())[0]].ch_info_
    sphere, outlines = prepare_layout(outlines, info=ch_info)
    _, pos, _, _, _, this_sphere, clip_origin = \
        mne.viz.topomap._prepare_topomap_plot(ch_info, 'eeg', sphere=sphere)

    mask = np.in1d(np.arange(ch_info['nchan']), scalp_roi)
    mask_params = dict(marker='+', markerfacecolor='k', markeredgecolor='k',
                       linewidth=0, markersize=1)

    topos = []
    for ax, (name, marker), unit in zip(axes, markers.items(), units):
        good_idx = mne.pick_channels(marker.ch_info_['ch_names'], include=[],
                                     exclude=marker.ch_info_['bads'])
        bad_idx = []
        if len(marker.ch_info_['bads']) > 0:
            bad_idx = mne.pick_channels(marker.ch_info_['ch_names'],
                                        include=marker.ch_info_['bads'])

        topo = marker.reduce_to_topo(reductions[name], picks[name])
        n_topo = np.zeros((len(good_idx) + len(bad_idx)), dtype=np.float)
        n_topo[good_idx] = topo
        if len(bad_idx) > 0:
            n_topo[bad_idx] = np.nan
        topos.append(n_topo)

    vmin = -np.inf
    vmax = np.inf
    if same_scale is True:
        vmin = np.nanmin(topos)
        vmax = np.nanmax(topos)

    for ax, (name, marker), unit, topo in zip(
            axes, markers.items(), units, topos):
        if same_scale is False:
            vmin = np.nanmin(topo)
            vmax = np.nanmax(topo)
        if non_scalp is not None and not ignore_non_scalp:
            topo[non_scalp] = vmin
        nan_idx = np.isnan(topo)

        plot_topomap_multi_cbar(topo[~nan_idx], pos[~nan_idx], ax,
                                _map_marker_to_text(marker),
                                cmap='viridis',
                                outlines=outlines, mask=mask,
                                mask_params=mask_params, sensors=False,
                                unit=unit, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    return fig


def plot_marker_topo(marker, reduction, picks=None, outlines='head',
                     unit=None, fig_kwargs=None):

    name = 'single'
    reductions = {name: reduction}
    markers = {name: marker}
    picks = {name: picks}
    return plot_markers_topos(markers, reductions, picks, outlines=outlines,
                              units=[unit], fig_kwargs=fig_kwargs)


# Update MNE one with mask, mask_params and sensors parameters
def plot_topomap_multi_cbar(
    data, pos, ax, title=None, unit=None, vmin=None, vmax=None, cmap='RdBu_r',
    colorbar=True, outlines='head', cbar_format='%0.3f', contours=0,
        extrapolate='local', mask=None, mask_params=None, sensors=True,
        sphere=None):
    """Low level plot topography for a single marker

    Parameters
    ----------
    data : numpy.ndarray of float, shape (n_sensors,)
        The data show.
    pos : numpy.ndarray of float, shape (n_sensors, 2)
        The positions of the sensors.
    ax : instance of Axis
        The axes to plot on.
    title : str | None
        The axes title to show. Defaults to None (no title will be shown).
    cbar_format : str
        The colorbar format. Defaults to '%0.3f'
    """
    mne.viz.topomap._hide_frame(ax)
    vmin = np.min(data) if vmin is None else vmin
    vmax = np.max(data) if vmax is None else vmax

    cmap = mne.viz.utils._setup_cmap(cmap)
    if title is not None:
        ax.set_title(title, fontsize=10)
    im, _ = mne.viz.plot_topomap(data, pos, vmin=vmin, vmax=vmax, axes=ax,
                                 cmap=cmap[0], image_interp='bilinear',
                                 contours=contours, outlines=outlines,
                                 show=False, mask=mask,
                                 mask_params=mask_params, sensors=sensors,
                                 extrapolate=extrapolate, sphere=sphere)

    if colorbar is True:
        cbar, cax = mne.viz.topomap._add_colorbar(
            ax, im, cmap, pad=.25, title=None, size="10%", format=cbar_format)
        cbar.set_ticks((vmin, vmax))
        if unit is not None:
            cbar.ax.set_title(unit, fontsize=8)
        cbar.ax.tick_params(labelsize=8)


def plot_topo_equipment(topo, equipment, ch_info=None, ax=None,
                        symmetric_scale=False, unit='', label='',
                        cmap='viridis'):

    scalp_roi = get_roi(config=equipment, roi_name='scalp')
    non_scalp = get_roi(config=equipment, roi_name='nonscalp')

    sphere, outlines, info = prepare_layout(
        equipment, info=ch_info, return_info=True)
    _, pos, _, _, _, _, _ = \
        mne.viz.topomap._prepare_topomap_plot(ch_info, 'eeg', sphere=sphere)

    mask = np.in1d(np.arange(len(pos)), scalp_roi)
    mask_params = dict(marker='+', markerfacecolor='k', markeredgecolor='k',
                       linewidth=0, markersize=1)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    vmin = np.min(topo[scalp_roi])
    vmax = np.max(topo[scalp_roi])
    neutral = vmin

    if symmetric_scale is True:
        absmax = max(abs(vmin), abs(vmax))
        vmin = -absmax
        vmax = absmax
        neutral = 0.0
        cmap = 'RdBu_r'

    if non_scalp is not None:
        topo[non_scalp] = neutral

    plot_topomap_multi_cbar(
        topo, pos, ax, label, cmap=cmap,
        outlines=outlines, mask=mask,
        mask_params=mask_params, sensors=False,
        unit=unit, vmin=vmin, vmax=vmax)

    return fig, ax


def plot_topos_equipments(topos, names, equipment, ch_info=None,
                          same_scale=True, symmetric_scale=False, units=None,
                          is_stat=False, ncols=None, fig_kwargs=None,
                          sns_kwargs=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from .utils import get_stat_colormap
    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()

    if units is None:
        units = ''
    if not isinstance(units, list):
        units = [units] * len(names)

    scalp_roi = get_roi(config=equipment, roi_name='scalp')
    non_scalp = get_roi(config=equipment, roi_name='nonscalp')
    vmin, vmax = 0, 0

    sphere, outlines, ch_info = prepare_layout(
        equipment, info=ch_info, return_info=True)
    _, pos, _, _, _, _, _ = \
        mne.viz.topomap._prepare_topomap_plot(ch_info, 'eeg', sphere=sphere)

    mask = np.in1d(np.arange(len(pos)), scalp_roi)
    mask_params = dict(marker='+', markerfacecolor='k', markeredgecolor='k',
                       linewidth=0, markersize=1)

    if same_scale is True:
        vmin = np.nanmin(np.c_[topos][:, scalp_roi])
        vmax = np.nanmax(np.c_[topos][:, scalp_roi])

    cmap = 'viridis'
    if symmetric_scale is True:
        cmap = 'RdBu_r'
        if same_scale is True:
            vabsmax = max(abs(vmin), abs(vmax))
            vmin = -vabsmax
            vmax = vabsmax

    n_axes = len(names)
    if ncols is None:
        ncols = n_axes
        nrows = 1
    else:
        nrows = int(np.ceil(n_axes / ncols))

    fig = None
    if fig_kwargs is None:
        fig_kwargs = dict(figsize=(3 * ncols if ncols > 1 else 4,
                                   3 * nrows if nrows > 1 else 4))
    fig, axes = plt.subplots(nrows, ncols, **fig_kwargs)
    if n_axes == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for ax, name, unit, topo in zip(axes, names, units, topos):
        if same_scale is False:
            vmin = np.nanmin(topo[scalp_roi])
            vmax = np.nanmax(topo[scalp_roi])
            if symmetric_scale is True:
                vabsmax = max(abs(vmin), abs(vmax))
                vmin = -vabsmax
                vmax = vabsmax
        if non_scalp is not None:
            if is_stat is True or symmetric_scale is True:
                topo[non_scalp] = 0
            else:
                topo[non_scalp] = vmin

        nan_idx = np.isnan(topo)

        if is_stat is True:
            vmin = np.log10(1)
            vmax = -np.log10(1e-5)
            psig = -np.log10(0.05)
            cmap = get_stat_colormap(psig, vmin, vmax)

            unit = r'$-log_{10}(p)$'

        plot_topomap_multi_cbar(topo[~nan_idx], pos[~nan_idx], ax,
                                name,
                                cmap=cmap,
                                outlines=outlines, mask=mask,
                                mask_params=mask_params, sensors=False,
                                unit=unit, vmin=vmin, vmax=vmax,
                                extrapolate='local')
    plt.tight_layout()
    return fig, axes
