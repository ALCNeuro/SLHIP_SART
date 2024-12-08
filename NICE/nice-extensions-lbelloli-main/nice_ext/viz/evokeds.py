# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import numpy as np
import scipy.stats as scistats

import mne
from mne.utils import logger

from .utils import get_stat_colormap
from .topos import plot_topomap_multi_cbar

from ..stats import compute_gfp

from ..equipments import (prepare_layout, get_roi, get_roi_ch_names,
                          get_ch_names)
from ..equipments.montages import get_ch_adjacency, get_ch_adjacency_montage

from ..api.reductions import trim_mean80
from ..utils import get_contrast, get_contrast_1samp


def plot_cluster_test(epochs, conditions, labels, p_threshold=1e-2,
                      f_threshold=10, shift_time=0, plot=True,
                      event_times=None, sns_kwargs=None):
    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from scipy import stats
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import seaborn as sns
    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()

    scalp_roi_names = get_roi_ch_names(epochs.info['description'], 'scalp')
    epochs = epochs.copy().pick_channels(scalp_roi_names)
    adjacency = get_ch_adjacency(epochs)
    X = [epochs[k].get_data() for k in conditions]
    X = [np.transpose(x, (0, 2, 1)) for x in X]

    if f_threshold == 'auto':
        f_threshold = stats.distributions.f.ppf(
            1. - 1e-5 / 2., X[0].shape[0] - 1,  X[1].shape[0] - 1)
        logger.info('Using generated f_threshold {} from p {}'.format(
            f_threshold, 1e-5))

    cluster_stats = mne.stats.spatio_temporal_cluster_test(
        X, n_permutations=1000, tail=0, n_jobs=-1, threshold=f_threshold,
        adjacency=adjacency)

    T_obs, clusters, p_values, _ = cluster_stats
    sort_idx = np.argsort(p_values)
    p_values = p_values[sort_idx]
    clusters = [clusters[x] for x in sort_idx]
    if len(clusters) == 0:
        logger.info('No clusters found')
        return None

    if plot is False:
        return T_obs, clusters, p_values

    n_clusters = np.sum(p_values < p_threshold)
    if n_clusters == 0:
        logger.info(
            'No significant cluster. Plotting lowest p-value ({}).'.format(
                p_values[0]))
        # Plot the most significant one
        n_clusters = 1
    else:
        logger.info('Found {} significant clusters.'.format(n_clusters))

    fig_cluster = plt.figure(figsize=(12, 3 * n_clusters))
    gs = gridspec.GridSpec(n_clusters, 2, width_ratios=[1, 3])
    sphere, outlines = prepare_layout(
        epochs.info['description'], info=epochs.info)
    for i in range(n_clusters):

        time_inds, space_inds = clusters[i]
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)
        cluster_ch_names = [epochs.ch_names[i] for i in ch_inds]
        f_map = T_obs[time_inds, ...].mean(axis=0)

        _, evokeds, evokeds_stderr, _ = get_contrast(
            epochs, conditions=conditions, method=np.mean,
            roi_name='cluster', roi_channels=cluster_ch_names)

        # Plot Topo
        ax_topo = plt.subplot(gs[2 * i])
        mask = np.zeros((f_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True
        sig_times = (epochs.times[time_inds] + shift_time) * 1000
        plot_topomap_multi_cbar(
            f_map, pos=epochs.info, outlines=outlines, colorbar=False,
            mask=mask, ax=ax_topo, cmap='Reds', sphere=sphere)
        image = ax_topo.images[0]
        divider = make_axes_locatable(ax_topo)
        ax_colorbar = divider.append_axes('right', size='4%', pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            'Averaged F-map ({:0.1f} - {:0.1f} ms)'.format(*sig_times[[0, -1]])
        )

        # Plot evokeds
        ax_evoked = plt.subplot(gs[2 * i + 1])
        sig_mask = [x in time_inds for x in range(evokeds[0].times.shape[0])]
        plot_evoked(
            evokeds,
            std_errs=evokeds_stderr,
            colors=['r', 'b'],
            labels=labels,
            ax=ax_evoked,
            shift_time=shift_time,
            sig_mask=sig_mask,
            event_times=event_times,
            sns_kwargs=sns_kwargs)

        ax_evoked.axvline(0, color='.5', lw=0.5)
        handles, labels = ax_evoked.get_legend_handles_labels()
        sig = mpl.patches.Patch(
            color='r', alpha=0.5, label='SEM {}'.format(labels[0]))
        handles.append(sig)
        sig = mpl.patches.Patch(
            color='b', alpha=0.5, label='SEM {}'.format(labels[1]))
        handles.append(sig)
        sig = mpl.patches.Patch(
            color='orange', label='Significant (p={:.2})'.format(
                p_values[i]))
        handles.append(sig)
        ax_evoked.legend(handles=handles, loc='upper left')

    plt.subplots_adjust(left=0.04, right=0.97)
    return fig_cluster


def plot_cnv(cnv, reduction_func, outlines='head', stat_psig=0.05,
             stat_pvmin=1, stat_pvmax=1e-5, stat_cmap=None, epochs=None,
             rois=None, n_permutations=10000, color='teal', event_times=None,
             fig_kwargs=None, sns_kwargs=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import seaborn as sns
    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()

    fig = None
    if fig_kwargs is None:
        fig_kwargs = dict(figsize=(14, 8))

    if rois is None:
        rois = ['Cz']

    stat_logpsig = -np.log10(stat_psig)
    stat_vmin = np.log10(stat_pvmin)
    stat_vmax = -np.log10(stat_pvmax)
    if stat_cmap is None:
        stat_cmap = get_stat_colormap(stat_logpsig, stat_vmin, stat_vmax)

    montage_names = get_ch_names(config=outlines)
    scalp_roi = get_roi(config=outlines, roi_name='scalp')
    non_scalp = get_roi(config=outlines, roi_name='nonscalp')
    rois_chs = [get_roi(config=outlines, roi_name=roi_name)
                for roi_name in rois]

    adjacency = get_ch_adjacency_montage(config=outlines,
                                         pick_names=montage_names)
    sphere, outlines = prepare_layout(outlines, info=cnv.ch_info_)

    topo = cnv.reduce_to_topo(reduction_func)
    if non_scalp is not None:
        topo[non_scalp] = 0.0

    slopes = cnv.data_
    intercepts = cnv.intercepts_

    # Mass univariate one sample t-test
    _, p_topo = scistats.ttest_1samp(cnv.data_, popmean=0, axis=0)
    p_topo = -np.log10(p_topo)
    p_topo[non_scalp] = 0.0

    # Permutation cluster
    n = cnv.data_.shape[0]
    t_threshold = scistats.distributions.t.ppf(1 - 0.05 / 2., n - 1)

    include = np.in1d(np.arange(cnv.data_.shape[1]), scalp_roi)
    exclude = ~include

    obs, clusters, p_clusters, _ = mne.stats.permutation_cluster_1samp_test(
        slopes, stat_fun=None, threshold=t_threshold,
        exclude=exclude, adjacency=adjacency, out_type='mask',
        n_permutations=n_permutations, seed=42, n_jobs=-1)

    sort_idx = np.argsort(p_clusters)
    p_clusters = p_clusters[sort_idx]
    clusters = [clusters[x] for x in sort_idx]

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(1 + len(rois), 6)

    cluster_mask = None
    ax_topo_cluster = None
    cmap_cl = None
    if len(clusters) == 0:
        logger.info('No cluster found')
        n_clusters = 0
        ax_topo = plt.subplot(gs[0, 1:3])
        ax_topo_stat = plt.subplot(gs[0, 3:5])
    else:
        n_clusters = np.sum(p_clusters < stat_psig)
        cluster_mask = clusters[0]
        if isinstance(cluster_mask, tuple):
            old_mask = cluster_mask[0]
            cluster_mask = np.zeros(cnv.data_.shape[1], dtype=np.bool)
            cluster_mask[old_mask] = True
        if n_clusters == 0:
            n_clusters = 1
            cmap_cl = 'Greys'
        else:
            cmap_cl = 'Reds'
        ax_topo = plt.subplot(gs[0, 0:2])
        ax_topo_stat = plt.subplot(gs[0, 2:4])
        ax_topo_cluster = plt.subplot(gs[0, 4:])

    mask = np.in1d(np.arange(len(topo)), scalp_roi)
    mask_params = dict(marker='+', markerfacecolor='k', markeredgecolor='k',
                       linewidth=0, markersize=1)

    vminmax = max(np.abs(np.min(topo)), np.abs(np.max(topo)))
    plot_topomap_multi_cbar(topo, pos=cnv.ch_info_, ax=ax_topo, title='CNV',
                            cmap='RdBu_r',
                            vmin=-vminmax, vmax=vminmax, outlines=outlines,
                            mask=mask, mask_params=mask_params, sensors=False,
                            unit=r"$\mu{V}$", sphere=sphere)

    plot_topomap_multi_cbar(p_topo, pos=cnv.ch_info_, ax=ax_topo_stat,
                            title='-log10(p)',
                            cmap=stat_cmap, vmin=stat_vmin, vmax=stat_vmax,
                            outlines=outlines, mask=mask,
                            mask_params=mask_params, sensors=False,
                            unit="-log10(p)", sphere=sphere)

    cbar = fig.axes[0].images[0].colorbar
    cbar.set_ticks([-vminmax, 0, vminmax])


    cbar = fig.axes[1].images[0].colorbar
    cbar.set_ticks([stat_vmin, stat_logpsig, stat_vmax])
    cbar.set_ticklabels([
        'p={}'.format(stat_pvmin),
        'p={}'.format(stat_psig),
        'p={}'.format(stat_pvmax)
    ])

    if n_clusters != 0:
        plot_topomap_multi_cbar(
            abs(obs), pos=cnv.ch_info_, outlines=outlines, colorbar=False,
            title='Cluster', mask=cluster_mask, ax=ax_topo_cluster,
            cmap=cmap_cl, sphere=sphere)
        image = ax_topo_cluster.images[0]
        divider = make_axes_locatable(ax_topo_cluster)
        ax_colorbar = divider.append_axes('right', size='4%', pad=0.05)
        cbar = plt.colorbar(image, cax=ax_colorbar)
        ax_colorbar.set_title(r'$\|T\|$')
        ax_topo_cluster.set_xlabel(
            "\nCluster\np-value={:0.4f}" .format(p_clusters[0]))

    for i, (roi_name, roi) in enumerate(zip(rois, rois_chs)):
        roi_cnv = slopes[:, roi].mean(axis=1)
        roi_intercept = intercepts[:, roi].mean(axis=1)
        _, p = scistats.ttest_1samp(roi_cnv, popmean=0)
        if p < 0.05:
            p_color = color
            if p < 1e-4:
                p = 'p < 0.0001'
            else:
                p = 'p = {}'.format(round(p, 4))
        else:
            p_color = 'silver'
            p = 'p = {}'.format(round(p, 4))

        mean_slope = roi_cnv.mean(axis=0)
        mean_intercept = roi_intercept.mean(axis=0)
        # sem_slope = scistats.sem(roi_cnv, axis=0)
        # sem_intercept = scistats.sem(roi_intercept, axis=0)
        cnv_line = [mean_intercept, mean_intercept + 0.6 * mean_slope]
        # cnv_sem1 = [(mean_intercept - sem_intercept),
        #             ((mean_intercept - sem_intercept)
        #                 + 0.6 * (mean_slope - sem_slope))]
        # cnv_sem2 = [(mean_intercept + sem_intercept),
        #             ((mean_intercept + sem_intercept)
        #                 + 0.6 * (mean_slope + sem_slope))]

        ax_roi = plt.subplot(gs[1 + i, 0:4])
        evoked, evoked_stderr = get_contrast_1samp(
            epochs, conditions='all', method=trim_mean80, roi_name=roi_name)

        plot_evoked(evoked, std_errs=evoked_stderr, colors=[color],
                    shift_time=0, ax=ax_roi, event_times=event_times,
                    sns_kwargs=sns_kwargs)
        ax_roi.set_title('Around {}'.format(roi_name), pad=20)
        ax_roi.plot([0, 600], cnv_line, color=p_color, ls='--')
    #     # ax_roi.fill_between([0, 600], cnv_sem1, cnv_sem2,
    #     #                     color=p_color, alpha=0.2)

        ax_roi.axhline(0, color='.5', lw=0.5, ls='--')

        # lab_cnv = mpl.patches.Patch(
        #     color=p_color, alpha=0.2, label=r'$CNV\/(\mu \pm SEM)$')
        lab_erp = mpl.patches.Patch(
            color=color, alpha=0.2, label=r'$ERP\/(\mu \pm SEM)$')
        lab_slope = mpl.lines.Line2D(
            [0], [0], color=p_color, ls='--', label='CNV slope')
        ax_roi.legend(handles=[lab_erp, lab_slope], loc='upper left')

        ax_hist = plt.subplot(gs[1 + i, 4:6])
        sns.distplot(roi_cnv, color=p_color, ax=ax_hist)
        ax_hist.axvline(mean_slope, color=p_color)
        ax_hist.axvline(0, color='black', lw=0.8, ls='--')
        ax_hist.set_title('CNV slope at {}'.format(roi_name), pad=20)
        ax_hist.set_xlabel(r'$CNV\ Slope\ (\mu{V})$')
        ax_hist.set_ylabel('Distribution')
        lab_slope = mpl.lines.Line2D(
            [0], [0], color=p_color,
            label='Slope = {:.2f}'.format(mean_slope))
        lab_p = mpl.patches.Patch(
            color=p_color, label=p)
        ax_hist.legend(handles=[lab_slope, lab_p], loc='upper left')
    plt.tight_layout()
    return fig


def plot_gfp(epochs, conditions=None, colors=None, linestyles=None,
             shift_time=0, labels=None, roi_name=None, ax=None,
             method=trim_mean80, sig_mask=None, event_times=None,
             fig_kwargs=None, sns_kwargs=None):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns
    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()
    fig = None
    if fig_kwargs is None:
        fig_kwargs = {}
    if ax is None:
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
    if conditions is None:
        conditions = epochs.events.keys()
    if colors is None:
        colors = [None for x in conditions]
    if linestyles is None:
        linestyles = ['-' for x in conditions]
    if labels is None:
        labels = [None for x in conditions]
    this_times = (epochs.times + shift_time) * 1e3
    
    for condition, color, ls, label in zip(conditions, colors, linestyles, labels):
        if label is None:
            label = '{}'.format(condition)
        data = epochs[condition].get_data()
        if roi_name != False:
            if roi_name is not None:
                roi = get_roi(config=epochs.info['description'], roi_name=roi_name)
                data = data[:, roi, :]
            else:
                roi = get_roi(config=epochs.info['description'], roi_name='scalp')
                data = data[:, roi, :]
        data = method(data, axis=0)
        gfp, ci1, ci2 = compute_gfp(data)
        lines = ax.plot(this_times, gfp * 1e6, color=color, linestyle=ls,
                        label=label)

        ax.fill_between(this_times, y1=ci1 * 1e6, y2=ci2 * 1e6,
                        color=lines[0].get_color(), alpha=0.5)
        if sig_mask is not None:
            for i in np.where(sig_mask)[0]:
                ax.axvline(this_times[i], alpha=0.5, color='orange')
    handles, labels = ax.get_legend_handles_labels()
    for color in np.unique(colors):
        sig = mpl.patches.Patch(color=color, alpha=0.5, label=r'$\chi^{2}$ CI')
        handles.append(sig)
    if event_times is not None:
        # times = this_times * 1e3
        # xticks = list(event_times.keys())
        # xticks.insert(0, times[0])
        # xticks.extend([times[-1]])
        # ax.set_xticks(np.unique(xticks))
        for t, s in event_times.items():
            t += (shift_time * 1e3)
            ax.axvline(t, color='black', lw=0.5, ls='--')
            ax.text(x=t, y=ax.get_ylim()[1], s=s, horizontalalignment='center')
    ax.set_xlim(this_times[[0, -1]])
    ax.set_ylabel(r'Evoked Response ($\mu{V}$)')
    ax.set_xlabel('Time (ms)')
    ax.legend(handles=handles, loc='upper left')
    return fig


def plot_evoked(evokeds, std_errs=None, colors=None, linestyles=None,
                shift_time=0, labels=None, ax=None, event_times=None,
                sig_mask=None, fig_kwargs=None, sns_kwargs=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()
    fig = None
    if fig_kwargs is None:
        fig_kwargs = {}
    if ax is None:
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
    if std_errs is None:
        std_errs = [None for x in evokeds]
    if colors is None:
        colors = [None for x in evokeds]
    if linestyles is None:
        linestyles = ['-' for x in evokeds]
    if labels is None:
        labels = ['{}'.format(i) for i in range(len(evokeds))]
    max_val = -np.inf
    min_val = np.inf
    this_times = None
    for evoked, color, ls, label, std_err in zip(
            evokeds, colors, linestyles, labels, std_errs):
        if label is None:
            label = '{}'.format(evoked)
        this_times = (evoked.times + shift_time) * 1e3
        data = np.squeeze(evoked.data)
        lines = ax.plot(this_times, data * 1e6,
                        color=color, linestyle=ls, label=label)
        this_max_val = np.max(data)
        this_min_val = np.min(data)
        if std_err is not None:
            ax.fill_between(this_times,
                            y1=(data + np.squeeze(std_err.data)) * 1e6,
                            y2=(data - np.squeeze(std_err.data)) * 1e6,
                            color=lines[0].get_color(), alpha=0.2)
            this_max_val += np.max(std_err.data)
            this_min_val -= np.max(std_err.data)
        max_val = max(this_max_val, max_val)
        min_val = min(this_min_val, min_val)

    if sig_mask is not None:
        for i in np.where(sig_mask)[0]:
            ax.axvline(this_times[i], alpha=0.3, color='orange')
    max_val = np.ceil(max_val * 1e6)
    min_val = np.floor(min_val * 1e6)
    step = 0.5
    if abs(max_val) + abs(min_val) >= 5:
        step = 1.0

    if abs(max_val) + abs(min_val) < step:
        step = (abs(max_val) + abs(min_val)) / 4

    if event_times is not None:
        # times = this_times
        # xticks = list(event_times.keys())
        # xticks.insert(0, times[0])
        # xticks.extend([times[-1]])
        # ax.set_xticks(np.unique(xticks))
        for t, s in event_times.items():
            t += (shift_time * 1e3)
            ax.axvline(t, color='black', lw=0.5, ls='--')
            ax.text(x=t, y=ax.get_ylim()[1], s=s, horizontalalignment='center')

    ax.set_yticks(np.arange(min_val, max_val + 0.1, step))
    ax.set_xlim(this_times[[0, -1]])
    ax.set_ylabel(r'Evoked Response ($\mu V$)')
    ax.set_xlabel('Time (ms)')
    if fig is not None:
        ax.legend(loc='upper left')
    return fig


def plot_butterfly(evoked, color=None, linestyle=None,
                   shift_time=0, labels=None, ax=None, fig_kwargs=None,
                   sns_kwargs=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()
    fig = None
    if fig_kwargs is None:
        fig_kwargs = {}
    if ax is None:
        fig, ax = plt.subplots(1, 1, **fig_kwargs)
    if linestyle is None:
        linestyle = '-'
    if color is None:
        color = 'b'

    this_times = (evoked.times + shift_time) * 1e3
    data = np.squeeze(evoked.data).T
    ax.plot(this_times, data * 1e6, color=color, linestyle=linestyle, lw=0.1)
    max_val = np.max(data)
    min_val = np.min(data)

    max_val = np.ceil(max_val * 1e6)
    min_val = np.floor(min_val * 1e6)
    step = 2
    if abs(max_val) + abs(min_val) >= 10:
        step = 5

    ax.set_yticks(np.arange(min_val, max_val + 0.1, step))
    ax.set_xlim(this_times[[0, -1]])
    ax.set_ylabel(r'Evoked Response ($\mu V$)')
    ax.set_xlabel('Time (ms)')
    return fig


def plot_evoked_topomap(evoked, **kwargs):
    t_evoked = evoked.copy()
    scalp_roi = get_roi(config=t_evoked.info['description'], roi_name='scalp')
    non_scalp = get_roi(config=t_evoked.info['description'],
                        roi_name='nonscalp')
    if non_scalp is not None:
        t_evoked.data[non_scalp, :] = 0.0

    sphere, outlines = prepare_layout(
        t_evoked.info['description'], info=t_evoked.info)
    nchans, ntimes = t_evoked.data.shape
    mask = np.in1d(np.arange(nchans), scalp_roi)
    mask = np.tile(mask[:, None], (1, ntimes))
    mask_params = dict(marker='+', markerfacecolor='k', markeredgecolor='k',
                       linewidth=0, markersize=1)

    kwargs['mask'] = mask
    kwargs['mask_params'] = mask_params
    kwargs['sensors'] = False
    kwargs['outlines'] = outlines
    kwargs['sphere'] = sphere
    sns_kwargs = kwargs.get('sns_kwargs', None)
    if sns_kwargs is not None:
        import seaborn as sns
        sns.set(**sns_kwargs)
    del kwargs['sns_kwargs']

    return mne.viz.plot_evoked_topomap(t_evoked, **kwargs)


def plot_ttest(p_vals, labels, ticks, times, n_times_thresh, n_chans_thresh,
               colors=None, sns_kwargs=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()
    st = int(np.floor((n_times_thresh - 1) / 2))
    end = int(np.ceil((n_times_thresh - 1) / 2))
    this_times = times[st:-end]
    fig_p_vals, ax_pvals = plt.subplots(1, 1, figsize=(12, 8))
    if colors is None:
        colors = [None] * len(p_vals)
    for p_val, label, col in zip(p_vals, labels, colors):
        values = np.zeros(len(this_times))
        for tick in ticks:
            mask = p_val < tick
            time_mask = [
                np.convolve(
                    mask[i, :], np.ones((n_times_thresh,)),
                    mode='valid') == n_times_thresh
                for i in range(p_val.shape[0])]
            this_count = np.sum(np.c_[time_mask], axis=0)
            values[this_count > n_chans_thresh] += 1
        ax_pvals.plot(this_times * 1e3, values, color=col)

    ax_pvals.set_xlabel('Time (ms)')
    ax_pvals.set_ylabel('p value')
    ax_pvals.set_xlim(this_times[[0, -1]] * 1e3)
    ax_pvals.set_ylim([0, len(ticks) + 1])
    ax_pvals.set_yticklabels([0] + ticks)
    ax_pvals.legend(ax_pvals.lines, labels, loc='upper left')

    return fig_p_vals
