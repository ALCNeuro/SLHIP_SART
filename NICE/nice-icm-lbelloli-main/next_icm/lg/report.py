import numpy as np

import mne
from mne.utils import logger

from nice_ext.api.modules import register_module
from nice_ext.api import get_reductions
from nice_ext.api.reductions import trim_mean80
from nice_ext.viz import utils as vizutils
from nice_ext import viz
from nice_ext.utils import get_contrast
from nice_ext.common import get_clf_name


from ..rs.report import _create_rs_report, _create_preprocessing_report


def register():
    register_module('report', 'icm/lg', _create_lg_report)
    register_module('report', 'icm/predict', _create_predictive_report)


def _create_lg_report(instance, report, config_params):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    epochs = None

    # Some config parameters
    stat_psig = 0.05
    stat_logpsig = -np.log10(stat_psig)
    stat_pvmin = 1
    stat_pvmax = 1e-5
    stat_vmin = 0
    stat_vmax = -np.log10(stat_pvmax)
    stat_cmap = vizutils.get_stat_colormap(stat_logpsig, stat_vmin,
                                           stat_vmax)
    cluster_f_threshold = config_params.get('cluster_f_threshold', 10)
    event_times = {
        0: 'I',
        150: 'II',
        300: 'III',
        450: 'IV',
        600: 'V'
    }
    outlines = 'head'
    title_extra = ''

    if 'epochs' in config_params:
        epochs = config_params['epochs']
        if 'baseline' in config_params:
            baseline = config_params['baseline']
            title_extra = f' (baseline {baseline[0]} - {baseline[1]})'
            logger.info(f'Applying baseline for report: {baseline}')
            epochs = epochs.copy().apply_baseline(baseline)

        if 'summary' in config_params and 'epochs' in config_params:
            _create_preprocessing_report(epochs, report=report,
                                         config_params=config_params)

        outlines = epochs.info['description']

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        mne.viz.plot_events(epochs.events, epochs.info['sfreq'],
                            event_id=epochs.event_id, axes=ax)

        # ax = fig.get_axes()[0]
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles[::-1], labels[::-1],
        #           loc='center left', bbox_to_anchor=(1, 0.5))
        # for legend in fig.findobj(mpl.legend.Legend):
        #     for text in legend.texts:
        #         text.set_fontsize(12)
        report.add_figs_to_section(
            figs=[fig], captions=[f'Local Global Paradigm{title_extra}'],
            section='Diagnostic')

        # ERP Diagnostic
        fig = viz.plot_gfp(
            epochs, conditions=['LSGS', 'LSGD', 'LDGS', 'LDGD'],
            colors=['b', 'b', 'r', 'r'],
            linestyles=['-', '--', '-', '--'],
            shift_time=-0.6,
            event_times=event_times,
            fig_kwargs=dict(figsize=(12, 6)),
            sns_kwargs=dict(style='darkgrid'))
        report.add_figs_to_section(
            figs=[fig],
            captions=[f'All Blocks: Global Field Power{title_extra}'],
            section='Diagnostic')

        # ERPs

        local_conditions = [['LDGS', 'LDGD'], ['LSGS', 'LSGD']]
        local_labels = ['Local Deviant', 'Local Standard']

        global_conditions = [['LDGD', 'LSGD'], ['LSGS', 'LDGS']]
        global_labels = ['Global Deviant', 'Global Standard']

        # -- Local Effect
        time_shift = -0.6
        plot_times = np.arange(.64, 1.336, .02) + time_shift

        fig_gfp = viz.plot_gfp(
            epochs,
            conditions=local_conditions,
            colors=['r', 'b'],
            labels=local_labels,
            shift_time=time_shift,
            event_times=event_times,
            fig_kwargs=dict(figsize=(12, 6)),
            sns_kwargs=dict(style='darkgrid'))

        evoked, _, _, local_contrast = get_contrast(
            epochs, conditions=local_conditions, method=trim_mean80)

        evoked.shift_time(time_shift)

        fig_topo = viz.plot_evoked_topomap(
            evoked, times=plot_times, ch_type='eeg', contours=0, cmap='RdBu_r',
            cbar_fmt='%0.3f', average=.04, units=r"$\mu{V}$",
            ncols=10, nrows='auto', extrapolate='local',
            sns_kwargs=dict(style='white'))

        local_contrast.mlog10_p_val.shift_time(time_shift)
        fig_topo_stat = viz.plot_evoked_topomap(
            local_contrast.mlog10_p_val, times=plot_times, ch_type='eeg',
            contours=0, cmap=stat_cmap, scalings=1,
            cbar_fmt='%0.3f', average=.04, vmin=stat_vmin,
            vmax=stat_vmax, units="-log10(p)", ncols=10, nrows='auto',
            extrapolate='local', sns_kwargs=dict(style='white'))

        cb = fig_topo_stat.axes[-1].images[0].colorbar
        cb.set_ticks([stat_vmin, stat_logpsig, stat_vmax])
        cb.set_ticklabels(['p={}'.format(stat_pvmin), 'p={}'.format(stat_psig),
                           'p={}'.format(stat_pvmax)])

        fig_cluster = viz.plot_cluster_test(
            epochs, local_conditions, local_labels, shift_time=-0.6,
            f_threshold=cluster_f_threshold, event_times=event_times,
            sns_kwargs=dict(style='white'))

        this_figs = [fig_gfp, fig_topo, fig_topo_stat]
        this_captions = [
            f'Local Effect: Global Field Power{title_extra}',
            f'Local Effect: Topographies{title_extra}',
            f'Local Effect: Topographies (-log10(p)){title_extra}']
        if fig_cluster is not None:
            this_figs.append(fig_cluster)
            this_captions.append(
                f'Local Effect: Cluster Permutation Test{title_extra}')

        report.add_figs_to_section(figs=this_figs, captions=this_captions,
                                   section='ERP')

        # -- Global Effect
        plot_times = np.arange(.64, 1.336, .02) + time_shift

        fig_gfp = viz.plot_gfp(
            epochs,
            conditions=global_conditions,
            colors=['r', 'b'],
            labels=global_labels,
            shift_time=time_shift,
            event_times=event_times,
            fig_kwargs=dict(figsize=(12, 6)),
            sns_kwargs=dict(style='darkgrid'))

        evoked, _, _, global_contrast = get_contrast(
            epochs, conditions=global_conditions, method=trim_mean80)

        evoked.shift_time(time_shift)

        fig_topo = viz.plot_evoked_topomap(
            evoked, times=plot_times, ch_type='eeg', contours=0, cmap='RdBu_r',
            cbar_fmt='%0.3f', average=.04, units=r"$\mu{V}$",
            ncols=10, nrows='auto', extrapolate='local',
            sns_kwargs=dict(style='white'))

        global_contrast.mlog10_p_val.shift_time(time_shift)
        fig_topo_stat = viz.plot_evoked_topomap(
            global_contrast.mlog10_p_val, times=plot_times, ch_type='eeg',
            contours=0, cmap=stat_cmap, scalings=1,
            cbar_fmt='%0.3f', average=.04, vmin=stat_vmin,
            vmax=stat_vmax, units="-log10(p)",
            ncols=10, nrows='auto',
            extrapolate='local', sns_kwargs=dict(style='white'))

        cb = fig_topo_stat.axes[-1].images[0].colorbar
        cb.set_ticks([stat_vmin, stat_logpsig, stat_vmax])
        cb.set_ticklabels(['p={}'.format(stat_pvmin), 'p={}'.format(stat_psig),
                           'p={}'.format(stat_pvmax)])

        fig_cluster = viz.plot_cluster_test(
            epochs, global_conditions, global_labels, shift_time=-0.6,
            f_threshold=cluster_f_threshold, event_times=event_times,
            sns_kwargs=dict(style='white'))

        this_figs = [fig_gfp, fig_topo, fig_topo_stat]
        this_captions = [
            f'Global Effect: Global Field Power{title_extra}',
            f'Global Effect: Topographies{title_extra}',
            f'Global Effect: Topographies (-log10(p)){title_extra}']

        if fig_cluster is not None:
            this_figs.append(fig_cluster)
            this_captions.append(
                f'Global Effect: Cluster Permutation Test{title_extra}')

        report.add_figs_to_section(figs=this_figs, captions=this_captions,
                                   section='ERP')

        n_chans_thresh = config_params.get('stats_n_chans', 10)
        n_times_thresh = config_params.get('stats_n_times', 5)
        p_vals_ticks = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        fig_stats = viz.plot_ttest(
            [local_contrast.p_val.data, global_contrast.p_val.data],
            ['Local Effect', 'Global Effect'],
            p_vals_ticks, (epochs.times - 0.6),
            n_times_thresh=n_times_thresh,
            n_chans_thresh=n_chans_thresh, colors=['b', 'r'])
        caption = 'ERP Statistical Analysis '\
            '(> {} channels, > {} samples)'.format(
                n_chans_thresh, n_times_thresh)

        report.add_figs_to_section(
            figs=[fig_stats], captions=[caption], section='ERP')

        rois = ['Fz', 'Cz', 'Pz']
        fig_rois, axes = plt.subplots(
            len(rois), 2, figsize=(14, 2 * len(rois)))

        for i, roi_name in enumerate(rois):
            this_evoked, evokeds, evokeds_stderr, this_contrast = get_contrast(
                epochs, local_conditions, method=trim_mean80,
                roi_name=roi_name)
            sig_mask = np.squeeze(this_contrast.p_val.data < stat_psig)
            viz.plot_evoked(
                evokeds,
                std_errs=evokeds_stderr,
                colors=['r', 'b'],
                labels=['Deviant', 'Standard'],
                shift_time=time_shift,
                ax=axes[i, 0],
                sig_mask=sig_mask,
                event_times=event_times,
                sns_kwargs=dict(style='darkgrid'))
            col_title = '' if i != 0 else 'Local Effect \n\n'
            axes[i, 0].set_title(f'{col_title}Around {roi_name}')

            this_evoked, evokeds, evokeds_stderr, this_contrast = get_contrast(
                epochs, global_conditions, method=trim_mean80,
                roi_name=roi_name)
            sig_mask = np.squeeze(this_contrast.p_val.data < stat_psig)
            viz.plot_evoked(
                evokeds,
                std_errs=evokeds_stderr,
                colors=['b', 'r'],
                labels=None,
                shift_time=time_shift,
                ax=axes[i, 1],
                sig_mask=sig_mask,
                event_times=event_times,
                sns_kwargs=dict(style='darkgrid'))
            col_title = '' if i != 0 else 'Global Effect \n\n'
            axes[i, 1].set_title(f'{col_title}Around {roi_name}')

        handles, labels = axes[0, 0].get_legend_handles_labels()
        sig = mpl.patches.Patch(
            color='r', alpha=0.5, label='Deviant SEM')
        handles.append(sig)
        sig = mpl.patches.Patch(
            color='b', alpha=0.5, label='Standard SEM')
        handles.append(sig)
        sig = mpl.patches.Patch(
            color='orange', label='Significative (p<{})'.format(stat_psig))
        handles.append(sig)
        axes[0, 0].legend(handles=handles, bbox_to_anchor=(-0.1, 1.0))
        plt.subplots_adjust(hspace=0.6, left=0.18, right=0.98)
        report.add_figs_to_section(
            figs=[fig_rois], captions=[f'ERP ROI Analysis{title_extra}'],
            section='ERP')

        # END epochs plots
    reduction_config = config_params.get(
        'reduction_lg', 'icm/lg/egi256/trim_mean80')

    reductions = get_reductions(config=reduction_config)
    # CNV
    reduction_func = reductions[
        'ContingentNegativeVariation']['reduction_func']
    cnv = instance['nice/marker/ContingentNegativeVariation/default']
    fig_cnv = viz.plot_cnv(
        cnv, reduction_func, outlines=outlines,
        stat_psig=stat_psig, stat_pvmax=stat_pvmax,
        stat_pvmin=stat_pvmin, stat_cmap=stat_cmap,
        epochs=epochs,
        event_times=event_times,
        sns_kwargs=dict(style='white'))
    report.add_figs_to_section(
        figs=[fig_cnv],
        captions=[f'Contingent Negative Variaton{title_extra}'],
        section='CNV')

    if 'summary' in config_params:
        del config_params['summary']
    report = _create_rs_report(instance, report, config_params)
    return report


def _create_predictive_report(instance, report, config_params):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import seaborn as sns
    sns.set()
    sns.set_color_codes()

    summary = instance.get('summary', None)
    if summary is not None:
        summary_table = vizutils.render_prediction_summary(summary)

        report.add_htmls_to_section(
            [summary_table], ['Prediction Summary'], section='Prediction')

    # Overall figure
    overall_figures = []
    overall_captions = []
    for clf_type in instance['multivariate'].keys():
        t_res = instance['multivariate'][clf_type]
        fig_overall = plt.figure(figsize=(8, 2))
        if clf_type == 'et-reduced':
            if t_res['MCS'] > 0.5:
                bins = [0, 1]
            else:
                bins = [1, 0]
            bars = plt.barh([0, 0], bins, 1, [0,  bins[0]],
                            color=['r', 'b'], alpha=0.7, edgecolor='0.2')
            sns.despine(left=True, bottom=True)
            ax = fig_overall.get_axes()[0]
            ax.set_xticklabels([])
            ax.set_yticks([0])
            ax.set_yticklabels(['Result:'])
            ax.set_xlim(0, 1)
            plt.subplots_adjust(bottom=0.5, top=0.6, left=0.45, right=0.55)
        else:
            bars = plt.barh([0, 0],
                            [t_res['VS/UWS'], t_res['MCS']],
                            1, [0,  t_res['VS/UWS']],
                            color=['r', 'b'], alpha=0.7, edgecolor='0.2')
            ax = fig_overall.get_axes()[0]
            [ax.axvline(x, color='0.2', lw=0.4)
             for x in np.arange(0, 1.01, 0.1)]
            plt.xticks(np.arange(0, 1.01, 0.1))

            # fig_overall.get_axes()[0].set_ylim(0, 1)
            sns.despine(left=True, bottom=False)
            ax.yaxis.set_visible(False)
            ax.set_xlim(0, 1)
            plt.subplots_adjust(bottom=0.5, top=0.6)

        plt.legend((bars[0], bars[1]), ['VS/UWS', 'MCS'], loc=9,
                   bbox_to_anchor=(0.5, -1.5), ncol=2)
        overall_figures.append(fig_overall)
        overall_captions.append(get_clf_name(clf_type))

    report.add_figs_to_section(
        overall_figures, captions=overall_captions, section='Multivariate')
    [plt.close(x) for x in overall_figures]

    # Violin plot
    probas = instance['univariate']
    stacked_df = probas.set_index(
        ['Marker', 'Reduction'])[['VS/UWS', 'MCS']].stack()
    stacked_df.index.names = ['Marker', 'Reduction', 'Label']
    stacked_df.name = 'P'
    stacked_df = stacked_df.reset_index()
    fig_violin = plt.figure(figsize=(8, 2))
    sns.violinplot(x='P', y='Label', data=stacked_df, orient='h', 
                   inner='quartile', palette=['r', 'b'], cut=0)
    plt.axvline(0.5, color='k', linestyle='--')
    plt.xlabel('Probability')
    plt.ylabel('Probability Density')
    # plt.xlim([0, 1])
    # plt.xticks([0], ['VS/UWS'])
    plt.title(
        'Summary of univariate prediction ({} markers)'.format(len(probas)))
    plt.subplots_adjust(bottom=0.25)

    report.add_figs_to_section(
        fig_violin, captions='Summary I', section='Univariate Summaries')

    groups = {
        'Information Theory': [
            'PermutationEntropy/default',
            'KolmogorovComplexity/default'
        ],
        'Spectral': [
            'PowerSpectralDensity/alpha',
            'PowerSpectralDensity/alphan',
            'PowerSpectralDensity/beta',
            'PowerSpectralDensity/betan',
            'PowerSpectralDensity/delta',
            'PowerSpectralDensity/deltan',
            'PowerSpectralDensity/theta',
            'PowerSpectralDensity/thetan',
            'PowerSpectralDensity/gamma',
            'PowerSpectralDensity/gamman',
            'PowerSpectralDensity/summary_se',
            'PowerSpectralDensitySummary/summary_msf',
            'PowerSpectralDensitySummary/summary_sef90',
            'PowerSpectralDensitySummary/summary_sef95',
        ],
        'Connectivity': [
            'SymbolicMutualInformation/weighted'
        ],
        'ERPs': [
            'ContingentNegativeVariation/default',
            'WindowDecoding/local',
            'WindowDecoding/global',
            'TimeLockedContrast/mmn',
            'TimeLockedContrast/p3a',
            'TimeLockedContrast/p3b',
            'TimeLockedTopography/p1',
            'TimeLockedTopography/p3a',
            'TimeLockedTopography/p3b',
            'TimeLockedContrast/LD-LS',
            'TimeLockedContrast/GD-GS',
            'TimeLockedContrast/LSGD-LDGS',
            'TimeLockedContrast/LSGS-LDGD',
        ]
    }

    group_orders = ['Information Theory', 'Connectivity', 'Spectral', 'ERPs']

    _icm_text_maps = {
        'WindowDecoding/local': 'Decod Local',
        'WindowDecoding/global': 'Decod Global',
        'TimeLockedContrast/p3a': r'$\Delta P3A$',
        'TimeLockedContrast/p3b': r'$\Delta P3B$',
        'TimeLockedContrast/mmn': r'$\Delta MMN$',
        'TimeLockedTopography/p1': 'P1',
        'TimeLockedTopography/p3a': 'P3A',
        'TimeLockedTopography/p3b': 'P3B',
        'TimeLockedContrast/GD-GS': 'GD',
        'TimeLockedContrast/LD-LS': 'LS',
        'TimeLockedContrast/LSGD-LDGS': 'LSGS-LDGS',
        'TimeLockedContrast/LSGS-LDGD': 'LSGS-LDGD',
    }

    _icm_text_maps.update(vizutils._text_maps)

    sns.set_style('white')
    probas = instance['univariate']

    n_reductions = len(np.unique(probas['Reduction']))

    groups_to_use = {}
    for k_g, markers in groups.items():
        _this_markers = [m for m in markers if 'nice/marker/{}'.format(m) in
                         probas['Marker'].values]
        if len(_this_markers) > 0:
            groups_to_use[k_g] = _this_markers

    group_orders = [k for k in group_orders if k in groups_to_use.keys()]

    group_sizes = [n_reductions * len(groups_to_use[x]) for x in group_orders]

    fig_bars = plt.figure(figsize=(12, 24))
    gs = gridspec.GridSpec(len(group_orders), 1, height_ratios=group_sizes)

    bar_height = 3
    for i_group, group in enumerate(group_orders):
        this_markers = groups_to_use[group]
        ax = plt.subplot(gs[i_group])
        _this_markers = ['nice/marker/{}'.format(m) for m in this_markers]

        idx = np.concatenate(
            [np.where(probas['Marker'].values == m)[0].ravel()
             for m in reversed(_this_markers)])
        this_probas = probas.iloc[np.array(idx)]
        ys = np.arange(len(this_probas)) * bar_height
        bottoms = np.c_[ys, ys].ravel()
        xs = np.c_[this_probas['VS/UWS'].values,
                   this_probas['MCS'].values].ravel()
        lefts = np.c_[
            np.zeros(len(this_probas)), this_probas['VS/UWS'].values].ravel()
        bars = ax.bar(
            x=lefts, height=3, width=xs, bottom=bottoms, edgecolor='0.2',
            color=['r', 'b'], alpha=0.7, orientation='horizontal')
        ax.set_ylim([-bar_height / 2,
                     bar_height * len(this_probas) - bar_height / 2])

        labels = []
        for m, r in zip(this_probas['Marker'], this_probas['Reduction']):
            r_label = vizutils._map_function_to_text(r.split('/')[-1])
            if 'gfp' in r.split('/')[-2]:
                r_label = r'{}(GFP)'.format(r_label)
            m_label = _icm_text_maps['/'.join(m.split('/')[-2:])]
            labels.append('{} ({})'.format(m_label, r_label))

        ax.set_yticks(range(0, bar_height * len(this_probas), bar_height))
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.1))
        ax.axvline(0.5, color='k', linestyle='--', linewidth=1)
        ax.axvline(0.25, color='k', linestyle='--', linewidth=1)
        ax.axvline(0.75, color='k', linestyle='--', linewidth=1)
        ax.set_title(group)
        if i_group == 0:
            ax.legend(bars, ['VS/UWS', 'MCS'], loc='upper right',
                      bbox_to_anchor=(1.13, 1.1))
    plt.subplots_adjust(bottom=0.05, top=0.92)

    plt.suptitle('Probability of being VS/UWS vs MCS')

    report.add_figs_to_section(
        fig_bars, captions='Summary II', section='Univariate Summaries')

    return report
