from collections import OrderedDict

from mne.utils import logger

from nice_ext.api.modules import register_module
from nice_ext.api import get_reductions
from nice_ext import viz
from nice_ext.viz import utils as vizutils


def register():
    register_module('report', 'icm/rs', _create_rs_report)
    register_module('report', 'icm/preprocess', _create_preprocessing_report)


def _create_rs_report(instance, report, config_params):
    import matplotlib.pyplot as plt
    epochs = None

    # Some config parameters
    # stat_psig = 0.05
    # stat_logpsig = -np.log10(stat_psig)
    # stat_pvmin = 1
    # stat_pvmax = 1e-5
    # stat_vmin = -np.log10(stat_pvmin)
    # stat_vmax = -np.log10(stat_pvmax)
    # # stat_cmap = vizutils.get_log_topomap(stat_logpsig, stat_vmin,
    #                                      stat_vmax)

    outlines = 'head'
    if 'epochs' in config_params:
        epochs = config_params['epochs']
        outlines = epochs.info['description']

    if 'summary' in config_params and 'epochs' in config_params:
        _create_preprocessing_report(epochs, report=report,
                                     config_params=config_params)

    reduction_config = config_params.get(
        'reduction_rs', 'icm/rs/egi256/trim_mean80')
    reductions = get_reductions(config=reduction_config)

    # Summaries
    summaries_n = ['nice/marker/PowerSpectralDensity/summary_se',
                   'nice/marker/PowerSpectralDensitySummary/summary_msf',
                   'nice/marker/PowerSpectralDensitySummary/summary_sef90',
                   'nice/marker/PowerSpectralDensitySummary/summary_sef95']

    s_reductions = {
        'nice/marker/PowerSpectralDensity/summary_se':
            reductions['PowerSpectralDensity/summary_se']['reduction_func'],
        'nice/marker/PowerSpectralDensitySummary/summary_msf':
            reductions['PowerSpectralDensitySummary']['reduction_func'],
        'nice/marker/PowerSpectralDensitySummary/summary_sef90':
            reductions['PowerSpectralDensitySummary']['reduction_func'],
        'nice/marker/PowerSpectralDensitySummary/summary_sef95':
            reductions['PowerSpectralDensitySummary']['reduction_func']
    }

    s_picks = {
        'nice/marker/PowerSpectralDensity/summary_se':
            reductions['PowerSpectralDensity/summary_se']['picks'],
        'nice/marker/PowerSpectralDensitySummary/summary_msf':
            reductions['PowerSpectralDensitySummary']['picks'],
        'nice/marker/PowerSpectralDensitySummary/summary_sef90':
            reductions['PowerSpectralDensitySummary']['picks'],
        'nice/marker/PowerSpectralDensitySummary/summary_sef95':
            reductions['PowerSpectralDensitySummary']['picks']
    }

    summaries = OrderedDict()
    for k in summaries_n:
        summaries[k] = instance[k]

    fig_summaries = viz.plot_markers_topos(
        summaries, s_reductions, s_picks, outlines=outlines)

    report.add_figs_to_section(
        figs=[fig_summaries], captions=['Spectral Summaries'],
        section='Spectral')
    plt.close(fig_summaries)

    # Normalised
    normalised_n = ['nice/marker/PowerSpectralDensity/deltan',
                    'nice/marker/PowerSpectralDensity/thetan',
                    'nice/marker/PowerSpectralDensity/alphan',
                    'nice/marker/PowerSpectralDensity/betan',
                    'nice/marker/PowerSpectralDensity/gamman']

    s_reductions = {k: reductions['PowerSpectralDensity']['reduction_func']
                    for k in normalised_n}
    s_picks = {k: reductions['PowerSpectralDensity']['picks']
               for k in normalised_n}

    normalised = OrderedDict()
    for k in normalised_n:
        normalised[k] = instance[k]

    fig_normalised = viz.plot_markers_topos(
        normalised, s_reductions, s_picks, outlines=outlines)

    report.add_figs_to_section(
        figs=[fig_normalised], captions=['Spectral Power (normalised)'],
        section='Spectral')
    plt.close(fig_normalised)

    # Non normalised
    non_normalised_n = ['nice/marker/PowerSpectralDensity/delta',
                        'nice/marker/PowerSpectralDensity/theta',
                        'nice/marker/PowerSpectralDensity/alpha',
                        'nice/marker/PowerSpectralDensity/beta',
                        'nice/marker/PowerSpectralDensity/gamma']

    s_reductions = {k: reductions['PowerSpectralDensity']['reduction_func']
                    for k in non_normalised_n}
    s_picks = {k: reductions['PowerSpectralDensity']['picks']
               for k in non_normalised_n}

    non_normalised = OrderedDict()
    for k in non_normalised_n:
        non_normalised[k] = instance[k]

    fig_non_normalised = viz.plot_markers_topos(
        non_normalised, s_reductions, s_picks, outlines=outlines)

    report.add_figs_to_section(
        figs=[fig_non_normalised], captions=['Spectral Power (log)'],
        section='Spectral')
    plt.close(fig_non_normalised)

    smis = [(k, v) for k, v in instance.items()
            if 'SymbolicMutualInformation' in k]
    smis = OrderedDict(smis)
    s_reductions = {
        k: reductions['SymbolicMutualInformation']['reduction_func']
        for k in smis.keys()}
    s_picks = {k: reductions['SymbolicMutualInformation']['picks']
               for k in smis.keys()}
    fig_smis = viz.plot_markers_topos(
        smis, s_reductions, s_picks, outlines=outlines)

    report.add_figs_to_section(
        figs=[fig_smis], captions=['Mutual Information'],
        section='Connectivity')
    plt.close(fig_smis)

    pes = [(k, v) for k, v in instance.items() if 'PermutationEntropy' in k]
    pes = OrderedDict(pes)
    s_reductions = {k: reductions['PermutationEntropy']['reduction_func']
                    for k in pes.keys()}
    s_picks = {k: reductions['PermutationEntropy']['picks']
               for k in pes.keys()}
    fig_pes = viz.plot_markers_topos(
        pes, s_reductions, s_picks, outlines=outlines)

    report.add_figs_to_section(
        figs=[fig_pes], captions=['Permutation Entropy'],
        section='Information Theory')
    plt.close(fig_pes)

    komp = instance['nice/marker/KolmogorovComplexity/default']
    reduction = reductions['KolmogorovComplexity']['reduction_func']
    picks = reductions['KolmogorovComplexity']['picks']
    fig_komp = viz.plot_marker_topo(
        komp, reduction, picks, outlines=outlines)
    report.add_figs_to_section(
        figs=[fig_komp], captions=['Kolmogorov-Chaitin Complexity'],
        section='Information Theory')
    plt.close(fig_komp)

    return report


def _create_preprocessing_report(instance, report, config_params):
    import matplotlib.pyplot as plt
    if 'summary' not in config_params:
        raise ValueError('Summary needed to create preprocessing report')
    summary = config_params['summary']
    logger.info('Creating preprocessing report')
    if 'bad_channels' in summary:
        fig_bad_channels = vizutils.plot_bad_channels(
            instance, summary['bad_channels'])

        report.add_figs_to_section(
            [fig_bad_channels], captions=['Bad Channels'],
            section='Preprocessing')
        plt.close(fig_bad_channels)
    if 'bad_epochs' in summary:
        bad_epochs = vizutils.render_bad_epochs(instance)

        report.add_htmls_to_section(
            [bad_epochs], ['Bad Epochs'], section='Preprocessing')
        plt.close(bad_epochs)
    if 'autoreject' in summary:
        bad_autoreject = vizutils.render_autoreject(instance, summary)
        report.add_htmls_to_section(
            [bad_autoreject], ['Bad Epochs'], section='Preprocessing')
        plt.close(bad_autoreject)

    summary_table = vizutils.render_preprocessing_summary(summary)

    report.add_htmls_to_section(
        [summary_table], ['Preprocessing Summary'], section='Preprocessing')

    logger.info('Preprocessing report done')
    return report
