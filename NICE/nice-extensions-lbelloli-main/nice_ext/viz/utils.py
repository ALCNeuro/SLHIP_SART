# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import numpy as np
import math
from collections import Counter, OrderedDict
from packaging import version

import mne
from mne.stats import spatio_temporal_cluster_1samp_test
from ..equipments import prepare_layout

_html_maps = {
    'PermutationEntropy/default': r'PE &Theta;',
    'PermutationEntropy/delta': r'PE &Delta;',
    'PermutationEntropy/theta': r'PE &Theta;',
    'PermutationEntropy/alpha': r'PE &Alpha;',
    'PermutationEntropy/beta': r'PE &Beta;',
    'PermutationEntropy/gamma': r'PE &Gamma;',
    'PowerSpectralDensity/delta': r'&delta;',
    'PowerSpectralDensity/deltan': r'&#x2016;&delta;&#x2016;',
    'PowerSpectralDensity/theta': r'&theta;',
    'PowerSpectralDensity/thetan': r'&#x2016;&theta;&#x2016;',
    'PowerSpectralDensity/alpha': r'&alpha;',
    'PowerSpectralDensity/alphan': r'&#x2016;&alpha;&#x2016;',
    'PowerSpectralDensity/beta': r'&beta;',
    'PowerSpectralDensity/betan': r'&#x2016;&beta;&#x2016;',
    'PowerSpectralDensity/gamma': r'&gamma;',
    'PowerSpectralDensity/gamman': r'&#x2016;&gamma;&#x2016;',
    'PowerSpectralDensity/highgamma': r'&Hgamma;',
    'PowerSpectralDensity/highgamman': r'&#x2016;H&gamma;&#x2016;',
    'SymbolicMutualInformation/default': r'SMI &Theta;',
    'SymbolicMutualInformation/delta': r'SMI &Delta;',
    'SymbolicMutualInformation/theta': r'SMI &Theta;',
    'SymbolicMutualInformation/alpha': r'SMI &Alpha;',
    'SymbolicMutualInformation/beta': r'SMI &Beta;',
    'SymbolicMutualInformation/gamma': r'SMI &Gamma;',
    'SymbolicMutualInformation/weighted': r'wSMI &Theta;',
    'SymbolicMutualInformation/delta_weighted': r'wSMI &Delta;',
    'SymbolicMutualInformation/theta_weighted': r'wSMI &Theta;',
    'SymbolicMutualInformation/alpha_weighted': r'wSMI &Alpha;',
    'SymbolicMutualInformation/beta_weighted': r'wSMI &Beta;',
    'SymbolicMutualInformation/gamma_weighted': r'wSMI &Gamma;',
    'ContingentNegativeVariation/default': r'CNV;',
    'PowerSpectralDensitySummary/summary_msf': r'MSF',
    'PowerSpectralDensity/summary_se': r'SE',
    'PowerSpectralDensitySummary/summary_sef90': r'SE90',
    'PowerSpectralDensitySummary/summary_sef95': r'SE95'
}

_text_maps = {
    'PermutationEntropy/default': r'PE $\theta$',
    'PermutationEntropy/delta': r'PE $\delta$',
    'PermutationEntropy/theta': r'PE $\theta$',
    'PermutationEntropy/alpha': r'PE $\alpha$',
    'PermutationEntropy/beta': r'PE $\beta$',
    'PermutationEntropy/gamma': r'PE $\gamma$',
    'PowerSpectralDensity/delta': r'$\delta$',
    'PowerSpectralDensity/deltan': r'$\|\delta\|$',
    'PowerSpectralDensity/theta': r'$\theta$',
    'PowerSpectralDensity/thetan': r'$\|\theta\|$',
    'PowerSpectralDensity/alpha': r'$\alpha$',
    'PowerSpectralDensity/alphan': r'$\|\alpha\|$',
    'PowerSpectralDensity/beta': r'$\beta$',
    'PowerSpectralDensity/betan': r'$\|\beta\|$',
    'PowerSpectralDensity/gamma': r'$\gamma$',
    'PowerSpectralDensity/gamman': r'$\|\gamma\|$',
    'PowerSpectralDensity/highgamma': r'$H\gamma$',
    'PowerSpectralDensity/highgamman': r'$\|H\gamma\|$',
    'SymbolicMutualInformation/default': r'SMI $\theta$',
    'SymbolicMutualInformation/delta': r'SMI $\delta$',
    'SymbolicMutualInformation/theta': r'SMI $\theta$',
    'SymbolicMutualInformation/alpha': r'SMI $\alpha$',
    'SymbolicMutualInformation/beta': r'SMI $\beta$',
    'SymbolicMutualInformation/gamma': r'SMI $\gamma$',
    'SymbolicMutualInformation/weighted': r'wSMI $\theta$',
    'SymbolicMutualInformation/delta_weighted': r'wSMI $\delta$',
    'SymbolicMutualInformation/theta_weighted': r'wSMI $\theta$',
    'SymbolicMutualInformation/alpha_weighted': r'wSMI $\alpha$',
    'SymbolicMutualInformation/beta_weighted': r'wSMI $\beta$',
    'SymbolicMutualInformation/gamma_weighted': r'wSMI $\gamma$',
    'ContingentNegativeVariation/default': r'CNV',
    'KolmogorovComplexity/default': r'K',
    'PowerSpectralDensitySummary/summary_msf': r'MSF',
    'PowerSpectralDensity/summary_se': r'SE',
    'PowerSpectralDensitySummary/summary_sef90': r'SE90',
    'PowerSpectralDensitySummary/summary_sef95': r'SE95',
    'TimeLockedContrast/p3b': r'P3b',
    'TimeLockedContrast/mmn': r'MMN'
}

_unit_maps = {
    'PermutationEntropy/default': r'$bits$',
    'PermutationEntropy/theta': r'$bits$',
    'PermutationEntropy/alpha': r'$bits$',
    'PermutationEntropy/beta': r'$bits$',
    'PermutationEntropy/gamma': r'$bits$',
    'PowerSpectralDensity/alpha': r'dB/Hz',
    'PowerSpectralDensity/alphan': r'',
    'PowerSpectralDensity/beta': r'dB/Hz',
    'PowerSpectralDensity/betan': r'',
    'PowerSpectralDensity/delta': r'dB/Hz',
    'PowerSpectralDensity/deltan': r'',
    'PowerSpectralDensity/gamma': r'dB/Hz',
    'PowerSpectralDensity/gamman': r'',
    'PowerSpectralDensity/theta': r'dB/Hz',
    'PowerSpectralDensity/thetan': r'',
    'SymbolicMutualInformation/default': r'',
    'SymbolicMutualInformation/theta': r'',
    'SymbolicMutualInformation/alpha': r'',
    'SymbolicMutualInformation/beta': r'',
    'SymbolicMutualInformation/gamma': r'',
    'SymbolicMutualInformation/weighted': r'',
    'SymbolicMutualInformation/theta_weighted': r'',
    'SymbolicMutualInformation/alpha_weighted': r'',
    'SymbolicMutualInformation/beta_weighted': r'',
    'SymbolicMutualInformation/gamma_weighted': r'',
    'ContingentNegativeVariation/default': r'$mV/s$',
    'KolmogorovComplexity/default': r'$bits$',
    'PowerSpectralDensitySummary/summary_msf': r'Hz',
    'PowerSpectralDensity/summary_se': r'$bits$',
    'PowerSpectralDensitySummary/summary_sef90': r'Hz',
    'PowerSpectralDensitySummary/summary_sef95': r'Hz',
    'TimeLockedContrast/p3b': r'$\mu{V}$',
    'TimeLockedContrast/mmn': r'$\mu{V}$',
}

_function_text_maps = {
    'trim_mean80': r'$\mu_{80}$',
    'trim_mean90': r'$\mu_{90}$',
    'mean': r'$\mu$',
    'std': r'$\sigma$',
}


def _map_function_to_text(func_name):
    return _function_text_maps[func_name]


def _map_key_to(name, to, marker=None):
    if to == 'html':
        _map = _html_maps
    elif to == 'text':
        _map = _text_maps
    elif to == 'unit':
        _map = _unit_maps
    else:
        raise ValueError('I do not know how to map to {}'.format(to))
    key_split = name.split('/')
    key = '/'.join(key_split[2:])
    if key in _map:
        text = _map[key]
    elif key_split[-1] == 'default':
        text = key_split[-2]
    else:
        text = key_split[-1]
    if callable(text):
        text = text(key)
    return text


def _map_key_to_html(name, marker=None):
    return _map_key_to(name, to='html', marker=marker)


def _map_key_to_text(name, marker=None):
    return _map_key_to(name, to='text', marker=marker)


def _map_key_to_unit(name, marker=None):
    return _map_key_to(name, to='unit', marker=marker)


def _map_key_to_section(name):
    if 'PowerSpectralDensity' in name:
        section = 'Spectral'
    elif 'SymbolicMutualInformation' in name:
        section = 'Connectivity'
    elif 'KolmogorovComplexity' in name:
        section = 'Information Theory'
    elif 'PermutationEntropy' in name:
        section = 'Information Theory'
    elif 'ContingentNegativeVariation' in name:
        section = 'Time Locked'
    elif 'TimeLockedTopography' in name:
        section = 'Time Locked'
    elif 'TimeLockedContrast' in name:
        section = 'Time Locked'
    else:
        raise NotImplementedError('Sorry, no default section for marker {}'
                                  .format(name))
    return section


def _map_marker_to_html(marker):
    name = marker._get_title()
    text = _map_key_to_html(name, marker)
    return text


def _map_marker_to_text(marker):
    name = marker._get_title()
    text = _map_key_to_text(name, marker)
    return text


def _map_marker_to_unit(marker):
    name = marker._get_title()
    text = _map_key_to_unit(name, marker)
    return text


def _map_marker_to_section(marker):
    section = None
    name = marker._get_title()
    section = _map_key_to_section(name)
    return section


def get_log_topomap(xsig, vmin, vmax):
    from mne.utils import logger
    logger.warning('This will be removed soon! Wrong name. '
                   'Call get_stat_colormap() instead')
    return get_stat_colormap(xsig, vmin, vmax)


def get_stat_colormap(xsig, vmin, vmax):
    from matplotlib.colors import LinearSegmentedColormap
    x = xsig / (vmax - vmin)
    blue = ((0.0, 1.0, 1.0), (x, 0.5, 0.0), (1.0, 0.0, 0.0))
    red = ((0.0, 1.0, 1.0), (x, 0.5, 1.0), (1.0, 1.0, 1.0))
    green = ((0.0, 1.0, 1.0), (x, 0.5, 1.0), (1.0, 0.0, 0.0))
    cdict = dict(red=red, green=green, blue=blue)
    logcolor = LinearSegmentedColormap('logcolor', cdict)

    return logcolor


def plot_bad_channels(epochs, bads, outlines='head', axes=None,
                      sns_kwargs=None):
    """Plot bad channels

    Parameters
    ----------
    evoked : instance of Evoked
        The evoked data.
    Returns
    -------
    fig_image : instance of matplotlib.figure.Figure
        The image plot of the bads
    fig_bad_topo :
        The topo plot of bad channels.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_style('white')
    sns.set_color_codes()
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    evoked = epochs.average()
    good_idx = [i for i, ch in enumerate(evoked.ch_names)
                if ch not in bads]
    vmin, vmax = (evoked.data[good_idx].min() * 1e6,
                  evoked.data[good_idx].max() * 1e6)
    clim = dict(eeg=[vmin, vmax])
    fig_all, axes = plt.subplots(1, 1, figsize=(16, 8))
    plt.title('Bad channels')
    fig_image = evoked.plot_image(
        picks='eeg', exclude=[], cmap='gray', clim=clim, axes=axes)
    times = evoked.times * 1e3
    has_bads = False
    im = None
    im2 = None
    data = None
    for ax in fig_image.axes:
        if 1 <= len(ax.images) <= 2:
            im = ax.images[0]
            # im.colorbar.set_label('regular')
            data = np.array(im.get_array())
            bad_idx = {evoked.ch_names.index(k) for k in bads
                       if k in evoked.ch_names}
            if len(bad_idx) > 0:
                has_bads = True
                mask = np.ones(data.shape, dtype=bool)
                mask[list(bad_idx)] = False
                data = np.ma.masked_array(data, mask)
                vmin_bad, vmax_bad = data.min(), data.max()
                im2 = ax.imshow(data, cmap='RdYlBu_r', interpolation='nearest',
                                origin='lower',
                                extent=[times[0], times[-1], 0, data.shape[0]],
                                aspect='auto', vmin=vmin_bad, vmax=vmax_bad)
            ax.set_xlabel('time (s)')
        elif len(ax.images) > 2:
            raise RuntimeError('More than two images found. Currently only '
                               'supported for one channel type at a time')
    # import pdb; pdb.set_trace()
    # fig_image.delaxes(fig_image.axes[1])
    fig_image.delaxes(fig_image.axes[1])
    divider = make_axes_locatable(fig_image.axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('regular', labelpad=-40, y=-0.05)
    cbar.ax.set_title(r'$\mu{V}$')
    if has_bads is True:
        cax = divider.append_axes("right", size="5%", pad=0.5)
        cbar = plt.colorbar(im2, cax=cax)
        cbar.set_label('bads', labelpad=-43, y=-0.05)
        cbar.ax.set_title(r'$\mu{V}$')
    # fig_image.canvas.draw()

    new_axes = divider.append_axes("right", size="100%", pad=0.5)

    bad_idx = {evoked.ch_names.index(k) for k in bads
               if k in evoked.ch_names}
    mask = np.zeros(data.shape, dtype=bool)
    mask[list(bad_idx)] = True

    sphere, outlines = prepare_layout(
        evoked.info.get('description', outlines),
        info=evoked.info)
    mne.viz.plot_topomap(np.ones((len(evoked.data))), pos=evoked.info,
                         mask=mask[:, 0], contours=0, cmap='gray',
                         names=evoked.ch_names, sensors=False, axes=new_axes,
                         outlines=outlines, show_names=lambda x: ' ' * 10 + x,
                         mask_params={'markeredgecolor': 'red', 'marker': 'x',
                                      'markersize': 7},
                         extrapolate='local', sphere=None)
    fig_all.subplots_adjust(wspace=0, right=1)
    fig_all.tight_layout()
    return fig_all

def plot_channels(epochs,
                  outlines='head',
                  show_names=False,
                  axes=None,
                  sns_kwargs=None):
    """Plot removed channels to subsample the number of used channels

    Parameters
    ----------
    epochs : instance of Epochs
        The epoched data.
    Returns
    -------
    fig_image : instance of matplotlib.figure.Figure
        The image plot of the bads
    fig_bad_topo :
        The topo plot of bad channels.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_style('white')
    sns.set_color_codes()

    total_n_channels = len(epochs.ch_names)

    # Plot epochs evoked to show an overview of the channel values
    evoked = epochs.average()
    vmin, vmax = (evoked.data.min() * 1e6, evoked.data.max() * 1e6)
    clim = dict(eeg=[vmin, vmax])
    fig_all, axes = plt.subplots(1, 2, figsize=(16, 8))
    evoked.plot_image(
        cmap='gray',
        clim=clim,
        axes=axes[0]
    )

    # Plot the electrode positions and show the selected ones as MCP subsamples channels
    sphere, outlines = prepare_layout(
        evoked.info.get('description', outlines),
        info=evoked.info
    )
    mne.viz.plot_topomap(
        np.ones(total_n_channels),
        pos=epochs.info,
        contours=0,
        cmap='gray',
        sensors=True,
        axes=axes[1],
        outlines=outlines,
        names=epochs.ch_names,
        show_names=show_names,
        mask=np.array([True for _ in epochs.ch_names]) if show_names else None,
        mask_params={'marker': ' '},
        extrapolate='local',
        sphere=None
    )

    # Adjust fogures
    fig_all.subplots_adjust(wspace=0, right=1)
    fig_all.tight_layout()

    return fig_all


def plot_events(epochs):
    fig = mne.viz.plot_events(epochs.events, epochs.info['sfreq'],
                              event_id=epochs.event_id)

    # Reverse labels
    ax = fig.get_axes()[0]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              loc='center left', bbox_to_anchor=(1, 0.5))
    import matplotlib as mpl
    for legend in fig.findobj(mpl.legend.Legend):
        for text in legend.texts:
            text.set_fontsize(12)


def render_bad_epochs(epochs, sns_kwargs=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_style('white')
    sns.set_color_codes()
    # import matplotlib.cm as cm
    table = _render_drop_log_table(epochs)
    ignore = 'IGNORED'
    drop_log = [ep for ep in epochs.drop_log if ignore not in ep]
    n_orig_epochs = len(drop_log)
    n_cols = 50
    n_rows = int(np.ceil(n_orig_epochs / n_cols))
    mask = np.zeros((n_rows, n_cols), dtype=np.bool)
    mask.ravel()[np.where(drop_log)[0]] = True

    masked_data = np.ma.masked_array(mask)
    last = n_orig_epochs % n_cols
    masked_data[:, :] = np.ma.masked
    masked_data[n_rows - 1, last:] = np.ma.nomask

    fig_bad_epochs, ax = plt.subplots()
    cmap = sns.light_palette('red', as_cmap=True)
    ax.imshow(mask, interpolation='nearest', cmap=cmap, alpha=1.0)
    ax.imshow(masked_data, interpolation='nearest', cmap='gray_r')
    ax.xaxis.tick_top()
    ax.set_yticklabels(['{}'.format(int(x)) for x in ax.get_yticks() * n_cols])

    if version.parse(mne.__version__) < version.parse('0.24'):
        html_image = mne.report._fig_to_img(fig=fig_bad_epochs)
    else:
        html_image = mne.report.report._fig_to_img(fig=fig_bad_epochs)

    html = '''
    <div style="display: table; width: 100%;">
        <div class="left" style="float:left; width:50%; display: table-row;">
            <div class="thumbnail" style="border: none;">
                <img alt="" src="data:image/png;base64,{}">
            </div>
        </div>
        <div class="right" style="float:right;width:50%;">{}</div>
    </div>
    '''.format(html_image, table)

    return html


def render_autoreject(epochs, summary, sns_kwargs=None):
    autoreject_summary = summary['autoreject']
    import matplotlib.pyplot as plt
    import seaborn as sns
    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_style('white')
    sns.set_color_codes()
    # import matplotlib.cm as cm
    table = _render_drop_log_table(epochs)

    # bad_sensor_counts = autoreject_summary.bad_sensors_counts()
    bad_sensor_counts = np.sum(
        np.logical_and(autoreject_summary.labels != 0,
                       ~np.isnan(autoreject_summary.labels)), axis=1)

    bad_epochs = autoreject_summary.bad_epochs

    n_orig_epochs = len(bad_epochs)
    n_cols = 50
    n_rows = int(np.ceil(n_orig_epochs / n_cols))
    chans_interp = bad_sensor_counts

    epochs_matrix = np.zeros((n_rows, n_cols), dtype=np.float)
    epochs_matrix.ravel()[:len(chans_interp)] = chans_interp

    last = n_orig_epochs % n_cols

    masked_matrix = np.ma.masked_array(epochs_matrix, fill_value=np.nan)
    masked_dropped = None
    if np.any(bad_epochs):
        epochs_matrix_dropped = np.zeros((n_rows, n_cols), dtype=np.bool)
        epochs_matrix_dropped.ravel()[:len(bad_epochs)][bad_epochs] = True
        masked_matrix[epochs_matrix_dropped] = np.ma.masked

        masked_dropped = np.ma.masked_array(epochs_matrix_dropped)
        masked_dropped[~epochs_matrix_dropped] = np.ma.masked
        masked_dropped[n_rows - 1, last:] = np.ma.masked

    masked_matrix[n_rows - 1, last:] = np.ma.masked

    cmap_drop = sns.cubehelix_palette(8, as_cmap=True, reverse=True)
    cmap_inter = sns.cubehelix_palette(8, as_cmap=True)
    fig_bad_epochs, ax = plt.subplots()
    im = ax.imshow(masked_matrix, interpolation='nearest',
                   cmap=cmap_inter, alpha=0.9)
    if np.any(bad_epochs):
        ax.imshow(masked_dropped, interpolation='nearest',
                  cmap=cmap_drop, alpha=0.9)
    cbar = fig_bad_epochs.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_ticks(np.arange(cbar.vmin, cbar.vmax + 1))
    cbar.set_ticklabels(list(np.arange(cbar.vmin, cbar.vmax)) + ['Rejected'])

    ax.xaxis.tick_top()
    ax.set_yticklabels(['{}'.format(int(x)) for x in ax.get_yticks() * n_cols])

    if version.parse(mne.__version__) < version.parse('0.24'):
        html_image = mne.report._fig_to_img(fig=fig_bad_epochs)
    else:
        html_image = mne.report.report._fig_to_img(fig=fig_bad_epochs)
    
    html = '''
    <div style="display: table; width: 100%;">
        <div class="left" style="float:left; width:50%; display: table-row;">
            <div class="thumbnail" style="border: none;">
                <img alt="" src="data:image/png;base64,{}">
            </div>
        </div>
        <div class="right" style="float:right;width:50%;">{}</div>
    </div>
    '''.format(html_image, table)

    return html


def _render_drop_log_table(epochs):
    drop_log = epochs.drop_log
    ignore = ['IGNORED']
    scores = Counter([ep for d in drop_log for ep in d if ep not in ignore])
    reasons = np.array(list(scores.keys()))
    percs = 100 * np.array(list(scores.values()), dtype=float) / len(drop_log)

    css = '''<style type="text/css">
    table.dropped_epochs {
    border: 1px solid black;
    margin-top: 20px;
    margin-bottom: 20px;
    margin-left: auto;
    margin-right: auto;}
    table.dropped_epochs th {
    width: 150px;
    text-align:left;
    padding-top: 10px;
    padding-left: 10px;
    font-size: 14px;}
    table.dropped_epochs td {
    text-align:left;
    padding-top: 10px;
    padding-left: 10px;
    font-size: 12px;}</style>'''

    header = '<tr><th>Reason</th><th># Epochs</th><th>Percentage</th></tr>'
    table_content = ''
    for reason, perc in zip(reasons, percs):
        value = scores[reason]
        table_content += \
            '<tr><td>{}</td><td>{}</td><td>{:.2f} %</td></tr>'.format(
                reason.title(), value, perc
            )

    value = len(drop_log) - np.array(list(scores.values())).sum()
    perc = 100 * value / len(drop_log)
    table_content += \
        '<tr><td>Good</td><td>{}</td><td>{:.2f} %</td></tr>'.format(
            value, perc
        )
    table = '''{}<div><h5 style="text-align:center;">Epochs count</h5></div>
        <table class="dropped_epochs">{}{}</table>'''.format(
        css, header, table_content)
    return table


def _params_to_string(params):
    out = ''
    for k, v in params.items():
        out += '{}: {}<br/>'.format(k, v)
    return out


def render_preprocessing_summary(summary):
    css = '''
    <style type="text/css">
        table.preprocess_summary {
            border: 1px solid black;
            margin-top: 20px;
            margin-bottom: 20px;
            margin-left: auto;
            margin-right: auto;}
            table.preprocess_summary th {
            width: 150px;
            text-align:left;
            padding-top: 10px;
            padding-left: 10px;
            font-size: 14px;}
            table.preprocess_summary td {
            text-align:left;
            padding-top: 10px;
            padding-left: 10px;
            font-size: 12px;}
            table.preprocess_summary td.params {
            width: 250px;
        }
    </style>'''

    header = '<tr><th>Step</th><th>Params</th><th>Result</th></tr>'
    table_content = ''

    for step in summary['steps']:
        method = step['step']
        params = _params_to_string(step['params'])
        result = ''
        if 'bad_chs' in step:
            result += f'{len(step["bad_chs"])} bad channels<br/>'
        if 'bad_epochs' in step:
            result += f'{len(step["bad_epochs"])} bad epochs'
        if 'droped_chs' in step:
            result += f'{len(step["droped_chs"])} droped channels<br/>'
        if 'droped_epochs' in step:
            result += f'{len(step["droped_epochs"])} droped epochs'

        table_content += (
            f'<tr><td>{method}</td>'
            f'<td class="params">{params}</td>'
            f'<td>{result}</td></tr>'
        )

    table = (
        f'{css}'
        '<table class="preprocess_summary">'
            f'{header}{table_content}'
        '</table>'
    )
    return table


def render_prediction_summary(summary):
    css = '''
    <style type="text/css">
        table.prediction_summary {
            border: 1px solid black;
            margin-top: 20px;
            margin-bottom: 20px;
            margin-left: auto;
            margin-right: auto;}
            table.prediction_summary th {
            width: 150px;
            text-align:left;
            padding-top: 10px;
            padding-left: 10px;
            font-size: 14px;}
            table.prediction_summary td {
            text-align:left;
            padding-top: 10px;
            padding-left: 10px;
            font-size: 12px;}
            table.prediction_summary td.params {
            width: 250px;
            }
            table.prediction_summary td.results {
            width: 350px;
        }
    </style>'''

    header = '<tr><th>Step</th><th>Params</th><th>Result</th></tr>'
    table_content = ''

    for step, params, result in zip(summary['step'],
                                    summary['params'],
                                    summary['result']):
        
        str_params = _params_to_string(params)
        str_results = '<br/>'.join(result)
        table_content += (
            f'<tr><td>{step}</td>'
            f'<td class="params">{str_params}</td>'
            f'<td class="results">{str_results}</td></tr>'
        )

    table = (
        f'{css}'
        '<table class="prediction_summary">'
            f'{header}{table_content}'
        '</table>'
    )
    return table


def plot_generalization_decoding(instances, shift_time=0, titles=None,
                                 axes=None, fig_kwargs=None, n_cols=1,
                                 vrange=None,
                                 sig_masks=False, sig_type='contour',
                                 sns_kwargs=None):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns
    if sns_kwargs is None:
        sns_kwargs = {}
    sns.set(**sns_kwargs)
    sns.set_color_codes()
    if not isinstance(instances, dict):
        if not isinstance(instances, list):
            instances = [instances]
        new_i = OrderedDict()
        for i in instances:
            new_i[i._get_title()] = i
        instances = new_i
    fig = None
    if fig_kwargs is None:
        fig_kwargs = {}
    if axes is None:
        n_rows = math.ceil(len(instances) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, **fig_kwargs)
        if len(instances) == 1:
            axes = np.array([axes])
    if titles is None:
        titles = list(instances.keys())

    compute_sig_mask = False
    if sig_masks is not False:
        if isinstance(sig_masks, str):
            if sig_masks == 'compute':
                compute_sig_mask = True
                sig_masks = [None] * len(instances)
    t_axes = axes.ravel()[:len(instances)]
    im2 = None
    for t_ax, t_i, t_title, t_sig in zip(
            t_axes, instances.values(), titles, sig_masks):
        t_data = t_i.data_.mean(axis=0)
        if vrange is None:
            vmin = np.min(t_data)
            vmax = np.max(t_data)
            absmax = max(vmax - 0.5, 0.5 - vmin)
            vmax = 0.5 + absmax
            vmin = 0.5 - absmax
        else:
            vmax = 0.5 + vrange
            vmin = 0.5 - vrange

        im = t_ax.imshow(t_data, vmin=vmin, vmax=vmax,
                         cmap='RdBu_r', origin='lower',
                         extent=[t_i.tmin, t_i.tmax, t_i.tmin, t_i.tmax])
        if compute_sig_mask is True:
            X = t_i.data_ - 0.5
            T_obs_, clusters, p_values, _ = spatio_temporal_cluster_1samp_test(
                X, out_type='mask', n_permutations=2**12, n_jobs=-1,
                verbose=True)
            p_values_ = np.ones_like(X[0]).T
            for cluster, pval in zip(clusters, p_values):
                p_values_[cluster.T] = pval
            t_sig = np.squeeze(p_values_).T < 0.05
        if t_sig is not None:
            if sig_type == 'contour':
                _interval = (t_i.tmax - t_i.tmin) / t_i.shape_[1]
                times = np.arange(t_i.tmin, t_i.tmax, _interval)
                xx, yy = np.meshgrid(times, times, copy=False, indexing='xy')
                t_ax.contour(xx, yy, t_sig, colors='.1', levels=[0],
                             linestyles='dotted', linewidths=0.7)
            elif sig_type == 'color':
                C = [(0, 0, 0), (1.0, 1.0, 1.0), (0, 0, 0)]
                cm = mpl.colors.LinearSegmentedColormap.from_list(
                    colors=C, name='bwb')
                im2 = t_ax.imshow(
                    np.ma.masked_array(t_data, mask=t_sig),
                    vmin=vmin, vmax=vmax,
                    cmap=cm, origin='lower',
                    extent=[t_i.tmin, t_i.tmax, t_i.tmin, t_i.tmax])
        t_ax.axhline(0., color='.1', lw=0.7)
        t_ax.axvline(0., color='.1', lw=0.7)
        t_ax.xaxis.set_ticks_position('bottom')
        t_ax.set_xlabel('Testing Time (s)')
        t_ax.set_ylabel('Training Time (s)')
        t_ax.set_title(t_title)
        t_ax.plot(np.array([t_i.tmin, t_i.tmax]),
                  np.array([t_i.tmin, t_i.tmax]),
                  color='.1', ls='--', lw=0.7)
        cbar = plt.colorbar(im, ax=t_ax, pad=-0.1, shrink=1, aspect=30,
                            format='%.2f')
        cbar.set_label('AUC')
        if t_sig is not None and sig_type == 'color':
            cbar2 = plt.colorbar(im2, ax=t_ax, shrink=1, aspect=30,
                                 pad=0.02, ticks=[], format='%.2f')
            cbar2.set_ticks([])
            cbar2.set_label('')

    return fig, axes
