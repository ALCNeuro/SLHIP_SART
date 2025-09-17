#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:42:27 2024

@author: arthurlecoz

SLHIP_config_ALC.py

"""
# %% Paths

rootpath = "/Volumes/DDE_ALC/PhD/SLHIP"

rawDataPath = rootpath + "/00_Raw"
cleanDataPath = rootpath + "/01_Preproc"
powerPath = rootpath + "/03_Power"
wavesPath = rootpath + "/04_Waves"
complexPath = rootpath + "/08_Complexity_Connectivity"
burstPath = rootpath + "/09_Bursts"

config_dict = {
      "file_format": "BrainVision",
      "load_and_preprocess": {
        "referencing" : "average",
        "montage": "standard_1020",
        "l_freq": 0.1,
        "h_freq": 100,
        "notch_freq": [50, 100],
        "f_resample" : 256,
        "channel_types" : {
            'VEOG' : 'eog', 'HEOG' : 'eog', 'ECG' : 'ecg', 'RESP' : 'resp'
            },
        "n_jobs" : -1
      },
      "channel_interpolation": {
        "method": "automatic"
      },
      "ica": {
        "n_components": 15,
        "l_freq": 1.0,
      }
    }

eeg_channels = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 
    'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8',
    'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4',
    'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1',
    'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6',
    'AF8', 'AF4', 'F2', 'Iz']

incomplete_subjects = ["HS_001", "HS_004"]

# %% Functions

    #### Utils

def find_closest_value(value1, arr2):
    """
    Find the closest value comparing a value to an array and returns index 
    and value of the closest value

    Parameters
    ----------
    value1 : float
        Value to find the closest element to.
    arr2 : np.array
        Array in which you want to find a close element.

    Returns
    -------
    index_closest_value : int
        Index of the closest value.
    closest_value : float
        Closest value.

    """
    import numpy as np
    index_closest_value = []
    closest_value = []
    d_array = [
        abs(value1 - value2) for i2, value2 in enumerate(arr2)
               ]
    index_closest_value.append(np.argmin(d_array))
    closest_value.append(arr2[np.argmin(d_array)])
    
    return(index_closest_value, closest_value)

def display_matrices_info(mat_type):
    """
    Parameters
    ----------
    mat_type : str : 'test_res' or 'probe_res'
        Display detail for either part of the matrix

    Returns
    -------
    None.

    """
    
    assert mat_type in ["test_res", "probe_res"], "Careful, mat_type can only be 'test_res' or 'probe_res'"
    if mat_type == "test_res" :
        print('''
        - test_res (matrices des résultats du SART, 1 ligne = 1 essai)
    
        Col 1: block number
        Col 2: block condition (always 2)
        Col 3: image set (always 3)
        Col 4: trial number
        Col 5: digit displayed
        Col 6: nogo digit
        Col 7: response key
        Col 8: stimulus onset (in seconds, PTB time)
        Col 9: duration presentation (in seconds)
        Col 10: response time (in seconds, PTB time)
        Col 11: correctness on nogo trial
        Col 12: correctness on go trial
        ''')
    elif mat_type == 'probe_res' :
        print('''
        - probe_res (résultats des probes, 1 ligne = 1 probe)
    
        Col 1: probe number
        Col 2: probe time  (in seconds, PTB time, theoretical)
        Col 3: probe time  (in seconds, PTB time, actual)
        Col 4: block number
        Col 5: block condition
        Col 6: trial number
        Col 7: Probe Question 1 - Response key
        Col 8: Probe Question 2 - Response key 
        Col 9: Probe Question 3 - Response key 
        Col 10: Probe Question 1 - Response time
        Col 11: Probe Question 2 - Response time 
        Col 12: Probe Question 3 - Response time 
        Col 13: Probe Question 1 - Question time
        Col 14: Probe Question 2 - Question time
        Col 15: Probe Question 3 - Question time
        Col 16: Probe Question 1 - Response value
        Col 17: Probe Question 2 - Response value
        Col 18: Probe Question 3 - Response value
    
        Probe Q1 : Etat d'esprit juste avant l'interruption.
            Ans :   1 - J'étais concentré-e sur la tâche 
                    2 - Je pensais à autre chose
                    3 - Je ne pensais à rien
                    4 - Je ne me souviens pas
    
        Probe Q2 : Sur quoi étiez-vous concentré-e?
                Sur quelque-chose:
            Ans :   1 - Dans la pièce
                    2 - Une pensee venant de vous
                    3 - Une pensée liée a la tâche
                    4 - Une hallucination dans mon champs de vision
    
        Probe Q3 : Notez votre vigilance
            Ans : 1 - 9 
        ''')

    #### Pipeline
    
import mne

def load_and_preprocess_data(file_path):
    """
    Load and minimally preprocess raw EEG data based on the settings in this script (config).

    Parameters
    ----------
    file_path : str
        Path to the data file.

    Returns
    -------
    raw : mne.io.Raw
        Minimally processed Raw object from the path.
    """
    # Load configuration
    
    file_format = config_dict['file_format']
    settings = config_dict['load_and_preprocess']
    supported_formats = ['BrainVision', 'EDF', 'FIF']
    channel_types = settings["channel_types"]
    
    # Ensure the file format is supported
    assert file_format in supported_formats, f"File format {file_format} not supported."

    # Load the raw data based on the file format
    if file_format == "BrainVision":
        raw = mne.io.read_raw_brainvision(file_path, preload=True)
    elif file_format == "EDF":
        raw = mne.io.read_raw_edf(file_path, preload=True)
    elif file_format == "FIF":
        raw = mne.io.read_fif(file_path, preload=True)
    
    # Apply preprocessing steps
    raw.set_channel_types(channel_types)
    mne.set_eeg_reference(raw, settings['referencing'], copy = False)
    raw.set_montage(
        mne.channels.make_standard_montage(settings['montage']), 
        on_missing='ignore'
        )
    raw.filter(
        settings['l_freq'], 
        settings['h_freq'], 
        fir_design='firwin',
        picks = ["eeg", "eog"],
        n_jobs = settings["n_jobs"]
        )
    raw.notch_filter(
        settings['notch_freq'], 
        filter_length='auto', 
        phase='zero', 
        n_jobs = settings["n_jobs"]
        )
    raw.resample(settings['f_resample'])
    
    # Return the preprocessed data
    return raw

def handle_events(raw, merge_dict=None):
    """
    Extract and handle events from raw EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The preprocessed raw data.
    merge_dict : dict, optional
        Dictionary with keys as new event IDs and values as lists of event IDs to be merged. 
        The default is None.

    Returns
    -------
    events : numpy.ndarray
        The events array after handling.
    event_id : dict
        The event dictionary after handling.

    """

    import mne
    # Extract events
    events, event_id = mne.events_from_annotations(raw)
    
    # Merge events if merge_dict is provided
    if merge_dict is not None:
        for new_id, old_ids in merge_dict.items():
            events = mne.merge_events(
                events, 
                old_ids, 
                new_id, 
                replace_events=True
                )

    return events, event_id

def automatic_ica(
        eeg_data, 
        sub_id, 
        output_dir, 
        n_components=15, 
        l_freq=1.0, 
        v_eog_ch=None, 
        h_eog_ch=None, 
        ecg_ch=None, 
        icalabel=False
        ):
    """
    Perform ICA on EEG data and automatically exclude artifacts.
 
    Parameters
    ----------
    eeg_data : mne.io.Raw or mne.Epochs
        Instance of Raw or Epochs containing the EEG data.
    sub_id : str
        Subject identifier.
    output_dir : str
        Directory to save the ICA outputs.
    n_components : int, optional
        Number of components for ICA. The default is 15.
    l_freq : float, optional
        High-pass filter cutoff frequency before ICA. The default is 1.0.
    v_eog_ch : str, optional
        Vertical EOG channel name for artifact detection. The default is None.
    h_eog_ch : str, optional
        Horizontal EOG channel name for artifact detection. The default is None.
    ecg_ch : str, optional
        ECG channel name for artifact detection. The default is None.
    icalabel : bool, optional
        Whether to use ICLabel for further classification and exclusion of components. 
        The default is False.
 
    Returns
    -------
    ica : mne.preprocessing.ICA
        ICA object fitted on the EEG data.
    """
    import os
    # import mne
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    # Preprocess: High-pass filter
    assert type(eeg_data) in [mne.io.brainvision.brainvision.RawBrainVision, mne.epochs.Epochs], "The class of the eeg_data is not supported..."
    filt_data = eeg_data.copy().filter(
        l_freq=l_freq, h_freq=None, n_jobs = -1
        )

    # Fit ICA
    
    ica = ICA(
        n_components=n_components, 
        random_state=97,
        method='infomax', 
        fit_params=dict(extended=True)
        )
    ica.fit(filt_data)

    # Automatic EOG/ECG artifact detection and labeling with ICLabel
    if h_eog_ch:
        _, weights_heog=ica.find_bads_eog(filt_data, ch_name = 'HEOG')
        bads_heog = [i for i, weight in enumerate(weights_heog)
                    if weight > weights_heog.mean()+3*weights_heog.std()]
        for i in bads_heog:
            ica.exclude.append(i)
    if v_eog_ch:
        _, weights_veog=ica.find_bads_eog(filt_data, ch_name = 'VEOG')
        bads_veog = [i for i, weight in enumerate(weights_veog)
                    if weight > weights_veog.mean()+1*weights_veog.std()]
        for i in bads_veog:
            ica.exclude.append(i)
    if ecg_ch:
        _, weights_ecg=ica.find_bads_ecg(filt_data, "ECG")
        bads_ecg = [i for i, weight in enumerate(weights_ecg)
                    if weight > weights_ecg.mean()+4*weights_ecg.std()
                    and weight > .1]
        for i in bads_ecg:
            ica.exclude.append(i)
    
    if icalabel or not any([h_eog_ch, v_eog_ch, ecg_ch]):
    # ICLabel for further classification and exclusion
        ica_labels = label_components(filt_data, ica, method='iclabel')
        for idx, (label, score) in enumerate(
                zip(ica_labels['labels'], ica_labels['y_pred_proba'])):
            if label not in ['brain', 'other']:
                ica.exclude.append(idx)

    # Save ICA object and exclusion info
    if output_dir : 
        ica.save(os.path.join(output_dir, f'{sub_id}-ica.fif'), overwrite = True)
        with open(os.path.join(output_dir, f'{sub_id}-ica-exclude.txt'), 'w') as f:
            f.write('\n'.join(map(str, ica.exclude)))
    return ica


#### Cluster Permutations



def prepare_neighbours_from_layout(info, ch_type='eeg'):
    """
    Create a neighbours structure similar to FieldTrip's ft_prepare_neighbours
    using Delaunay triangulation.

    Parameters
    ----------
    info : MNE Info object
        contains sensor locations..
    ch_type : str, optional
        Type of channels to consider (e.g., 'eeg'). The default is 'eeg'.

    Returns
    -------
    neighbours_list : A list of dicts.
        With each channel's label and its neighbours.

    """
    from scipy.spatial import Delaunay
    import numpy as np
    # Get the 2D positions of the channels from info
    pos = []
    labels = []
    for ch in info['chs']:
        # Filter by channel type if needed, e.g., check ch['kind'] or use mne.pick_types.
        # Here we simply assume that the info is for the ch_type of interest.
        if 'loc' in ch:
            # Use the first two coordinates from the sensor location as 2D projection
            pos.append(ch['loc'][:2])
            labels.append(ch['ch_name'])
    
    pos = np.array(pos)
    
    # Perform Delaunay triangulation
    tri = Delaunay(pos)
    
    # Create a dictionary where each channel has a set of neighbours
    neighbours = {label: set() for label in labels}
    
    # For each simplex (triangle) in the triangulation, add edges between sensors
    for simplex in tri.simplices:
        for i in range(3):
            ch_i = labels[simplex[i]]
            for j in range(i + 1, 3):
                ch_j = labels[simplex[j]]
                neighbours[ch_i].add(ch_j)
                neighbours[ch_j].add(ch_i)
    
    # Format the neighbours info as a list of dictionaries (similar to FieldTrip's structure)
    neighbours_list = []
    for label in labels:
        neighbours_list.append({
            'label': label,
            'neighblabel': list(neighbours[label])
        })
    
    return neighbours_list

# ============================================================================
# 1. Run mixed-effects model for a given channel (unchanged)
def run_mixedlm(data, channel, interest, model):    
    """
    Run a mixed linear model for a given channel

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing the values to run LME.
    channel : List
        List of channels in the correct order for later plotting.
    interest : str
        Name of effect of interest to collect p and t values.
    model : str
        Models to run LME.

    Returns
    -------
    p_values_oi : list
        p values of interest.
    t_values_oi : list
        t values of interest.
    """
    import statsmodels.formula.api as smf
    import numpy as np
    
    subdf = data.loc[data.channel == channel].dropna()
    md = smf.mixedlm(
        model, subdf, groups=subdf['sub_id']
        )
    try:
        mdf = md.fit(method='lbfgs', reml=False)
        t_values_oi = mdf.tvalues[interest]
        p_values_oi = mdf.pvalues[interest]
    except Exception as e:
        print(f"Model failed for channel {channel}: {e}")
        t_values_oi = np.nan
        p_values_oi = 1
    
    return (p_values_oi, t_values_oi)

# ============================================================================
# 2. Updated helper function: Cluster significant channels using spatial neighbours,
#    ensuring only candidate channels (uncorrected p < clus_alpha) are included.
def cluster_significant_channels(
        channels, 
        pvals, 
        tvals, 
        neighbours, 
        clus_alpha, 
        min_cluster_size, 
        sign='pos'
        ):
    """
    Create clusters of significant channels based solely on candidate channels.
    1) Select candidate channels (p < clus_alpha, correct sign).
    2) Seed one‐channel clusters for each candidate.
    3) Iteratively merge any two clusters if any channel in A is a neighbour
       of any channel in B (using the supplied neighbours map).
    4) Discard clusters smaller than min_cluster_size.

    Parameters
    ----------
    channels : List
        List of channel labels.
    pvals : List
        List of p-values for each channel.
    tvals : List
        List of t-values for each channel.
    neighbours : List of dicts
        List of dicts. Each dict has keys 'label' (channel label)
        and 'neighblabel' (list of neighbouring channel labels).
    clus_alpha : float
        The uncorrected p-value threshold.
    min_cluster_size : float
        Minimum number of channels required for a valid cluster.
    sign : str, optional
        'pos' for positive effects, 'neg' for negative. The default is 'pos'.

    Returns
    -------
    clusters : List
        A list of clusters. Each cluster is a dict with keys:
        'labels'  : a set of channel labels that are candidates and belong to the cluster,
        'tstats'  : a list of t-values for those channels,
        'neighbs' : the union of candidate neighbour labels for all channels in the cluster.
    """
    import numpy as np
    # 1) Build candidate set
    candidate_set = {
        channels[i]
        for i in range(len(channels))
        if (pvals[i] < clus_alpha)
           and ((sign=='pos' and tvals[i]>0) or (sign=='neg' and tvals[i]<0))
           }
    # Quick neighbour lookup: channel -> set of its neighbours (intersected with candidates)
    neigh_map = {
        n['label']: set(n['neighblabel']).intersection(candidate_set)
        for n in neighbours
        if n['label'] in candidate_set
        }
    # 2) Seed initial one‐channel clusters
    clusters = []
    for ch in candidate_set:
        clusters.append({
            'labels': {ch},
            'tstats': [tvals[np.where(np.asarray(channels) == ch)[0][0]]]
        })
    
    # 3) Iteratively merge any two clusters that touch
    merged = True
    while merged:
        merged = False
        new_clusters = []
        used = [False]*len(clusters)
        for i, ci in enumerate(clusters):
            if used[i]:
                continue
            # try to absorb any later cluster that’s adjacent
            for j in range(i+1, len(clusters)):
                if used[j]:
                    continue
                cj = clusters[j]
                # check adjacency: any channel in ci is neighbour of any in cj?
                if any(
                    (label in neigh_map and neigh_map[label] & cj['labels'])
                    for label in ci['labels']
                ) or any(
                    (label in neigh_map and neigh_map[label] & ci['labels'])
                    for label in cj['labels']
                ):
                    # fuse j into i
                    ci['labels'] |= cj['labels']
                    ci['tstats']  += cj['tstats']
                    used[j] = True
                    merged = True
            new_clusters.append(ci)
        clusters = new_clusters
    
    # 4) Filter by minimum cluster size
    clusters = [c for c in clusters if len(c['labels']) >= min_cluster_size]
    return clusters

# ============================================================================
# 3. Permutation procedure with clustering using neighbour information
def permute_and_cluster(
        data, 
        model,
        interest, 
        to_permute,
        num_permutations, 
        neighbours, 
        clus_alpha, 
        min_cluster_size,
        channels
        ):
    """
    Compute original channel p-values and t-values, build clusters based on 
    neighbours, and then generate a null distribution via permutation clustering.

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing the data.
    model : str
        The model to run linear mixed models.
    interest : str
        The effect of interest (e.g., 'n_session:C(difficulty)[T.HARD]').
    num_permutations : int
        Number of permutations.
    neighbours : List of Dict
        Neighbours structure (list of dicts as produced by e.g., prepare_neighbours_from_layout).
    clus_alpha : float
        Uncorrected p-value threshold (e.g., 0.05).
    min_cluster_size : int
        Minimum number of channels per cluster..
    channels : list or np.array
        List of channels in the right order for later 

    Returns
    -------
    clusters_pos : List (to complete)
        Clusters from the real (non-permuted) data (for positive effects)..
    clusters_neg : List (to complete)
        Clusters from the real (non-permuted) data (for negative effects)..
    perm_stats_pos : List
        Lists of maximum cluster stats from each permutation.
    perm_stats_neg : List
        Lists of maximum cluster stats from each permutation.
    original_pvals : List
        Original channel-level statistics.
    original_tvals : List
        Original channel-level statistics.

    """
    import numpy as np
    
    original_pvals = []
    original_tvals = []
    for chan in channels:
        p, t = run_mixedlm(data, chan, interest, model)
        original_pvals.append(p)
        original_tvals.append(t)
    
    if np.any(np.isnan(original_tvals)) :
        for pos in np.where(np.isnan(original_tvals))[0] :
            original_tvals[pos] = np.nanmean(original_tvals)
    
    # Form clusters separately for positive and negative effects.
    clusters_pos = cluster_significant_channels(
        channels, 
        original_pvals, 
        original_tvals,
        neighbours, 
        clus_alpha, 
        min_cluster_size, 
        sign='pos'
        )
    clusters_neg = cluster_significant_channels(
        channels, 
        original_pvals, 
        original_tvals,
        neighbours, 
        clus_alpha, 
        min_cluster_size, 
        sign='neg'
        )
    
    perm_stats_pos = []  # one value per permutation: maximum cluster t-sum (for positive clusters)
    perm_stats_neg = []  # one value per permutation: minimum (most negative) cluster t-sum (for negative clusters)
    
    for _ in range(num_permutations):
        shuffled_data = data.copy()
        shuffled_data[to_permute] = np.random.permutation(shuffled_data[to_permute].values)
        perm_pvals = []
        perm_tvals = []
        for chan in channels:
            p, t = run_mixedlm(shuffled_data, chan, interest, model)
            perm_pvals.append(p)
            perm_tvals.append(t)
        
        perm_clusters_pos = cluster_significant_channels(
            channels, 
            perm_pvals, 
            perm_tvals,
            neighbours, 
            clus_alpha, 
            min_cluster_size, 
            sign='pos')
        perm_clusters_neg = cluster_significant_channels(
            channels,
            perm_pvals, 
            perm_tvals,
            neighbours, 
            clus_alpha, 
            min_cluster_size, 
            sign='neg'
            )
        
        if perm_clusters_pos:
            perm_stat_pos = max(sum(clust['tstats']) for clust in perm_clusters_pos)
        else:
            perm_stat_pos = 0
        perm_stats_pos.append(perm_stat_pos)
        
        if perm_clusters_neg:
            perm_stat_neg = min(sum(clust['tstats']) for clust in perm_clusters_neg)
        else:
            perm_stat_neg = 0
        perm_stats_neg.append(perm_stat_neg)
    
    return clusters_pos, clusters_neg, perm_stats_pos, perm_stats_neg, original_pvals, original_tvals

# ============================================================================
# 4. Determine which real clusters are significant via permutation comparison
def identify_significant_clusters(
        clusters_pos,
        clusters_neg, 
        perm_stats_pos, 
        perm_stats_neg, 
        montecarlo_alpha, 
        num_permutations
        ):
    """
    Compare each original cluster statistic against its permutation distribution and return
    those clusters that are significant.

    Parameters
    ----------
    clusters_pos : List (to complete)
        Clusters from the real (non-permuted) data (for positive effects)..
    clusters_neg : List (to complete)
        Clusters from the real (non-permuted) data (for negative effects)..
    perm_stats_pos : List
        Lists of maximum cluster stats from each permutation.
    perm_stats_neg : List
        Lists of maximum cluster stats from each permutation.
    montecarlo_alpha : TYPE
        DESCRIPTION.
    num_permutations : TYPE
        DESCRIPTION.

    Returns
    -------
    significant_clusters : A list of tuple
        (sign, cluster_labels, cluster_stat, p_value)
        where sign is 'pos' or 'neg'.

    """
    import numpy as np
    
    significant_clusters = []
    for clust in clusters_pos:
        stat = sum(clust['tstats'])
        p_value = (np.sum(np.array(perm_stats_pos) >= stat) + 1) / (num_permutations + 1)
        if p_value < montecarlo_alpha:
            significant_clusters.append(('pos', clust['labels'], stat, p_value))
    
    for clust in clusters_neg:
        stat = sum(clust['tstats'])
        p_value = (np.sum(np.array(perm_stats_neg) <= stat) + 1) / (num_permutations + 1)
        if p_value < montecarlo_alpha:
            significant_clusters.append(('neg', clust['labels'], stat, p_value))
    
    return significant_clusters


# ============================================================================
# 5. Visualization function
def visualize_clusters(tvals, channels, significant_mask, info, savepath, vlims = None):
    """
    Visualize significant clusters using topomap.

    Parameters
    ----------
    tvals : TYPE
        DESCRIPTION.
    channels : TYPE
        DESCRIPTION.
    significant_mask : TYPE
        DESCRIPTION.
    info : TYPE
        DESCRIPTION.
    savepath : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    fig, ax = plt.subplots(figsize=(4, 4))
    
    im, cm = mne.viz.plot_topomap(
        data=np.array(tvals),
        pos=info,  # expects sensor positions from the info object
        mask=significant_mask,
        axes=ax,
        show=False,
        contours=2,
        mask_params=dict(
            marker='o',
            markerfacecolor='w',
            markeredgecolor='k',
            linewidth=0,
            markersize=10
        ),
        cmap="coolwarm",
        vlim = vlims
        )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    cax.set_title("t-values", fontsize=12, pad = 10)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=10)
    fig.colorbar(im, cax=cax)
    fig.tight_layout(pad=1)
    # ax.set_title("Interaction Effect", fontweight="bold")
    # fig.suptitle("T-values, Cluster Permutation Corrected", fontsize="xx-large", fontweight="bold")
    plt.savefig(savepath, dpi = 300)
    plt.show()


