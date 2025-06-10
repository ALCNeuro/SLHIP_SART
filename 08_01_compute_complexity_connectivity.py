#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 18:33:00 2025

@author: arthurlecoz

08_01_compute_complexity_connectivity.py
"""
# %% Paths
import mne 
import os 
import pickle

import numpy as np
import pandas as pd
import SLHIP_config_ALC as config

from glob import glob


cleanDataPath = config.cleanDataPath
powerPath = config.powerPath

complexityPath =  config.complexPath
features_path = os.path.join(complexityPath, "features")

channels = np.array(config.eeg_channels)

subtypes = ["C1", "HI", "N1"]

files = glob(os.path.join(cleanDataPath, "epochs_probes", "*.fif"))

# %% NICE Fun

import time
import numpy as np
import zlib

from mne import pick_types
from mne.utils import logger, _time_mask


def epochs_compute_komplexity(epochs, nbins, tmin=None, tmax=None,
                              backend='python', method_params=None):
    """Compute complexity (K)

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs on which to compute the wSMI.
    nbins : int
        Number of bins to use for symbolic transformation
    method_params : dictionary.
        Overrides default parameters for the backend used.
        OpenMP specific {'nthreads'}
    backend : {'python', 'openmp'}
        The backend to be used. Defaults to 'python'.
    """
    picks = pick_types(epochs.info, meg=True, eeg=True)

    if method_params is None:
        method_params = {}

    data = epochs.get_data()[:, picks if picks is not None else Ellipsis]
    time_mask = _time_mask(epochs.times, tmin, tmax)
    data = data[:, :, time_mask]
    logger.info("Running KolmogorovComplexity")

    if backend == 'python':
        start_time = time.time()
        komp = _komplexity_python(data, nbins)
        elapsed_time = time.time() - start_time
        logger.info("Elapsed time {} sec".format(elapsed_time))
    elif backend == 'openmp':
        from ..optimizations.ompk import komplexity as _ompk_k
        nthreads = (method_params['nthreads']
                    if 'nthreads' in method_params else 1)
        if nthreads == 'auto':
            try:
                import mkl
                nthreads = mkl.get_max_threads()
                logger.info(
                    'Autodetected number of threads {}'.format(nthreads))
            except:
                logger.info('Cannot autodetect number of threads')
                nthreads = 1
        start_time = time.time()
        komp = _ompk_k(data, nbins, nthreads)
        elapsed_time = time.time() - start_time
        logger.info("Elapsed time {} sec".format(elapsed_time))
    else:
        raise ValueError('backend %s not supported for KolmogorovComplexity'
                         % backend)
    return komp


def _symb_python(signal, nbins):
    """Compute symbolic transform"""
    ssignal = np.sort(signal)
    items = signal.shape[0]
    first = int(items / 10)
    last = items - first if first > 1 else items - 1
    lower = ssignal[first]
    upper = ssignal[last]
    bsize = (upper - lower) / nbins

    osignal = np.zeros(signal.shape, dtype=np.uint8)
    maxbin = nbins - 1

    for i in range(items):
        tbin = int((signal[i] - lower) / bsize)
        osignal[i] = ((0 if tbin < 0 else maxbin
                       if tbin > maxbin else tbin) + ord('A'))

    return osignal.tostring()


def _komplexity_python(data, nbins):
    """Compute komplexity (K)"""
    ntrials, nchannels, nsamples = data.shape
    k = np.zeros((nchannels, ntrials), dtype=np.float64)
    for trial in range(ntrials):
        for channel in range(nchannels):
            string = _symb_python(data[trial, channel, :], nbins)
            cstring = zlib.compress(string)
            k[channel, trial] = float(len(cstring)) / float(len(string))

    return k


import math
import numpy as np
from itertools import permutations
from scipy.signal import butter, filtfilt

def epochs_compute_pe(epochs, kernel, tau, tmin=None, tmax=None,
                      backend='python', method_params=None):
    """Compute Permutation Entropy (PE)

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs on which to compute the PE.
    kernel : int
        The number of samples to use to transform to a symbol
    tau : int
        The number of samples left between the ones that defines a symbol.
    backend : {'python', 'c'}
        The backend to be used. Defaults to 'python'.
    """
    if method_params is None:
        method_params = {}

    freq = epochs.info['sfreq']

    data = epochs.pick(['eeg']).get_data()
    n_epochs = len(data)

    data = np.hstack(data)

    if 'filter_freq' in method_params:
        filter_freq = method_params['filter_freq']
    else:
        filter_freq = np.double(freq) / kernel / tau
    logger.info('Filtering  at %.2f Hz' % filter_freq)
    b, a = butter(6, 2.0 * filter_freq / np.double(freq), 'lowpass')

    fdata = np.transpose(np.array(
        np.split(filtfilt(b, a, data), n_epochs, axis=1)), [1, 2, 0])

    time_mask = _time_mask(epochs.times, tmin, tmax)
    fdata = fdata[:, time_mask, :]

    if backend == 'python':
        logger.info("Performing symbolic transformation")
        sym, count = _symb_python_2(fdata, kernel, tau)
        pe = np.nan_to_num(-np.nansum(count * np.log(count), axis=1))
    elif backend == 'c':
        from ..optimizations.jivaro import pe as jpe
        pe, sym = jpe(fdata, kernel, tau)
    else:
        raise ValueError('backend %s not supported for PE'
                         % backend)
    nsym = math.factorial(kernel)
    pe = pe / np.log(nsym)
    return pe, sym


def _define_symbols(kernel):
    result_dict = dict()
    total_symbols = math.factorial(kernel)
    cursymbol = 0
    for perm in permutations(range(kernel)):
        order = ''.join(map(str, perm))
        if order not in result_dict:
            result_dict[order] = cursymbol
            cursymbol = cursymbol + 1
            result_dict[order[::-1]] = total_symbols - cursymbol
    result = []
    for v in range(total_symbols):
        for symbol, value in result_dict.items():
            if value == v:
                result += [symbol]
    return result


# Performs symbolic transformation accross 1st dimension
def _symb_python_2(data, kernel, tau):
    """Compute symbolic transform"""
    symbols = _define_symbols(kernel)
    dims = data.shape

    signal_sym_shape = list(dims)
    signal_sym_shape[1] = data.shape[1] - tau * (kernel - 1)
    signal_sym = np.zeros(signal_sym_shape, np.int32)

    count_shape = list(dims)
    count_shape[1] = len(symbols)
    count = np.zeros(count_shape, np.int32)

    for k in range(signal_sym_shape[1]):
        subsamples = range(k, k + kernel * tau, tau)
        ind = np.argsort(data[:, subsamples], 1)
        signal_sym[:, k, ] = np.apply_along_axis(
            lambda x: symbols.index(''.join(map(str, x))), 1, ind)

    count = np.double(np.apply_along_axis(
        lambda x: np.bincount(x, minlength=len(symbols)), 1, signal_sym))

    return signal_sym, (count / signal_sym_shape[1])

def compute_complexity(
        epochs,
        reduction=None,
        compute_gamma=True):
    """
    Parameters
    ----------
    epochs : TYPE
        DESCRIPTION.
    reduction : TYPE, optional
        DESCRIPTION. The default is None.
    compute_gamma : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    #Modules
    import numpy as np
    from mne_features.univariate import compute_app_entropy,compute_samp_entropy

    #Define frequency bands
    if compute_gamma==True:
        frequencies= [(epochs.info['highpass'], 4,16,'delta'),
            (4,8,8,'theta'), (8,12,4,'alpha'),(12,30,2,'beta'),
            (30,epochs.info['lowpass'],1,'gamma')]  #format fmin,fmax,tau,name
    else:
        frequencies= [(epochs.info['highpass'], 4,16,'delta'),
            (4,8,8,'theta'), (8,12,4,'alpha'),(12,30,2,'beta')]

    #Create dic for storing markers
    markers= {}


    #Kolmogorov
    print('Computing Kolmogorov complexity')
    komp= epochs_compute_komplexity(epochs,nbins=32)
    komp= komp.transpose(1,0)
    markers['Kolmogorov']= komp

    #Approximative entropy
    print('Computing app entropy')
    approximative_entropies= []
    for i in range(len(epochs)):
        epochs_i= epochs[i]
        app_ent= compute_app_entropy(epochs_i.get_data()[0])
        approximative_entropies.append(app_ent)
    markers['Approximative_Entropy']=np.array(approximative_entropies)

    #Sample entropy
    print('Computing sample entropy')
    sample_entropies= []
    for i in range(len(epochs)):
        epochs_i= epochs[i]
        samp_ent= compute_samp_entropy(epochs_i.get_data()[0])
        sample_entropies.append(samp_ent)
    markers['Sample_Entropy']=np.array(sample_entropies)

    #Permutation entropies
    print('Computing permutation entropies en different fbands')
    for _,_,tau,fname in frequencies:
        perm_entropy= epochs_compute_pe(epochs,kernel= 3, tau= tau)
        perm_entropy= perm_entropy[0].transpose(1,0)
        markers['Permutation_Entropy_{}'.format(fname)]= perm_entropy

    #Reductions
    if reduction==None:
        return markers
    
    elif reduction=='topo':
        markers_topo= {}
        for k,v in markers.items():
            markers_topo[k]= v.mean(axis=0)
        return markers_topo
   
    elif reduction== 'epoch':
        markers_epoch= {}
        for k,v in markers.items():
            markers_epoch[k]= v.mean(axis=1)
        return markers_epoch    

def _get_weights_matrix(nsym):
    """Aux function"""
    wts = np.ones((nsym, nsym))
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    return wts


def epochs_compute_wsmi(epochs, kernel, tau, tmin=None, tmax=None,
                        backend='python', method_params=None, n_jobs='auto'):
    """Compute weighted mutual symbolic information (wSMI)

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs on which to compute the wSMI.
    kernel : int
        The number of samples to use to transform to a symbol
    tau : int
        The number of samples left between the ones that defines a symbol.
    method_params : dictionary.
        Overrides default parameters.
        OpenMP specific {'nthreads'}
    backend : {'python', 'openmp'}
        The backend to be used. Defaults to 'pytho'.
    """
    if method_params is None:
        method_params = {}

    if n_jobs == 'auto':
        try:
            import multiprocessing as mp
            mp.set_start_method('forkserver')
            import mkl
            n_jobs = int(mp.cpu_count() / mkl.get_max_threads())
            logger.info(
                'Autodetected number of jobs {}'.format(n_jobs))
        except Exception:
            logger.info('Cannot autodetect number of jobs')
            n_jobs = 1

    if 'bypass_csd' in method_params and method_params['bypass_csd'] is True:
        logger.info('Bypassing CSD')
        csd_epochs = epochs.pick(['eeg'])
    else:
        logger.info('Computing CSD')
        csd_epochs = mne.preprocessing.compute_current_source_density(
            epochs, lambda2=1e-5)

    freq = csd_epochs.info['sfreq']

    data = csd_epochs.get_data()
    n_epochs = len(data)

    if 'filter_freq' in method_params:
        filter_freq = method_params['filter_freq']
    else:
        filter_freq = np.double(freq) / kernel / tau
    logger.info('Filtering  at %.2f Hz' % filter_freq)
    b, a = butter(6, 2.0 * filter_freq / np.double(freq), 'lowpass')
    data = np.hstack(data)

    fdata = np.transpose(np.array(
        np.split(filtfilt(b, a, data), n_epochs, axis=1)), [1, 2, 0])

    time_mask = _time_mask(epochs.times, tmin, tmax)
    fdata = fdata[:, time_mask, :]
    if backend == 'python':
        logger.info("Performing symbolic transformation")
        sym, count = _symb_python_2(fdata, kernel, tau)
        nsym = count.shape[1]
        wts = _get_weights_matrix(nsym)
        logger.info("Running wsmi with python...")
        wsmi, smi = _wsmi_python(sym, count, wts)
    elif backend == 'openmp':
        from .optimizations.jivaro import wsmi as jwsmi
        nsym = np.math.factorial(kernel)
        wts = _get_weights_matrix(nsym)
        nthreads = (method_params['nthreads'] if 'nthreads' in
                    method_params else 1)
        if nthreads == 'auto':
            try:
                import mkl
                nthreads = mkl.get_max_threads()
                logger.info(
                    'Autodetected number of threads {}'.format(nthreads))
            except Exception:
                logger.info('Cannot autodetect number of threads')
                nthreads = 1
        wsmi, smi, sym, count = jwsmi(fdata, kernel, tau, wts, nthreads)
    else:
        raise ValueError('backend %s not supported for wSMI'
                         % backend)

    return wsmi, smi, sym, count


def _wsmi_python(data, count, wts):
    """Compute wsmi"""
    nchannels, nsamples, ntrials = data.shape
    nsymbols = count.shape[1]
    smi = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)
    wsmi = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)
    for trial in range(ntrials):
        for channel1 in range(nchannels):
            for channel2 in range(channel1 + 1, nchannels):
                pxy = np.zeros((nsymbols, nsymbols))
                for sample in range(nsamples):
                    pxy[data[channel1, sample, trial],
                        data[channel2, sample, trial]] += 1
                pxy = pxy / nsamples
                for sc1 in range(nsymbols):
                    for sc2 in range(nsymbols):
                        if pxy[sc1, sc2] > 0:
                            aux = pxy[sc1, sc2] * np.log(
                                pxy[sc1, sc2] /  # noqa
                                count[channel1, sc1, trial] /  # noqa
                                count[channel2, sc2, trial])
                            smi[channel1, channel2, trial] += aux
                            wsmi[channel1, channel2, trial] += \
                                (wts[sc1, sc2] * aux)
    wsmi = wsmi / np.log(nsymbols)
    smi = smi / np.log(nsymbols)
    return wsmi, smi

# %% Single Process

mindstates = ['ON', 'MW', 'HALLU', 'MB', 'FORGOT']
redo = 0
    
coi = [
    'sub_id', 'subtype', 'daytime', 'n_epoch', 'mindstate', "channel",
    'Kolmogorov', 'Approximative_Entropy', 'Sample_Entropy', 
    'Permutation_Entropy_delta', 'Permutation_Entropy_theta', 
    'Permutation_Entropy_alpha', 'Permutation_Entropy_beta', 
    'Permutation_Entropy_gamma', 
    'WSMI_delta','WSMI_theta','WSMI_alpha','WSMI_beta'
       ]

frequencies= [(0.5,4,32,'delta'),
              (4,8,16,'theta'),
              (8,12,8,'alpha'),
              (12,30,4,'beta')]

for file in files :
    
    sub_id = file.split('probes/')[-1].split('_epo')[0]
    daytime = sub_id[-2:]
    sub_id = sub_id[:-3]
    subtype = sub_id[:2]
    
    this_subject_savepath = os.path.join(
        features_path, f"{sub_id}.csv"
        )
    
    if not os.path.exists(this_subject_savepath) or redo : 
    
        temp_dic = {c : [] for c in coi}
        
        print(f"...processing {sub_id}")
        
        epochs = mne.read_epochs(file, preload = True)
        metadata = epochs.metadata
        
        for ms in mindstates:
            print(f'processing {ms}')
            if ms not in metadata.mindstate.unique() : continue
            else : 
                this_epochs = epochs[metadata.mindstate == ms]                
                
                complexity = compute_complexity(this_epochs)
                markers= {}
                
                for fmin,fmax,tau, name in frequencies:
                    wsmi,_,_,_= epochs_compute_wsmi(
                        this_epochs,
                        kernel=3,
                        tau=tau,
                        n_jobs=-1,
                        backend='python',
                        method_params= {'bypass_csd':False}
                        )
                    wsmi= wsmi.transpose(2,0,1)
                    print(wsmi.shape)
                    markers['WSMI_{}'.format(name)]= wsmi
                
                for i_ch, channel in enumerate(channels) :
                    print(f'processing channel {channel}')
                    for i_epoch in range(
                            len(epochs[epochs.metadata.mindstate == ms])
                            ) :
                        
                        temp_dic['sub_id'].append(sub_id)
                        temp_dic['subtype'].append(subtype)
                        temp_dic['daytime'].append(daytime)
                        temp_dic['n_epoch'].append(i_epoch)
                        temp_dic['channel'].append(channel)
                        temp_dic['mindstate'].append(ms)
                        temp_dic['Kolmogorov'].append(
                            complexity['Kolmogorov'][i_epoch, i_ch]
                            )
                        temp_dic['Approximative_Entropy'].append(
                            complexity['Approximative_Entropy'][i_epoch, i_ch]
                            )
                        temp_dic['Sample_Entropy'].append(
                            complexity['Sample_Entropy'][i_epoch, i_ch]
                            )
                        for pe in ["delta", "theta", "alpha", "beta", "gamma"]:
                            temp_dic[f'Permutation_Entropy_{pe}'].append(
                                complexity[f'Permutation_Entropy_{pe}'][i_epoch, i_ch]
                                )
                        for wsmi in ["delta", "theta", "alpha", "beta"]:
                            median_wsmi = np.median(
                                markers[f'WSMI_{wsmi}'][i_epoch, i_ch, :]
                                )
                            temp_dic[f'WSMI_{wsmi}'].append(median_wsmi)
                            
        df = pd.DataFrame.from_dict(temp_dic)
        df.to_csv(this_subject_savepath)   
        print(f"\n{sub_id} dataframe saved successfuly.")


