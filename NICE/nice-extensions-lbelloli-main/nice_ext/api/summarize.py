# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

from pathlib import Path
from copy import deepcopy
import os

import numpy as np
from scipy import io as sio

import pandas as pd

from mne.utils import logger

from . reductions import get_reductions
from nice.collection import Markers, read_markers


class Summary(object):

    def __init__(self):
        self._scalars = None

        self._topo_epoch_names = None
        self._epoch_names = None
        self._topo_names = None

        self._epochs = None
        self._topos = None
        self._topo_epochs = None

        self._topo_subjects = None
        self._epoch_subjects = None
        self._topo_epoch_subjects = None

    @property
    def _scalar_names(self):
        names = None
        if self._scalars is not None:
            names = [x for x in self._scalars.columns if x.startswith('nice')]
        return names

    @property
    def subjects(self):
        return self._topo_subjects

    def add_topo_epoch(self, names, values, reduction_name):
        if self._topo_epoch_names is None:
            self._topo_epoch_names = names
        elif names != self._topo_epoch_names:
            raise ValueError('Summary topo epoch names do not match')

        if self._topo_epochs is None:
            self._topo_epochs = {}
        self._topo_epochs[reduction_name] = values[..., None]

    def add_epoch(self, names, values, reduction_name):
        if self._epoch_names is None:
            self._epoch_names = names
        elif names != self._epoch_names:
            raise ValueError('Summary epoch names do not match')

        if self._epochs is None:
            self._epochs = {}
        self._epochs[reduction_name] = values[..., None]

    def add_topo(self, names, values, reduction_name):
        if self._topo_names is None:
            self._topo_names = names
        # TODO(Lao): shouldn't be sorted to compare? If yes, then the same happen in add_epoch and add_topo_epoch
        elif names != self._topo_names:
            raise ValueError('Summary topo names do not match')
        if self._topos is None:
            self._topos = {}
        self._topos[reduction_name] = values[..., None]

    def add_scalar(self, names, values, reduction_name):
        if (self._scalar_names is not None and  # noqa
                sorted(names) != sorted(self._scalar_names)):
            raise ValueError('Summary scalar names do not match')
        data = dict(zip(names, values))
        data['Reduction'] = reduction_name
        if self._scalars is None:
            self._scalars = pd.DataFrame(data, index=[0])
        else:
            ts = pd.DataFrame(data, index=[len(self._scalars)])
            self._scalars = pd.concat([self._scalars, ts])

    def topo_epoch_names(self):
        return self._topo_epoch_names

    def epoch_names(self):
        return self._epoch_names

    def topo_names(self):
        return self._topo_names
    
    def scalar_names(self):
        return self._scalar_names

    def scalars(self):
        return self._scalars

    def topos(self):
        return self._topos

    def epochs(self):
        return self._epochs

    def topo_epochs(self):
        return self._topo_epochs

    def __repr__(self):
        if self._scalar_names is None:
            s_string = 'Empty'
        else:
            s_string = '({}): {}'.format(
                len(self._scalar_names), len(self._scalars))

        if self._topo_names is None:
            t_string = 'Empty'
        else:
            t_string = '({}): {}'.format(
                len(self._topo_names), len(self._topos))

        n_subjects = 1
        if self._topo_subjects is not None:
            n_subjects = len(self._topo_subjects)
        s = '<Summary | Scalars: {} - Topos: {} - Subjects: {} >'
        s = s.format(s_string, t_string, n_subjects)

        return s

    def save(self, prefix):
        self._scalars.to_csv(f'{prefix}_scalars.csv', sep=';')

        if self._topos is not None:
            copy = {k: v for k, v in self._topos.items()}
            copy['names'] = self._topo_names
            if self._topo_subjects is not None:
                copy['subjects'] = self._topo_subjects
        sio.savemat('{}_topos.mat'.format(prefix), copy)

        if self._epochs is not None:
            copy = {k: v for k, v in self._epochs.items()}
            copy['names'] = self._epoch_names
            # TODO: check where is the _topo_subjects asigned and used to replicate the same with epochs
            if self._epoch_subjects is not None:
                copy['subjects'] = self._epoch_subjects
        sio.savemat('{}_epochs.mat'.format(prefix), copy)

        if self._topo_epochs is not None:
            copy = {k: v for k, v in self._topo_epochs.items()}
            copy['names'] = self._topo_epoch_names
            # TODO: check where is the _topo_subjects asigned and used to replicate the same with topo epochs
            if self._topo_epoch_subjects is not None:
                copy['subjects'] = self._topo_epoch_subjects
        sio.savemat('{}_topoepochs.mat'.format(prefix), copy)

    #TODO(Lao): Add copy for epoch and topo_epoch
    def copy(self):
        out = Summary()
        if self._scalars is not None:
            out._scalars = self._scalars.copy(deep=True)

        if self._topos is not None:
            out._topo_names = list(self._topo_names)
            out._topos = deepcopy(self._topos)
        if self._topo_subjects is not None:
            out._topo_subjects = list(self._topo_subjects)
        return out

    # TODO(Lao): Include epoch and topo here. Check how and if they must match the reductions as they do with topo and scalar
    def filter(self, reductions=None, subjects=None, markers=None):
        """ Filter a summary

        Parameters
        ----------
        reductions : list of str
            reductions to keep, if None (default), will keep all
        subjects : list of str
            subjects to keep, if None (default), will keep all
        markers : list of str
            markers to keep, if None (default), will keep all

        Returns
        -------
        out : instance of summary

        """
        out = self.copy()
        if out._scalars is None:
            return out
        if reductions is None and subjects is None and markers is None:
            logger.warning('Nothing to filter here, returning a copy')
        if reductions is not None:
            n_orig_s = len(np.unique(out._scalars['Reduction'].values))
            n_orig_t = len(out._topos.keys())
            out._scalars = out._scalars[
                out._scalars['Reduction'].isin(reductions)]
            for k in list(out._topos.keys()):
                if k not in reductions:
                    del out._topos[k]
            n_new_s = len(np.unique(out._scalars['Reduction'].values))
            n_new_t = len(out._topos.keys())
            logger.info('Filtering reductions: {} out of {} scalars and '
                        '{} out of {} topos'.format(
                            n_new_s, n_orig_s, n_new_t, n_orig_t))
        if subjects is not None:
            # Here we guess we have a subject to filter
            out._scalars = out._scalars[out._scalars['Subject'].isin(subjects)]
            idx, names = zip(*[(i, v) for i, v in
                               enumerate(out._topo_subjects) if v in subjects])
            idx = np.array(idx)
            for k in out._topos.keys():
                out._topos[k] = out._topos[k][..., idx]
            n_orig = len(out._topo_subjects)
            out._topo_subjects = names
            n_new = len(names)
            logger.info(
                'Filtering subjects: {} out of {}'.format(n_new, n_orig))
        if markers is not None:
            cols = out._scalars.columns
            n_orig_s = len([x for x in cols if x.startswith('nice')])
            n_orig_t = len(out._topo_names)
            cols = [x for x in cols if not (x.startswith('nice') and  # noqa
                                            x not in markers)]
            out._scalars = out._scalars[cols]
            idx, names = zip(*[(i, v) for i, v in
                               enumerate(out._topo_names) if v in markers])
            idx = np.array(idx)
            for k in out._topos.keys():
                out._topos[k] = out._topos[k][idx, ...]
            out._topo_names = names
            n_new_s = len([x for x in cols if x.startswith('nice')])
            n_new_t = len(out._topo_names)
            logger.info('Filtering markers: {} out of {} scalars and '
                        ' {} out of {} topos'.format(
                            n_new_s, n_orig_s, n_new_t, n_orig_t))
        return out

    def append_info(self, subject_info):
        self._scalars = pd.merge(
            self._scalars, subject_info, how='inner', on='Subject')

    #TODO(Lao): Add epoch and topo epochs to subject checks and add the system to add the subjects to them as well
    def _check_integrity(self):
        # Check data:
        #   - Same Subjects
        #   - Topos shape are correct
        t_subjects = self._topo_subjects
        n_subjects = len(t_subjects) if t_subjects is not None else 1
        if 'Subject' in self._scalars.columns:
            if t_subjects is None and self._topos is not None:
                raise ValueError('Subjects lists do not match')
            else:
                s_subjects = np.unique(self._scalars['Subject'].values)
                if sorted(t_subjects) != sorted(s_subjects):
                    raise ValueError('Subjects list items do not match')

        if self._topos is not None:
            n_topos = len(self._topo_names)
            for k, v in self._topos.items():
                if v.shape[-1] != n_subjects:
                    raise ValueError('Number of subjects do not match topos')
                if v.shape[0] != n_topos:
                    raise ValueError('Number of topos do not match object')

    def _make_back_compat(self):
        # COMPAT
        # Previous single subject topos summaries were 2D
        for k, v in self._topos.items():
            if v.ndim != 3:
                self._topos[k] = v[..., None]

        # This two columns has caps
        if 'reduction' in self._scalars.columns:
            self._scalars.rename(
                columns={'reduction': 'Reduction'}, inplace=True)
        if 'subject' in self._scalars.columns:
            self._scalars.rename(
                columns={'subject': 'Subject'}, inplace=True)
            if len(np.unique(self._scalars['Subject'].values)) == 1:
                del self._scalars['Subject']

# TODO(Lao): Also concatenate epoch and topo epoch
def _concatenate_summaries(summaries, names):
    """Concatenate list of summaries using subject from names

    Parameters
    ----------
    summaries : list of summary
        The input summaries.
    names : list of str
        list of subjects names

    Returns
    -------
    out : instance of summary
    """
    results = Summary()
    scalar_reductions = None
    for t_s, t_n in zip(summaries, names):
        t_df = t_s._scalars.copy(deep=True)
        t_df['Subject'] = t_n
        if results._scalars is None:
            results._scalars = t_df
            scalar_reductions = np.unique(t_s._scalars['Reduction'].values)
        else:
            if (t_s._scalar_names is None or # noqa
                    sorted(results._scalar_names) !=  # noqa
                    sorted(t_s._scalar_names)):
                raise ValueError('Scalars do not have the same markers')
            elif (sorted(np.unique(t_df['Reduction'].values)) !=  # noqa
                    sorted(scalar_reductions)):
                raise ValueError('Scalars reductions do not match')
            else:
                results._scalars = pd.concat([results._scalars, t_df])

        if results._topos is None:
            results._topo_names = list(t_s._topo_names)
            results._topos = t_s._topos.copy()
            results._topo_subjects = [t_n]
        else:
            if results._topo_names != t_s._topo_names:
                raise ValueError('Topos do not have the same markers')
            for k, v in results._topos.items():
                results._topos[k] = np.concatenate([v, t_s._topos[k]], axis=-1)
            results._topo_subjects.append(t_n)
    return results

def read_summary(
    prefix,
    s_suffix='_scalars.csv',
    t_suffix='_topos.mat',
    e_suffix='_epochs.mat',
    te_suffix='_topoepochs.mat'):

    if isinstance(prefix, Path):
        prefix = prefix.as_posix()
    
    for suffix in [s_suffix, t_suffix, e_suffix, te_suffix]:
        if prefix.endswith(suffix):
            prefix = prefix.replace(suffix, '')

    suma = Summary()
    
    # Read scalar reductions
    df_path = f'{prefix}{s_suffix}'
    df = pd.read_csv(df_path, sep=';', index_col=0)
    if len(df.columns) == 0:  # COMPAT
        # Might be old CSV, separated by ','
        df = pd.read_csv(df_path.format(prefix), sep=',', index_col=0)
    suma._scalars = df

    # Read topo reductions (channels)
    mc = sio.loadmat(f'{prefix}{t_suffix}')    
    topo_names = [x.strip() for x in mc['names']]
    topo_subjects = None
    if 'subjects' in mc:
        topo_subjects = [x.strip() for x in mc['subjects']]
    
    suma._topo_subjects = topo_subjects
    suma._topo_names = topo_names
    k = [
        x for x in mc.keys()
        if not (x in ['names', 'subjects'] or x.startswith('_'))
    ]
    suma._topos = {t_k: mc[t_k] for t_k in k}

    # Read epoch reductions if exists
    if os.path.exists(f'{prefix}{e_suffix}'):
        mc = sio.loadmat(f'{prefix}{e_suffix}')    
        epoch_names = [x.strip() for x in mc['names']]
        epoch_subjects = None
        if 'subjects' in mc:
            epoch_subjects = [x.strip() for x in mc['subjects']]
        
        suma._epoch_subjects = epoch_subjects
        suma._epoch_names = epoch_names
        k = [
            x for x in mc.keys()
            if not (x in ['names', 'subjects'] or x.startswith('_'))
        ]
        suma._epochs = {t_k: mc[t_k] for t_k in k}

    # Read toco epoch reductions if exists
    if os.path.exists(f'{prefix}{te_suffix}'):
        mc = sio.loadmat(f'{prefix}{te_suffix}')    
        topo_epoch_names = [x.strip() for x in mc['names']]
        topo_epoch_subjects = None
        if 'subjects' in mc:
            topo_epoch_subjects = [x.strip() for x in mc['subjects']]
        
        suma._topo_epoch_subjects = topo_epoch_subjects
        suma._topo_epoch_names = topo_epoch_names
        k = [
            x for x in mc.keys()
            if not (x in ['names', 'subjects'] or x.startswith('_'))
        ]
        suma._topo_epochs = {t_k: mc[t_k] for t_k in k}

    suma._make_back_compat()  # COMPAT
    suma._check_integrity()
    return suma

# TODO(Lao): Make this function public (remove _ prefix) as it's used by other modules to avoid future conflicts
def _try_read_summary(
    path, prefix=None,
    s_suffix='_scalars.csv',
    t_suffix='_topos.mat',
    e_suffix='_epochs.mat',
    te_suffix='_topoepochs.mat'):
    
    if not isinstance(path, Path):
        path = Path(path)
        
    # Use None for default prefix for backward compatibility
    if prefix is None:
        prefix = '*'

    # Try to load previous results
    scalars_csv = sorted(path.glob(f'{prefix}{s_suffix}'))
    if len(scalars_csv) > 0:
        if len(scalars_csv) > 1:
            logger.warning(f'More than one {s_suffix} file in {path}. Using last one')
    else:
        return False

    topo_mat = sorted(path.glob(f'{prefix}{t_suffix}'))
    if len(topo_mat) > 0:
        if len(topo_mat) > 1:
            logger.warning(f'More than one {t_suffix} file in {path}. Using last one')
    # TODO(Lao): Here shouldn't be a else branch to return False as with the csv_s. 
    # If not this double if could just check the len(mat_s) > 1

    epoch_mat = sorted(path.glob(f'{prefix}{e_suffix}'))
    if len(epoch_mat) > 0:
        if len(epoch_mat) > 1:
            logger.warning(f'More than one {e_suffix} file in {path}. Using last one')
    #TODO(Lao): Same than before if before is a bug

    topo_epoch_mat = sorted(path.glob(f'{prefix}{te_suffix}'))
    if len(topo_epoch_mat) > 0:
        if len(topo_epoch_mat) > 1:
            logger.warning(f'More than one {te_suffix} file in {path}. Using last one')
    #TODO(Lao): Same than before if before is a bug

    s_prefix = scalars_csv[-1].as_posix()[:-len(s_suffix)]
    t_prefix = topo_mat[-1].as_posix()[:-len(t_suffix)]
    if s_prefix != t_prefix:
        raise ValueError('Different prefix for scalars and topos.'
                         ' Clean database or Recompute')

    logger.info('Reading previous results')
    summary = read_summary(
        s_prefix,
        s_suffix=s_suffix,
        t_suffix=t_suffix,
        e_suffix=e_suffix,
        te_suffix=te_suffix
    )
    return summary


def _try_get_markers(markers):
    if not isinstance(markers, Path):
        markers = Path(markers)
    if markers.suffix == 'hdf5':
        return markers
    fc_files = sorted(markers.glob('*.hdf5'))
    if len(fc_files) > 1:
        logger.warning('More than one HDF5 file for {}.'
                       ' Using last one'.format(markers))
    elif len(fc_files) == 0:
        logger.warning('No HDF5 file in {}.'.format(markers))
        return None
    return fc_files[-1]


def summarize_subject(markers, reductions, reduction_params=None,
                      out_path=None, recompute=False, out_prefix=None):
    """Summarizes one subject

    Parameters
    ----------
    markers instance of marker collection or string: 
        The input data. If string, look for and HDF5 marker collection.
    reductions list(str): list of reductions to use
    reductions dict: Parameters to pass to the reductions
    out_path str: 
        path to store the summary object. If None (default),
        results will not be saved.
    recompute bool:
        If False, try to use previously saved results (default). If true,
        recompute and overwrite. If 'skip', look only for precomputed
        reductions. Results will be read from out_path.
    
    Returns
    -------
    out : instance of summary
    """
    reductions_to_do = reductions
    summary = False
    
    # If is set to not recompute and the marker is a path, the read the 
    # already calculated data from the path
    data_is_path = out_path is not None or isinstance(markers, (str, Path))
    if recompute in [False, 'skip'] and data_is_path:
        summary = False  # not Found

        if isinstance(markers, (str, Path)):
            logger.info(f'Trying to get summary from {markers}')
            summary = _try_read_summary(markers, prefix=out_prefix)
            if summary is False:
                logger.info('Summary not found in markers path ')
        if summary is False and out_path is not None:
            logger.info(f'Trying to get summary from {out_path}')
            summary = _try_read_summary(out_path, prefix=out_prefix)
            if summary is False:
                logger.info('Summary not found in out_path path ')
        if summary is not False:
            s_reductions = np.unique(summary.scalars()['Reduction'].values)
            t_reductions = list(summary.topos().keys())
            # TODO(Lao): add the epoch and topo epoch reduction that should match?
            if sorted(s_reductions) != sorted(t_reductions):
                raise ValueError('Scalar and topos reductions do not match. '
                                 'Recompute')

            reductions_to_do = [x for x in reductions if x not in s_reductions]
            logger.info('Using previous reductions.')
            logger.info('Missing {}'.format(reductions_to_do))

    if len(reductions_to_do) == 0:
        logger.info('All reductions were done')
        if summary is not False:
            summary = summary.filter(reductions)
        return summary
    elif summary is False and recompute == 'skip':
        logger.info('No summary, skipping recompute')
        return False
    elif summary is False:
        summary = Summary()

    #TODO(Lao): This should be a elif right? is the complement of the previous if and elif
    if recompute == 'skip':
        logger.info('Skipping recompute')
        summary = summary.filter(reductions)
        return summary

    if out_prefix is None:
        out_prefix = 'default'

    if isinstance(markers, (str, Path)):
        fc_name = _try_get_markers(markers)
        if fc_name is None:
            return None
        logger.info(f'Reading markers from {fc_name}')
        fc = read_markers(fc_name)
        if fc_name.name.endswith('_markers.hdf5'):
            out_prefix = fc_name.name[:-14]
        else:
            out_prefix = fc_name.name[:-5]
        out_prefix = out_prefix.split('/')[-1]
    elif isinstance(markers, Markers):
        fc = markers
    else:
        raise ValueError(
            'Marker must be either a Marker instance or a path to'
            ' a valid file where to read a Marker instance'
        )

    logger.info('Proceding with following reductions:')
    for reduction in reductions_to_do:
        logger.info(f'\t{reduction}')
    
    where = None
    if out_path is not None:
        if not isinstance(out_path, Path):
            out_path = Path(out_path)
        where = out_path / out_prefix
        logger.info(f'Saving summary to {where}')
    
    for reduction_name in reductions_to_do:
        logger.info(f'Applying {reduction_name}')
        reduction = get_reductions(
            reduction_name,
            config_params=reduction_params
        )
        scalars = fc.reduce_to_scalar(reduction)
        topos = fc.reduce_to_topo(reduction)
        epochs = fc.reduce_to_epoch(reduction)
        topo_epochs = fc.reduce_to_topo_epoch(reduction)

        topo_names = fc.topo_names()
        scalar_names = fc.scalar_names()
        epoch_names = fc.epoch_names()
        topo_epoch_names = fc.topo_epoch_names()

        summary.add_scalar(scalar_names, scalars, reduction_name)
        summary.add_topo(topo_names, topos, reduction_name)
        summary.add_epoch(epoch_names, epochs, reduction_name)
        summary.add_topo_epoch(topo_epoch_names, topo_epochs, reduction_name)

        if out_path is not None:
            summary.save(where)
    
    if out_path is not None:
        summary.save(where)

    # Filter only reductions that we were actually asked
    out = summary.filter(reductions)
    return out


def summarize_run(in_path, reductions, subject_extra_info=None, out_path=None,
                  recompute=False):
    """Summarizes a folder with several subjects results subfolders

    Parameters
    ----------
    in_path : str
        Path to subjects' results subfolders
    reductions : list of str
        list of reductions to use
    subject_extra_info : pandas.DataFrame object
        If not None (default) it will use the 'Subject' column to get the list
        of subjects to include and it will append all the other columns
        to the summary dataframe
    out_path : str
        path to store the summary object. If None (default),
        results will not be saved.
    recompute : bool
        If true, try to use previously saved results. If false,
        recompute and overwrite. Results will be read from out_path

    NOTE:
         All subject results will be stored next to the subjects result file
    Returns
    -------
    out : instance of summary
    """
    if not isinstance(in_path, Path):
        in_path = Path(in_path)
    if not isinstance(out_path, Path):
        out_path = Path(out_path)

    if subject_extra_info is not None:
        subjects = list(subject_extra_info['Subject'].values)
    else:
        subjects = [x.as_posix().split('/')[-1]
                    for x in in_path.glob('*') if x.is_dir()]
        subjects = [x for x in subjects if len(x) > 0]
    logger.info(f'Summarizing {len(subjects)} subjects')

    summaries = []
    subjects_done = []
    for subject in subjects:
        sin_path = in_path / subject
        summary = summarize_subject(
            sin_path, reductions,
            out_path=sin_path, recompute=recompute
        )
        
        valid_summary = (
            summary is not None
            and summary is not False
            and len(summary.scalars()) > 0
        )
        
        if valid_summary:
            summaries.append(summary)
            subjects_done.append(subject)
    
    logger.info(
        f'Could summarize {len(subjects_done)} '
        f'subjects out of {len(subjects)}'
    )
    
    global_summary = None
    if len(subjects_done) > 0:
        global_summary = _concatenate_summaries(summaries, subjects_done)
        if subject_extra_info is not None:
            global_summary.append_info(subject_extra_info)
        if out_path is not None:
            global_summary.save(out_path / 'all')

    return global_summary
