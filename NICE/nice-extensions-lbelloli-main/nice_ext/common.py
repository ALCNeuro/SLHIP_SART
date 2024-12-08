# Copyright (C) Federico Raimondo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

import numpy as np
import pandas as pd

# import os.path as op
# from glob import glob
from collections import OrderedDict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score)
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression

from mne.utils import logger


from .api.summarize import Summary, _try_read_summary, summarize_subject


def get_clf_name(clf_type):
    name = None
    if clf_type == 'gssvm':
        name = 'GridSearch SVM'
    elif clf_type == 'svm':
        name = 'SVM'
    elif clf_type == 'et-reduced':
        name = 'ExtraTrees (reduced depth)'
    return name


def _get_cv(y, seed):
    min_elems = min(np.sum(y), np.sum(1 - y))
    n_splits = 5
    if min_elems < n_splits:
        logger.warning('Using {} splits instead of {}'.format(
            min_elems - 1, n_splits))
        n_splits = min_elems - 1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return skf


def _get_clf(clf_type, seed, clf_params, clf_select, y):
    if clf_type == 'gssvm':
        skf = _get_cv(y, seed)
        cost_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        svc_params = dict(kernel='linear', probability=True,
                          random_state=seed,
                          class_weight='balanced')
        svc_params.update(clf_params)
        logger.info('Predict using GSSVM with {}'.format(svc_params))
        gc_fit_params = {'C': cost_range}
        logger.info('GridSearch on {}'.format(gc_fit_params))
        GSSVM = GridSearchCV(SVC(**svc_params),
                             gc_fit_params, cv=skf, scoring='roc_auc')
        clf_model = GSSVM
        if clf_select is not None:
            logger.info('Selecting {}% of markers'.format(clf_select))
            clf = Pipeline([
                ('scaler', StandardScaler()),
                ('select', SelectPercentile(f_classif, percentile=clf_select)),
                (clf_type, clf_model)])
        else:
            clf = Pipeline([
                ('scaler', StandardScaler()),
                (clf_type, clf_model)])
    elif clf_type == 'svm':
        target_weight = np.sum(y == 1) / y.shape[0]
        class_weight = {0: 1 - target_weight, 1: target_weight}
        svc_params = dict(kernel='linear', probability=True,
                          random_state=seed, class_weight=class_weight)
        svc_params.update(clf_params)
        logger.info(f'Predict using SVM with {svc_params}')
        clf_model = SVC(**svc_params)
        if clf_select is not None:
            logger.info(f'Selecting {clf_select}% of markers')
            clf = Pipeline([
                ('scaler', StandardScaler()),
                ('select', SelectPercentile(f_classif, percentile=clf_select)),
                (clf_type, clf_model)])
        else:
            clf = Pipeline([
                ('scaler', StandardScaler()),
                (clf_type, clf_model)])
    elif clf_type == 'et-reduced':
        et_params = dict(n_jobs=-1, n_estimators=2000,
                         max_features=1, max_depth=4,
                         random_state=seed, class_weight='balanced',
                         criterion='entropy')
        et_params.update(clf_params)
        logger.info(f'Predict using ET (reduced depth) with {et_params}')
        clf_model = ExtraTreesClassifier(**et_params)
        clf = Pipeline([
            ('scaler', RobustScaler()),
            (clf_type, clf_model)])
    else:
        raise ValueError(f'Unknown classifier: {clf_type}')

    return clf


def _predict(markers, summary, config_params=None):
    if config_params is None:
        config_params = {}

    pred_summary = None
    if 'summary' in config_params:
        if config_params['summary'] is True:
            pred_summary = dict(step=[], params=[], result=[])

    # Parameters to predict
    ml_classes = config_params.get('ml_classes', ['VS/UWS', 'MCS'])
    target = config_params.get('target', 'Diagnosis')

    logger.info('Predicting {} to {}'.format(target, ml_classes))

    # Default, use all reductions in group summary
    reductions = config_params.get('reductions', None)
    clf = config_params.get('clf', None)
    clf_type = config_params.get('clf_type', 'gssvm')
    clf_params = config_params.get('clf_params', {})
    clf_select = config_params.get('clf_select', 90.)

    logger.info(f'Using {reductions}')

    seed = config_params.get('random_state', 42)

    reduction_map = config_params.get('reduction_map', None)

    result = {}

    # get summary
    if isinstance(summary, str):
        summ = _try_read_summary(summary)
        if summ is False:
            raise ValueError('Can not get a summary from {}'.format(summary))
        summary = summ
    elif isinstance(summary, Summary):
        pass
    else:
        raise ValueError('What are you trying to predict against?')

    if reductions is None:
        reductions = np.unique(summary.scalars()['Reduction'].values)
    else:
        summary = summary.filter(reductions=reductions)

    s_reductions = reductions
    if reduction_map is not None:
        all_mapped = [k in reductions for k, v in reduction_map.items()]
        if not all(all_mapped):
            raise ValueError('All reductions should be mapped')
        i_reduction_map = {v: k for k, v in reduction_map.items()}
        s_reductions = [reduction_map[x] for x in reductions]

    # get markers summary
    s_summary = summarize_subject(
        markers,
        s_reductions,
        out_prefix=config_params.get('out_prefix', None)
    )

    # This allows to predict base on a subset of the group summary markers
    summary = summary.filter(markers=s_summary.scalar_names())

    # TODO: check that markers match (s_summary is subset of summary)

    # Get scalar values
    scalars = summary.scalars().sort_values(by=['Subject', 'Reduction'])
    if reduction_map is not None:
        scalars.replace({'Reduction': i_reduction_map}, inplace=True)
    labels = scalars[['Subject', target]].drop_duplicates().sort_values(by='Subject')
    markers = summary.scalar_names()

    # do mutivariate SVC
    if len(ml_classes) != 2:
        raise ValueError('Currently multilabel machine-learning based '
                         'prediction is not supported')

    if target not in scalars.columns:
        raise ValueError('Target column {} not in scalars info'.format(target))

    # X = get marker columns and rotate by 'Reduction'. Group by subject.
    ml_scalars = scalars[scalars[target].isin(ml_classes)]
    ml_scalars = ml_scalars[markers + ['Subject', 'Reduction']]
    ml_scalars = ml_scalars.pivot(index='Subject', columns='Reduction')
    X = ml_scalars.values
    summary_markers = [x for x in ml_scalars.columns]
    logger.info('Using {} markers'.format(len(summary_markers)))
    for marker in summary_markers:
        logger.info('\t{}'.format(marker))

    # y = binarize target column
    ml_labels = labels[labels[target].isin(ml_classes)]
    y = (ml_labels[target] == ml_classes[1]).values.astype(np.int)

    s_scalars = s_summary.scalars().sort_values(by=['Reduction'])
    s_ml_scalars = s_scalars[markers + ['Reduction']]
    s_ml_scalars = s_ml_scalars.assign(fakeidx=np.zeros(len(s_ml_scalars)))
    s_ml_scalars = s_ml_scalars.pivot(index='fakeidx', columns='Reduction')
    s_X = s_ml_scalars.values

    subject_markers = [x for x in s_ml_scalars.columns]
    logger.info('Using {} markers (Subject)'.format(len(subject_markers)))
    for marker in subject_markers:
        logger.info('\t{}'.format(marker))

    for a, b in zip(subject_markers, summary_markers):
        if a != b:
            raise ValueError('Markers do not match')

    # Handle Nans
    nans = np.any(np.isnan(X), axis=1)
    if np.sum(nans) > 0:
        logger.warning(
            'Removing {} subjects with NAN values'.format(np.sum(nans)))
        X = X[~nans]
        y = y[~nans]

    logger.info('Predicting using multivariate approach')
    # XXX not suported, but might be useful? {'sample_weight': sample_weight}
    # XXX TODO: sigmoid kernel + 20% feature selection
    logger.info('Using random seed {}'.format(seed))

    if clf is None:
        if not isinstance(clf_type, list):
            clf_select = {clf_type: clf_select}
            clf_params = {clf_type: clf_params}
            clf_type = [clf_type]
    else:
        clf_type = ['custom']
        clf_select = {'custom': None}
        clf_params = {'custom': {}}

    if pred_summary is not None:
        pred_summary['step'].append('')
        pred_summary['params'].append(
            {'Classes': ml_classes, 'Target': target}
        )
        n_samples = ['{}={}'.format(k, np.sum(y == i)) for i, k in
                     enumerate(ml_classes)]
        pred_summary['result'].append(
            ['N Samples = {}'.format(' '.join(n_samples))]
        )

    mv_result = OrderedDict()
    for t_type in clf_type:
        if t_type != 'custom':
            t_params = clf_params.get(t_type, {})
            t_select = clf_select.get(t_type, None)
            t_clf = _get_clf(t_type, seed, t_params, t_select, y)
        else:
            t_select = None
            t_params = {}
            t_clf = ['custom']
        cv_scores = []
        for seed in [43, 56, 23, 12, 32]:
            cv_skf = _get_cv(y, seed)
            t_score = cross_val_score(
                t_clf, X=X, y=y, cv=cv_skf, scoring='roc_auc')
            cv_scores.append(np.mean(t_score))

        logger.info('CV Score = {}'.format(np.mean(cv_scores)))

        logger.info('Fitting {} markers ({} samples)'.format(
            X.shape[1], X.shape[0]))
        t_clf.fit(X, y)
        in_sample_score = t_clf.score(X, y)
        logger.info('In-sample score = {}'.format(in_sample_score))
        probas = t_clf.predict_proba(s_X)

        if pred_summary is not None:
            pred_summary['step'].append('Multivariate')
            pred_summary['params'].append(
                {'Feature Selection (%)':
                    'None' if t_select is None else t_select,
                 'Classifier': t_type,
                 'N Markers': X.shape[1]}
            )
            predicted_probas = ['{}={:.2f}'.format(k, p * 100) for k, p in
                                zip(ml_classes, probas[0, :])]
            pred_summary['result'].append(
                ['In-sample AUC = {:.2f}'.format(in_sample_score),
                 'SKFold (K={}, iter=5) AUC = {:.2f}'.format(
                    cv_skf.n_splits, np.mean(cv_scores)),
                 'Predicted Probas: {}'.format(' '.join(predicted_probas))]
            )

        t_mv_result = {'score': in_sample_score,
                       'cv_score': cv_scores,
                       ml_classes[0]: probas[0, 0],
                       ml_classes[1]: probas[0, 1]}
        logger.info('Predicted probas ({}): {} {} - {} {}'.format(
            t_type, probas[0, 0], ml_classes[0], probas[0, 1], ml_classes[1]))
        mv_result[t_type] = t_mv_result
    result['multivariate'] = mv_result

    # do univariate logreg
    logger.info('Predicting using univariate approach')
    # prealocate results
    uv_result = pd.DataFrame(
        index=np.arange(0, ml_scalars.columns.shape[0]),
        columns=['Marker', 'Reduction', 'Score'] + ml_classes,
        dtype=np.float)
    # Univariate classification
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(random_state=seed))
    ])

    if pred_summary is not None:
        pred_summary['step'].append('Univariate')
        pred_summary['params'].append(
            {'Classifier': 'LogisticRegression'}
        )

        pred_summary['result'].append(['See figure'])

    for idx, (marker, reduction) in enumerate(ml_scalars.columns):
        logger.info('Predicting on {} {}'.format(marker, reduction))
        clf.fit(X[:, idx][:, None], y)
        score = clf.score(X[:, idx][:, None], y)
        probas = clf.predict_proba(s_X[:, idx][:, None])
        uv_result.iloc[idx] = [marker, reduction, score,
                               probas[0, 0], probas[0, 1]]
        logger.info('Predicted probas: {} {} - {} {}'.format(
            probas[0, 0], ml_classes[0], probas[0, 1], ml_classes[1]))

    result['univariate'] = uv_result

    if pred_summary is not None:
        result['summary'] = pred_summary
    # result['ml_classes'] = ml_classes

    # TODO: do topo distance

    # return predictive results
    return result
