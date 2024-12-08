from nice import Markers
from nice.markers import (PowerSpectralDensity,
                          KolmogorovComplexity,
                          PermutationEntropy,
                          SymbolicMutualInformation,
                          PowerSpectralDensitySummary,
                          PowerSpectralDensityEstimator)

from nice_ext.api.modules import register_module


def register():
    register_module('markers', 'icm/rs', _get_resting_state)


def _get_resting_state(config_params):
    
    optimized_backend = config_params.get('optimized_backend', True)

    if optimized_backend:
        backend_c = 'c'
        backend_openmp = 'openmp'
    else:
        backend_c = 'python'
        backend_openmp = 'python'
    
    n_jobs = config_params.get('n_jobs', 'auto')
    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs=n_jobs, nperseg=128)

    base_psd = PowerSpectralDensityEstimator(
        psd_method='welch', tmin=None, tmax=0.6, fmin=1., fmax=45.,
        psd_params=psds_params, comment='default')
    f_list = [
        PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                             normalize=False, comment='delta'),
        PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                             normalize=True, comment='deltan'),
        PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                             normalize=False, comment='theta'),
        PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                             normalize=True, comment='thetan'),
        PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                             normalize=False, comment='alpha'),
        PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                             normalize=True, comment='alphan'),
        PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                             normalize=False, comment='beta'),
        PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                             normalize=True, comment='betan'),
        PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                             normalize=False, comment='gamma'),
        PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                             normalize=True, comment='gamman'),

        PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=45.,
                             normalize=True, comment='summary_se'),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                    percentile=.5, comment='summary_msf'),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                    percentile=.9, comment='summary_sef90'),
        PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                    percentile=.95, comment='summary_sef95'),

        PermutationEntropy(tmin=None, tmax=0.6, backend=backend_c),

        SymbolicMutualInformation(
            tmin=None, tmax=0.6, method='weighted', backend=backend_openmp,
            method_params={'nthreads': n_jobs}, comment='weighted'),

        KolmogorovComplexity(tmin=None, tmax=0.6, backend=backend_openmp,
                             method_params={'nthreads': n_jobs}),
    ]

    fc = Markers(f_list)
    return fc
