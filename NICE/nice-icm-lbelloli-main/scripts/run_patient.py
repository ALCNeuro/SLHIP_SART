import os
import os.path as op
import time
import sys
import traceback

from argparse import ArgumentParser

import mne
from mne.utils import logger

from nice_ext.api import (utils, read, preprocess, fit, create_report, predict,
                          summarize_subject)


default_path = '/media/data2/new_patients'
default_path = '/Volumes/Databases/ICM/doc_stable/'
default_path = '/Users/fraimondo/data/icm/test'
default_predict_path = '/media/data/old_patients/group_results/baseline_brain'
default_predict_path = ('/Volumes/Databases/ICM/doc_stable/group_results/'
                        'baseline_stable_20210128')

start_time = time.time()

parser = ArgumentParser(description='Run the pipeline on the selected suject')

parser.add_argument('--io', metavar='io', type=str, nargs='?',
                    default='icm/lg/raw/egi',
                    help='IO to use (default = icm/lg/raw/egi)')
parser.add_argument('--preprocess', metavar='preprocess', type=str, nargs='?',
                    default='icm/lg/raw/egi',
                    help='Preprocessing to run (default = icm/lg/raw/egi)')
parser.add_argument('--fit', metavar='fit', type=str, nargs='?',
                    default='icm/lg',
                    help='Fit to run (default = icm/lg)')
parser.add_argument('--report', metavar='report', type=str, nargs='?',
                    default='icm/lg',
                    help='Report to create (default = icm/lg)')
parser.add_argument('--path', metavar='path', nargs=1, type=str,
                    help='Path with the database.',
                    default=default_path)
parser.add_argument('--subject', metavar='subject', nargs=1, type=str,
                    help='Subject name', required=True)
parser.add_argument('--predict-path', metavar='pred_path', nargs=1, type=str,
                    help='Path with the database. to predict',
                    default=default_predict_path)
parser.add_argument('--runid', metavar='runid', type=str, nargs='?',
                    default=None,
                    help='Run id (default = generated)')

args = parser.parse_args()
db_path = args.path
subject = args.subject
io_config = args.io
fit_config = args.fit
report_config = args.report
preprocess_config = args.preprocess
predict_path = args.predict_path

if isinstance(db_path, list):
    db_path = db_path[0]

if isinstance(subject, list):
    subject = subject[0]

if isinstance(predict_path, list):
    predict_path = predict_path[0]

if args.runid is not None:
    run_id = args.runid
    if isinstance(run_id, list):
        run_id = run_id[0]
else:
    run_id = utils.get_run_id()
s_path = op.join(db_path, 'subjects', subject)

if not op.exists(op.join(db_path, 'results')):
    os.mkdir(op.join(db_path, 'results'))

results_dir = op.join(db_path, 'results', run_id)
if not op.exists(results_dir):
    os.mkdir(results_dir)

if not op.exists(op.join(results_dir, subject)):
    os.mkdir(op.join(results_dir, subject))

now = time.strftime('%Y_%m_%d_%H_%M_%S')
log_suffix = '_{}.log'.format(now)
mne.utils.set_log_file(op.join(results_dir,
                               subject,
                               subject + log_suffix))
utils.configure_logging()
utils.log_versions()

logger.info('Running {}'.format(subject))
report = None
# try:
if True:

    # Read
    io_params, _ = utils.parse_params_from_config(io_config)
    data = read(s_path, io_config, io_params)

    # Preprocess
    preprocess_params, preprocess_config = \
        utils.parse_params_from_config(preprocess_config)
    preprocess_params.update({'summary': True})
    epochs, summary = preprocess(data, preprocess_config, preprocess_params)

    # Preprocess Report
    report = mne.report.Report(title=subject)
    create_report(epochs, config='icm/preprocess',
                  config_params=dict(summary=summary), report=report)

    # Fit
    fit_params, fit_config = utils.parse_params_from_config(fit_config)
    fc = fit(epochs, config=fit_config)
    out_fname = '{}_{}_markers.hdf5'.format(subject, now)
    # fc.save(op.join(results_dir, subject, out_fname), overwrite=True)

    # Fit report
    report_params, report_config = utils.parse_params_from_config(report_config)
    report_params.update(dict(epochs=epochs))
    create_report(fc, config=report_config,
                  config_params=report_params,
                  report=report)

    # # Summarize
    summary = summarize_subject(
        fc,
        reductions=[
            '{}/egi256/trim_mean80'.format(fit_config),
            '{}/egi256/std'.format(fit_config),
            '{}/egi256gfp/trim_mean80'.format(fit_config),
            '{}/egi256gfp/std'.format(fit_config)
        ],
        out_path=op.join(results_dir, subject))

    # Predict
    predictions = predict(op.join(results_dir, subject),
                          predict_path, config='icm/lg',
                          config_params={'summary': True})

    # Predict report
    create_report(predictions, config='icm/predict', report=report)

# except Exception as err:
#     msg = traceback.format_exc()
#     logger.info(msg + '\nRunning subject failed ("%s")' % subject)
#     sys.exit(-4)
# finally:
    if report is not None:
        out_fname = '{}_{}_report.html'.format(subject, now)
        report.save(op.join(results_dir, subject, out_fname),
                    overwrite=True, open_browser=False)

    elapsed_time = time.time() - start_time
    logger.info('Elapsed time {}'.format(
        time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))
    logger.info('Running pipeline done')
    utils.remove_file_logging()
