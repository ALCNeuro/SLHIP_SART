# EEG Data Processing and Analysis

## Overview
This repository contains Python scripts for preprocessing, analyzing, and visualizing EEG data collected during cognitive neuroscience experiments. The scripts are designed to handle behavioral data, extract and clean EEG epochs, remove artifacts using ICA, and compute power spectral densities.

## Project Structure

### 1. Configuration File
- **`SLHIP_config_ALC.py`**
  This script defines key settings for EEG data processing, including:
  - **File Paths**: Specifies directories for raw, preprocessed, and analyzed data.
  - **EEG Preprocessing**: Defines filtering parameters, referencing methods, and resampling frequency.
  - **Channel Configurations**: Assigns EEG, EOG, and ECG channel types.
  - **Artifact Rejection & ICA**: Configures automatic channel interpolation and ICA parameters.

### 2. Preprocessing EEG Data
- **`0_preproc.py`**  
  Automates preprocessing of EEG recordings for sleep-wake detection and classification. It prepares raw files for further processing by computing block timings and applying standardized channel montages.

- **`02_01_Peprocessing_trials.py`**  
  Processes trial-based EEG epochs, applies automatic artifact rejection, and performs Independent Component Analysis (ICA) to clean the data. This script is tailored for cleaning epochs extracted during a sustained attention task (SART).

- **`02_02_epochs_probes.py`**  
  Extracts and preprocesses EEG epochs related to experimental probe responses. It cleans the data with a thresholding approach (e.g., 300µV) combined with ICA.

- **`02_03_probes_for_scoring_v2.py`**  
  Prepares a scorable 30-second window around each probe event. The script extracts only the necessary channels (such as F3, C3, O1, HEOG, and VEOG) for manual scoring based on AASM criteria.

### 3. Behavioral Data Analysis
- **`01_01_behav_20s_MS.py`**  
  Analyzes task performance data, focusing on response times, accuracy, and probe responses. It processes behavioral logs to extract key metrics per trial and probe.

- **`01_02_behav_Interprobe_MS.py`**  
  Explores behavioral performance across different probe intervals. This script provides an alternative perspective on the behavioral data, similar to the previous script but with a focus on interprobe comparisons.

### 4. Power Spectrum Analysis
- **Global Power Analysis:**
  - **`03_01_compute_global_power.py`**  
    Computes global power spectral density (PSD) across different cognitive or mental states. This script uses methods such as Welch’s algorithm to estimate the PSD over defined frequency ranges.
  
  - **`03_02_explore_global_power.py`**  
    Analyzes and visualizes global power metrics. It incorporates statistical tests and visualizations (using libraries like Matplotlib and Seaborn) to explore differences across conditions.

- **Periodic Power Analysis:**
  - **`03_03_compute_periodic_power.py`**  
    Computes periodic components of the EEG power spectrum by separating out the aperiodic (background) activity using the FOOOF algorithm. This helps in isolating oscillatory activity from the overall PSD.
  
  - **`03_04_explore_periodic_power.py`**  
    Explores and visualizes the periodic power characteristics across experimental conditions. The script further examines frequency bands of interest (e.g., delta, theta) and presents comparisons using various statistical tools.

- **Additional Power Analysis:**
  - **`03_05_compute_band_power.py`**  
    Computes power within specific frequency bands (e.g., delta, theta, alpha) and compares it across conditions.
  
  - **`03_06_compute_fooof_params.py`**  
    Extracts parameters from the FOOOF model, such as aperiodic slope and oscillatory peak properties.
  
  - **`03_07_explore_fooof_params.py`**  
    Visualizes and statistically analyzes FOOOF-derived parameters across different experimental conditions.

### 5. Slow Wave Analysis
- **`04_01_swdet.py`**  
  Detects slow waves from EEG data and extracts their temporal properties.

- **`04_02_compute_slowwaves.py`**  
  Computes slow-wave properties such as amplitude, duration, and slope.

- **`04_03_compute_swdensity_v2.py`**  
  Analyzes slow-wave density and links it to behavioral responses.

- **`04_04_explore_sw_v2.py`**  
  Visualizes slow-wave characteristics across different conditions.

- **`04_05_explore_aw.py`**  
  Explores associations between slow waves and wakefulness-related variables.

- **`04_06_sw_erp.py`**  
  Computes event-related potentials (ERPs) associated with slow-wave activity.

### 6. Event-Related Potential (ERP) Analysis
- **`05_01_explore_P300.py`**  
  Analyzes P300 event-related potentials and their relationship to cognitive states.

## Dependencies
The scripts require the following Python libraries:
- `numpy`
- `pandas`
- `mne`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `fooof`
- `scipy`

## Usage
1. **Preprocessing:**  
   Run `0_preproc.py` to prepare your raw EEG files. Then use `02_01_Peprocessing_trials.py` and `02_02_epochs_probes.py` to extract and clean the EEG epochs.

2. **Behavioral Analysis:**  
   Analyze your task performance data with `01_01_behav_20s_MS.py` or `01_02_behav_Interprobe_MS.py`.

3. **Power Spectrum Analysis:**  
   Use `03_01_compute_global_power.py` and `03_02_explore_global_power.py` to compute and explore global power. For periodic power components, run `03_03_compute_periodic_power.py` and then visualize the findings using `03_04_explore_periodic_power.py`.

4. **Slow Wave & ERP Analysis:**  
   Use scripts in the `04_*` series for slow wave detection and `05_01_explore_P300.py` for ERP analysis.

## Contributions
Contributions are welcome! Feel free to submit pull requests or open issues for bug reports and feature requests.
