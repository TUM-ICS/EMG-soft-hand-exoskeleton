# EMG-soft-hand-exoskeleton


## Overview

This repository contains code and data for activity onset detection and post-hoc error correction in the EMG signal of healthy individuals and persons with hand impairment. It provides scripts for training and evaluating 1D Convolutional Neural Network (CNN) models, as well as an adaptive threshold onset detector for comparison.

The models are designed to work with RMS-processed EMG data at a sampling rate of 34.81 Hz, and can be applied to both healthy participants and individuals with hand impairments (ALS, SMA).

**Main Replication Script:** `1_evaluate_cnn_onset_detection.py` is the primary script for replicating CNN onset detection results with models trained on different datasets (i.e. individual with ALS, healthy participants, mixed). See the [Replicating Core Results](#replicating-core-results) section for detailed instructions.

## Installation

### Prerequisites

- Python 3.12.1 (or compatible version)
- pip package manager

### Installing Dependencies

To install all required libraries, navigate to the repository root directory and run:

```bash
pip install -r requirements.txt
```

This will install the following packages:
- `numpy==1.26.4` - Numerical computing
- `pandas==2.2.3` - Data manipulation and analysis
- `matplotlib==3.10.6` - Plotting and visualization
- `torch==2.7.1` - PyTorch deep learning framework
- `scikit-learn==1.7.0` - Machine learning utilities
- `scipy==1.16.0` - Scientific computing
- `joblib==1.5.1` - Parallel processing and model persistence

**Note:** If you encounter compatibility issues, you may need to adjust the Python version or package versions. The repository was tested with Python 3.12.1.

## Data Structure

The `EMG_data` folder contains all experimental data organized into two main subdirectories:

### `EMG_data/signal_data/`

Contains RMS-processed EMG signal data in CSV format. Each file contains time-series data with timestamps and RMS amplitude values.

**Healthy Participants:**
- `RMS_healthy_P1_34p81hz_processed_cleaned.csv` through `RMS_healthy_P15_34p81hz_processed_cleaned.csv`
- 15 healthy participants (P1-P15) used for training and testing

**Hand-impaired Participants:**
- `RMS_ALS_block1.csv`, `RMS_ALS_block2.csv`, `RMS_ALS_block3.csv`, `RMS_ALS_block4.csv` - ALS patient data blocks
- `RMS_SMA_34p81hz_processed_cleaned.csv` - SMA (Spinal Muscular Atrophy) patient data

All signal data files are sampled at 34.81 Hz and have been processed and cleaned.

### `EMG_data/label_data/`

Contains ground truth labels (peak timestamps) for activity onsets, used for training and evaluation.

**Healthy Participant Labels:**
- `peaks_P1_interactive_final.csv` through `peaks_P15_interactive_final.csv` - Peak timestamps for each healthy participant
- `peaks_P1_final.png` through `peaks_P15_final.png` - Visualization plots of detected peaks

**Pathological Participant Labels:**
- `peaks_ALS_block1.csv` through `peaks_ALS_block4.csv` - Peak timestamps for ALS blocks
- `peaks_ALS_block1.png` through `peaks_ALS_block4.png` - Visualization plots
- `peaks_SMA_clinical.csv` and `peaks_SMA_clinical.png` - SMA clinical data labels

Each CSV file contains a `timestamp` column indicating the time points where muscle activity peaks were manually annotated or automatically detected.

## Scripts Documentation

All analysis and training scripts are located in the `Scripts_Training_and_Analysis/` folder. The scripts are organized by functionality:

### Training Scripts

#### `train_cnn_onset_detection.py`
**Purpose:** Trains a 1D CNN model for peak detection in 1162ms windows with a 162ms peak detection zone.

**Functionality:**
- Trains on healthy participants P1-P11
- Evaluates on P12-P15
- Uses sliding windows without class balancing
- Saves trained models to `trained_models/` directory
- Generates training curves and evaluation metrics

**How to run:**
```bash
cd Scripts_Training_and_Analysis
python train_cnn_onset_detection.py
```

**Output:**
- Trained model: `trained_models/cnn_healthy.pth`
- Training visualizations in `experiment_results/`

#### `train_cnn_onset_detection_allhealthy.py`
**Purpose:** Similar to `train_cnn_onset_detection.py` but trains on all healthy participants (P1-P15).

**Functionality:**
- Trains on all 15 healthy participants
- Uses the same 1162ms window architecture
- Useful for models that will be applied to pathological data

**How to run:**
```bash
cd Scripts_Training_and_Analysis
python train_cnn_onset_detection_allhealthy.py
```

#### `train_cnn_error.py`
**Purpose:** Trains a 1D CNN model for muscle activity detection in 200ms windows.

**Functionality:**
- Classifies 200ms windows as containing muscle activity or baseline
- Activity phases defined as -200ms to +800ms around detected peaks
- Used for error correction and activity detection tasks
- Generates training metrics and saves models

**How to run:**
```bash
cd Scripts_Training_and_Analysis
python train_cnn_error.py
```

**Output:**
- Trained models: `trained_models/error_*.pth` (various dataset combinations)
- Training visualizations in `experiment_results/`

### Evaluation Scripts

#### `1_evaluate_cnn_onset_detection.py` ⭐ **MAIN SCRIPT FOR REPLICATING RESULTS**
**Purpose:** **This is the primary script for replicating CNN onset detection results with models trained on different datasets.** Evaluates trained CNN models on test participants using movement phase analysis.

**Functionality:**
- Separates data into "movement phases" (-500ms before peak to +800ms after peak) and "baseline phases"
- Evaluates using sliding window detection (simulates real-time operation)
- Measures baseline phase accuracy (no false positives)
- Measures movement phase accuracy (at least one detection per movement)
- Calculates average delay between first detection and actual peak time
- **Interactive model selection from available trained models** - allows comparison of models trained on different datasets (healthy only, healthy+ALS, healthy+SMA, etc.)
- **Supports both CNN model detection and adaptive threshold detection** for comparison
- Can evaluate on healthy test participants (P12-P15) or ALS patient data (blocks 3-4)

**How to run:**
```bash
cd Scripts_Training_and_Analysis
python 1_evaluate_cnn_onset_detection.py
```

**Output:**
- Evaluation plots in `experiment_results/` showing movement phase evaluation results
- Performance metrics printed to console (baseline accuracy, movement phase accuracy, average detection delay)
- Summary statistics table comparing all evaluated participants

**Key Features:**
- Compare models trained on different datasets (healthy, healthy+ALS, healthy+SMA, ALS-only, SMA-only)
- Evaluate on healthy test set (P12-P15) or ALS patient data
- Compare CNN-based detection with adaptive threshold method
- Generate publication-ready visualizations

#### `evaluate_cnn_onset_detection_wholeBaseline.py`
**Purpose:** Similar to `1_evaluate_cnn_onset_detection.py` but with different movement phase definitions (-100ms before peak to +400ms after peak).

**Functionality:**
- Uses narrower movement phase windows
- Evaluates baseline and movement phase performance
- Useful for comparing different evaluation criteria

**How to run:**
```bash
cd Scripts_Training_and_Analysis
python evaluate_cnn_onset_detection_wholeBaseline.py
```

### Analysis Scripts

#### `activity_detection.py`
**Purpose:** Uses a trained error detection model to classify ALS data blocks 3 and 4 into activity and baseline phases.

**Functionality:**
- Uses 200ms detection windows
- Visualizes results with real peaks marked
- Classifies continuous signal into activity vs. baseline periods

**How to run:**
```bash
cd Scripts_Training_and_Analysis
python activity_detection.py
```

**Output:**
- Visualization plots in `experiment_results/` showing activity detection results

#### `error_correction.py`
**Purpose:** Analyzes error correction patterns by detecting onsets and classifying the 250ms-500ms interval after each detected onset.

**Functionality:**
- Detects onsets using activity window (last 162ms) > baseline window (first 1000ms) + threshold
- Uses trained error model with 200ms windows to classify post-onset intervals
- Identifies false positives and error patterns
- Marks errors when 3+ samples classified as baseline after detection

**How to run:**
```bash
cd Scripts_Training_and_Analysis
python error_correction.py
```

**Output:**
- Error correction analysis plots in `experiment_results/`
- Visualizations showing detection patterns and error indicators

#### `fast_recovery.py`
**Purpose:** Analyzes recovery patterns after false positive detections during baseline phases.

**Functionality:**
- Tracks what happens in the 500ms following each false positive detection
- Evaluates both CNN models and adaptive threshold methods
- Analyzes how quickly the system recovers from false positives
- Separates movement phases (-200ms to +800ms around peaks) from baseline

**How to run:**
```bash
cd Scripts_Training_and_Analysis
python fast_recovery.py
```

**Output:**
- Recovery analysis plots in `experiment_results/`
- Statistics on recovery patterns

#### `gradcam_analysis.py`
**Purpose:** Uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which time points in input windows the model focuses on.

**Functionality:**
- Provides interpretability insights into model decision-making
- Supports both MuscleActivityCNN (200ms windows) and PeakDetectionCNN (1162ms windows)
- Computes temporal importance maps showing model attention
- Visualizes which signal features drive predictions

**How to run:**
```bash
cd Scripts_Training_and_Analysis
python gradcam_analysis.py
```

**Output:**
- Grad-CAM visualization plots in `experiment_results/`
- Temporal importance maps showing model focus areas

#### `plot_error_correction.py`
**Purpose:** Generates visualization plots for error correction analysis results.

**Functionality:**
- Creates publication-ready plots from error correction data
- Visualizes detection patterns, errors, and corrections
- Supports multiple model comparisons

**How to run:**
```bash
cd Scripts_Training_and_Analysis
python plot_error_correction.py
```

**Output:**
- Error correction visualization plots in `experiment_results/`

## Usage Workflow

### Typical Workflow for Scientific Reviewers

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train models (if needed):**
   ```bash
   cd Scripts_Training_and_Analysis
   python train_cnn_onset_detection.py
   python train_cnn_error.py
   ```

3. **Evaluate models (main replication script):**
   ```bash
   python 1_evaluate_cnn_onset_detection.py
   ```

4. **Run analyses:**
   ```bash
   python error_correction.py
   python fast_recovery.py
   python gradcam_analysis.py
   ```

5. **View results:**
   - All plots and visualizations are saved in `Scripts_Training_and_Analysis/experiment_results/`
   - Trained models are saved in `Scripts_Training_and_Analysis/trained_models/`

### Notes for Usage

- All scripts are designed to be run from the `Scripts_Training_and_Analysis/` directory
- Scripts automatically create necessary output directories (`experiment_results/`, `trained_models/`)
- Most evaluation scripts provide interactive prompts to select which trained model to use
- Data paths are relative to the script location - ensure you're in the correct directory
- Pre-trained models may already be available in `trained_models/` - check before training

## Model Architectures

### PeakDetectionCNN
- **Input:** 1162ms windows (approximately 40 samples at 34.81 Hz)
- **Output:** Binary classification (peak in last 162ms or not)
- **Architecture:** 1D CNN with 3 convolutional layers, batch normalization, and dropout
- **Use case:** Activity onset detection

### MuscleActivityCNN
- **Input:** 200ms windows (approximately 7 samples at 34.81 Hz)
- **Output:** Binary classification (activity or baseline)
- **Architecture:** 1D CNN with 3 convolutional layers, no pooling to preserve sequence length
- **Use case:** Activity vs. baseline classification, error detection


## Replicating Core Results

### Main Replication Script: `1_evaluate_cnn_onset_detection.py`

**This is the primary script for replicating all CNN onset detection results reported in the publication.** It allows you to evaluate models trained on different datasets and compare their performance on healthy test participants or ALS patient data.

### Example: Replicating Healthy-to-ALS Transfer Results

**Result to replicate:** "When training a 1D convolutional neural network (CNN) model on the flexor pollicis longus EMG-RMS data of 15 healthy participants, and evaluating it on the ALS EMG-RMS data, we obtained a sensitivity of 100% and a specificity of 90.6%, i.e. the model could detect grasping intentions accurately, but resulted in a large number of false-positives."

**Replication steps:**
1. Navigate to the scripts directory:
   ```bash
   cd Scripts_Training_and_Analysis
   ```

2. Run the main evaluation script:
   ```bash
   python 1_evaluate_cnn_onset_detection.py
   ```

3. Follow the interactive prompts:
   - Select option **2** (evaluating on the ALS patient dataset)
   - Select option **1** (CNN Model Detection)
   - Select the trained model corresponding to "healthy" or "healthy plus sma" (typically option 2 or 3, depending on available models)

4. The script will:
   - Generate evaluation plots for ALS blocks 3 and 4 (close each plot window to continue)
   - Display a summary table in the terminal with baseline accuracy (specificity) and movement phase accuracy (sensitivity)
   - Save detailed visualizations in `experiment_results/`

**Note:** The movement phase accuracy corresponds to sensitivity, and baseline phase accuracy corresponds to specificity. The final values will be output in a summary table in the terminal.

### Comparing Different Training Datasets

The script allows you to compare models trained on different datasets:
- **Healthy only** (`cnn_healthy.pth`) - trained on P1-P11 or P1-P15
- **Healthy + ALS** (`cnn_healthy_plus_als.pth`) - trained on healthy participants plus ALS blocks 1-2
- **Healthy + SMA** (`cnn_healthy_plus_sma.pth`) - trained on healthy participants plus SMA data
- **ALS only** (`cnn_als.pth`) - trained only on ALS blocks 1-2
- **SMA only** (`cnn_sma.pth`) - trained only on SMA data

Simply select different models when prompted to compare their performance on the test data. 



## Citation

If you use this code or data in your research, please cite the associated scientific publication (paper reference will be added after publication).

