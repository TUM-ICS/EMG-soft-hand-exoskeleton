"""
Fast Recovery Analysis Script
============================

This script analyzes the recovery patterns after false positive detections during baseline phases.
It evaluates a trained CNN model or adaptive threshold method and tracks what happens in the
500ms following each false positive detection during baseline periods.

Analysis criteria:
- Movement phases: -200ms before peak to +800ms after peak
- Baseline phases: Everything that is NOT in a movement phase
- False positive: Detection during baseline phase
- Recovery window: 500ms after each false positive detection
- Recovery analysis: Track detection patterns in the recovery window
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('experiment_results', exist_ok=True)

# Import the CNN model and dataset from the main script
from train_cnn_onset_detection import PeakDetectionCNN, PeakDetectionDataset


def list_available_models():
    """List all available trained models."""
    models_dir = 'trained_models'
    if not os.path.exists(models_dir):
        print(f"No trained_models directory found")
        return []
    
    model_files = glob.glob(os.path.join(models_dir, 'cnn_*.pth'))
    if not model_files:
        print(f"No trained models found in {models_dir}")
        return []
    
    # Extract model names and sort them
    model_names = []
    for model_file in model_files:
        filename = os.path.basename(model_file)
        # Extract dataset name from filename (remove 'cnn_' prefix and '.pth' suffix)
        dataset_name = filename[4:-4].replace('_', ' ').replace('plus', '+')
        model_names.append((model_file, dataset_name))
    
    model_names.sort(key=lambda x: x[1])  # Sort by dataset name
    return model_names


def select_model():
    """Allow user to select a trained model or adaptive threshold method."""
    available_models = list_available_models()
    
    print("\nDetection method selection:")
    print("=" * 40)
    print("1. CNN Model Detection")
    if available_models:
        print("2. Adaptive Threshold Detection")
    else:
        print("2. Adaptive Threshold Detection (No trained models available)")
    
    while True:
        try:
            choice = input(f"\nSelect detection method (1-2): ").strip()
            
            if choice == "1":
                if not available_models:
                    print(" No trained models available. Please train a model first.")
                    return None, None, "cnn"
                
                print("\nAvailable trained models:")
                print("=" * 40)
                
                for i, (model_path, dataset_name) in enumerate(available_models, 1):
                    print(f"{i}. {dataset_name}")
                
                while True:
                    try:
                        model_choice = int(input(f"\nSelect model (1-{len(available_models)}): ")) - 1
                        if 0 <= model_choice < len(available_models):
                            selected_model_path, selected_dataset_name = available_models[model_choice]
                            print(f"Selected: {selected_dataset_name}")
                            return selected_model_path, selected_dataset_name, "cnn"
                        else:
                            print(f"Please enter a number between 1 and {len(available_models)}")
                    except ValueError:
                        print("Please enter a valid number")
                    except KeyboardInterrupt:
                        print("\n Operation cancelled")
                        return None, None, "cnn"
                        
            elif choice == "2":
                print(" Selected adaptive threshold detection")
                # Ask for lambda threshold
                while True:
                    try:
                        lambda_input = input("Enter lambda threshold (default: 2.0): ").strip()
                        if lambda_input == "":
                            lambda_threshold = 2.0
                        else:
                            lambda_threshold = float(lambda_input)
                        if lambda_threshold > 0:
                            break
                        else:
                            print(" Lambda threshold must be positive")
                    except ValueError:
                        print(" Please enter a valid number")
                    except KeyboardInterrupt:
                        print("\n Operation cancelled")
                        return None, None, "cnn"
                
                print(f"Lambda threshold set to: {lambda_threshold}")
                return None, f"adaptive_threshold_lambda_{lambda_threshold}", "adaptive", lambda_threshold
            else:
                print(" Please enter 1 or 2")
        except ValueError:
            print(" Please enter a valid number")
        except KeyboardInterrupt:
            print("\n Operation cancelled")
            return None, None, "cnn"


def load_signal_data(signal_file):
    """Load signal data from CSV file."""
    try:
        df = pd.read_csv(signal_file)
        return df
    except Exception as e:
        print(f"Error loading signal data: {e}")
        return None


def load_peak_labels(peak_file):
    """Load peak labels from CSV file."""
    try:
        df = pd.read_csv(peak_file)
        peak_timestamps = df['timestamp'].values
        return peak_timestamps
    except Exception as e:
        print(f"Error loading peak labels: {e}")
        return None


def load_test_data():
    """Load test data from P12-P15."""
    print("Loading Test Data (P12-P15)")
    print("=" * 30)
    
    signal_dir = '../EMG_data/signal_data'
    label_dir = '../EMG_data/label_data'
    
    test_data = []
    
    # Load P12-P15
    for participant_num in range(12, 16):
        signal_file = os.path.join(signal_dir, f'RMS_healthy_P{participant_num}_34p81hz_processed_cleaned.csv')
        peak_file = os.path.join(label_dir, f'peaks_P{participant_num}_interactive_final.csv')
        
        signal_data = load_signal_data(signal_file)
        peak_timestamps = load_peak_labels(peak_file)
        
        if signal_data is not None and peak_timestamps is not None:
            test_data.append({
                'participant': f'P{participant_num}',
                'signal_data': signal_data,
                'peak_timestamps': peak_timestamps
            })
            print(f"P{participant_num}: {len(signal_data)} samples, {len(peak_timestamps)} peaks")
        else:
            print(f"Failed to load P{participant_num}")
    
    print(f"\nTotal test participants: {len(test_data)}")
    
    return test_data


def load_als_data():
    """Load the last two blocks of ALS data."""
    print("Loading ALS Data (Last Two Blocks)")
    print("=" * 35)
    
    signal_dir = '../EMG_data/signal_data'
    label_dir = '../EMG_data/label_data'
    
    als_data = []
    
    # Load the last two blocks of ALS data (block3 and block4)
    # als_blocks = ['block3', 'block4']  # Last two blocks
    als_blocks = ['block3']  # Last two blocks
    
    for block in als_blocks:
        signal_file = os.path.join(signal_dir, f'RMS_ALS_{block}.csv')
        peak_file = os.path.join(label_dir, f'peaks_ALS_{block}.csv')
        
        signal_data = load_signal_data(signal_file)
        peak_timestamps = load_peak_labels(peak_file)
        
        if signal_data is not None and peak_timestamps is not None:
            als_data.append({
                'participant': f'ALS_{block}',
                'signal_data': signal_data,
                'peak_timestamps': peak_timestamps
            })
            print(f"ALS {block}: {len(signal_data)} samples, {len(peak_timestamps)} peaks")
        else:
            print(f"Failed to load ALS {block}")
    
    print(f"\nTotal ALS blocks: {len(als_data)}")
    
    return als_data


def create_movement_phases(peak_timestamps, pre_peak_ms=200, post_peak_ms=800):
    """
    Create movement phases around peaks.
    
    Args:
        peak_timestamps: Array of peak timestamps in seconds
        pre_peak_ms: Time before peak to include in movement phase (ms)
        post_peak_ms: Time after peak to include in movement phase (ms)
    
    Returns:
        List of (start_time, end_time) tuples for movement phases
    """
    movement_phases = []
    
    for peak_time in peak_timestamps:
        start_time = peak_time - (pre_peak_ms / 1000.0)  # Convert ms to seconds
        end_time = peak_time + (post_peak_ms / 1000.0)
        movement_phases.append((start_time, end_time))
    
    return movement_phases


def is_in_movement_phase(timestamp, movement_phases):
    """Check if a timestamp falls within any movement phase."""
    for start_time, end_time in movement_phases:
        if start_time <= timestamp <= end_time:
            return True
    return False


def is_in_baseline_phase(timestamp, baseline_phases):
    """Check if a timestamp falls within any baseline phase."""
    for start_time, end_time in baseline_phases:
        if start_time <= timestamp <= end_time:
            return True
    return False


def create_movement_phases(peak_timestamps, pre_peak_ms=200, post_peak_ms=800):
    """
    Create movement phases only. Everything else is considered baseline.
    
    Movement phases: -200ms before peak to +800ms after peak
    Baseline: Everything that is NOT in a movement phase
    
    Args:
        peak_timestamps: Array of peak timestamps in seconds
        pre_peak_ms: Time before peak to include in movement phase (ms)
        post_peak_ms: Time after peak to include in movement phase (ms)
    
    Returns:
        List of (start_time, end_time) tuples for movement phases
    """
    if len(peak_timestamps) == 0:
        return []
    
    # Sort peak timestamps
    sorted_peaks = np.sort(peak_timestamps)
    
    # Create movement phases for each peak
    movement_phases = []
    for peak_time in sorted_peaks:
        start_time = peak_time - (pre_peak_ms / 1000.0)  # Convert ms to seconds
        end_time = peak_time + (post_peak_ms / 1000.0)
        movement_phases.append((start_time, end_time))
    
    # Debug information
    print(f" Created {len(movement_phases)} movement phases (-{pre_peak_ms}ms to +{post_peak_ms}ms)")
    print(f" Everything else will be treated as baseline")
    
    return movement_phases


def adaptive_threshold_detection(signal_data, movement_phases, window_size_ms=1162, 
                                peak_detection_ms=162, sampling_rate=34.81, lambda_threshold=2.0):
    """
    Detect activity onsets using adaptive threshold approach.
    
    Args:
        signal_data: DataFrame with signal data
        movement_phases: List of movement phase time ranges
        window_size_ms: Size of detection window in ms
        peak_detection_ms: Size of peak detection zone in ms
        sampling_rate: Sampling rate in Hz
        lambda_threshold: Threshold multiplier for standard deviation
    
    Returns:
        Dictionary with evaluation results
    """
    timestamps = signal_data['timestamp'].values
    signal_column = 'rms' if 'rms' in signal_data.columns else 'emg'
    signal_values = signal_data[signal_column].values
    
    window_size_samples = int(window_size_ms * sampling_rate / 1000)
    peak_detection_samples = int(peak_detection_ms * sampling_rate / 1000)
    rest_samples = window_size_samples - peak_detection_samples
    
    print(f" Adaptive threshold detection:")
    print(f" Signal duration: {timestamps[-1] - timestamps[0]:.2f}s")
    print(f" Window size: {window_size_ms}ms, Activity window: {peak_detection_ms}ms, Rest window: {rest_samples/sampling_rate*1000:.0f}ms")
    print(f" Lambda threshold: {lambda_threshold}")
    print(f" Movement phases: {len(movement_phases)}")
    
    # Debug: Show phase details
    if len(movement_phases) > 0:
        print(f" Movement phase range: {movement_phases[0][0]:.2f}s - {movement_phases[-1][1]:.2f}s")
        print(f" Movement phases: {[(f'{start:.2f}s', f'{end:.2f}s') for start, end in movement_phases]}")
    else:
        print(f"   No movement phases created! This could be due to:")
        print(f"    - No peaks detected")
        print(f"    - Empty peak timestamps")
    
    # Evaluate movement phases
    movement_phase_results = []
    detection_delays = []
    
    # Evaluate all windows with adaptive threshold (sample-level evaluation)
    all_detections = []
    
    print(f" Evaluating all windows with adaptive threshold (sample-level)")
    
    for j in range(len(signal_values) - window_size_samples + 1):
        window_start_time = timestamps[j]
        window_end_time = timestamps[j + window_size_samples - 1]
        
        # Get current window
        window = signal_values[j:j + window_size_samples]
        
        # Split window into activity and rest parts
        activity_window = window[-peak_detection_samples:]  # Last 162ms
        rest_window = window[:-peak_detection_samples]      # First 1000ms
        
        # Calculate statistics
        activity_mean = np.mean(activity_window)
        rest_mean = np.mean(rest_window)
        rest_std = np.std(rest_window)
        
        # Adaptive threshold: rest_mean + lambda * rest_std
        threshold = rest_mean + lambda_threshold * rest_std
        
        # Detection decision
        detected = activity_mean > threshold
        probability = min(1.0, max(0.0, (activity_mean - rest_mean) / (rest_std + 1e-8)))  # Normalized probability
        
        # Check if detection time is in movement phase
        is_movement = is_in_movement_phase(window_end_time, movement_phases)
        is_baseline = not is_movement
        
        all_detections.append({
            'timestamp': window_end_time,
            'activity_mean': activity_mean,
            'rest_mean': rest_mean,
            'rest_std': rest_std,
            'threshold': threshold,
            'probability': probability,
            'detected': detected,
            'is_movement_phase': is_movement,
            'is_baseline_phase': is_baseline
        })
        
        # Calculate delay for movement phase detections
        if detected and is_movement:
            # Find the closest movement phase for this detection
            for start_time, end_time in movement_phases:
                if start_time <= window_end_time <= end_time:
                    # Calculate delay (use middle of movement phase as reference)
                    phase_middle = (start_time + end_time) / 2
                    delay = window_end_time - phase_middle
                    detection_delays.append(delay)
                    break
    
    # Separate movement and baseline detections
    movement_detections = [d for d in all_detections if d['is_movement_phase']]
    baseline_detections = [d for d in all_detections if d['is_baseline_phase']]
    
    # Movement phase evaluation (sample-level - should have detections)
    movement_detected = [d for d in movement_detections if d['detected']]
    movement_correct = len(movement_detected)
    movement_total = len(movement_detections)
    movement_accuracy = movement_correct / movement_total if movement_total > 0 else 0
    
    print(f" Movement Phase (sample-level):")
    print(f"   Total samples: {movement_total}")
    print(f"   Correct (detected): {movement_correct}")
    print(f"   Missed detections: {movement_total - movement_correct}")
    print(f"   Accuracy: {movement_accuracy:.3f}")
    
    # Create movement phase results for compatibility
    for i, (start_time, end_time) in enumerate(movement_phases):
        phase_detections = [d for d in movement_detections if start_time <= d['timestamp'] <= end_time]
        phase_detected = [d for d in phase_detections if d['detected']]
        
        movement_phase_results.append({
            'phase_id': i,
            'start_time': start_time,
            'end_time': end_time,
            'correct': len(phase_detected) > 0,
            'first_detection_time': min(d['timestamp'] for d in phase_detected) if phase_detected else None,
            'detections': phase_detections,
            'max_activity_mean': max(d['activity_mean'] for d in phase_detections) if phase_detections else 0.0,
            'max_threshold': max(d['threshold'] for d in phase_detections) if phase_detections else 0.0
        })
    
    # Calculate baseline metrics (sample-level - should have no detections)
    baseline_detected = [d for d in baseline_detections if d['detected']]
    baseline_correct = len(baseline_detections) - len(baseline_detected)  # Correct if no detection
    baseline_total = len(baseline_detections)
    baseline_accuracy = baseline_correct / baseline_total if baseline_total > 0 else 1.0
    
    print(f" Baseline Phase (sample-level):")
    print(f"   Total samples: {baseline_total}")
    print(f"   Correct (no detection): {baseline_correct}")
    print(f"   False positives: {baseline_total - baseline_correct}")
    print(f"   Accuracy: {baseline_accuracy:.3f}")
    
    if baseline_total == 0:
        print(f"   No baseline samples found! Possible reasons:")
        print(f"    - No baseline phases created (peaks too close or only one peak)")
        print(f"    - Signal duration too short for baseline phases")
        print(f"    - Window size too large for available data")
    
    # Movement phase metrics are already calculated above (sample-level)
    
    avg_delay = np.mean(detection_delays) if detection_delays else 0
    
    return {
        'baseline_accuracy': baseline_accuracy,
        'baseline_correct': baseline_correct,
        'baseline_total': baseline_total,
        'movement_phase_accuracy': movement_accuracy,
        'movement_phase_correct': movement_correct,
        'movement_phase_total': movement_total,
        'average_delay_ms': avg_delay * 1000,  # Convert to milliseconds
        'detection_delays': detection_delays,
        'movement_phase_results': movement_phase_results,
        'baseline_detections': baseline_detections,
        'all_detections': all_detections,
        'method': 'adaptive_threshold',
        'lambda_threshold': lambda_threshold
    }


def sliding_window_evaluation(model, signal_data, movement_phases, window_size_ms=1162, 
                            peak_detection_ms=162, sampling_rate=34.81, threshold=0.5):
    """
    Evaluate model using sliding window approach similar to real-time.
    
    Args:
        model: Trained CNN model
        signal_data: DataFrame with signal data
        movement_phases: List of movement phase time ranges
        window_size_ms: Size of detection window in ms
        peak_detection_ms: Size of peak detection zone in ms
        sampling_rate: Sampling rate in Hz
        threshold: Detection threshold for peak classification
    
    Returns:
        Dictionary with evaluation results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    timestamps = signal_data['timestamp'].values
    signal_column = 'rms' if 'rms' in signal_data.columns else 'emg'
    signal_values = signal_data[signal_column].values
    
    window_size_samples = int(window_size_ms * sampling_rate / 1000)
    
    # Results storage
    detections = []  # (timestamp, probability, is_in_movement_phase)
    baseline_phases = []
    movement_phase_results = []
    detection_delays = []
    
    print(f" Evaluating with sliding window (step size: 1 sample)")
    print(f" Window size: {window_size_ms}ms, Peak detection zone: {peak_detection_ms}ms")
    
    # Sliding window evaluation
    for i in range(len(signal_values) - window_size_samples + 1):
        # Get current window
        window = signal_values[i:i + window_size_samples]
        window_start_time = timestamps[i]  # Start of window
        window_timestamp = timestamps[i + window_size_samples - 1]  # End of window (detection time)
        
        # Check if detection time is in movement phase
        is_movement = is_in_movement_phase(window_timestamp, movement_phases)
        is_baseline = not is_movement  # Everything not in movement is baseline
        
        # Get model prediction
        with torch.no_grad():
            window_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, window_size)
            output = model(window_tensor)
            probability = torch.sigmoid(output).cpu().item()
        
        # Store detection
        detections.append({
            'timestamp': window_timestamp,
            'probability': probability,
            'is_movement_phase': is_movement,
                'is_baseline_phase': is_baseline,
            'detected': probability > threshold
        })
    
    # Analyze results - convert to structured array
    # Extract values from dictionaries
    timestamps = np.array([d['timestamp'] for d in detections])
    probabilities = np.array([d['probability'] for d in detections])
    is_movement_phases = np.array([d['is_movement_phase'] for d in detections])
    is_baseline_phases = np.array([d['is_baseline_phase'] for d in detections])
    detected_flags = np.array([d['detected'] for d in detections])
    
    # Create structured array
    detections = np.array(list(zip(timestamps, probabilities, is_movement_phases, is_baseline_phases, detected_flags)), 
                         dtype=[('timestamp', 'f8'), ('probability', 'f8'), 
                                ('is_movement_phase', 'bool'), ('is_baseline_phase', 'bool'), ('detected', 'bool')])
    
    # Separate baseline and movement phase detections
    is_movement_phase = detections['is_movement_phase']
    is_baseline_phase = detections['is_baseline_phase']
    baseline_detections = detections[is_baseline_phase]
    movement_detections = detections[is_movement_phase]
    
    # Baseline phase evaluation (sample-level - should have no detections)
    baseline_detected = baseline_detections['detected']
    baseline_correct = np.sum(~baseline_detected)
    baseline_total = len(baseline_detections)
    baseline_accuracy = baseline_correct / baseline_total if baseline_total > 0 else 1.0
    
    print(f" Baseline Phase (sample-level):")
    print(f"   Total samples: {baseline_total}")
    print(f"   Correct (no detection): {baseline_correct}")
    print(f"   False positives: {baseline_total - baseline_correct}")
    print(f"   Accuracy: {baseline_accuracy:.3f}")
    
    if baseline_total == 0:
        print(f"   No baseline samples found! Possible reasons:")
        print(f"    - No baseline phases created (peaks too close or only one peak)")
        print(f"    - Signal duration too short for baseline phases")
        print(f"    - Window size too large for available data")
    
    # Movement phase evaluation (should have at least one detection per phase)
    movement_phase_accuracy = 0
    movement_phase_total = len(movement_phases)
    
    for start_time, end_time in movement_phases:
        # Find detections within this movement phase
        phase_mask = (movement_detections['timestamp'] >= start_time) & (movement_detections['timestamp'] <= end_time)
        phase_detections = movement_detections[phase_mask]
        
        if len(phase_detections) > 0:
            phase_detected = phase_detections['detected']
            if np.any(phase_detected):
                movement_phase_accuracy += 1
                
                # Calculate delay for this phase
                first_detection_idx = np.where(phase_detected)[0][0]
                first_detection_time = phase_detections[first_detection_idx]['timestamp']
                actual_peak_time = (start_time + end_time) / 2  # Approximate peak time
                delay = first_detection_time - actual_peak_time
                detection_delays.append(delay)
    
    movement_phase_accuracy = movement_phase_accuracy / movement_phase_total if movement_phase_total > 0 else 0
    
    # Calculate average delay
    avg_delay = np.mean(detection_delays) if detection_delays else 0
    
    return {
        'baseline_accuracy': baseline_accuracy,
        'baseline_correct': baseline_correct,
        'baseline_total': baseline_total,
        'movement_phase_accuracy': movement_phase_accuracy,
        'movement_phase_correct': movement_phase_accuracy * movement_phase_total,
        'movement_phase_total': movement_phase_total,
        'average_delay_ms': avg_delay * 1000,  # Convert to milliseconds
        'detection_delays': detection_delays,
        'baseline_detections': baseline_detections,
        'all_detections': detections
    }


def analyze_fast_recovery(signal_data, movement_phases, detections, recovery_window_ms=500, 
                         sampling_rate=34.81):
    """
    Analyze recovery patterns after first false positive detections and first movement detections.
    
    Args:
        signal_data: DataFrame with signal data
        movement_phases: List of movement phase time ranges
        detections: List of detection dictionaries or structured array
        recovery_window_ms: Time window to analyze after detection (ms)
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Dictionary with recovery analysis results
    """
    timestamps = signal_data['timestamp'].values
    recovery_window_samples = int(recovery_window_ms * sampling_rate / 1000)
    
    # Convert detections to list of dictionaries if it's a structured array
    if isinstance(detections, np.ndarray):
        detections_list = []
        for i in range(len(detections)):
            detection_dict = {
                'timestamp': detections[i]['timestamp'],
                'detected': detections[i]['detected'],
                'is_movement_phase': detections[i]['is_movement_phase'],
                'is_baseline_phase': detections[i]['is_baseline_phase']
            }
            # Add probability if it exists
            if 'probability' in detections.dtype.names:
                detection_dict['probability'] = detections[i]['probability']
            else:
                detection_dict['probability'] = 0.5  # Default value
            detections_list.append(detection_dict)
        detections = detections_list
    
    # Find all detections (both false positives and correct detections)
    all_detections = [d for d in detections if d['detected']]
    false_positives = [d for d in all_detections if d['is_baseline_phase']]
    movement_detections = [d for d in all_detections if d['is_movement_phase']]
    
    print(f" Found {len(false_positives)} false positive detections during baseline phases")
    print(f" Found {len(movement_detections)} correct detections during movement phases")
    
    # Analyze first false positives (not within 500ms of previous false positive)
    first_false_positives = []
    covered_times = set()  # Track times already covered by previous false positives
    
    for fp in false_positives:
        fp_time = fp['timestamp']
        
        # Check if this false positive is within 500ms of any previous false positive
        is_within_500ms = False
        for covered_time in covered_times:
            if abs(fp_time - covered_time) <= 0.5:  # 500ms = 0.5s
                is_within_500ms = True
                break
        
        if not is_within_500ms:
            first_false_positives.append(fp)
            covered_times.add(fp_time)
    
    print(f" Found {len(first_false_positives)} first false positives (not within 500ms of previous)")
    
    # Analyze first false positives
    first_fp_analyses = []
    for i, fp in enumerate(first_false_positives):
        fp_time = fp['timestamp']
        
        # Find all detections in the following 500ms
        recovery_end_time = fp_time + (recovery_window_ms / 1000)  # Convert to seconds
        following_detections = [d for d in all_detections if fp_time < d['timestamp'] <= recovery_end_time]
        
        # Count detections by phase
        movement_count = len([d for d in following_detections if d['is_movement_phase']])
        baseline_count = len([d for d in following_detections if d['is_baseline_phase']])
        total_count = len(following_detections)
        
        first_fp_analyses.append({
            'fp_time': fp_time,
            'fp_probability': fp['probability'],
            'total_detections': total_count,
            'movement_detections': movement_count,
            'baseline_detections': baseline_count,
            'following_detections': following_detections
        })
        
        print(f"   First FP {i+1}: {fp_time:.2f}s -> {total_count} detections in 500ms "
              f"(M:{movement_count}, B:{baseline_count})")
    
    # Calculate averages for first false positives
    if first_fp_analyses:
        avg_fp_total = np.mean([fp['total_detections'] for fp in first_fp_analyses])
        avg_fp_movement = np.mean([fp['movement_detections'] for fp in first_fp_analyses])
        avg_fp_baseline = np.mean([fp['baseline_detections'] for fp in first_fp_analyses])
    else:
        avg_fp_total = 0
        avg_fp_movement = 0
        avg_fp_baseline = 0
    
    print(f" First FP Averages: {avg_fp_total:.1f} total, {avg_fp_movement:.1f} movement, {avg_fp_baseline:.1f} baseline")
    
    # Analyze first detections in each movement phase
    first_movement_analyses = []
    for i, (start_time, end_time) in enumerate(movement_phases):
        # Find first detection in this movement phase
        phase_detections = [d for d in movement_detections if start_time <= d['timestamp'] <= end_time]
        
        if phase_detections:
            first_detection = min(phase_detections, key=lambda x: x['timestamp'])
            first_time = first_detection['timestamp']
            
            # Find all detections in the following 500ms
            recovery_end_time = first_time + (recovery_window_ms / 1000)  # Convert to seconds
            following_detections = [d for d in all_detections if first_time < d['timestamp'] <= recovery_end_time]
            
            # Count detections by phase
            movement_count = len([d for d in following_detections if d['is_movement_phase']])
            baseline_count = len([d for d in following_detections if d['is_baseline_phase']])
            total_count = len(following_detections)
            
            first_movement_analyses.append({
                'phase_id': i,
                'phase_start': start_time,
                'phase_end': end_time,
                'first_detection_time': first_time,
                'first_detection_probability': first_detection['probability'],
                'total_detections': total_count,
                'movement_detections': movement_count,
                'baseline_detections': baseline_count,
                'following_detections': following_detections
            })
            
            print(f"   Movement Phase {i+1}: First detection at {first_time:.2f}s -> {total_count} detections in 500ms "
                  f"(M:{movement_count}, B:{baseline_count})")
        else:
            print(f"   Movement Phase {i+1}: No detections found")
    
    # Calculate averages for first movement detections
    if first_movement_analyses:
        avg_movement_total = np.mean([m['total_detections'] for m in first_movement_analyses])
        avg_movement_movement = np.mean([m['movement_detections'] for m in first_movement_analyses])
        avg_movement_baseline = np.mean([m['baseline_detections'] for m in first_movement_analyses])
    else:
        avg_movement_total = 0
        avg_movement_movement = 0
        avg_movement_baseline = 0
    
    print(f" First Movement Averages: {avg_movement_total:.1f} total, {avg_movement_movement:.1f} movement, {avg_movement_baseline:.1f} baseline")
    
    return {
        'first_false_positives': first_false_positives,
        'first_fp_analyses': first_fp_analyses,
        'first_movement_analyses': first_movement_analyses,
        'summary': {
            'total_first_fps': len(first_false_positives),
            'total_movement_phases': len(movement_phases),
            'avg_fp_total': avg_fp_total,
            'avg_fp_movement': avg_fp_movement,
            'avg_fp_baseline': avg_fp_baseline,
            'avg_movement_total': avg_movement_total,
            'avg_movement_movement': avg_movement_movement,
            'avg_movement_baseline': avg_movement_baseline
        }
    }


def analyze_detection_frequency_histograms(signal_data, movement_phases, detections, 
                                         analysis_window_ms=500, sampling_rate=34.81):
    """
    Analyze detection frequency histograms for false positives and correct detections.
    
    Args:
        signal_data: DataFrame with signal data
        movement_phases: List of movement phase time ranges
        detections: List of detection dictionaries or structured array
        analysis_window_ms: Time window to analyze after each detection (ms)
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Dictionary with histogram analysis results
    """
    timestamps = signal_data['timestamp'].values
    analysis_window_samples = int(analysis_window_ms * sampling_rate / 1000)
    
    # Convert detections to list of dictionaries if it's a structured array
    if isinstance(detections, np.ndarray):
        detections_list = []
        for i in range(len(detections)):
            detection_dict = {
                'timestamp': detections[i]['timestamp'],
                'detected': detections[i]['detected'],
                'is_movement_phase': detections[i]['is_movement_phase'],
                'is_baseline_phase': detections[i]['is_baseline_phase']
            }
            # Add probability if it exists
            if 'probability' in detections.dtype.names:
                detection_dict['probability'] = detections[i]['probability']
            else:
                detection_dict['probability'] = 0.5  # Default value
            detections_list.append(detection_dict)
        detections = detections_list
    
    # Find false positive detections (detected during baseline phases)
    false_positives = [d for d in detections if d['detected'] and d['is_baseline_phase']]
    
    # Find correct detections (detected during movement phases)
    correct_detections = [d for d in detections if d['detected'] and d['is_movement_phase']]
    
    print(f" Detection Frequency Analysis:")
    print(f" Found {len(false_positives)} false positive detections in baseline phases")
    print(f" Found {len(correct_detections)} correct detections in movement phases")
    
    # Analyze false positives
    fp_following_detections = []
    for fp in false_positives:
        fp_time = fp['timestamp']
        fp_idx = np.argmin(np.abs(timestamps - fp_time))
        
        # Define analysis window
        analysis_start_idx = fp_idx
        analysis_end_idx = min(fp_idx + analysis_window_samples, len(timestamps) - 1)
        analysis_end_time = timestamps[analysis_end_idx]
        
        # Find all detections in the analysis window
        following_detections = []
        for d in detections:
            if fp_time < d['timestamp'] <= analysis_end_time:  # Exclude the original detection
                following_detections.append(d)
        
        fp_following_detections.append(len(following_detections))
    
    # Analyze correct detections
    correct_following_detections = []
    for cd in correct_detections:
        cd_time = cd['timestamp']
        cd_idx = np.argmin(np.abs(timestamps - cd_time))
        
        # Define analysis window
        analysis_start_idx = cd_idx
        analysis_end_idx = min(cd_idx + analysis_window_samples, len(timestamps) - 1)
        analysis_end_time = timestamps[analysis_end_idx]
        
        # Find all detections in the analysis window
        following_detections = []
        for d in detections:
            if cd_time < d['timestamp'] <= analysis_end_time:  # Exclude the original detection
                following_detections.append(d)
        
        correct_following_detections.append(len(following_detections))
    
    # Calculate statistics
    fp_avg_following = np.mean(fp_following_detections) if fp_following_detections else 0
    correct_avg_following = np.mean(correct_following_detections) if correct_following_detections else 0
    
    print(f" False positives - Average detections in following {analysis_window_ms}ms: {fp_avg_following:.2f}")
    print(f" Correct detections - Average detections in following {analysis_window_ms}ms: {correct_avg_following:.2f}")
    
    return {
        'false_positives': {
            'detections': false_positives,
            'following_detection_counts': fp_following_detections,
            'average_following_detections': fp_avg_following
        },
        'correct_detections': {
            'detections': correct_detections,
            'following_detection_counts': correct_following_detections,
            'average_following_detections': correct_avg_following
        },
        'analysis_window_ms': analysis_window_ms
    }


def visualize_detection_frequency_histograms(participant_name, histogram_results, 
                                           dataset_name="Unknown", method="cnn", save_all=False):
    """Visualize detection frequency histograms for false positives and correct detections."""
    
    fp_counts = histogram_results['false_positives']['following_detection_counts']
    correct_counts = histogram_results['correct_detections']['following_detection_counts']
    analysis_window_ms = histogram_results['analysis_window_ms']
    
    if not fp_counts and not correct_counts:
        print(f" No detections found for {participant_name} - skipping histogram visualization")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot false positive histogram
    if fp_counts:
        # Create histogram bins
        max_count = max(max(fp_counts) if fp_counts else 0, max(correct_counts) if correct_counts else 0)
        bins = np.arange(0, max_count + 2) - 0.5
        
        ax1.hist(fp_counts, bins=bins, alpha=0.7, color='red', edgecolor='black')
        ax1.set_xlabel(f'Number of Detections in Following {analysis_window_ms}ms')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'False Positive Detections\n(n={len(fp_counts)}, avg={np.mean(fp_counts):.2f})')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        ax1.text(0.7, 0.9, f'Mean: {np.mean(fp_counts):.2f}\nStd: {np.std(fp_counts):.2f}\nMax: {np.max(fp_counts)}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax1.text(0.5, 0.5, 'No false positive detections', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('False Positive Detections\n(n=0)')
    
    # Plot correct detection histogram
    if correct_counts:
        ax2.hist(correct_counts, bins=bins, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel(f'Number of Detections in Following {analysis_window_ms}ms')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Correct Detections\n(n={len(correct_counts)}, avg={np.mean(correct_counts):.2f})')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        ax2.text(0.7, 0.9, f'Mean: {np.mean(correct_counts):.2f}\nStd: {np.std(correct_counts):.2f}\nMax: {np.max(correct_counts)}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No correct detections', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Correct Detections\n(n=0)')
    
    plt.tight_layout()
    
    # Save plot based on user preference
    if save_all:
        plot_filename = f"experiment_results/{participant_name}_detection_frequency_histogram_{dataset_name}_{method}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved histogram plot: {plot_filename}")
    
    plt.show()


def interactive_false_positive_explorer(participant_name, signal_data, peak_timestamps, recovery_results, 
                                      dataset_name="Unknown", method="cnn", save_all=False):
    """Interactive exploration of first false positives with scrolling through each one."""
    
    if not recovery_results or recovery_results['summary']['total_first_fps'] == 0:
        print(f" No first false positives found for {participant_name} - skipping interactive exploration")
        return
    
    first_fp_analyses = recovery_results['first_fp_analyses']
    print(f"\n  Interactive First False Positive Explorer for {participant_name}")
    print(f" Found {len(first_fp_analyses)} first false positives to explore")
    print(f" Use 'n' for next, 'p' for previous, 'q' to quit, or number to jump to specific FP")
    
    # Create movement phases for visualization
    movement_phases = create_movement_phases(peak_timestamps, pre_peak_ms=200, post_peak_ms=800)
    
    # Get signal data
    signal_column = 'rms' if 'rms' in signal_data.columns else 'emg'
    timestamps = signal_data['timestamp'].values
    signal_values = signal_data[signal_column].values
    
    current_fp = 0
    
    while True:
        if current_fp < 0 or current_fp >= len(first_fp_analyses):
            print(f" Invalid FP index: {current_fp}. Please use 0-{len(first_fp_analyses)-1}")
            current_fp = max(0, min(current_fp, len(first_fp_analyses) - 1))
            continue
        
        fp_analysis = first_fp_analyses[current_fp]
        fp_time = fp_analysis['fp_time']
        total_detections = fp_analysis['total_detections']
        movement_detections = fp_analysis['movement_detections']
        baseline_detections = fp_analysis['baseline_detections']
        
        # Calculate recovery window
        recovery_start = fp_time
        recovery_end = fp_time + 0.5  # 500ms = 0.5s
        
        print(f"\n  First FP {current_fp + 1}/{len(first_fp_analyses)}: t={fp_time:.2f}s")
        print(f"   Detections in following 500ms: {total_detections} (M:{movement_detections}, B:{baseline_detections})")
        
        # Create plot for this false positive
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Define time window for this false positive
        time_margin = 1.0  # seconds
        plot_start = max(recovery_start - time_margin, timestamps[0])
        plot_end = min(recovery_end + time_margin, timestamps[-1])
        
        # Find indices for plotting
        start_idx = np.argmin(np.abs(timestamps - plot_start))
        end_idx = np.argmin(np.abs(timestamps - plot_end))
        
        # Plot signal
        plot_times = timestamps[start_idx:end_idx]
        plot_signal = signal_values[start_idx:end_idx]
        ax.plot(plot_times, plot_signal, 'b-', alpha=0.7, linewidth=0.8, label='RMS EMG Signal')
        
        # Highlight movement phases in this range
        for start_time, end_time in movement_phases:
            if start_time <= plot_end and end_time >= plot_start:
                ax.axvspan(max(start_time, plot_start), min(end_time, plot_end), 
                          alpha=0.3, color='red', label='Movement Phase' if start_time == movement_phases[0][0] else "")
        
        # Mark the false positive
        ax.axvline(fp_time, color='red', linestyle='--', linewidth=2, label='False Positive')
        
        # Mark recovery window
        ax.axvspan(recovery_start, recovery_end, alpha=0.2, color='orange', 
                  label=f'Analysis Window (500ms)')
        
        # Plot actual peaks in this range
        peak_times_in_range = [pt for pt in peak_timestamps if plot_start <= pt <= plot_end]
        if peak_times_in_range:
            peak_values = []
            for peak_time in peak_times_in_range:
                closest_idx = np.argmin(np.abs(timestamps - peak_time))
                peak_values.append(signal_values[closest_idx])
            
            ax.scatter(peak_times_in_range, peak_values, 
                      color='red', s=50, label=f'Actual Peaks ({len(peak_times_in_range)})', zorder=5)
        
        # Plot detections in recovery window
        following_detections = fp_analysis['following_detections']
        if following_detections:
            detection_times = [d['timestamp'] for d in following_detections]
            detection_probs = [d['probability'] for d in following_detections]
            
            if detection_times:
                # Scale probabilities to signal range
                prob_scale = np.max(plot_signal) * 0.8
                scaled_probs = [p * prob_scale for p in detection_probs]
                
                ax.scatter(detection_times, scaled_probs, 
                          color='green', s=15, label=f'Following Detections ({len(detection_times)})', zorder=5)
        
        # Formatting
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('RMS EMG (V)')
        ax.set_title(f'First FP {current_fp + 1}/{len(first_fp_analyses)}: {participant_name} - False Positive Analysis\n'
                    f't={fp_time:.2f}s, Following Detections: {total_detections} (M:{movement_detections}, B:{baseline_detections})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot based on user preference
        if save_all:
            plot_filename = f"experiment_results/{participant_name}_fp_{current_fp+1}_analysis_{dataset_name}_{method}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved FP analysis plot: {plot_filename}")
        
        plt.show()
        
        # Get user input
        user_input = input(f"\n  First FP {current_fp + 1}/{len(first_fp_analyses)} - Command (n/p/q/number): ").strip().lower()
        
        if user_input == 'q':
            print("  Exiting false positive explorer")
            break
        elif user_input == 'n':
            current_fp = min(current_fp + 1, len(first_fp_analyses) - 1)
        elif user_input == 'p':
            current_fp = max(current_fp - 1, 0)
        elif user_input.isdigit():
            fp_num = int(user_input) - 1
            if 0 <= fp_num < len(first_fp_analyses):
                current_fp = fp_num
            else:
                print(f" Invalid FP number. Please use 1-{len(first_fp_analyses)}")
        else:
            print("  Invalid command. Use 'n' (next), 'p' (previous), 'q' (quit), or number (1-{})".format(len(first_fp_analyses)))


def evaluate_participant(model, participant_data, window_size_ms=1162, peak_detection_ms=162, method="cnn", lambda_threshold=2.0, save_all=False):
    """Evaluate a single participant."""
    participant_name = participant_data['participant']
    signal_data = participant_data['signal_data']
    peak_timestamps = participant_data['peak_timestamps']
    
    print(f"\nEvaluating {participant_name}")
    print("-" * 20)
    
    # Create movement phases (everything else is baseline)
    movement_phases = create_movement_phases(peak_timestamps, pre_peak_ms=200, post_peak_ms=800)
    print(f"Created {len(movement_phases)} movement phases")
    print(f"Everything else will be treated as baseline")
    
    # Run evaluation based on selected method
    if method == "adaptive":
        results = adaptive_threshold_detection(
            signal_data, movement_phases,
            window_size_ms, peak_detection_ms, lambda_threshold=lambda_threshold
        )
    else:  # CNN method
        results = sliding_window_evaluation(
            model, signal_data, movement_phases,
            window_size_ms, peak_detection_ms
        )
    
    # Print results
    print(f" Baseline Phase:")
    print(f"   Accuracy: {results['baseline_accuracy']:.3f} ({results['baseline_correct']}/{results['baseline_total']})")
    print(f" Movement Phase:")
    print(f"   Accuracy: {results['movement_phase_accuracy']:.3f} ({results['movement_phase_correct']:.0f}/{results['movement_phase_total']})")
    print(f" Average Detection Delay: {results['average_delay_ms']:.1f} ms")
    
    # Perform fast recovery analysis
    print(f"\n  Fast Recovery Analysis:")
    print(f" " + "="*30)
    
    # Get detections for recovery analysis
    if method == "adaptive":
        # For adaptive method, combine baseline and movement detections
        all_detections = []
        baseline_detections = results.get('baseline_detections', [])
        movement_detections = results.get('movement_phase_results', [])
        
        print(f" Debug: Found {len(baseline_detections)} baseline detections")
        print(f" Debug: Found {len(movement_detections)} movement phase results")
        
        # Debug: Show structure of first baseline detection
        if baseline_detections:
            print(f" Debug: First baseline detection keys: {list(baseline_detections[0].keys())}")
        
        # Add baseline detections
        for d in baseline_detections:
            all_detections.append({
                'timestamp': d['timestamp'],
                'probability': d.get('probability', 0.5),  # Default if not present
                'detected': d['detected'],
                'is_movement_phase': False,
                'is_baseline_phase': True
            })
        
        # Add movement detections
        for phase_result in movement_detections:
            for d in phase_result.get('detections', []):
                all_detections.append({
                    'timestamp': d['timestamp'],
                    'probability': d.get('probability', 0.5),  # Default if not present
                    'detected': d['detected'],
                    'is_movement_phase': True,
                    'is_baseline_phase': False
                })
    else:
        # For CNN method, use all_detections
        all_detections = results.get('all_detections', [])
    
    # Perform recovery analysis
    recovery_results = analyze_fast_recovery(signal_data, movement_phases, all_detections)
    
    # Perform detection frequency histogram analysis
    histogram_results = analyze_detection_frequency_histograms(signal_data, movement_phases, all_detections)
    
    # Add analysis results to main results
    results['recovery_analysis'] = recovery_results
    results['histogram_analysis'] = histogram_results
    
    return results


def visualize_participant_results(participant_name, signal_data, peak_timestamps, results, 
                                window_size_ms=1162, peak_detection_ms=162, dataset_name="Unknown", method="cnn", save_all=False):
    """Visualize evaluation results for a participant."""
    
    # Create movement phases for visualization (everything else is baseline)
    movement_phases = create_movement_phases(peak_timestamps, pre_peak_ms=200, post_peak_ms=800)
    
    # Get signal data
    signal_column = 'rms' if 'rms' in signal_data.columns else 'emg'
    timestamps = signal_data['timestamp'].values
    signal_values = signal_data[signal_column].values
    
    # Get detections based on method
    if method == "adaptive":
        # For adaptive threshold, we need to create detections from baseline_detections
        baseline_detections = results.get('baseline_detections', [])
        movement_results = results.get('movement_phase_results', [])
        
        # Create detections list for visualization
        detections = []
        
        # Add baseline detections
        for det in baseline_detections:
            detections.append({
                'timestamp': det['timestamp'],
                'probability': det.get('probability', 0.0),
                'detected': det['detected']
            })
        
        # Add movement phase detections (if any)
        for result in movement_results:
            if result.get('first_detection_time') is not None:
                detections.append({
                    'timestamp': result['first_detection_time'],
                    'probability': 1.0,  # High probability for detected
                    'detected': True
                })
    else:
        # For CNN method, use all_detections
        detections = results.get('all_detections', [])
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Signal with movement phases and detections
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, signal_values, 'b-', alpha=0.7, linewidth=0.8, label='RMS EMG Signal')
    
    # Highlight movement phases
    for i, (start_time, end_time) in enumerate(movement_phases):
        plt.axvspan(start_time, end_time, alpha=0.3, color='red', 
                   label='Movement Phase' if i == 0 else "")
    
    # Add baseline label
    if len(movement_phases) > 0:
        plt.axvspan(timestamps[0], timestamps[-1], alpha=0.1, color='green', 
                   label='Baseline (everything else)')
    
    # Plot actual peaks
    if len(peak_timestamps) > 0:
        peak_values = []
        for peak_time in peak_timestamps:
            closest_idx = np.argmin(np.abs(timestamps - peak_time))
            peak_values.append(signal_values[closest_idx])
        
        plt.scatter(peak_timestamps, peak_values, 
                   color='red', s=50, label=f'Actual Peaks ({len(peak_timestamps)})', zorder=5)
    
    # Plot detections
    if method == "adaptive":
        # For adaptive method, detections is a list of dictionaries
        detection_times = [d['timestamp'] for d in detections if d['detected']]
        detection_probs = [d['probability'] for d in detections if d['detected']]
    else:
        # For CNN method, detections is a structured array
        detection_times = detections['timestamp'][detections['detected']]
        detection_probs = detections['probability'][detections['detected']]
    
    if len(detection_times) > 0:
        plt.scatter(detection_times, np.max(signal_values) * 0.8 * np.ones_like(detection_times), 
                   color='green', s=15, label=f'Detections ({len(detection_times)})', zorder=5)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('RMS EMG (V)')
    plt.title(f'{participant_name} - Movement Phase Evaluation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Detection probabilities over time
    plt.subplot(3, 1, 2)
    if method == "adaptive":
        # For adaptive method, detections is a list of dictionaries
        detection_timestamps = [d['timestamp'] for d in detections]
        detection_probabilities = [d['probability'] for d in detections]
    else:
        # For CNN method, detections is a structured array
        detection_timestamps = detections['timestamp']
        detection_probabilities = detections['probability']
    
    plt.plot(detection_timestamps, detection_probabilities, 'g-', alpha=0.7, linewidth=1, label='Detection Probability')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Detection Threshold (0.5)')
    
    # Highlight movement phases
    for start_time, end_time in movement_phases:
        plt.axvspan(start_time, end_time, alpha=0.3, color='red')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Detection Probability')
    plt.title('Detection Probabilities Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Plot 3: Detection delays histogram
    plt.subplot(3, 1, 3)
    if results['detection_delays']:
        delays_ms = np.array(results['detection_delays']) * 1000
        plt.hist(delays_ms, bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(x=np.mean(delays_ms), color='red', linestyle='--', 
                   label=f'Mean Delay: {np.mean(delays_ms):.1f} ms')
        plt.xlabel('Detection Delay (ms)')
        plt.ylabel('Frequency')
        plt.title('Detection Delay Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No detections in movement phases', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Detection Delay Distribution')
    
    plt.tight_layout()
    
    # Save plot based on user preference
    if save_all:
        result_filename = f'{participant_name.lower()}_movement_phase_evaluation_{dataset_name.replace(" ", "_").replace("+", "plus").lower()}.png'
        result_path = os.path.join('experiment_results', result_filename)
        plt.savefig(result_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {result_path}")
    
    plt.show()


def visualize_fast_recovery(participant_name, signal_data, peak_timestamps, recovery_results, 
                           dataset_name="Unknown", method="cnn", save_all=False):
    """Visualize fast recovery analysis results."""
    
    if not recovery_results or recovery_results['summary']['total_false_positives'] == 0:
        print(f" No false positives found for {participant_name} - skipping recovery visualization")
        return
    
    # Create movement phases for visualization
    movement_phases = create_movement_phases(peak_timestamps, pre_peak_ms=200, post_peak_ms=800)
    
    # Get signal data
    signal_column = 'rms' if 'rms' in signal_data.columns else 'emg'
    timestamps = signal_data['timestamp'].values
    signal_values = signal_data[signal_column].values
    
    # Create figure with subplots for each false positive
    num_fps = len(recovery_results['recovery_analyses'])
    fig, axes = plt.subplots(num_fps, 1, figsize=(15, 4 * num_fps))
    if num_fps == 1:
        axes = [axes]
    
    for i, recovery_analysis in enumerate(recovery_results['recovery_analyses']):
        ax = axes[i]
        
        # Define time window for this false positive
        fp_time = recovery_analysis['fp_time']
        recovery_start = recovery_analysis['recovery_start_time']
        recovery_end = recovery_analysis['recovery_end_time']
        
        # Find time range to plot (extend a bit before and after)
        time_margin = 0.5  # seconds
        plot_start = max(recovery_start - time_margin, timestamps[0])
        plot_end = min(recovery_end + time_margin, timestamps[-1])
        
        # Find indices for plotting
        start_idx = np.argmin(np.abs(timestamps - plot_start))
        end_idx = np.argmin(np.abs(timestamps - plot_end))
        
        # Plot signal
        plot_times = timestamps[start_idx:end_idx]
        plot_signal = signal_values[start_idx:end_idx]
        ax.plot(plot_times, plot_signal, 'b-', alpha=0.7, linewidth=0.8, label='RMS EMG Signal')
        
        # Highlight movement phases in this range
        for start_time, end_time in movement_phases:
            if start_time <= plot_end and end_time >= plot_start:
                ax.axvspan(max(start_time, plot_start), min(end_time, plot_end), 
                          alpha=0.3, color='red', label='Movement Phase' if start_time == movement_phases[0][0] else "")
        
        # Mark the false positive
        ax.axvline(fp_time, color='red', linestyle='--', linewidth=2, label='False Positive')
        
        # Mark recovery window
        ax.axvspan(recovery_start, recovery_end, alpha=0.2, color='orange', 
                  label=f'Recovery Window ({recovery_analysis["recovery_duration"]:.2f}s)')
        
        # Plot actual peaks in this range
        peak_times_in_range = [pt for pt in peak_timestamps if plot_start <= pt <= plot_end]
        if peak_times_in_range:
            peak_values = []
            for peak_time in peak_times_in_range:
                closest_idx = np.argmin(np.abs(timestamps - peak_time))
                peak_values.append(signal_values[closest_idx])
            
            ax.scatter(peak_times_in_range, peak_values, 
                      color='red', s=50, label=f'Actual Peaks ({len(peak_times_in_range)})', zorder=5)
        
        # Plot detections in recovery window
        recovery_detections = recovery_analysis['recovery_detections']
        if recovery_detections:
            detection_times = [d['timestamp'] for d in recovery_detections if d['detected']]
            detection_probs = [d['probability'] for d in recovery_detections if d['detected']]
            
            if detection_times:
                # Scale probabilities to signal range
                prob_scale = np.max(plot_signal) * 0.8
                scaled_probs = [p * prob_scale for p in detection_probs]
                
                ax.scatter(detection_times, scaled_probs, 
                          color='green', s=15, label=f'Recovery Detections ({len(detection_times)})', zorder=5)
        
        # Formatting
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('RMS EMG (V)')
        ax.set_title(f'FP {i+1}: Recovery Analysis (t={fp_time:.2f}s, '
                    f'Detections: {recovery_analysis["total_detections"]}, '
                    f'Rate: {recovery_analysis["detection_rate"]:.2f}/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot based on user preference
    if save_all:
        plot_filename = f"experiment_results/{participant_name}_fast_recovery_analysis_{dataset_name}_{method}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved recovery analysis plot: {plot_filename}")
    
    plt.show()


def main():
    """Main fast recovery analysis function."""
    print("Fast Recovery Analysis")
    print("=" * 40)
    print("Analyzing recovery patterns after false positive detections")
    print("Tracking detection behavior in 500ms following baseline false positives")
    print("=" * 40)
    
    # Ask user if they want to save all figures
    save_all_figures = input("\nSave all figures automatically? (y/n): ").strip().lower()
    save_all = save_all_figures in ['y', 'yes']
    if save_all:
        print(" All figures will be saved automatically")
    else:
        print(" You will be asked before saving each figure")
    
    # Ask user to choose dataset
    print("\nChoose dataset to evaluate:")
    print("1. Healthy participants (P12-P15)")
    print("2. ALS patients (last two blocks)")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == '1':
            dataset_name = "Healthy"
            test_data = load_test_data()
            break
        elif choice == '2':
            dataset_name = "ALS"
            test_data = load_als_data()
            break
        else:
            print("Please enter 1 for healthy data or 2 for ALS data.")
    
    try:
        if not test_data:
            print(f"Failed to load {dataset_name.lower()} test data")
            return
        
        # Select detection method and model
        model_selection = select_model()
        if model_selection is None:
            return
        
        if len(model_selection) == 4:
            model_path, model_dataset_name, method, lambda_threshold = model_selection
        else:
            model_path, model_dataset_name, method = model_selection
            lambda_threshold = 2.0
        
        # Initialize model if using CNN
        model = None
        if method == "cnn":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = PeakDetectionCNN(input_length=1162, num_filters=64, dropout=0.3)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        
            print(f"Loaded trained model: {model_dataset_name}")
            print(f"Model path: {model_path}")
            print(f"Using device: {device}")
        else:
            print(f"Using adaptive threshold detection")
            print(f"Lambda threshold: {lambda_threshold}")
        
        # Evaluate each participant
        all_results = []
        
        for participant_data in test_data:
            results = evaluate_participant(model, participant_data, method=method, lambda_threshold=lambda_threshold, save_all=save_all)
            results['participant'] = participant_data['participant']
            results['signal_data'] = participant_data['signal_data']
            results['peak_timestamps'] = participant_data['peak_timestamps']
            all_results.append(results)
            
            # Create visualization
            visualize_participant_results(
                participant_data['participant'], 
                participant_data['signal_data'],
                participant_data['peak_timestamps'],
                results,
                dataset_name=model_dataset_name,
                method=method,
                save_all=save_all
            )
            
            # Create detection frequency histogram visualization
            if 'histogram_analysis' in results:
                visualize_detection_frequency_histograms(
                    participant_data['participant'],
                    results['histogram_analysis'],
                    dataset_name=model_dataset_name,
                    method=method,
                    save_all=save_all
                )
            
            # Interactive false positive explorer
            if 'recovery_analysis' in results:
                interactive_false_positive_explorer(
                    participant_data['participant'],
                    participant_data['signal_data'],
                    participant_data['peak_timestamps'],
                    results['recovery_analysis'],
                    dataset_name=model_dataset_name,
                    method=method,
                    save_all=save_all
            )
        
        # Print summary results
        print(f"\n{'='*60}")
        print(f"MOVEMENT PHASE EVALUATION SUMMARY - {dataset_name.upper()} DATA")
        print(f"{'='*60}")
        
        print(f"\n{'Participant':<12} {'Baseline Acc':<12} {'Movement Acc':<12} {'Avg Delay (ms)':<15}")
        print("-" * 60)
        
        total_baseline_correct = 0
        total_baseline = 0
        total_movement_correct = 0
        total_movement = 0
        all_delays = []
        
        for result in all_results:
            print(f"{result['participant']:<12} {result['baseline_accuracy']:<12.3f} "
                  f"{result['movement_phase_accuracy']:<12.3f} {result['average_delay_ms']:<15.1f}")
            
            total_baseline_correct += result['baseline_correct']
            total_baseline += result['baseline_total']
            total_movement_correct += result['movement_phase_correct']
            total_movement += result['movement_phase_total']
            all_delays.extend(result['detection_delays'])
        
        # Calculate overall averages
        overall_baseline_acc = total_baseline_correct / total_baseline if total_baseline > 0 else 0
        overall_movement_acc = total_movement_correct / total_movement if total_movement > 0 else 0
        overall_avg_delay = np.mean([d * 1000 for d in all_delays]) if all_delays else 0
        
        print("-" * 60)
        print(f"{'Overall':<12} {overall_baseline_acc:<12.3f} {overall_movement_acc:<12.3f} {overall_avg_delay:<15.1f}")
        
        print(f"\nDetailed Statistics:")
        print(f" Total baseline samples: {total_baseline}")
        print(f" Total movement phases: {total_movement}")
        print(f" Overall baseline accuracy: {overall_baseline_acc:.3f}")
        print(f" Overall movement phase accuracy: {overall_movement_acc:.3f}")
        print(f" Overall average delay: {overall_avg_delay:.1f} ms")
        print(f" Delay standard deviation: {np.std([d * 1000 for d in all_delays]):.1f} ms")
        
        print(f"\n {dataset_name} data evaluation completed successfully!")
        print(f"Model used: {model_dataset_name}")
        print(f"Individual visualizations available for saving in: experiment_results/")
        print(f"Files: *_movement_phase_evaluation_{model_dataset_name.replace(' ', '_').replace('+', 'plus').lower()}.png")
        print(f"Files: *_detection_frequency_histogram_{model_dataset_name.replace(' ', '_').replace('+', 'plus').lower()}.png")
        print(f"Files: *_fp_*_analysis_{model_dataset_name.replace(' ', '_').replace('+', 'plus').lower()}.png")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
