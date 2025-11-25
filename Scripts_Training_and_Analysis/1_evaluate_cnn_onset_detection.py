"""
CNN Movement Phase Evaluation Script
===================================

This script evaluates the trained CNN model on test participants using movement phase analysis.
It separates the test data into "movement phases" (from -500ms before a peak until +800ms after a peak)
and "baseline phases" (everything else). The model is evaluated using sliding window detection
similar to real-time operation.

Evaluation criteria:
- Baseline phase: Correct if no peak detected while detection window head is in baseline
- Movement phase: Correct if at least one peak detected while detection window head is in movement phase
- Delay measurement: Average delay between first detection and actual peak time
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
    als_blocks = ['block3', 'block4']  # Last two blocks
    
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


def create_movement_phases(peak_timestamps, pre_peak_ms=500, post_peak_ms=800):
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


def create_movement_phases(peak_timestamps, pre_peak_ms=500, post_peak_ms=800):
    """
    Create movement phases only. Everything else is considered baseline.
    
    Movement phases: -500ms before peak to +800ms after peak
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
    
    for i, (start_time, end_time) in enumerate(movement_phases):
        print(f" Evaluating movement phase {i+1}: {start_time:.2f}s - {end_time:.2f}s")
        
        # Find all windows that overlap with this movement phase
        phase_detections = []
        
        for j in range(len(signal_values) - window_size_samples + 1):
            window_start_time = timestamps[j]
            window_end_time = timestamps[j + window_size_samples - 1]
            
            # Check if window overlaps with movement phase
            if window_start_time <= end_time and window_end_time >= start_time:
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
                
                phase_detections.append({
                    'timestamp': window_end_time,
                    'activity_mean': activity_mean,
                    'rest_mean': rest_mean,
                    'rest_std': rest_std,
                    'threshold': threshold,
                    'probability': probability,
                    'detected': detected
                })
        
        # Check if any detection occurred in this movement phase
        if phase_detections:
            phase_detected = [d['detected'] for d in phase_detections]
            if any(phase_detected):
                # Find first detection
                first_detection_idx = next(i for i, detected in enumerate(phase_detected) if detected)
                first_detection_time = phase_detections[first_detection_idx]['timestamp']
                
                # Calculate delay (use middle of movement phase as reference)
                phase_middle = (start_time + end_time) / 2
                delay = first_detection_time - phase_middle
                detection_delays.append(delay)
                
                movement_phase_results.append({
                    'phase_id': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'correct': True,
                    'first_detection_time': first_detection_time,
                    'delay': delay,
                    'max_activity_mean': max(d['activity_mean'] for d in phase_detections),
                    'max_threshold': max(d['threshold'] for d in phase_detections)
                })
                print(f"    Detection found at {first_detection_time:.2f}s (delay: {delay*1000:.1f}ms)")
            else:
                movement_phase_results.append({
                    'phase_id': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'correct': False,
                    'first_detection_time': None,
                    'delay': None,
                    'max_activity_mean': max(d['activity_mean'] for d in phase_detections),
                    'max_threshold': max(d['threshold'] for d in phase_detections)
                })
                print(f"   ✗ No detection found (max activity: {max(d['activity_mean'] for d in phase_detections):.3f}, max threshold: {max(d['threshold'] for d in phase_detections):.3f})")
        else:
            movement_phase_results.append({
                'phase_id': i,
                'start_time': start_time,
                'end_time': end_time,
                'correct': False,
                'first_detection_time': None,
                'delay': None,
                'max_activity_mean': 0.0,
                'max_threshold': 0.0
            })
            print(f"   ✗ No windows found for this phase")
    
    # Evaluate baseline samples (everything NOT in movement phases)
    baseline_detections = []
    total_windows_checked = 0
    baseline_windows_found = 0
    
    for j in range(len(signal_values) - window_size_samples + 1):
        window_start_time = timestamps[j]
        window_end_time = timestamps[j + window_size_samples - 1]
        total_windows_checked += 1
        
        # Check if detection time is NOT in any movement phase (i.e., it's baseline)
        is_movement = is_in_movement_phase(window_end_time, movement_phases)
        is_baseline = not is_movement
        
        if is_baseline:
            baseline_windows_found += 1
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
            
            baseline_detections.append({
                'timestamp': window_end_time,
                'activity_mean': activity_mean,
                'rest_mean': rest_mean,
                'rest_std': rest_std,
                'threshold': threshold,
                'detected': detected
            })
    
    # Calculate baseline metrics (sample-level)
    baseline_detected = [d['detected'] for d in baseline_detections]
    baseline_correct = sum(1 for detected in baseline_detected if not detected)  # Correct if no detection
    baseline_total = len(baseline_detections)
    baseline_accuracy = baseline_correct / baseline_total if baseline_total > 0 else 1.0
    
    print(f" Baseline Phase (sample-level):")
    print(f"   Total windows checked: {total_windows_checked}")
    print(f"   Windows in baseline phases: {baseline_windows_found}")
    print(f"   Total samples: {baseline_total}")
    print(f"   Correct (no detection): {baseline_correct}")
    print(f"   False positives: {baseline_total - baseline_correct}")
    print(f"   Accuracy: {baseline_accuracy:.3f}")
    
    if baseline_total == 0:
        print(f"   No baseline samples found! Possible reasons:")
        print(f"    - No baseline phases created (peaks too close or only one peak)")
        print(f"    - Signal duration too short for baseline phases")
        print(f"    - Window size too large for available data")
    
    # Calculate movement phase metrics
    movement_correct = sum(1 for r in movement_phase_results if r['correct'])
    movement_total = len(movement_phase_results)
    movement_accuracy = movement_correct / movement_total if movement_total > 0 else 0
    
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


def evaluate_participant(model, participant_data, window_size_ms=1162, peak_detection_ms=162, method="cnn", lambda_threshold=2.0):
    """Evaluate a single participant."""
    participant_name = participant_data['participant']
    signal_data = participant_data['signal_data']
    peak_timestamps = participant_data['peak_timestamps']
    
    print(f"\nEvaluating {participant_name}")
    print("-" * 20)
    
    # Create movement phases (everything else is baseline)
    movement_phases = create_movement_phases(peak_timestamps, pre_peak_ms=500, post_peak_ms=800)
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
    
    return results


def visualize_participant_results(participant_name, signal_data, peak_timestamps, results, 
                                window_size_ms=1162, peak_detection_ms=162, dataset_name="Unknown", method="cnn"):
    """Visualize evaluation results for a participant."""
    
    # Create movement phases for visualization (everything else is baseline)
    movement_phases = create_movement_phases(peak_timestamps, pre_peak_ms=500, post_peak_ms=800)
    
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
    result_filename = f'{participant_name.lower()}_movement_phase_evaluation_{dataset_name.replace(" ", "_").replace("+", "plus").lower()}.png'
    result_path = os.path.join('experiment_results', result_filename)
    plt.savefig(result_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main evaluation function."""
    print("CNN Movement Phase Evaluation")
    print("=" * 40)
    print("Evaluating trained model on movement phases vs baseline phases")
    print("Using sliding window approach similar to real-time operation")
    print("=" * 40)
    
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
            results = evaluate_participant(model, participant_data, method=method, lambda_threshold=lambda_threshold)
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
                method=method
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
        print(f"Individual visualizations saved in: experiment_results/")
        print(f"Files: *_movement_phase_evaluation_{model_dataset_name.replace(' ', '_').replace('+', 'plus').lower()}.png")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
