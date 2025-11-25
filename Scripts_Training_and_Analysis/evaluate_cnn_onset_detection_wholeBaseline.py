"""
CNN Movement Phase Evaluation Script
===================================

This script evaluates the trained CNN model on test participants using movement phase analysis.
It separates the test data into "movement phases" (from -100ms before a peak until +400ms after a peak)
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
    """Allow user to select a trained model."""
    available_models = list_available_models()
    
    if not available_models:
        print("No trained models available. Please train a model first.")
        return None
    
    print("\nAvailable trained models:")
    print("=" * 40)
    
    for i, (model_path, dataset_name) in enumerate(available_models, 1):
        print(f"{i}. {dataset_name}")
    
    while True:
        try:
            choice = int(input(f"\nSelect model (1-{len(available_models)}): ")) - 1
            if 0 <= choice < len(available_models):
                selected_model_path, selected_dataset_name = available_models[choice]
                print(f"Selected: {selected_dataset_name}")
                return selected_model_path, selected_dataset_name
            else:
                print(f"Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("Please enter a valid number")


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


def create_phases(peak_timestamps, pre_peak_ms=200, post_peak_ms=800):
    """
    Create movement phases and baseline phases with non-overlapping phases.
    
    Movement phases: -200ms before peak to +800ms after peak
    Baseline phases: from peak+800ms to next peak-200ms
    
    Args:
        peak_timestamps: Array of peak timestamps in seconds
        pre_peak_ms: Time before peak to include in movement phase (ms)
        post_peak_ms: Time after peak to include in movement phase (ms)
    
    Returns:
        Tuple of (movement_phases, baseline_phases) where each is a list of (start_time, end_time) tuples
    """
    if len(peak_timestamps) == 0:
        return [], []
    
    # Sort peak timestamps
    sorted_peaks = np.sort(peak_timestamps)
    
    # Create movement phases for each peak
    movement_phases = []
    for peak_time in sorted_peaks:
        start_time = peak_time - (pre_peak_ms / 1000.0)  # Convert ms to seconds
        end_time = peak_time + (post_peak_ms / 1000.0)
        movement_phases.append((start_time, end_time))
    
    # Create baseline phases between peaks
    baseline_phases = []
    for i in range(len(sorted_peaks) - 1):
        current_peak = sorted_peaks[i]
        next_peak = sorted_peaks[i + 1]
        
        # Baseline starts 800ms after current peak
        baseline_start = current_peak + (post_peak_ms / 1000.0)
        # Baseline ends 200ms before next peak
        baseline_end = next_peak - (pre_peak_ms / 1000.0)
        
        # Only add baseline if there's a gap between movement phases
        if baseline_end > baseline_start:
            baseline_phases.append((baseline_start, baseline_end))
    
    # Debug information
    print(f" Created {len(movement_phases)} movement phases (-{pre_peak_ms}ms to +{post_peak_ms}ms)")
    print(f" Created {len(baseline_phases)} baseline phases (peak+{post_peak_ms}ms to next peak-{pre_peak_ms}ms)")
    
    return movement_phases, baseline_phases


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


def sliding_window_evaluation(model, signal_data, movement_phases, baseline_phases, window_size_ms=1162, 
                            peak_detection_ms=162, sampling_rate=34.81, threshold=0.5):
    """
    Evaluate model using sliding window approach with discrete phases.
    
    Args:
        model: Trained CNN model
        signal_data: DataFrame with signal data
        movement_phases: List of movement phase time ranges
        baseline_phases: List of baseline phase time ranges
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
    
    print(f" Evaluating with sliding window (step size: 1 sample)")
    print(f" Window size: {window_size_ms}ms, Peak detection zone: {peak_detection_ms}ms")
    print(f" Movement phases: {len(movement_phases)}, Baseline phases: {len(baseline_phases)}")
    
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
                # Get model prediction
                window = signal_values[j:j + window_size_samples]
                with torch.no_grad():
                    window_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)
                    output = model(window_tensor)
                    probability = torch.sigmoid(output).cpu().item()
                
                phase_detections.append({
                    'timestamp': window_end_time,
                    'probability': probability,
                    'detected': probability > threshold
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
                    'max_probability': max(d['probability'] for d in phase_detections)
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
                    'max_probability': max(d['probability'] for d in phase_detections)
                })
                print(f"   ✗ No detection found (max prob: {max(d['probability'] for d in phase_detections):.3f})")
        else:
            movement_phase_results.append({
                'phase_id': i,
                'start_time': start_time,
                'end_time': end_time,
                'correct': False,
                'first_detection_time': None,
                'delay': None,
                'max_probability': 0.0
            })
            print(f"   ✗ No windows found for this phase")
    
    # Evaluate baseline phases
    baseline_phase_results = []
    
    for i, (start_time, end_time) in enumerate(baseline_phases):
        print(f" Evaluating baseline phase {i+1}: {start_time:.2f}s - {end_time:.2f}s")
        
        # Find all windows that overlap with this baseline phase
        phase_detections = []
        
        for j in range(len(signal_values) - window_size_samples + 1):
            window_start_time = timestamps[j]
            window_end_time = timestamps[j + window_size_samples - 1]
            
            # Check if window overlaps with baseline phase
            if window_start_time <= end_time and window_end_time >= start_time:
                # Get model prediction
                window = signal_values[j:j + window_size_samples]
                with torch.no_grad():
                    window_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)
                    output = model(window_tensor)
                    probability = torch.sigmoid(output).cpu().item()
                
                phase_detections.append({
                    'timestamp': window_end_time,
                    'probability': probability,
                    'detected': probability > threshold
                })
        
        # Check if any detection occurred in this baseline phase (false positive)
        if phase_detections:
            phase_detected = [d['detected'] for d in phase_detections]
            has_false_positive = any(phase_detected)
            
            baseline_phase_results.append({
                'phase_id': i,
                'start_time': start_time,
                'end_time': end_time,
                'correct': not has_false_positive,
                'false_positive_count': sum(phase_detected),
                'max_probability': max(d['probability'] for d in phase_detections)
            })
            
            if has_false_positive:
                print(f"   ✗ False positive detected ({sum(phase_detected)} detections, max prob: {max(d['probability'] for d in phase_detections):.3f})")
            else:
                print(f"    No false positives (max prob: {max(d['probability'] for d in phase_detections):.3f})")
        else:
            baseline_phase_results.append({
                'phase_id': i,
                'start_time': start_time,
                'end_time': end_time,
                'correct': True,
                'false_positive_count': 0,
                'max_probability': 0.0
            })
            print(f"    No windows found for this phase")
    
    # Calculate overall metrics
    movement_correct = sum(1 for r in movement_phase_results if r['correct'])
    movement_total = len(movement_phase_results)
    movement_accuracy = movement_correct / movement_total if movement_total > 0 else 0
    
    baseline_correct = sum(1 for r in baseline_phase_results if r['correct'])
    baseline_total = len(baseline_phase_results)
    baseline_accuracy = baseline_correct / baseline_total if baseline_total > 0 else 1.0
    
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
        'baseline_phase_results': baseline_phase_results
    }


def evaluate_participant(model, participant_data, window_size_ms=1162, peak_detection_ms=162):
    """Evaluate a single participant."""
    participant_name = participant_data['participant']
    signal_data = participant_data['signal_data']
    peak_timestamps = participant_data['peak_timestamps']
    
    print(f"\nEvaluating {participant_name}")
    print("-" * 20)
    
    # Create movement and baseline phases
    movement_phases, baseline_phases = create_phases(peak_timestamps, pre_peak_ms=200, post_peak_ms=800)
    print(f"Created {len(movement_phases)} movement phases and {len(baseline_phases)} baseline phases")
    
    # Run sliding window evaluation
    results = sliding_window_evaluation(
        model, signal_data, movement_phases, baseline_phases,
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
                                window_size_ms=1162, peak_detection_ms=162, dataset_name="Unknown"):
    """Visualize evaluation results for a participant."""
    
    # Create movement and baseline phases for visualization
    movement_phases, baseline_phases = create_phases(peak_timestamps, pre_peak_ms=200, post_peak_ms=800)
    
    # Get signal data
    signal_column = 'rms' if 'rms' in signal_data.columns else 'emg'
    timestamps = signal_data['timestamp'].values
    signal_values = signal_data[signal_column].values
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Signal with movement and baseline phases
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, signal_values, 'b-', alpha=0.7, linewidth=0.8, label='RMS EMG Signal')
    
    # Highlight movement phases
    for i, (start_time, end_time) in enumerate(movement_phases):
        plt.axvspan(start_time, end_time, alpha=0.3, color='red', 
                   label='Movement Phase' if i == 0 else "")
    
    # Highlight baseline phases
    for i, (start_time, end_time) in enumerate(baseline_phases):
        plt.axvspan(start_time, end_time, alpha=0.2, color='green', 
                   label='Baseline Phase' if i == 0 else "")
    
    # Plot actual peaks
    if len(peak_timestamps) > 0:
        peak_values = []
        for peak_time in peak_timestamps:
            closest_idx = np.argmin(np.abs(timestamps - peak_time))
            peak_values.append(signal_values[closest_idx])
        
        plt.scatter(peak_timestamps, peak_values, 
                   color='red', s=50, label=f'Actual Peaks ({len(peak_timestamps)})', zorder=5)
    
    # Plot detections from movement phase results
    movement_results = results.get('movement_phase_results', [])
    detection_times = [r['first_detection_time'] for r in movement_results if r['first_detection_time'] is not None]
    if detection_times:
        plt.scatter(detection_times, np.max(signal_values) * 0.8 * np.ones_like(detection_times), 
                   color='green', s=30, label=f'Detections ({len(detection_times)})', zorder=5)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('RMS EMG (V)')
    plt.title(f'{participant_name} - Movement Phase Evaluation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Phase evaluation results
    plt.subplot(3, 1, 2)
    
    # Plot movement phase results
    movement_results = results.get('movement_phase_results', [])
    for i, result in enumerate(movement_results):
        phase_center = (result['start_time'] + result['end_time']) / 2
        if result['correct']:
            plt.scatter(phase_center, 1, color='green', s=100, marker='o', label='Movement Phase (Correct)' if i == 0 else "")
        else:
            plt.scatter(phase_center, 1, color='red', s=100, marker='x', label='Movement Phase (Missed)' if i == 0 else "")
    
    # Plot baseline phase results
    baseline_results = results.get('baseline_phase_results', [])
    for i, result in enumerate(baseline_results):
        phase_center = (result['start_time'] + result['end_time']) / 2
        if result['correct']:
            plt.scatter(phase_center, 0, color='green', s=100, marker='o', label='Baseline Phase (Correct)' if i == 0 else "")
        else:
            plt.scatter(phase_center, 0, color='red', s=100, marker='x', label='Baseline Phase (False Positive)' if i == 0 else "")
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Phase Type')
    plt.title('Phase Evaluation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 1.5)
    plt.yticks([0, 1], ['Baseline', 'Movement'])
    
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
        
        # Select trained model
        model_selection = select_model()
        if model_selection is None:
            return
        
        model_path, model_dataset_name = model_selection
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PeakDetectionCNN(input_length=1162, num_filters=64, dropout=0.3)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        print(f"Loaded trained model: {model_dataset_name}")
        print(f"Model path: {model_path}")
        print(f"Using device: {device}")
        
        # Evaluate each participant
        all_results = []
        
        for participant_data in test_data:
            results = evaluate_participant(model, participant_data)
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
                dataset_name=model_dataset_name
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
