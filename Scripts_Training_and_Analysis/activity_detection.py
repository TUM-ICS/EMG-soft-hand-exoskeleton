"""
Activity Detection Script
========================

This script uses a trained error detection model to classify ALS data blocks 3 and 4
into activity and baseline phases using 200ms detection windows. It visualizes the 
results with real peaks marked.
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

# Import the CNN model from the training script
from train_cnn_error import MuscleActivityCNN, MuscleActivityDataset


def load_als_blocks_3_4():
    """Load ALS data blocks 3 and 4."""
    print("Loading ALS data blocks 3 and 4...")
    
    # Define file paths for blocks 3 and 4
    block3_file = '../EMG_data/signal_data/RMS_ALS_block3.csv'
    block4_file = '../EMG_data/signal_data/RMS_ALS_block4.csv'
    
    # Load signal data
    block3_data = pd.read_csv(block3_file)
    block4_data = pd.read_csv(block4_file)
    
    # Load peak timestamps
    block3_peaks_file = '../EMG_data/label_data/peaks_ALS_block3.csv'
    block4_peaks_file = '../EMG_data/label_data/peaks_ALS_block4.csv'
    
    block3_peaks = pd.read_csv(block3_peaks_file)['timestamp'].values
    block4_peaks = pd.read_csv(block4_peaks_file)['timestamp'].values
    
    return [
        {
            'participant': 'ALS_Block3',
            'signal_data': block3_data,
            'peak_timestamps': block3_peaks
        },
        {
            'participant': 'ALS_Block4', 
            'signal_data': block4_data,
            'peak_timestamps': block4_peaks
        }
    ]


def load_trained_model(model_path):
    """Load the trained error detection model."""
    print(f"Loading trained model from {model_path}...")
    
    # Initialize model
    model = MuscleActivityCNN(input_length=200, num_filters=64, dropout=0.3)
    
    # Load model weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully on {device}")
    return model, device


def classify_activity_phases(model, device, signal_data, peak_timestamps, window_size_ms=200, threshold=0.5):
    """Classify signal into activity and baseline phases using sliding windows."""
    print(f"Classifying activity phases with {window_size_ms}ms windows...")
    
    # Get signal data
    timestamps = signal_data['timestamp'].values
    signal_column = 'rms' if 'rms' in signal_data.columns else 'emg'
    signal_values = signal_data[signal_column].values
    
    # Convert window size to samples
    sampling_rate = 34.81  # Hz
    window_size_samples = int(window_size_ms * sampling_rate / 1000)
    
    # Create sliding windows
    all_predictions = []
    all_probabilities = []
    window_centers = []
    
    for i in range(len(signal_values) - window_size_samples + 1):
        window = signal_values[i:i + window_size_samples]
        window_center_time = timestamps[i + window_size_samples // 2]
        
        # Prepare window for model
        window_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        window_tensor = window_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            logits = model(window_tensor)
            probability = torch.sigmoid(logits).cpu().numpy()[0]
            prediction = 1 if probability > threshold else 0
        
        all_predictions.append(prediction)
        all_probabilities.append(probability)
        window_centers.append(window_center_time)
    
    return np.array(window_centers), np.array(all_predictions), np.array(all_probabilities)


def create_activity_phases_from_predictions(window_centers, predictions, min_activity_duration_ms=1000):
    """Convert window predictions into continuous activity phases."""
    print(f"Creating activity phases from predictions...")
    
    # Convert to time-based representation
    activity_phases = []
    in_activity = False
    activity_start = None
    
    min_activity_duration_s = min_activity_duration_ms / 1000.0
    
    for i, (time, pred) in enumerate(zip(window_centers, predictions)):
        if pred == 1 and not in_activity:
            # Start of activity phase
            in_activity = True
            activity_start = time
        elif pred == 0 and in_activity:
            # End of activity phase
            activity_duration = time - activity_start
            if activity_duration >= min_activity_duration_s:
                activity_phases.append((activity_start, time))
            in_activity = False
            activity_start = None
    
    # Handle case where activity continues to end of signal
    if in_activity and activity_start is not None:
        activity_duration = window_centers[-1] - activity_start
        if activity_duration >= min_activity_duration_s:
            activity_phases.append((activity_start, window_centers[-1]))
    
    print(f"✓ Created {len(activity_phases)} activity phases")
    return activity_phases


def visualize_activity_detection(participant_name, signal_data, peak_timestamps, 
                                window_centers, predictions, probabilities, 
                                activity_phases, model_name="error_als"):
    """Visualize the activity detection results."""
    print(f"Creating visualization for {participant_name}...")
    
    # Get signal data
    timestamps = signal_data['timestamp'].values
    signal_column = 'rms' if 'rms' in signal_data.columns else 'emg'
    signal_values = signal_data[signal_column].values
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Signal with activity phases and peaks
    ax1.plot(timestamps, signal_values, 'b-', alpha=0.7, linewidth=0.8, label='RMS EMG Signal')
    
    # Highlight detected activity phases
    for start_time, end_time in activity_phases:
        ax1.axvspan(start_time, end_time, alpha=0.3, color='red', 
                   label='Detected Activity Phase' if start_time == activity_phases[0][0] else "")
    
    # Plot real peaks
    if len(peak_timestamps) > 0:
        peak_values = []
        for peak_time in peak_timestamps:
            closest_idx = np.argmin(np.abs(timestamps - peak_time))
            peak_values.append(signal_values[closest_idx])
        
        ax1.scatter(peak_timestamps, peak_values, color='red', s=50, 
                   label=f'Real Peaks ({len(peak_timestamps)})', zorder=5)
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('RMS EMG (V)')
    ax1.set_title(f'Activity Detection: {participant_name}\nModel: {model_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction probabilities over time
    ax2.plot(window_centers, probabilities, 'g-', alpha=0.7, linewidth=1, label='Activity Probability')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold (0.5)')
    
    # Highlight activity phases in probability plot
    for start_time, end_time in activity_phases:
        ax2.axvspan(start_time, end_time, alpha=0.2, color='red')
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Activity Probability')
    ax2.set_title('Activity Detection Probabilities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"experiment_results/{participant_name}_activity_detection_{model_name}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_filename}")
    
    plt.show()
    
    # Print statistics
    total_activity_time = sum(end - start for start, end in activity_phases)
    total_time = timestamps[-1] - timestamps[0]
    activity_percentage = (total_activity_time / total_time) * 100
    
    print(f"\nActivity Detection Statistics:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Activity phases: {len(activity_phases)}")
    print(f"  Total activity time: {total_activity_time:.2f}s")
    print(f"  Activity percentage: {activity_percentage:.1f}%")
    print(f"  Real peaks: {len(peak_timestamps)}")
    print(f"  Average probability: {np.mean(probabilities):.3f}")


def main():
    """Main activity detection function."""
    print("Activity Detection using Trained Error Model")
    print("=" * 50)
    
    # Load ALS blocks 3 and 4
    test_data = load_als_blocks_3_4()
    
    # Load trained model
    model_path = 'trained_models/error_als.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first using train_cnn_error.py")
        return
    
    model, device = load_trained_model(model_path)
    
    # Process each block
    for participant_data in test_data:
        participant_name = participant_data['participant']
        signal_data = participant_data['signal_data']
        peak_timestamps = participant_data['peak_timestamps']
        
        print(f"\n{'='*60}")
        print(f"Processing {participant_name}")
        print(f"{'='*60}")
        
        # Classify activity phases
        window_centers, predictions, probabilities = classify_activity_phases(
            model, device, signal_data, peak_timestamps
        )
        
        # Create continuous activity phases
        activity_phases = create_activity_phases_from_predictions(
            window_centers, predictions, min_activity_duration_ms=1000
        )
        
        # Visualize results
        visualize_activity_detection(
            participant_name, signal_data, peak_timestamps,
            window_centers, predictions, probabilities, activity_phases
        )
    
    print(f"\n{'='*60}")
    print("Activity Detection Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
