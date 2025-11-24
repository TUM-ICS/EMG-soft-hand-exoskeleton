"""
Grad-CAM Analysis for CNN Models
================================

This script uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize
which time points in the input windows the model focuses on when making predictions.
This provides interpretability insights into the model's decision-making process.

Supports two model architectures:
- MuscleActivityCNN: 200ms windows for activity detection (uses conv3)
- PeakDetectionCNN: 1162ms windows for peak detection (uses conv4)

Grad-CAM computes the gradient of the output score with respect to the feature maps
of the last convolutional layer, then weights the feature maps by these gradients
to create a temporal importance map.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
import os
import warnings
warnings.filterwarnings('ignore')

# Import model and dataset classes
from train_cnn_error import MuscleActivityCNN, MuscleActivityDataset
from train_cnn_onset_detection import PeakDetectionCNN, PeakDetectionDataset

# Create necessary directories
os.makedirs('experiment_results', exist_ok=True)


class GradCAM:
    """Grad-CAM implementation for 1D CNNs."""
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The trained CNN model
            target_layer: The convolutional layer to analyze (e.g., model.conv3 or model.conv4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hook_layers()
    
    def hook_layers(self):
        """Register forward and backward hooks to capture activations and gradients."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap for the input.
        
        Args:
            input_tensor: Input tensor of shape (1, 1, sequence_length)
            class_idx: Class index to generate CAM for (None for binary classification)
        
        Returns:
            cam: Class activation map (importance scores for each time point)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # For binary classification, use the output directly
        if class_idx is None:
            # Use the positive class (activity)
            score = output
        else:
            score = output[0, class_idx]
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # Shape: (num_channels, sequence_length)
        activations = self.activations[0]  # Shape: (num_channels, sequence_length)
        
        # Global average pooling of gradients (weights) across spatial dimension
        weights = torch.mean(gradients, dim=1, keepdim=True)  # Shape: (num_channels, 1)
        
        # Weighted combination of feature maps
        cam = torch.sum(weights * activations, dim=0)  # Shape: (sequence_length,)
        
        # Apply ReLU to get positive contributions only
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().detach().numpy()


def detect_model_type(model_path):
    """Detect model type based on file path or try loading."""
    # Check if it's an error model (200ms) or onset detection model (1162ms)
    if 'error' in model_path.lower():
        return 'error', 200
    elif 'cnn_als' in model_path.lower() or 'onset' in model_path.lower():
        return 'onset', 1162
    else:
        # Try to infer from model state dict
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            # Check for conv4 (PeakDetectionCNN) vs conv3 only (MuscleActivityCNN)
            if 'conv4.weight' in state_dict:
                return 'onset', 1162
            else:
                return 'error', 200
        except:
            # Default to error model
            return 'error', 200


def load_trained_model(model_path):
    """Load the trained model (supports both MuscleActivityCNN and PeakDetectionCNN)."""
    print(f"Loading trained model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Detect model type
    model_type, window_size_ms = detect_model_type(model_path)
    
    # Initialize appropriate model
    if model_type == 'error':
        model = MuscleActivityCNN(input_length=window_size_ms, num_filters=64, dropout=0.3)
        print(f"✓ Detected MuscleActivityCNN model (window size: {window_size_ms}ms)")
    else:  # onset
        model = PeakDetectionCNN(input_length=window_size_ms, num_filters=64, dropout=0.3)
        print(f"✓ Detected PeakDetectionCNN model (window size: {window_size_ms}ms)")
    
    # Load model weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully on {device}")
    return model, device, model_type, window_size_ms


def load_test_data():
    """Load ALS test data (blocks 3 and 4) for analysis."""
    print("Loading test data...")
    
    # Define file paths
    block3_file = '../EMG_data/signal_data/RMS_ALS_block3.csv'
    block4_file = '../EMG_data/signal_data/RMS_ALS_block4.csv'
    
    block3_peaks_file = '../EMG_data/label_data/peaks_ALS_block3.csv'
    block4_peaks_file = '../EMG_data/label_data/peaks_ALS_block4.csv'
    
    test_data = []
    
    # Load block 3
    if os.path.exists(block3_file):
        block3_data = pd.read_csv(block3_file)
        block3_peaks = pd.read_csv(block3_peaks_file)['timestamp'].values if os.path.exists(block3_peaks_file) else []
        test_data.append({
            'participant': 'ALS_Block3',
            'signal_data': block3_data,
            'peak_timestamps': block3_peaks
        })
        print(f"✓ Loaded ALS Block 3: {len(block3_data)} samples, {len(block3_peaks)} peaks")
    
    # Load block 4
    if os.path.exists(block4_file):
        block4_data = pd.read_csv(block4_file)
        block4_peaks = pd.read_csv(block4_peaks_file)['timestamp'].values if os.path.exists(block4_peaks_file) else []
        test_data.append({
            'participant': 'ALS_Block4',
            'signal_data': block4_data,
            'peak_timestamps': block4_peaks
        })
        print(f"✓ Loaded ALS Block 4: {len(block4_data)} samples, {len(block4_peaks)} peaks")
    
    return test_data


def analyze_windows_with_gradcam(model, device, model_type, signal_data, peak_timestamps, 
                                 num_samples=20, window_size_ms=200, sampling_rate=34.81):
    """
    Analyze multiple windows using Grad-CAM.
    
    Args:
        model: Trained model
        device: Device (cpu/cuda)
        model_type: Type of model ('error' or 'onset')
        signal_data: DataFrame with signal data
        peak_timestamps: Array of peak timestamps
        num_samples: Number of windows to analyze
        window_size_ms: Window size in milliseconds
        sampling_rate: Sampling rate in Hz
    
    Returns:
        results: List of analysis results for each window
    """
    # Initialize Grad-CAM (target the last conv layer before global pooling)
    if model_type == 'error':
        # MuscleActivityCNN: last conv is conv3
        gradcam = GradCAM(model, model.conv3)
    else:
        # PeakDetectionCNN: last conv is conv4
        gradcam = GradCAM(model, model.conv4)
    
    # Get signal data
    timestamps = signal_data['timestamp'].values
    signal_column = 'rms' if 'rms' in signal_data.columns else 'emg'
    signal_values = signal_data[signal_column].values
    
    # Convert window size to samples
    window_size_samples = int(window_size_ms * sampling_rate / 1000)
    
    # Create appropriate dataset
    if model_type == 'error':
        dataset = MuscleActivityDataset(signal_data, peak_timestamps, window_size_ms, sampling_rate)
    else:
        # PeakDetectionDataset uses peak_detection_ms parameter
        dataset = PeakDetectionDataset(signal_data, peak_timestamps, window_size_ms, 
                                      peak_detection_ms=162, sampling_rate=sampling_rate)
    
    # Select diverse samples: mix of activity and baseline windows
    activity_indices = np.where(dataset.labels == 1)[0]
    baseline_indices = np.where(dataset.labels == 0)[0]
    
    num_activity = min(num_samples // 2, len(activity_indices))
    num_baseline = min(num_samples // 2, len(baseline_indices))
    
    selected_indices = (
        list(np.random.choice(activity_indices, num_activity, replace=False)) +
        list(np.random.choice(baseline_indices, num_baseline, replace=False))
    )
    
    results = []
    
    print(f"\nAnalyzing {len(selected_indices)} windows with Grad-CAM...")
    
    for idx in selected_indices:
        window = dataset.windows[idx]
        label = dataset.labels[idx]
        
        # Get window timing information
        # Windows are created with step size = 1, so window idx corresponds to start position
        window_start_idx = idx
        if window_start_idx + window_size_samples > len(timestamps):
            continue
        window_times = timestamps[window_start_idx:window_start_idx + window_size_samples]
        
        # Prepare input tensor
        input_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        # Generate CAM
        cam = gradcam.generate_cam(input_tensor)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output).cpu().item()
            prediction = 1 if probability > 0.5 else 0
        
        # Interpolate CAM to match window length if needed (due to pooling in PeakDetectionCNN)
        if len(cam) != len(window):
            # Use interpolation to match the original window length
            cam_original_indices = np.linspace(0, len(window) - 1, len(cam))
            cam_new_indices = np.arange(len(window))
            interp_func = interp1d(cam_original_indices, cam, kind='linear', 
                                 bounds_error=False, fill_value=(cam[0], cam[-1]))
            cam = interp_func(cam_new_indices)
        
        # Store results
        results.append({
            'window_idx': idx,
            'window': window,
            'window_times': window_times[:len(window)],
            'cam': cam,
            'label': label,
            'prediction': prediction,
            'probability': probability,
            'correct': (prediction == label)
        })
    
    return results


def visualize_gradcam_results(results, participant_name, signal_data, peak_timestamps, 
                              num_plots=10, save_path=None):
    """
    Visualize Grad-CAM results.
    
    Args:
        results: List of analysis results
        participant_name: Name of participant/block
        signal_data: Full signal data
        peak_timestamps: Peak timestamps
        num_plots: Number of windows to visualize
        save_path: Path to save the figure
    """
    # Select diverse samples for visualization
    activity_results = [r for r in results if r['label'] == 1]
    baseline_results = [r for r in results if r['label'] == 0]
    
    num_activity_plots = min(num_plots // 2, len(activity_results))
    num_baseline_plots = min(num_plots // 2, len(baseline_results))
    
    selected_results = (
        activity_results[:num_activity_plots] +
        baseline_results[:num_baseline_plots]
    )
    
    # Get full signal for context
    timestamps = signal_data['timestamp'].values
    signal_column = 'rms' if 'rms' in signal_data.columns else 'emg'
    signal_values = signal_data[signal_column].values
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(selected_results), 1, figsize=(15, 3 * len(selected_results)))
    if len(selected_results) == 1:
        axes = [axes]
    
    for ax, result in zip(axes, selected_results):
        window = result['window']
        window_times = result['window_times']
        cam = result['cam']
        label = result['label']
        prediction = result['prediction']
        probability = result['probability']
        correct = result['correct']
        
        # Plot signal
        ax.plot(window_times, window, 'b-', linewidth=1.5, label='EMG Signal', alpha=0.7)
        
        # Plot Grad-CAM heatmap as overlay
        # Normalize CAM to match signal scale for visualization
        cam_scaled = cam * (window.max() - window.min()) + window.min()
        ax.fill_between(window_times, window.min(), cam_scaled, alpha=0.4, 
                        color='red', label='Grad-CAM Importance')
        
        # Add vertical line at center (typical detection point)
        center_time = window_times[len(window_times) // 2]
        ax.axvline(center_time, color='green', linestyle='--', alpha=0.5, linewidth=1)
        
        # Title with prediction info
        status = "✓" if correct else "✗"
        title = f"{status} True: {'Activity' if label == 1 else 'Baseline'} | "
        title += f"Pred: {'Activity' if prediction == 1 else 'Baseline'} ({probability:.3f})"
        ax.set_title(title, fontsize=10, fontweight='bold' if not correct else 'normal')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Grad-CAM Analysis: {participant_name}\n'
                 f'Red overlay shows temporal importance (higher = more important for prediction)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Grad-CAM visualization: {save_path}")
    
    plt.show()


def analyze_temporal_importance_patterns(results):
    """
    Analyze patterns in temporal importance across all windows.
    
    Args:
        results: List of analysis results
    
    Returns:
        analysis: Dictionary with analysis statistics
    """
    print("\nAnalyzing temporal importance patterns...")
    
    # Separate by class
    activity_results = [r for r in results if r['label'] == 1]
    baseline_results = [r for r in results if r['label'] == 0]
    
    # Separate by prediction correctness
    correct_activity = [r for r in activity_results if r['correct']]
    incorrect_activity = [r for r in activity_results if not r['correct']]
    correct_baseline = [r for r in baseline_results if r['correct']]
    incorrect_baseline = [r for r in baseline_results if not r['correct']]
    
    def compute_statistics(result_list, name):
        if len(result_list) == 0:
            return None
        
        cams = np.array([r['cam'] for r in result_list])
        
        # Find peak importance location (normalized position in window)
        peak_positions = np.array([np.argmax(r['cam']) / len(r['cam']) for r in result_list])
        
        # Average CAM across all windows
        avg_cam = np.mean(cams, axis=0)
        
        # Importance distribution (early, middle, late)
        window_length = len(avg_cam)
        early_importance = np.mean(avg_cam[:window_length // 3])
        middle_importance = np.mean(avg_cam[window_length // 3:2 * window_length // 3])
        late_importance = np.mean(avg_cam[2 * window_length // 3:])
        
        return {
            'name': name,
            'count': len(result_list),
            'avg_cam': avg_cam,
            'avg_peak_position': np.mean(peak_positions),
            'std_peak_position': np.std(peak_positions),
            'early_importance': early_importance,
            'middle_importance': middle_importance,
            'late_importance': late_importance,
            'total_importance': np.sum(avg_cam)
        }
    
    stats = {}
    
    if len(activity_results) > 0:
        stats['activity_all'] = compute_statistics(activity_results, 'Activity (All)')
        if len(correct_activity) > 0:
            stats['activity_correct'] = compute_statistics(correct_activity, 'Activity (Correct)')
        if len(incorrect_activity) > 0:
            stats['activity_incorrect'] = compute_statistics(incorrect_activity, 'Activity (Incorrect)')
    
    if len(baseline_results) > 0:
        stats['baseline_all'] = compute_statistics(baseline_results, 'Baseline (All)')
        if len(correct_baseline) > 0:
            stats['baseline_correct'] = compute_statistics(correct_baseline, 'Baseline (Correct)')
        if len(incorrect_baseline) > 0:
            stats['baseline_incorrect'] = compute_statistics(incorrect_baseline, 'Baseline (Incorrect)')
    
    # Print summary
    print("\n" + "="*60)
    print("TEMPORAL IMPORTANCE ANALYSIS SUMMARY")
    print("="*60)
    
    for key, stat in stats.items():
        if stat is None:
            continue
        print(f"\n{stat['name']} (n={stat['count']}):")
        print(f"  Average peak importance position: {stat['avg_peak_position']:.2f} ± {stat['std_peak_position']:.2f} (0=start, 1=end)")
        print(f"  Early window importance: {stat['early_importance']:.3f}")
        print(f"  Middle window importance: {stat['middle_importance']:.3f}")
        print(f"  Late window importance: {stat['late_importance']:.3f}")
        print(f"  Total importance: {stat['total_importance']:.3f}")
    
    return stats


def visualize_average_importance_patterns(stats, save_path=None):
    """Visualize average importance patterns across different categories."""
    if not stats:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot 1: Activity vs Baseline average CAM
    if 'activity_all' in stats and 'baseline_all' in stats:
        ax = axes[plot_idx]
        ax.plot(stats['activity_all']['avg_cam'], 'r-', linewidth=2, label='Activity Windows')
        ax.plot(stats['baseline_all']['avg_cam'], 'b-', linewidth=2, label='Baseline Windows')
        ax.set_xlabel('Time Point in Window')
        ax.set_ylabel('Average Importance')
        ax.set_title('Average Temporal Importance: Activity vs Baseline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 2: Correct vs Incorrect predictions for Activity
    if 'activity_correct' in stats and 'activity_incorrect' in stats:
        ax = axes[plot_idx]
        ax.plot(stats['activity_correct']['avg_cam'], 'g-', linewidth=2, label='Correct Predictions')
        ax.plot(stats['activity_incorrect']['avg_cam'], 'r--', linewidth=2, label='Incorrect Predictions')
        ax.set_xlabel('Time Point in Window')
        ax.set_ylabel('Average Importance')
        ax.set_title('Activity Windows: Correct vs Incorrect')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 3: Correct vs Incorrect predictions for Baseline
    if 'baseline_correct' in stats and 'baseline_incorrect' in stats:
        ax = axes[plot_idx]
        ax.plot(stats['baseline_correct']['avg_cam'], 'g-', linewidth=2, label='Correct Predictions')
        ax.plot(stats['baseline_incorrect']['avg_cam'], 'r--', linewidth=2, label='Incorrect Predictions')
        ax.set_xlabel('Time Point in Window')
        ax.set_ylabel('Average Importance')
        ax.set_title('Baseline Windows: Correct vs Incorrect')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 4: Importance distribution (early/middle/late)
    if 'activity_all' in stats and 'baseline_all' in stats:
        ax = axes[plot_idx]
        categories = ['Early', 'Middle', 'Late']
        activity_vals = [
            stats['activity_all']['early_importance'],
            stats['activity_all']['middle_importance'],
            stats['activity_all']['late_importance']
        ]
        baseline_vals = [
            stats['baseline_all']['early_importance'],
            stats['baseline_all']['middle_importance'],
            stats['baseline_all']['late_importance']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, activity_vals, width, label='Activity', color='red', alpha=0.7)
        ax.bar(x + width/2, baseline_vals, width, label='Baseline', color='blue', alpha=0.7)
        ax.set_xlabel('Window Region')
        ax.set_ylabel('Average Importance')
        ax.set_title('Importance Distribution Across Window Regions')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Grad-CAM Temporal Importance Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved pattern analysis: {save_path}")
    
    plt.show()


def main():
    """Main function for Grad-CAM analysis."""
    print("Grad-CAM Analysis for Muscle Activity CNN")
    print("=" * 60)
    
    # Load model
    #model_path = 'trained_models/error_als.pth'
    model_path = 'trained_models/cnn_als.pth'
    model, device, model_type, window_size_ms = load_trained_model(model_path)
    
    # Load test data
    test_data = load_test_data()
    
    if not test_data:
        print("❌ No test data loaded")
        return
    
    # Analyze each participant/block
    all_results = []
    
    for participant_data in test_data:
        participant_name = participant_data['participant']
        signal_data = participant_data['signal_data']
        peak_timestamps = participant_data['peak_timestamps']
        
        print(f"\n{'='*60}")
        print(f"Analyzing {participant_name}")
        print(f"{'='*60}")
        
        # Analyze windows with Grad-CAM
        results = analyze_windows_with_gradcam(
            model, device, model_type, signal_data, peak_timestamps, 
            num_samples=50,  # Analyze 50 windows per participant
            window_size_ms=window_size_ms
        )
        
        all_results.extend(results)
        
        # Visualize individual windows
        save_path = f'experiment_results/gradcam_{participant_name.lower()}_windows.png'
        visualize_gradcam_results(
            results, participant_name, signal_data, peak_timestamps,
            num_plots=10, save_path=save_path
        )
    
    # Overall pattern analysis
    print(f"\n{'='*60}")
    print("Overall Pattern Analysis")
    print(f"{'='*60}")
    
    stats = analyze_temporal_importance_patterns(all_results)
    
    # Visualize patterns
    pattern_save_path = 'experiment_results/gradcam_importance_patterns.png'
    visualize_average_importance_patterns(stats, save_path=pattern_save_path)
    
    print(f"\n{'='*60}")
    print("Grad-CAM Analysis Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

