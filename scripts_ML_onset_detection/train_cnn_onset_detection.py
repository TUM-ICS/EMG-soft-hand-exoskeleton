"""
1D CNN Model for Peak Detection in 1162ms Windows with 162ms Peak Detection Zone
==============================================================================

This module implements a 1D CNN model to predict whether 1162ms windows
contain a peak in the last 162ms or not. The model is trained on healthy 
participants P1-P11 and evaluated on P12-P15 using all sliding windows
without class balancing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.utils import resample
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('experiment_results', exist_ok=True)
os.makedirs('trained_models', exist_ok=True)


class PeakDetectionCNN(nn.Module):
    """1D CNN model for peak detection in 1162ms windows with 162ms peak detection zone."""
    
    def __init__(self, input_length=1162, num_filters=64, dropout=0.3):
        super(PeakDetectionCNN, self).__init__()
        
        self.input_length = input_length
        
        # 1D Convolutional layers with less aggressive pooling
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        # Remove the fourth pooling layer to prevent size becoming too small
        
        # Since we use global average pooling, the input to FC layers is just the number of channels
        self.fc_input_size = 256  # Number of channels from conv4
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc4 = nn.Linear(64, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        # Fourth conv block (no pooling)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        # Global average pooling instead of flattening
        x = torch.mean(x, dim=2)  # (batch_size, 256)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        # Remove sigmoid activation since we're using BCEWithLogitsLoss
        
        return x.squeeze(-1)


class PeakDetectionDataset(torch.utils.data.Dataset):
    """Dataset for peak detection in 1162ms windows with 162ms peak detection zone."""
    
    def __init__(self, signal_data, peak_timestamps, window_size_ms=1162, peak_detection_ms=162, sampling_rate=34.81):
        """
        Args:
            signal_data: DataFrame with 'timestamp' and 'rms'/'emg' columns
            peak_timestamps: Array of peak timestamps in seconds
            window_size_ms: Window size in milliseconds (default 1162ms)
            peak_detection_ms: Peak detection zone in milliseconds (default 162ms)
            sampling_rate: Sampling rate in Hz
        """
        self.signal_data = signal_data
        self.peak_timestamps = peak_timestamps
        self.window_size_ms = window_size_ms
        self.peak_detection_ms = peak_detection_ms
        self.sampling_rate = sampling_rate
        
        # Convert window size to samples
        self.window_size_samples = int(window_size_ms * sampling_rate / 1000)
        self.peak_detection_samples = int(peak_detection_ms * sampling_rate / 1000)
        
        # Create windows and labels
        self.windows, self.labels = self._create_windows_and_labels()
        
    def _create_windows_and_labels(self):
        """Create 1162ms windows and check for peaks in the last 162ms."""
        timestamps = self.signal_data['timestamp'].values
        
        # Handle both 'rms' and 'emg' column names
        if 'rms' in self.signal_data.columns:
            signal_values = self.signal_data['rms'].values
        elif 'emg' in self.signal_data.columns:
            signal_values = self.signal_data['emg'].values
        else:
            raise ValueError("Signal data must have either 'rms' or 'emg' column")
        
        windows = []
        labels = []
        
        # Create sliding windows
        for i in range(len(signal_values) - self.window_size_samples + 1):
            window = signal_values[i:i + self.window_size_samples]
            window_start_time = timestamps[i]
            window_end_time = timestamps[i + self.window_size_samples - 1]
            
            # Check if there's a peak in the last 162ms of the window
            has_peak = self._check_peak_in_last_162ms(window_start_time, window_end_time)
            
            windows.append(window)
            labels.append(1 if has_peak else 0)
        
        return np.array(windows), np.array(labels)
    
    def _check_peak_in_last_162ms(self, window_start_time, window_end_time):
        """Check if there's a peak in the last 162ms of the window."""
        # Calculate the time range for the last 162ms of the window
        peak_detection_start_time = window_end_time - (self.peak_detection_ms / 1000.0)
        
        for peak_time in self.peak_timestamps:
            if peak_detection_start_time <= peak_time <= window_end_time:
                return True
        return False
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = torch.FloatTensor(self.windows[idx])
        label = torch.FloatTensor([self.labels[idx]])
        return window, label


class CNNTrainer:
    """Trainer class for CNN peak detection."""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            # Apply sigmoid to get probabilities for threshold comparison
            pred_probs = torch.sigmoid(output)
            pred = (pred_probs > 0.5).float()
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
        return total_loss / len(dataloader), 100. * correct / total
    
    def evaluate(self, dataloader, criterion):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                # Apply sigmoid to get probabilities for threshold comparison
                pred_probs = torch.sigmoid(output)
                pred = (pred_probs > 0.5).float()
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_predictions.extend(pred_probs.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_loss = total_loss / len(dataloader)
        val_acc = 100. * correct / total
        
        return (val_loss, val_acc, np.array(all_predictions), np.array(all_targets))
    


def load_signal_data(signal_file):
    """Load signal data from CSV file."""
    try:
        df = pd.read_csv(signal_file)
        return df
    except Exception as e:
        print(f"❌ Error loading signal data: {e}")
        return None


def load_peak_labels(peak_file):
    """Load peak labels from CSV file."""
    try:
        df = pd.read_csv(peak_file)
        peak_timestamps = df['timestamp'].values
        return peak_timestamps
    except Exception as e:
        print(f"❌ Error loading peak labels: {e}")
        return None


def load_healthy_training_data():
    """Load healthy training data from P1-P11."""
    print("Loading Healthy Training Data (P1-P11)")
    print("=" * 40)
    
    signal_dir = '../_data_for_ml/signal_data'
    label_dir = '../_data_for_ml/label_data'
    
    all_signal_data = []
    all_peak_timestamps = []
    
    # Load P1-P11
    for participant_num in range(1, 12):
        signal_file = os.path.join(signal_dir, f'RMS_healthy_P{participant_num}_34p81hz_processed_cleaned.csv')
        peak_file = os.path.join(label_dir, f'peaks_P{participant_num}_interactive_final.csv')
        
        signal_data = load_signal_data(signal_file)
        peak_timestamps = load_peak_labels(peak_file)
        
        if signal_data is not None and peak_timestamps is not None:
            all_signal_data.append(signal_data)
            all_peak_timestamps.extend(peak_timestamps)
            print(f"✓ P{participant_num}: {len(signal_data)} samples, {len(peak_timestamps)} peaks")
        else:
            print(f"❌ Failed to load P{participant_num}")
    
    print(f"\nTotal healthy participants: {len(all_signal_data)}")
    print(f"Total healthy peaks: {len(all_peak_timestamps)}")
    
    return all_signal_data, all_peak_timestamps


def load_als_training_data():
    """Load ALS training data from first two blocks."""
    print("Loading ALS Training Data (Block1 & Block2)")
    print("=" * 40)
    
    signal_dir = '../_data_for_ml/signal_data'
    label_dir = '../_data_for_ml/label_data'
    
    all_signal_data = []
    all_peak_timestamps = []
    
    # Load first two blocks of ALS data
    als_blocks = ['block1', 'block2']
    
    for block in als_blocks:
        signal_file = os.path.join(signal_dir, f'RMS_ALS_{block}.csv')
        peak_file = os.path.join(label_dir, f'peaks_ALS_{block}.csv')
        
        signal_data = load_signal_data(signal_file)
        peak_timestamps = load_peak_labels(peak_file)
        
        if signal_data is not None and peak_timestamps is not None:
            all_signal_data.append(signal_data)
            all_peak_timestamps.extend(peak_timestamps)
            print(f"ALS {block}: {len(signal_data)} samples, {len(peak_timestamps)} peaks")
        else:
            print(f"Failed to load ALS {block}")
    
    print(f"\nTotal ALS blocks: {len(all_signal_data)}")
    print(f"Total ALS peaks: {len(all_peak_timestamps)}")
    
    return all_signal_data, all_peak_timestamps


def load_sma_training_data():
    """Load SMA training data."""
    print("Loading SMA Training Data")
    print("=" * 30)
    
    signal_dir = '../_data_for_ml/signal_data'
    label_dir = '../_data_for_ml/label_data'
    
    all_signal_data = []
    all_peak_timestamps = []
    
    # Load SMA data
    signal_file = os.path.join(signal_dir, 'RMS_SMA_34p81hz_processed_cleaned.csv')
    peak_file = os.path.join(label_dir, 'peaks_SMA_clinical.csv')
    
    signal_data = load_signal_data(signal_file)
    peak_timestamps = load_peak_labels(peak_file)
    
    if signal_data is not None and peak_timestamps is not None:
        all_signal_data.append(signal_data)
        all_peak_timestamps.extend(peak_timestamps)
        print(f"✓ SMA: {len(signal_data)} samples, {len(peak_timestamps)} peaks")
    else:
        print(f"Failed to load SMA data")
    
    print(f"\nTotal SMA datasets: {len(all_signal_data)}")
    print(f"Total SMA peaks: {len(all_peak_timestamps)}")
    
    return all_signal_data, all_peak_timestamps


def combine_training_data(healthy_data=None, als_data=None, sma_data=None):
    """Combine different training datasets."""
    all_signal_data = []
    all_peak_timestamps = []
    
    if healthy_data is not None:
        healthy_signal, healthy_peaks = healthy_data
        all_signal_data.extend(healthy_signal)
        all_peak_timestamps.extend(healthy_peaks)
        print(f"✓ Added {len(healthy_signal)} healthy participants with {len(healthy_peaks)} peaks")
    
    if als_data is not None:
        als_signal, als_peaks = als_data
        all_signal_data.extend(als_signal)
        all_peak_timestamps.extend(als_peaks)
        print(f"✓ Added {len(als_signal)} ALS blocks with {len(als_peaks)} peaks")
    
    if sma_data is not None:
        sma_signal, sma_peaks = sma_data
        all_signal_data.extend(sma_signal)
        all_peak_timestamps.extend(sma_peaks)
        print(f"✓ Added {len(sma_signal)} SMA datasets with {len(sma_peaks)} peaks")
    
    print(f"\nCombined training dataset:")
    print(f"  Total participants/blocks: {len(all_signal_data)}")
    print(f"  Total peaks: {len(all_peak_timestamps)}")
    
    return all_signal_data, all_peak_timestamps


def load_training_data():
    """Load training data from P1-P11 (for backward compatibility)."""
    return load_healthy_training_data()


def load_test_data():
    """Load test data from P12-P15."""
    print("\nLoading Test Data (P12-P15)")
    print("=" * 30)
    
    signal_dir = '../_data_for_ml/signal_data'
    label_dir = '../_data_for_ml/label_data'
    
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
            print(f"✓ P{participant_num}: {len(signal_data)} samples, {len(peak_timestamps)} peaks")
        else:
            print(f"Failed to load P{participant_num}")
    
    print(f"\nTotal test participants: {len(test_data)}")
    
    return test_data


def visualize_signal_characteristics(all_signal_data, all_peak_timestamps, window_size_ms=1162, peak_detection_ms=162, dataset_name="Unknown"):
    """Visualize mean and variance of windows with/without peaks in the activity area."""
    print(f"\nVisualizing Signal Characteristics")
    print("=" * 40)
    
    # Collect all windows and their labels
    all_windows = []
    all_labels = []
    
    for i, signal_data in enumerate(all_signal_data):
        participant_name = f"P{i+1}"
        print(f"Processing {participant_name} for visualization...")
        
        # Create dataset for this participant
        dataset = PeakDetectionDataset(signal_data, all_peak_timestamps, window_size_ms, peak_detection_ms)
        
        all_windows.extend(dataset.windows)
        all_labels.extend(dataset.labels)
    
    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)
    
    # Separate windows with and without peaks
    peak_windows = all_windows[all_labels == 1]
    non_peak_windows = all_windows[all_labels == 0]
    
    print(f"✓ Peak windows: {len(peak_windows)}")
    print(f"✓ Non-peak windows: {len(non_peak_windows)}")
    
    # Calculate mean and variance for each time point
    peak_mean = np.mean(peak_windows, axis=0)
    peak_var = np.var(peak_windows, axis=0)
    non_peak_mean = np.mean(non_peak_windows, axis=0)
    non_peak_var = np.var(non_peak_windows, axis=0)
    
    # Create time axis (in milliseconds)
    time_ms = np.linspace(0, window_size_ms, len(peak_mean))
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Signal Characteristics: Windows with vs without Peaks in Activity Area', fontsize=16)
    
    # Plot 1: Mean signals
    axes[0, 0].plot(time_ms, peak_mean, 'r-', linewidth=2, label=f'With Peaks (n={len(peak_windows)})')
    axes[0, 0].plot(time_ms, non_peak_mean, 'b-', linewidth=2, label=f'Without Peaks (n={len(non_peak_windows)})')
    axes[0, 0].axvline(x=window_size_ms - peak_detection_ms, color='gray', linestyle='--', alpha=0.7, label='Activity Area Start')
    axes[0, 0].axvline(x=window_size_ms, color='gray', linestyle='-', alpha=0.7, label='Activity Area End')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Mean RMS EMG (V)')
    axes[0, 0].set_title('Mean Signal Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Variance signals
    axes[0, 1].plot(time_ms, peak_var, 'r-', linewidth=2, label=f'With Peaks (n={len(peak_windows)})')
    axes[0, 1].plot(time_ms, non_peak_var, 'b-', linewidth=2, label=f'Without Peaks (n={len(non_peak_windows)})')
    axes[0, 1].axvline(x=window_size_ms - peak_detection_ms, color='gray', linestyle='--', alpha=0.7, label='Activity Area Start')
    axes[0, 1].axvline(x=window_size_ms, color='gray', linestyle='-', alpha=0.7, label='Activity Area End')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Variance RMS EMG (V²)')
    axes[0, 1].set_title('Variance Signal Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Mean difference
    mean_diff = peak_mean - non_peak_mean
    axes[1, 0].plot(time_ms, mean_diff, 'g-', linewidth=2, label='Mean Difference')
    axes[1, 0].axvline(x=window_size_ms - peak_detection_ms, color='gray', linestyle='--', alpha=0.7, label='Activity Area Start')
    axes[1, 0].axvline(x=window_size_ms, color='gray', linestyle='-', alpha=0.7, label='Activity Area End')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Mean Difference (V)')
    axes[1, 0].set_title('Mean Signal Difference (Peak - Non-Peak)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Variance difference
    var_diff = peak_var - non_peak_var
    axes[1, 1].plot(time_ms, var_diff, 'purple', linewidth=2, label='Variance Difference')
    axes[1, 1].axvline(x=window_size_ms - peak_detection_ms, color='gray', linestyle='--', alpha=0.7, label='Activity Area Start')
    axes[1, 1].axvline(x=window_size_ms, color='gray', linestyle='-', alpha=0.7, label='Activity Area End')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('Variance Difference (V²)')
    axes[1, 1].set_title('Variance Signal Difference (Peak - Non-Peak)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    signal_filename = f'signal_characteristics_{dataset_name.replace(" ", "_").replace("+", "plus").lower()}.png'
    signal_path = os.path.join('experiment_results', signal_filename)
    plt.savefig(signal_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Peak windows mean RMS: {np.mean(peak_mean):.4f} ± {np.std(peak_mean):.4f} V")
    print(f"  Non-peak windows mean RMS: {np.mean(non_peak_mean):.4f} ± {np.std(non_peak_mean):.4f} V")
    print(f"  Peak windows variance: {np.mean(peak_var):.6f} ± {np.std(peak_var):.6f} V²")
    print(f"  Non-peak windows variance: {np.mean(non_peak_var):.6f} ± {np.std(non_peak_var):.6f} V²")
    
    # Activity area statistics
    activity_start_idx = int((window_size_ms - peak_detection_ms) * len(peak_mean) / window_size_ms)
    activity_end_idx = len(peak_mean)
    
    peak_activity_mean = np.mean(peak_mean[activity_start_idx:activity_end_idx])
    non_peak_activity_mean = np.mean(non_peak_mean[activity_start_idx:activity_end_idx])
    peak_activity_var = np.mean(peak_var[activity_start_idx:activity_end_idx])
    non_peak_activity_var = np.mean(non_peak_var[activity_start_idx:activity_end_idx])
    
    print(f"\nActivity Area ({peak_detection_ms}ms) Statistics:")
    print(f"  Peak windows mean in activity area: {peak_activity_mean:.4f} V")
    print(f"  Non-peak windows mean in activity area: {non_peak_activity_mean:.4f} V")
    print(f"  Peak windows variance in activity area: {peak_activity_var:.6f} V²")
    print(f"  Non-peak windows variance in activity area: {non_peak_activity_var:.6f} V²")
    print(f"  Mean difference in activity area: {peak_activity_mean - non_peak_activity_mean:.4f} V")
    print(f"  Variance difference in activity area: {peak_activity_var - non_peak_activity_var:.6f} V²")
    
    print(f"\n✓ Visualization saved as: signal_characteristics_visualization.png")


def create_training_dataset(all_signal_data, all_peak_timestamps, window_size_ms=1162):
    """Create training dataset from all training participants using all sliding windows."""
    print(f"\nCreating Training Dataset (All Sliding Windows)")
    print("=" * 50)
    
    all_windows = []
    all_labels = []
    
    for i, signal_data in enumerate(all_signal_data):
        participant_name = f"P{i+1}"
        print(f"Processing {participant_name}...")
        
        # Create dataset for this participant
        dataset = PeakDetectionDataset(signal_data, all_peak_timestamps, window_size_ms)
        
        all_windows.extend(dataset.windows)
        all_labels.extend(dataset.labels)
        
        print(f"  ✓ Windows: {len(dataset.windows)}")
        print(f"  ✓ Peak rate: {np.mean(dataset.labels)*100:.2f}%")
    
    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)
    
    print(f"\nCombined training dataset:")
    print(f"  ✓ Total windows: {len(all_windows)}")
    print(f"  ✓ Overall peak rate: {np.mean(all_labels)*100:.2f}%")
    print(f"  ✓ Peak windows: {np.sum(all_labels)}")
    print(f"  ✓ Non-peak windows: {len(all_labels) - np.sum(all_labels)}")
    print(f"  ✓ Class imbalance ratio: {(len(all_labels) - np.sum(all_labels)) / np.sum(all_labels):.2f}:1")
    
    return all_windows, all_labels


def train_cnn_model(all_signal_data, all_peak_timestamps, window_size_ms=1162, dataset_name="Unknown"):
    """Train the CNN model on training data."""
    print(f"\nTraining CNN Model")
    print("=" * 25)
    
    # Visualize signal characteristics before training
    visualize_signal_characteristics(all_signal_data, all_peak_timestamps, window_size_ms, peak_detection_ms=162, dataset_name=dataset_name)
    
    # Create training dataset (all sliding windows, no balancing)
    X_all, y_all = create_training_dataset(all_signal_data, all_peak_timestamps, window_size_ms)
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    print(f"✓ Training: {len(X_train)} windows ({np.mean(y_train)*100:.2f}% peaks)")
    print(f"✓ Validation: {len(X_val)} windows ({np.mean(y_val)*100:.2f}% peaks)")
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).unsqueeze(1),  # Add channel dimension
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val).unsqueeze(1),  # Add channel dimension
        torch.FloatTensor(y_val)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = PeakDetectionCNN(input_length=window_size_ms, num_filters=64, dropout=0.3)
    
    trainer = CNNTrainer(model, device)
    
    # Training setup with class weights for imbalanced data
    # Calculate class weights based on the training data
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    # Convert to tensor and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    # Use weighted BCE loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1] / class_weights_tensor[0])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print(f"✓ Class weights: {class_weights}")
    print(f"✓ Using weighted BCE loss for imbalanced classes")
    
    # Training loop
    print(f"\nTraining Model (User-controlled stopping every 50 epochs)")
    print("=" * 55)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    max_epochs = 1000  # Set a high maximum to allow user control
    epoch = 0
    
    while epoch < max_epochs:
        # Train for 50 epochs
        for i in range(50):
            if epoch >= max_epochs:
                break
                
            train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc, _, _ = trainer.evaluate(val_loader, criterion)
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            epoch += 1
        
        # Ask user if they want to continue
        if epoch < max_epochs:
            print(f"\nCompleted {epoch} epochs.")
            print(f"Current performance: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}")
            
            while True:
                user_input = input("Continue training for another 50 epochs? (y/n): ").lower().strip()
                if user_input in ['y', 'yes']:
                    print("Continuing training...")
                    break
                elif user_input in ['n', 'no']:
                    print("Stopping training.")
                    max_epochs = epoch  # Stop training
                    break
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
    
    print(f"\nTraining completed after {epoch} epochs.")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training')
    plt.plot(val_losses, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training')
    plt.plot(val_accuracies, label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    curves_filename = f'training_curves_{dataset_name.replace(" ", "_").replace("+", "plus").lower()}.png'
    curves_path = os.path.join('experiment_results', curves_filename)
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Final validation loss: {val_loss:.4f}")
    
    # Save the trained model with dataset name
    model_filename = f'cnn_{dataset_name.replace(" ", "_").replace("+", "plus").lower()}.pth'
    model_path = os.path.join('trained_models', model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved as: {model_path}")
    
    return model, trainer


def evaluate_on_test_participants(model, test_data, window_size_ms=1162):
    """Evaluate the model on test participants P12-P15."""
    print(f"\nEvaluating CNN on Test Participants (P12-P15)")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_results = []
    
    for participant_data in test_data:
        participant_name = participant_data['participant']
        signal_data = participant_data['signal_data']
        peak_timestamps = participant_data['peak_timestamps']
        
        print(f"\nEvaluating {participant_name}")
        print("-" * 20)
        
        # Create dataset
        dataset = PeakDetectionDataset(signal_data, peak_timestamps, window_size_ms)
        
        print(f"✓ Created {len(dataset)} windows")
        print(f"✓ Peak rate: {np.mean(dataset.labels)*100:.2f}%")
        print(f"✓ Peak windows: {np.sum(dataset.labels)}")
        print(f"✓ Non-peak windows: {len(dataset.labels) - np.sum(dataset.labels)}")
        
        # Use all windows for evaluation (no balancing)
        X_eval = dataset.windows
        y_eval = dataset.labels
        
        print(f"  Using all {len(X_eval)} windows for evaluation")
        
        # Create data loader
        test_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_eval).unsqueeze(1),  # Add channel dimension
                torch.FloatTensor(y_eval)
            ),
            batch_size=64,
            shuffle=False
        )
        
        # Get predictions
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                output = model(data)
                # Apply sigmoid to get probabilities
                pred_probs = torch.sigmoid(output)
                all_predictions.extend(pred_probs.cpu().numpy())
                all_targets.extend(target.numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(all_targets, all_predictions)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Evaluate with optimal threshold
        pred_binary = (all_predictions > optimal_threshold).astype(int)
        accuracy = np.mean(pred_binary == all_targets) * 100
        
        # Calculate metrics
        true_positives = np.sum((pred_binary == 1) & (all_targets == 1))
        false_positives = np.sum((pred_binary == 1) & (all_targets == 0))
        false_negatives = np.sum((pred_binary == 0) & (all_targets == 1))
        true_negatives = np.sum((pred_binary == 0) & (all_targets == 0))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        auc_score = roc_auc_score(all_targets, all_predictions)
        
        print(f"  Optimal threshold: {optimal_threshold:.3f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1_score:.3f}")
        print(f"  AUC Score: {auc_score:.3f}")
        
        all_results.append({
            'participant': participant_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc_score': auc_score,
            'optimal_threshold': optimal_threshold,
            'predictions': all_predictions,
            'targets': all_targets,
            'signal_data': signal_data,
            'peak_timestamps': peak_timestamps
        })
    
    return all_results


def visualize_results(all_results, dataset_name="Unknown"):
    """Visualize the evaluation results."""
    print(f"\nCreating Visualizations")
    print("=" * 25)
    
    for result in all_results:
        participant_name = result['participant']
        signal_data = result['signal_data']
        peak_timestamps = result['peak_timestamps']
        predictions = result['predictions']
        targets = result['targets']
        optimal_threshold = result['optimal_threshold']
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Signal with peaks
        plt.subplot(3, 1, 1)
        signal_column = 'rms' if 'rms' in signal_data.columns else 'emg'
        plt.plot(signal_data['timestamp'], signal_data[signal_column], 
                 'b-', alpha=0.7, linewidth=0.8, label='RMS EMG Signal')
        
        # Plot actual peaks
        if len(peak_timestamps) > 0:
            peak_values = []
            for peak_time in peak_timestamps:
                closest_idx = np.argmin(np.abs(signal_data['timestamp'] - peak_time))
                peak_values.append(signal_data[signal_column].iloc[closest_idx])
            
            plt.scatter(peak_timestamps, peak_values, 
                       color='red', s=50, label=f'Actual Peaks ({len(peak_timestamps)})', zorder=5)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('RMS EMG (V)')
        plt.title(f'{participant_name} - CNN Peak Detection Results (1162ms windows, 162ms peak detection)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Prediction probabilities
        plt.subplot(3, 1, 2)
        plt.plot(predictions, 'g-', alpha=0.7, linewidth=1, label='Peak Detection Probability')
        plt.axhline(y=optimal_threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Optimal Threshold ({optimal_threshold:.3f})')
        plt.xlabel('Window Index')
        plt.ylabel('Peak Detection Probability')
        plt.title('Peak Detection Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot 3: Confusion matrix
        plt.subplot(3, 1, 3)
        pred_binary = (predictions > optimal_threshold).astype(int)
        
        confusion_matrix = np.array([[np.sum((pred_binary == 0) & (targets == 0)), 
                                     np.sum((pred_binary == 1) & (targets == 0))],
                                    [np.sum((pred_binary == 0) & (targets == 1)), 
                                     np.sum((pred_binary == 1) & (targets == 1))]])
        
        plt.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                plt.text(j, i, confusion_matrix[i, j], ha='center', va='center', fontsize=12)
        
        plt.xticks([0, 1], ['No Peak', 'Peak'])
        plt.yticks([0, 1], ['No Peak', 'Peak'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        result_filename = f'cnn_{participant_name.lower()}_peak_detection_results_{dataset_name.replace(" ", "_").replace("+", "plus").lower()}.png'
        result_path = os.path.join('experiment_results', result_filename)
        plt.savefig(result_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ {participant_name} visualization saved as {result_filename}")


def main():
    """Main function."""
    print("CNN Peak Detection Training")
    print("1162ms windows with 162ms peak detection zone (no class balancing)")
    print("=" * 70)
    
    # Ask user to choose training data combination
    print("\nChoose training data combination:")
    print("1. Healthy only (P1-P11)")
    print("2. Healthy + ALS (P1-P11 + ALS Block1&2)")
    print("3. ALS only (ALS Block1&2)")
    print("4. SMA only")
    print("5. Healthy + SMA (P1-P11 + SMA)")
    print("6. Healthy + SMA + ALS (P1-P11 + SMA + ALS Block1&2)")
    
    while True:
        choice = input("Enter your choice (1-6): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6']:
            break
        else:
            print("Please enter a number between 1 and 6.")
    
    try:
        # Load training data based on choice
        print(f"\nLoading training data for option {choice}...")
        
        if choice == '1':  # Healthy only
            all_signal_data, all_peak_timestamps = load_healthy_training_data()
            dataset_name = "Healthy"
            
        elif choice == '2':  # Healthy + ALS
            healthy_data = load_healthy_training_data()
            als_data = load_als_training_data()
            all_signal_data, all_peak_timestamps = combine_training_data(healthy_data, als_data)
            dataset_name = "Healthy + ALS"
            
        elif choice == '3':  # ALS only
            all_signal_data, all_peak_timestamps = load_als_training_data()
            dataset_name = "ALS"
            
        elif choice == '4':  # SMA only
            all_signal_data, all_peak_timestamps = load_sma_training_data()
            dataset_name = "SMA"
            
        elif choice == '5':  # Healthy + SMA
            healthy_data = load_healthy_training_data()
            sma_data = load_sma_training_data()
            all_signal_data, all_peak_timestamps = combine_training_data(healthy_data, sma_data)
            dataset_name = "Healthy + SMA"
            
        elif choice == '6':  # Healthy + SMA + ALS
            healthy_data = load_healthy_training_data()
            sma_data = load_sma_training_data()
            als_data = load_als_training_data()
            all_signal_data, all_peak_timestamps = combine_training_data(healthy_data, sma_data, als_data)
            dataset_name = "Healthy + SMA + ALS"
        
        if not all_signal_data:
            print("Failed to load training data")
            return
        
        # Load test data (P12-P15)
        test_data = load_test_data()
        if not test_data:
            print("Failed to load test data")
            return
        
        # Train CNN model
        model, trainer = train_cnn_model(all_signal_data, all_peak_timestamps, dataset_name=dataset_name)
        
        # Evaluate on test participants
        all_results = evaluate_on_test_participants(model, test_data)
        
        # Visualize results
        visualize_results(all_results, dataset_name)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"CNN PEAK DETECTION TRAINING AND EVALUATION COMPLETED")
        print(f"Training Dataset: {dataset_name}")
        print(f"{'='*70}")
        
        print(f"\nSummary Results:")
        print(f"{'Participant':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<8}")
        print("-" * 70)
        
        for result in all_results:
            print(f"{result['participant']:<12} {result['accuracy']:<10.2f} {result['precision']:<10.3f} "
                  f"{result['recall']:<10.3f} {result['f1_score']:<10.3f} {result['auc_score']:<8.3f}")
        
        # Calculate overall average
        avg_accuracy = np.mean([r['accuracy'] for r in all_results])
        avg_precision = np.mean([r['precision'] for r in all_results])
        avg_recall = np.mean([r['recall'] for r in all_results])
        avg_f1 = np.mean([r['f1_score'] for r in all_results])
        avg_auc = np.mean([r['auc_score'] for r in all_results])
        
        print("-" * 70)
        print(f"{'Average':<12} {avg_accuracy:<10.2f} {avg_precision:<10.3f} "
              f"{avg_recall:<10.3f} {avg_f1:<10.3f} {avg_auc:<8.3f}")
        
        print(f"\n✓ Files saved:")
        print(f"  - trained_models/cnn_{dataset_name.replace(' ', '_').replace('+', 'plus').lower()}.pth")
        print(f"  - experiment_results/training_curves_{dataset_name.replace(' ', '_').replace('+', 'plus').lower()}.png")
        print(f"  - experiment_results/signal_characteristics_{dataset_name.replace(' ', '_').replace('+', 'plus').lower()}.png")
        print(f"  - experiment_results/cnn_p*_peak_detection_results_{dataset_name.replace(' ', '_').replace('+', 'plus').lower()}.png")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
