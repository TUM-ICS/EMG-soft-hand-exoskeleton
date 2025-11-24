"""
1D CNN Model for Muscle Activity Detection in 200ms Windows
==========================================================

This module implements a 1D CNN model to predict whether 200ms windows
contain muscle activity or not. Activity phases are defined as -200ms to +800ms
around detected activity peaks, with the rest being baseline. A training window
is labeled as activity only if it is fully inside an activity phase.
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


class MuscleActivityCNN(nn.Module):
    """1D CNN model for muscle activity detection in 200ms windows."""
    
    def __init__(self, input_length=200, num_filters=64, dropout=0.3):
        super(MuscleActivityCNN, self).__init__()
        
        self.input_length = input_length
        
        # Simplified architecture for 200ms windows (about 7 samples at 34.81 Hz)
        # Use smaller kernels and no pooling to preserve sequence length
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Global average pooling to get fixed-size output regardless of input length
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Since we use global average pooling, the input to FC layers is just the number of channels
        self.fc_input_size = 128  # Number of channels from conv3
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 64)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(32, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # Shape: (batch_size, 128, 1)
        x = x.squeeze(-1)  # Shape: (batch_size, 128)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        
        # Remove sigmoid activation since we're using BCEWithLogitsLoss
        
        return x.squeeze(-1)


class MuscleActivityDataset(torch.utils.data.Dataset):
    """Dataset for muscle activity detection in 200ms windows."""
    
    def __init__(self, signal_data, peak_timestamps, window_size_ms=200, sampling_rate=34.81):
        """
        Args:
            signal_data: DataFrame with 'timestamp' and 'rms'/'emg' columns
            peak_timestamps: Array of peak timestamps in seconds
            window_size_ms: Window size in milliseconds (default 200ms)
            sampling_rate: Sampling rate in Hz
        """
        self.signal_data = signal_data
        self.peak_timestamps = peak_timestamps
        self.window_size_ms = window_size_ms
        self.sampling_rate = sampling_rate
        
        # Convert window size to samples
        self.window_size_samples = int(window_size_ms * sampling_rate / 1000)
        
        # Create activity phases (-200ms to +800ms around peaks)
        self.activity_phases = self._create_activity_phases()
        
        # Create windows and labels
        self.windows, self.labels = self._create_windows_and_labels()
        
    def _create_activity_phases(self):
        """Create activity phases as -200ms to +800ms around each peak."""
        activity_phases = []
        for peak_time in self.peak_timestamps:
            start_time = peak_time - 0.2  # -200ms
            end_time = peak_time + 0.8    # +800ms
            activity_phases.append((start_time, end_time))
        return activity_phases
        
    def _create_windows_and_labels(self):
        """Create 500ms windows and label based on activity phase overlap."""
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
            
            # Check if more than half of the window is within an activity phase
            is_activity = self._is_window_activity(window_start_time, window_end_time)
            
            windows.append(window)
            labels.append(1 if is_activity else 0)
        
        return np.array(windows), np.array(labels)
    
    def _is_window_activity(self, window_start_time, window_end_time):
        """Check if the window is fully inside an activity phase."""
        # Check if the entire window is contained within any activity phase
        for phase_start, phase_end in self.activity_phases:
            if phase_start <= window_start_time and window_end_time <= phase_end:
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
    
    signal_dir = '../EMG_data/signal_data'
    label_dir = '../EMG_data/label_data'
    
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
    
    signal_dir = '../EMG_data/signal_data'
    label_dir = '../EMG_data/label_data'
    
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
    
    signal_dir = '../EMG_data/signal_data'
    label_dir = '../EMG_data/label_data'
    
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
            print(f"✓ P{participant_num}: {len(signal_data)} samples, {len(peak_timestamps)} peaks")
        else:
            print(f"Failed to load P{participant_num}")
    
    print(f"\nTotal test participants: {len(test_data)}")
    
    return test_data


def visualize_signal_characteristics(all_signal_data, all_peak_timestamps, window_size_ms=500, dataset_name="Unknown"):
    """Visualize mean and variance of windows with/without muscle activity."""
    print(f"\nVisualizing Signal Characteristics")
    print("=" * 40)
    
    # Collect all windows and their labels
    all_windows = []
    all_labels = []
    
    for i, signal_data in enumerate(all_signal_data):
        participant_name = f"P{i+1}"
        print(f"Processing {participant_name} for visualization...")
        
        # Create dataset for this participant
        dataset = MuscleActivityDataset(signal_data, all_peak_timestamps, window_size_ms)
        
        all_windows.extend(dataset.windows)
        all_labels.extend(dataset.labels)
    
    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)
    
    # Separate windows with and without muscle activity
    activity_windows = all_windows[all_labels == 1]
    baseline_windows = all_windows[all_labels == 0]
    
    print(f"✓ Activity windows: {len(activity_windows)}")
    print(f"✓ Baseline windows: {len(baseline_windows)}")
    
    # Calculate mean and variance for each time point
    activity_mean = np.mean(activity_windows, axis=0)
    activity_var = np.var(activity_windows, axis=0)
    baseline_mean = np.mean(baseline_windows, axis=0)
    baseline_var = np.var(baseline_windows, axis=0)
    
    # Create time axis (in milliseconds)
    time_ms = np.linspace(0, window_size_ms, len(activity_mean))
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Signal Characteristics: Windows with vs without Muscle Activity', fontsize=16)
    
    # Plot 1: Mean signals
    axes[0, 0].plot(time_ms, activity_mean, 'r-', linewidth=2, label=f'Activity Windows (n={len(activity_windows)})')
    axes[0, 0].plot(time_ms, baseline_mean, 'b-', linewidth=2, label=f'Baseline Windows (n={len(baseline_windows)})')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Mean RMS EMG (V)')
    axes[0, 0].set_title('Mean Signal Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Variance signals
    axes[0, 1].plot(time_ms, activity_var, 'r-', linewidth=2, label=f'Activity Windows (n={len(activity_windows)})')
    axes[0, 1].plot(time_ms, baseline_var, 'b-', linewidth=2, label=f'Baseline Windows (n={len(baseline_windows)})')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Variance RMS EMG (V²)')
    axes[0, 1].set_title('Variance Signal Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Mean difference
    mean_diff = activity_mean - baseline_mean
    axes[1, 0].plot(time_ms, mean_diff, 'g-', linewidth=2, label='Mean Difference')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Mean Difference (V)')
    axes[1, 0].set_title('Mean Signal Difference (Activity - Baseline)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Variance difference
    var_diff = activity_var - baseline_var
    axes[1, 1].plot(time_ms, var_diff, 'purple', linewidth=2, label='Variance Difference')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('Variance Difference (V²)')
    axes[1, 1].set_title('Variance Signal Difference (Activity - Baseline)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    signal_filename = f'signal_characteristics_{dataset_name.replace(" ", "_").replace("+", "plus").lower()}.png'
    signal_path = os.path.join('experiment_results', signal_filename)
    plt.savefig(signal_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Activity windows mean RMS: {np.mean(activity_mean):.4f} ± {np.std(activity_mean):.4f} V")
    print(f"  Baseline windows mean RMS: {np.mean(baseline_mean):.4f} ± {np.std(baseline_mean):.4f} V")
    print(f"  Activity windows variance: {np.mean(activity_var):.6f} ± {np.std(activity_var):.6f} V²")
    print(f"  Baseline windows variance: {np.mean(baseline_var):.6f} ± {np.std(baseline_var):.6f} V²")
    
    print(f"\n✓ Visualization saved as: {signal_filename}")


def create_training_dataset(all_signal_data, all_peak_timestamps, window_size_ms=200):
    """Create training dataset from all training participants using all sliding windows."""
    print(f"\nCreating Training Dataset (All Sliding Windows)")
    print("=" * 50)
    
    all_windows = []
    all_labels = []
    
    for i, signal_data in enumerate(all_signal_data):
        participant_name = f"P{i+1}"
        print(f"Processing {participant_name}...")
        
        # Create dataset for this participant
        dataset = MuscleActivityDataset(signal_data, all_peak_timestamps, window_size_ms)
        
        all_windows.extend(dataset.windows)
        all_labels.extend(dataset.labels)
        
        print(f"  ✓ Windows: {len(dataset.windows)}")
        print(f"  ✓ Activity rate: {np.mean(dataset.labels)*100:.2f}%")
    
    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)
    
    print(f"\nCombined training dataset:")
    print(f"  ✓ Total windows: {len(all_windows)}")
    print(f"  ✓ Overall activity rate: {np.mean(all_labels)*100:.2f}%")
    print(f"  ✓ Activity windows: {np.sum(all_labels)}")
    print(f"  ✓ Baseline windows: {len(all_labels) - np.sum(all_labels)}")
    print(f"  ✓ Class imbalance ratio: {(len(all_labels) - np.sum(all_labels)) / np.sum(all_labels):.2f}:1")
    
    return all_windows, all_labels


def train_cnn_model(all_signal_data, all_peak_timestamps, window_size_ms=200, dataset_name="Unknown"):
    """Train the CNN model on training data."""
    print(f"\nTraining CNN Model")
    print("=" * 25)
    
    # Visualize signal characteristics before training
    visualize_signal_characteristics(all_signal_data, all_peak_timestamps, window_size_ms, dataset_name=dataset_name)
    
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
    
    model = MuscleActivityCNN(input_length=window_size_ms, num_filters=64, dropout=0.3)
    
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
    model_filename = f'error_{dataset_name.replace(" ", "_").replace("+", "plus").lower()}.pth'
    model_path = os.path.join('trained_models', model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved as: {model_path}")
    
    return model, trainer


def evaluate_on_test_participants(model, test_data, window_size_ms=200):
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
        dataset = MuscleActivityDataset(signal_data, peak_timestamps, window_size_ms)
        
        print(f"✓ Created {len(dataset)} windows")
        print(f"✓ Activity rate: {np.mean(dataset.labels)*100:.2f}%")
        print(f"✓ Activity windows: {np.sum(dataset.labels)}")
        print(f"✓ Baseline windows: {len(dataset.labels) - np.sum(dataset.labels)}")
        
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


def visualize_evaluation_results(all_results, dataset_name="Unknown"):
    """Visualize evaluation results with activity/baseline classification background colors."""
    print(f"\nCreating Evaluation Visualizations")
    print("=" * 35)
    
    for result in all_results:
        participant_name = result['participant']
        signal_data = result['signal_data']
        peak_timestamps = result['peak_timestamps']
        predictions = result['predictions']
        targets = result['targets']
        optimal_threshold = result['optimal_threshold']
        
        # Get signal data
        timestamps = signal_data['timestamp'].values
        signal_column = 'rms' if 'rms' in signal_data.columns else 'emg'
        signal_values = signal_data[signal_column].values
        
        # Create binary predictions
        pred_binary = (predictions > optimal_threshold).astype(int)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Signal with activity/baseline background colors
        ax1.plot(timestamps, signal_values, 'k-', alpha=0.8, linewidth=0.8, label='RMS EMG Signal')
        
        # Create activity phases for visualization
        activity_phases = []
        for peak_time in peak_timestamps:
            activity_phases.append((peak_time - 0.2, peak_time + 0.8))  # -200ms to +800ms
        
        # Highlight activity phases
        for start_time, end_time in activity_phases:
            ax1.axvspan(start_time, end_time, alpha=0.3, color='red', 
                       label='Activity Phase' if start_time == activity_phases[0][0] else "")
        
        # Plot actual peaks
        if len(peak_timestamps) > 0:
            peak_values = []
            for peak_time in peak_timestamps:
                closest_idx = np.argmin(np.abs(timestamps - peak_time))
                peak_values.append(signal_values[closest_idx])
            
            ax1.scatter(peak_timestamps, peak_values, color='red', s=50, 
                       label=f'Actual Peaks ({len(peak_timestamps)})', zorder=5)
        
        # Add background colors for detected activity/baseline windows
        window_size_s = 0.2  # 200ms in seconds
        window_centers = []
        
        for i in range(len(pred_binary)):
            # Calculate window center time
            window_center_idx = i + int(0.2 * 34.81 / 2)  # Approximate center
            if window_center_idx < len(timestamps):
                window_center_time = timestamps[window_center_idx]
                window_centers.append(window_center_time)
                
                # Add background color based on prediction
                window_start = window_center_time - window_size_s / 2
                window_end = window_center_time + window_size_s / 2
                
                if pred_binary[i] == 1:  # Activity detected
                    ax1.axvspan(window_start, window_end, alpha=0.2, color='red')
                else:  # Baseline detected
                    ax1.axvspan(window_start, window_end, alpha=0.2, color='blue')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('RMS EMG (V)')
        ax1.set_title(f'Activity Detection Results: {participant_name}\n'
                     f'Red background = Activity detected, Blue background = Baseline detected')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction probabilities over time
        if window_centers:
            ax2.plot(window_centers, predictions, 'b-', alpha=0.7, linewidth=1, label='Activity Probability')
            ax2.axhline(y=optimal_threshold, color='r', linestyle='--', alpha=0.8, label=f'Threshold ({optimal_threshold:.3f})')
            ax2.fill_between(window_centers, 0, optimal_threshold, alpha=0.2, color='blue', label='Baseline Region')
            ax2.fill_between(window_centers, optimal_threshold, 1, alpha=0.2, color='red', label='Activity Region')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Activity Probability')
        ax2.set_title(f'Activity Probabilities: {participant_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"experiment_results/error_{participant_name}_evaluation_{dataset_name}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved evaluation plot: {plot_filename}")
        
        plt.show()


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
    print("CNN Muscle Activity Detection Training")
    print("200ms windows with full activity phase labeling (no class balancing)")
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
        
        # Visualize results with activity/baseline background colors
        visualize_evaluation_results(all_results, dataset_name)
        
        # Visualize results
        visualize_results(all_results, dataset_name)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"CNN MUSCLE ACTIVITY DETECTION TRAINING AND EVALUATION COMPLETED")
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
        print(f"  - trained_models/error_{dataset_name.replace(' ', '_').replace('+', 'plus').lower()}.pth")
        print(f"  - experiment_results/training_curves_{dataset_name.replace(' ', '_').replace('+', 'plus').lower()}.png")
        print(f"  - experiment_results/signal_characteristics_{dataset_name.replace(' ', '_').replace('+', 'plus').lower()}.png")
        print(f"  - experiment_results/cnn_p*_peak_detection_results_{dataset_name.replace(' ', '_').replace('+', 'plus').lower()}.png")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
