#!/usr/bin/env python3
"""
Temporal Gesture Recognition - LSTM/GRU/TCN models for dynamic gestures
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import pickle
import json
import os
from collections import deque

@dataclass
class TemporalGestureSequence:
    """A sequence of gesture features for temporal modeling."""
    features: List[List[float]]  # Sequence of feature vectors
    label: str                   # Gesture label
    timestamps: List[float]      # Timestamps for each frame
    sequence_length: int         # Length of the sequence
    hand_type: str = "Right"     # Hand type

class GestureSequenceDataset(Dataset):
    """PyTorch Dataset for gesture sequences."""
    
    def __init__(self, sequences: List[TemporalGestureSequence], 
                 max_sequence_length: int = 50):
        self.sequences = sequences
        self.max_length = max_sequence_length
        
        # Build label vocabulary
        self.labels = list(set(seq.label for seq in sequences))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Determine feature dimension
        if sequences:
            self.feature_dim = len(sequences[0].features[0])
        else:
            self.feature_dim = 0

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Pad or truncate sequence
        features = sequence.features[:self.max_length]
        
        # Pad with zeros if needed
        while len(features) < self.max_length:
            features.append([0.0] * self.feature_dim)
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.LongTensor([self.label_to_idx[sequence.label]])
        length_tensor = torch.LongTensor([min(sequence.sequence_length, self.max_length)])
        
        return features_tensor, label_tensor, length_tensor

class LSTMGestureClassifier(nn.Module):
    """LSTM-based temporal gesture classifier."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, num_classes: int = 10, 
                 dropout: float = 0.3, bidirectional: bool = True):
        super(LSTMGestureClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension adjustment for bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

    def forward(self, x, lengths=None):
        batch_size, seq_len, _ = x.size()
        
        # LSTM forward pass
        if lengths is not None:
            # Pack padded sequences for efficiency
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_x)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            lstm_output, (hidden, cell) = self.lstm(x)
        
        # Layer normalization
        lstm_output = self.layer_norm(lstm_output)
        
        # Self-attention
        attended_output, attention_weights = self.attention(
            lstm_output, lstm_output, lstm_output
        )
        
        # Global average pooling over sequence dimension
        if lengths is not None:
            # Mask out padded positions
            mask = torch.arange(seq_len).expand(batch_size, seq_len) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float().to(x.device)
            attended_output = attended_output * mask
            pooled_output = attended_output.sum(dim=1) / lengths.unsqueeze(1).float()
        else:
            pooled_output = attended_output.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits, attention_weights

class GRUGestureClassifier(nn.Module):
    """GRU-based temporal gesture classifier."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, num_classes: int = 10, 
                 dropout: float = 0.3, bidirectional: bool = True):
        super(GRUGestureClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension adjustment for bidirectional
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, lengths=None):
        # GRU forward pass
        if lengths is not None:
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, hidden = self.gru(packed_x)
            gru_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
        else:
            gru_output, hidden = self.gru(x)
        
        # Use final hidden state for classification
        if self.bidirectional:
            # Concatenate forward and backward final states
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]
        
        # Classification
        logits = self.classifier(final_hidden)
        
        return logits, None

class TCNBlock(nn.Module):
    """Temporal Convolutional Network block."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super(TCNBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=(kernel_size - 1) * dilation, dilation=dilation)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        
        # First convolution
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(residual)
        
        return self.relu(out + residual)

class TCNGestureClassifier(nn.Module):
    """Temporal Convolutional Network for gesture classification."""
    
    def __init__(self, input_dim: int, num_channels: List[int] = [64, 128, 256], 
                 kernel_size: int = 3, num_classes: int = 10, dropout: float = 0.2):
        super(TCNGestureClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, num_channels[0], 1)
        
        # TCN blocks
        self.tcn_blocks = nn.ModuleList()
        dilation = 1
        
        for i in range(len(num_channels)):
            in_channels = num_channels[i]
            out_channels = num_channels[i] if i == len(num_channels) - 1 else num_channels[i + 1]
            
            if i < len(num_channels) - 1:
                out_channels = num_channels[i + 1]
            else:
                out_channels = num_channels[i]
            
            self.tcn_blocks.append(
                TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )
            dilation *= 2
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, lengths=None):
        # Transpose for Conv1d (batch, features, sequence)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # TCN blocks
        for block in self.tcn_blocks:
            x = block(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits, None

class TemporalGestureRecognizer:
    """High-level interface for temporal gesture recognition."""
    
    def __init__(self, model_type: str = "lstm", device: str = "auto"):
        self.model_type = model_type
        self.device = self._get_device(device)
        self.model = None
        self.dataset = None
        self.is_trained = False
        
        # Training parameters
        self.max_sequence_length = 50
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.patience = 10
        
        # Feature buffer for real-time prediction
        self.feature_buffer = deque(maxlen=self.max_sequence_length)
        self.timestamp_buffer = deque(maxlen=self.max_sequence_length)

    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for training/inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)

    def prepare_training_data(self, sequences: List[TemporalGestureSequence]) -> GestureSequenceDataset:
        """Prepare training data."""
        self.dataset = GestureSequenceDataset(sequences, self.max_sequence_length)
        
        # Initialize model
        if self.model_type == "lstm":
            self.model = LSTMGestureClassifier(
                input_dim=self.dataset.feature_dim,
                num_classes=len(self.dataset.labels)
            )
        elif self.model_type == "gru":
            self.model = GRUGestureClassifier(
                input_dim=self.dataset.feature_dim,
                num_classes=len(self.dataset.labels)
            )
        elif self.model_type == "tcn":
            self.model = TCNGestureClassifier(
                input_dim=self.dataset.feature_dim,
                num_classes=len(self.dataset.labels)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.to(self.device)
        return self.dataset

    def train(self, sequences: List[TemporalGestureSequence], 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the temporal model."""
        
        # Prepare data
        dataset = self.prepare_training_data(sequences)
        
        # Split data
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                              shuffle=False, num_workers=0)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for features, labels, lengths in train_loader:
                features = features.to(self.device)
                labels = labels.squeeze().to(self.device)
                lengths = lengths.squeeze().to(self.device)
                
                optimizer.zero_grad()
                
                logits, _ = self.model(features, lengths)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for features, labels, lengths in val_loader:
                    features = features.to(self.device)
                    labels = labels.squeeze().to(self.device)
                    lengths = lengths.squeeze().to(self.device)
                    
                    logits, _ = self.model(features, lengths)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    
                    predicted = torch.argmax(logits, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_temporal_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_temporal_model.pth'))
        self.is_trained = True
        
        return {
            "model_type": self.model_type,
            "epochs_trained": epoch + 1,
            "best_val_loss": best_val_loss,
            "final_val_accuracy": val_accuracies[-1],
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "num_classes": len(dataset.labels),
            "classes": dataset.labels
        }

    def predict_realtime(self, features: List[float], timestamp: float = None) -> Tuple[str, float]:
        """Predict gesture from streaming features."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Add to buffer
        self.feature_buffer.append(features)
        if timestamp is not None:
            self.timestamp_buffer.append(timestamp)
        
        # Need minimum sequence length for prediction
        if len(self.feature_buffer) < 10:
            return "unknown", 0.0
        
        # Prepare sequence
        sequence = list(self.feature_buffer)
        while len(sequence) < self.max_sequence_length:
            sequence.append([0.0] * len(features))
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        length_tensor = torch.LongTensor([len(self.feature_buffer)]).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(features_tensor, length_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            class_name = self.dataset.idx_to_label[predicted_class]
            
            return class_name, confidence

    def predict_sequence(self, sequence: TemporalGestureSequence) -> Dict[str, float]:
        """Predict probabilities for a complete sequence."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare sequence
        features = sequence.features[:self.max_sequence_length]
        while len(features) < self.max_sequence_length:
            features.append([0.0] * len(sequence.features[0]))
        
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        length_tensor = torch.LongTensor([min(sequence.sequence_length, self.max_sequence_length)]).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            logits, attention_weights = self.model(features_tensor, length_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            # Create probability dictionary
            result = {}
            for i, class_name in enumerate(self.dataset.labels):
                result[class_name] = probabilities[0, i].item()
            
            return result

    def save_model(self, filepath: str):
        """Save trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "dataset_labels": self.dataset.labels,
            "dataset_label_to_idx": self.dataset.label_to_idx,
            "dataset_feature_dim": self.dataset.feature_dim,
            "max_sequence_length": self.max_sequence_length,
            "model_config": {
                "input_dim": self.dataset.feature_dim,
                "num_classes": len(self.dataset.labels)
            }
        }
        
        torch.save(model_data, filepath)

    def load_model(self, filepath: str):
        """Load trained model."""
        model_data = torch.load(filepath, map_location=self.device)
        
        # Recreate dataset info
        class MockDataset:
            def __init__(self, labels, label_to_idx, feature_dim):
                self.labels = labels
                self.label_to_idx = label_to_idx
                self.idx_to_label = {v: k for k, v in label_to_idx.items()}
                self.feature_dim = feature_dim
        
        self.dataset = MockDataset(
            model_data["dataset_labels"],
            model_data["dataset_label_to_idx"],
            model_data["dataset_feature_dim"]
        )
        
        self.model_type = model_data["model_type"]
        self.max_sequence_length = model_data["max_sequence_length"]
        
        # Recreate model
        config = model_data["model_config"]
        if self.model_type == "lstm":
            self.model = LSTMGestureClassifier(**config)
        elif self.model_type == "gru":
            self.model = GRUGestureClassifier(**config)
        elif self.model_type == "tcn":
            self.model = TCNGestureClassifier(**config)
        
        self.model.load_state_dict(model_data["model_state_dict"])
        self.model.to(self.device)
        self.is_trained = True

    def reset_buffer(self):
        """Reset the feature buffer for new gesture sequence."""
        self.feature_buffer.clear()
        self.timestamp_buffer.clear()
