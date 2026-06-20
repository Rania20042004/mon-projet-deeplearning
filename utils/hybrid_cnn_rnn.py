# hybrid_cnn_rnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridCNNRNN(nn.Module):
    """
    Modèle hybride CNN + RNN pour la classification d'images.
    Combine l'extraction de caractéristiques spatiales (CNN) 
    avec la modélisation séquentielle (RNN).
    """
    
    def __init__(self, num_classes=10, cnn_channels=[32, 64, 128], 
                 rnn_hidden_size=128, rnn_layers=2, dropout=0.5):
        super(HybridCNNRNN, self).__init__()
        
        # === PARTIE CNN : Extraction des caractéristiques spatiales ===
        self.cnn_layers = nn.ModuleList()
        
        # Bloc 1
        self.cnn_layers.append(nn.Sequential(
            nn.Conv2d(3, cnn_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ))
        
        # Bloc 2
        self.cnn_layers.append(nn.Sequential(
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ))
        
        # Bloc 3
        self.cnn_layers.append(nn.Sequential(
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ))
        
        # === PARTIE RNN : Modélisation séquentielle ===
        # Après les convolutions, la taille de l'image est réduite
        # Pour CIFAR-10 (32x32) après 3 maxpool (x2) : 32/8 = 4x4
        # Donc les features sont de taille [batch, 128, 4, 4]
        # On va les transformer en séquence de 16 timesteps de 128 features
        
        self.feature_size = cnn_channels[-1] * 4 * 4  # 128 * 16 = 2048
        self.rnn_hidden_size = rnn_hidden_size
        
        # Projection des features CNN vers l'entrée RNN
        self.feature_projection = nn.Linear(self.feature_size, rnn_hidden_size)
        
        # LSTM pour la partie séquentielle
        self.rnn = nn.LSTM(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0
        )
        
        # === PARTIE CLASSIFICATION ===
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size, rnn_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden_size // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # === 1. Extraction CNN ===
        for layer in self.cnn_layers:
            x = layer(x)
        
        # x: [batch, 128, 4, 4]
        
        # === 2. Transformation en séquence ===
        # Aplatir les dimensions spatiales
        x = x.view(batch_size, -1)  # [batch, 128*4*4]
        
        # Projection vers l'espace RNN
        x = self.feature_projection(x)  # [batch, rnn_hidden_size]
        
        # Ajouter la dimension de séquence (1 timestep)
        x = x.unsqueeze(1)  # [batch, 1, rnn_hidden_size]
        
        # === 3. Modélisation RNN ===
        rnn_output, (hidden, cell) = self.rnn(x)
        
        # Prendre le dernier état caché
        hidden = hidden[-1]  # [batch, rnn_hidden_size]
        
        # === 4. Classification ===
        output = self.classifier(hidden)
        
        return output

    def get_cnn_features(self, x):
        """Extrait les caractéristiques CNN sans la partie RNN"""
        for layer in self.cnn_layers:
            x = layer(x)
        return x.view(x.size(0), -1)


class HybridCNNGRU(HybridCNNRNN):
    """Version avec GRU au lieu de LSTM"""
    
    def __init__(self, num_classes=10, cnn_channels=[32, 64, 128], 
                 rnn_hidden_size=128, rnn_layers=2, dropout=0.5):
        super(HybridCNNRNN, self).__init__()
        
        # === PARTIE CNN ===
        self.cnn_layers = nn.ModuleList()
        for i, channels in enumerate(cnn_channels):
            in_channels = 3 if i == 0 else cnn_channels[i-1]
            self.cnn_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            ))
        
        self.feature_size = cnn_channels[-1] * 4 * 4
        self.rnn_hidden_size = rnn_hidden_size
        
        self.feature_projection = nn.Linear(self.feature_size, rnn_hidden_size)
        
        # GRU au lieu de LSTM
        self.rnn = nn.GRU(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size, rnn_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN
        for layer in self.cnn_layers:
            x = layer(x)
        
        # Transformation
        x = x.view(batch_size, -1)
        x = self.feature_projection(x)
        x = x.unsqueeze(1)
        
        # GRU
        rnn_output, hidden = self.rnn(x)
        hidden = hidden[-1]
        
        # Classification
        output = self.classifier(hidden)
        return output