# train_hybrid.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import os
import sys

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importer depuis utils
from utils.hybrid_cnn_rnn import HybridCNNRNN

def train_hybrid():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Créer le dossier models
    os.makedirs('models', exist_ok=True)
    
    # Charger CIFAR-10
    print("📦 Chargement de CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    full_test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Sous-ensembles (entraînement rapide)
    train_indices = list(range(2000))
    val_indices = list(range(2000, 2500))
    test_indices = list(range(1000))
    
    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_train, val_indices)
    test_dataset = Subset(full_test, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Modèle hybride
    model = HybridCNNRNN(num_classes=10).to(device)
    print(f"📊 Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Entraînement
    epochs = 20
    print(f"\n🚀 Entraînement sur {epochs} époques...")
    print("="*50)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_loss = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
        
        train_acc = train_correct / len(train_dataset)
        val_acc = val_correct / len(val_dataset)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/hybrid_cnn_rnn_best.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    # Sauvegarde finale
    torch.save(model.state_dict(), 'models/hybrid_cnn_rnn.pth')
    print(f"\n✅ Modèle hybride sauvegardé dans 'models/hybrid_cnn_rnn.pth'")
    print(f"✅ Meilleure validation accuracy: {best_val_acc:.4f}")
    
    return model

if __name__ == '__main__':
    train_hybrid()