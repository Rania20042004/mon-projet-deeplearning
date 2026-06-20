# utils/data_utils.py
"""
Data loading and preprocessing utilities for all three deep learning models.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import unicodedata
import re
import os


def load_breast_cancer_data(test_size=0.15, val_size=0.15, random_state=42):
    """
    Load and preprocess Breast Cancer Wisconsin dataset.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, scaler, feature_names, target_names)
    """
    # Chargement des données
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print(f"Breast Cancer dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {data.target_names}")
    print(f"Distribution: {np.bincount(y)}")
    
    # Split en train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, random_state=random_state, stratify=y_temp
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Conversion en tenseurs
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Création des DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return (train_loader, val_loader, test_loader, scaler, 
            data.feature_names, data.target_names)


def load_cifar10_data(batch_size=64, augment=True):
    """
    Load and preprocess CIFAR-10 dataset.
    
    Args:
        batch_size: Batch size for DataLoaders
        augment: Whether to use data augmentation
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, classes)
    """
    # Transformations
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Téléchargement
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"CIFAR-10 loaded: Train={len(train_dataset)}, Val={val_size}, Test={len(test_dataset)}")
    print(f"Classes: {classes}")
    
    return train_loader, val_loader, test_loader, classes


def unicode_to_ascii(s):
    """Convert Unicode string to ASCII."""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    """Normalize string: lowercase, trim, remove punctuation."""
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


class Lang:
    """Language vocabulary class for Seq2Seq."""
    
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1, "UNK": 2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3
    
    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def prepare_tatoeba_data(file_path, max_pairs=10000, max_length=20, reverse=False):
    """
    Prepare Tatoeba dataset for translation.
    
    Args:
        file_path: Path to fra.txt file
        max_pairs: Maximum number of sentence pairs to use
        max_length: Maximum sentence length
        reverse: Reverse direction (eng->fra if True)
    
    Returns:
        tuple: (input_lang, output_lang, pairs)
    """
    print(f"Loading Tatoeba data from {file_path}...")
    
    lines = open(file_path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')[:2]] for l in lines]
    
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang('eng')
        output_lang = Lang('fra')
    else:
        input_lang = Lang('fra')
        output_lang = Lang('eng')
    
    # Filter by length
    def filter_pair(p):
        return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length
    
    pairs = [pair for pair in pairs if filter_pair(pair)]
    pairs = pairs[:max_pairs]
    
    # Build vocabularies
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    
    print(f"Input language ({input_lang.name}): {input_lang.n_words} words")
    print(f"Output language ({output_lang.name}): {output_lang.n_words} words")
    print(f"Number of sentence pairs: {len(pairs)}")
    
    return input_lang, output_lang, pairs


def create_dataloaders(dataset, batch_size=32, shuffle=True):
    """Create DataLoader from PyTorch dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
# utils/data_utils.py (ajouter cette fonction à la fin du fichier)
# ... gardez le code existant et ajoutez :

def normalize_features(X, method='standard'):
    """
    Normalize features using StandardScaler or MinMaxScaler.
    
    Args:
        X: array-like, features to normalize
        method: 'standard' or 'minmax'
    
    Returns:
        tuple: (normalized_features, scaler)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_normalized = scaler.fit_transform(X)
    return X_normalized, scaler