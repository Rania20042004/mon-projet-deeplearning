# utils/__init__.py
"""
Utils package for Deep Learning Project
EMSI Casablanca - 2025/2026
"""

from .hybrid_cnn_rnn import HybridCNNRNN, HybridCNNGRU
from .agent_optimizer import HyperparameterAgent, LearningRateAgent

# Imports conditionnels pour les fonctions de data_utils
try:
    from .data_utils import (
        load_breast_cancer_data,
        load_cifar10_data,
        prepare_tatoeba_data,
        normalize_features,
        create_dataloaders,
        Lang
    )
except ImportError:
    print("⚠️ data_utils.py incomplet - certaines fonctions non disponibles")

try:
    from .training_utils import (
        train_epoch,
        evaluate,
        train_model,
        EarlyStopping,
        plot_training_history,
        save_model,
        load_model
    )
except ImportError:
    print("⚠️ training_utils.py incomplet - certaines fonctions non disponibles")

__all__ = [
    # Modèles
    'HybridCNNRNN',
    'HybridCNNGRU',
    # Agents
    'HyperparameterAgent',
    'LearningRateAgent',
    # Data utils
    'load_breast_cancer_data',
    'load_cifar10_data',
    'prepare_tatoeba_data',
    'normalize_features',
    'create_dataloaders',
    'Lang',
    # Training utils
    'train_epoch',
    'evaluate',
    'train_model',
    'EarlyStopping',
    'plot_training_history',
    'save_model',
    'load_model'
]