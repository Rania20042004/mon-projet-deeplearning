# agent_optimizer.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import random
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
import json
import time
import os

class HyperparameterAgent:
    """
    Agent d'optimisation des hyperparamètres pour les modèles de deep learning.
    Supporte Grid Search et Recherche Aléatoire.
    """
    
    def __init__(self, model_class, param_grid, device='cpu', n_trials=20):
        """
        Args:
            model_class: Classe du modèle à optimiser
            param_grid: Dictionnaire des hyperparamètres à tester
            device: 'cpu' ou 'cuda'
            n_trials: Nombre de combinaisons à tester (random search)
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.device = device
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = -float('inf')
        self.results = []
        
    def generate_combinations(self, method='random'):
        """Génère les combinaisons d'hyperparamètres"""
        if method == 'grid':
            return list(ParameterGrid(self.param_grid))
        else:  # random
            combinations = []
            param_keys = list(self.param_grid.keys())
            for _ in range(self.n_trials):
                combo = {}
                for key in param_keys:
                    values = self.param_grid[key]
                    if isinstance(values, list):
                        combo[key] = random.choice(values)
                    elif isinstance(values, range):
                        combo[key] = random.choice(list(values))
                    else:
                        combo[key] = values
                combinations.append(combo)
            return combinations
    
    def evaluate_model(self, params, train_loader, val_loader, epochs=10):
        """Évalue une combinaison d'hyperparamètres"""
        try:
            # Créer le modèle
            model = self.model_class(**params).to(self.device)
            
            # Optimizer
            lr = params.get('lr', 0.001)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            # Entraînement rapide
            model.train()
            for epoch in range(epochs):
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            # Évaluation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            accuracy = accuracy_score(all_labels, all_preds)
            return accuracy
            
        except Exception as e:
            print(f"Erreur avec les paramètres {params}: {e}")
            return 0.0
    
    def optimize(self, train_loader, val_loader, method='random', epochs=10, verbose=True):
        """
        Lance l'optimisation des hyperparamètres.
        
        Returns:
            dict: Meilleurs paramètres trouvés
        """
        combinations = self.generate_combinations(method)
        
        if verbose:
            print(f"🔍 Optimisation {method} sur {len(combinations)} combinaisons")
            print("="*60)
        
        for i, params in enumerate(combinations):
            start_time = time.time()
            
            # Évaluer cette combinaison
            score = self.evaluate_model(params, train_loader, val_loader, epochs)
            
            # Enregistrer les résultats
            result = {
                'params': params,
                'score': score,
                'time': time.time() - start_time
            }
            self.results.append(result)
            
            # Mettre à jour le meilleur
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                
            if verbose:
                print(f"[{i+1}/{len(combinations)}] Score: {score:.4f} | Meilleur: {self.best_score:.4f}")
                print(f"  {params}")
        
        if verbose:
            print("="*60)
            print(f"✅ Meilleurs paramètres: {self.best_params}")
            print(f"✅ Meilleur score: {self.best_score:.4f}")
        
        return self.best_params
    
    def save_results(self, filepath='hyperparameter_results.json'):
        """Sauvegarde les résultats de l'optimisation"""
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
            'param_grid': self.param_grid
        }
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Résultats sauvegardés dans {filepath}")
    
    def plot_results(self):
        """Visualise les résultats de l'optimisation"""
        import matplotlib.pyplot as plt
        
        if not self.results:
            print("Aucun résultat à visualiser")
            return
        
        scores = [r['score'] for r in self.results]
        
        plt.figure(figsize=(12, 5))
        
        # 1. Courbe d'évolution
        plt.subplot(1, 2, 1)
        plt.plot(scores, 'b-', alpha=0.7)
        plt.axhline(y=self.best_score, color='r', linestyle='--', 
                    label=f'Meilleur: {self.best_score:.4f}')
        plt.xlabel('Essai')
        plt.ylabel('Accuracy')
        plt.title('Évolution de la performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Distribution
        plt.subplot(1, 2, 2)
        plt.hist(scores, bins=10, alpha=0.7, color='green')
        plt.axvline(x=self.best_score, color='r', linestyle='--',
                    label=f'Meilleur: {self.best_score:.4f}')
        plt.xlabel('Accuracy')
        plt.ylabel('Fréquence')
        plt.title('Distribution des performances')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('static/hyperparameter_optimization.png', dpi=150)
        plt.show()


class LearningRateAgent:
    """Agent spécialisé pour trouver le meilleur learning rate"""
    
    @staticmethod
    def find_lr(model, train_loader, criterion, device='cpu', 
                start_lr=1e-7, end_lr=10, num_iters=100):
        """
        Trouve un bon learning rate en utilisant la méthode de Leslie Smith.
        """
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=start_lr)
        
        lrs = []
        losses = []
        
        # Multiplicateur géométrique
        multiplier = (end_lr / start_lr) ** (1 / num_iters)
        
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            if i >= num_iters:
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Mettre à jour le LR
            current_lr = start_lr * (multiplier ** i)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            lrs.append(current_lr)
            losses.append(loss.item())
        
        # Trouver le LR optimal (où la perte diminue le plus)
        import numpy as np
        losses = np.array(losses)
        lrs = np.array(lrs)
        
        # Lissage
        from scipy.signal import savgol_filter
        if len(losses) > 20:
            losses_smooth = savgol_filter(losses, min(11, len(losses)//2*2+1), 3)
        else:
            losses_smooth = losses
        
        # Trouver le LR où la perte est minimale
        best_idx = np.argmin(losses_smooth)
        best_lr = lrs[best_idx]
        
        return best_lr, lrs, losses_smooth