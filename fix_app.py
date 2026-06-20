#!/usr/bin/env python3
"""Script pour corriger les problèmes de l'application"""
import torch
import os

# Lire app.py
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remplacer la section de chargement du MLP par une version qui gère les architectures dynamiques
old_mlp_loading = """# Chemins CORRIGES pour trouver les modèles MLP
mlp_paths = [
    'notebooks/models/mlp_improved.pth',
    'models/best_mlp_model.pth',
    'models/mlp_model.pth',
    'web_app/models/best_mlp_model.pth'
]

for path in mlp_paths:
    if os.path.exists(path):
        try:
            checkpoint = torch.load(path, map_location=device)
            # Déterminer l'input_dim à partir du state dict
            state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            first_layer_weight = state_dict.get('network.0.weight', None) if isinstance(state_dict, dict) else None
            if first_layer_weight is not None:
                input_dim = first_layer_weight.shape[1]
                print(f"ℹ️ MLP: détecté {input_dim} features d'entrée")
            else:
                input_dim = 30  # default
            
            mlp_model = MLPModel(input_dim=input_dim, output_dim=2).to(device)
            if 'model_state_dict' in checkpoint:
                mlp_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                mlp_model.load_state_dict(checkpoint['state_dict'])
            else:
                mlp_model.load_state_dict(checkpoint)
            mlp_model.eval()
            
            if isinstance(checkpoint, dict) and 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
                mlp_scaler_mean = np.array(checkpoint['scaler_mean'])
                mlp_scaler_scale = np.array(checkpoint['scaler_scale'])
            else:
                data = load_breast_cancer()
                scaler = StandardScaler()
                scaler.fit(data.data)
                mlp_scaler_mean = scaler.mean_
                mlp_scaler_scale = scaler.scale_
                print("ℹ️ MLP: utilisation du scaling calculé sur le dataset Breast Cancer")
            
            print(f"✅ MLP chargé depuis {path}")
            break
        except Exception as e:
            print(f"⚠️ Erreur MLP {path}: {e}")
            import traceback
            traceback.print_exc()

if mlp_model is None:
    print("⚠️ MLP non trouvé - création d'un modèle par défaut pour 30 features")
    mlp_model = MLPModel(input_dim=30, output_dim=2).to(device)
    mlp_model.eval()"""

new_mlp_loading = """# Chemins CORRIGES pour trouver les modèles MLP
mlp_paths = [
    'web_app/models/best_mlp_model.pth',
    'models/best_mlp_model.pth',
    'models/mlp_model.pth',
    'notebooks/models/mlp_improved.pth'
]

for path in mlp_paths:
    if os.path.exists(path):
        try:
            checkpoint = torch.load(path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            
            # Extraire les dimensions depuis le state dict
            linear_keys = sorted([(int(k.split('.')[1]), k) for k in state_dict.keys() if 'network.' in k and '.weight' in k])
            if not linear_keys:
                raise ValueError("No linear layers found in checkpoint")
            
            # Construire architecture: input_dim, hidden_dims, output_dim
            dims = []
            for idx, (_, key) in enumerate(linear_keys):
                w = state_dict[key]
                out_dim, in_dim = w.shape
                if idx == 0:
                    dims.append(in_dim)
                dims.append(out_dim)
            
            input_dim = dims[0]
            output_dim = dims[-1]
            hidden_dims = dims[1:-1] if len(dims) > 2 else [max(64, input_dim // 4)]
            
            print(f"  MLP: input={input_dim}, hidden={hidden_dims}, output={output_dim}")
            
            mlp_model = MLPModel(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim).to(device)
            mlp_model.load_state_dict(state_dict if not isinstance(checkpoint, dict) else checkpoint['model_state_dict'])
            mlp_model.eval()
            
            # Charger ou calculer le scaler
            if isinstance(checkpoint, dict) and 'scaler_mean' in checkpoint:
                mlp_scaler_mean = np.array(checkpoint['scaler_mean'])
                mlp_scaler_scale = np.array(checkpoint['scaler_scale'])
                print("  Scaler: chargé du checkpoint")
            elif input_dim == 30:
                data = load_breast_cancer()
                scaler = StandardScaler()
                scaler.fit(data.data)
                mlp_scaler_mean = scaler.mean_
                mlp_scaler_scale = scaler.scale_
                print("  Scaler: calculé sur Breast Cancer dataset")
            else:
                mlp_scaler_mean = np.zeros(input_dim)
                mlp_scaler_scale = np.ones(input_dim)
                print("  Scaler: par défaut (pas de normalisation)")
            
            print(f"✅ MLP chargé depuis {path}")
            break
        except Exception as e:
            print(f"⚠️ Erreur MLP {path}: {str(e)[:100]}")

if mlp_model is None:
    print("⚠️ MLP non trouvé - modèle par défaut")
    mlp_model = MLPModel(input_dim=30, hidden_dims=[64, 32], output_dim=2).to(device)
    mlp_model.eval()"""

content = content.replace(old_mlp_loading, new_mlp_loading)

# Écrire app.py corrigé
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ app.py corrigé")
