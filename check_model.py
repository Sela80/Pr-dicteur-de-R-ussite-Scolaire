"""
Script pour vérifier les features attendues par le modèle
"""
import joblib
import numpy as np

# Charger le modèle
model = joblib.load('linear_regression_model.pkl')

print("=== Informations sur le modèle ===\n")
print(f"Type de modèle: {type(model)}")

# Vérifier si c'est un pipeline
if hasattr(model, 'named_steps'):
    print("\nC'est un Pipeline avec les étapes suivantes:")
    for name, step in model.named_steps.items():
        print(f"  - {name}: {type(step)}")
    
    # Vérifier le nombre de features attendues
    if hasattr(model, 'n_features_in_'):
        print(f"\nNombre de features attendues: {model.n_features_in_}")
    
    # Si c'est un ColumnTransformer, afficher les colonnes
    if 'columntransformer' in model.named_steps or 'preprocessor' in model.named_steps:
        transformer_name = 'columntransformer' if 'columntransformer' in model.named_steps else 'preprocessor'
        transformer = model.named_steps[transformer_name]
        print(f"\n=== Configuration du {transformer_name} ===")
        print(f"Nombre de features attendues: {transformer.n_features_in_}")
        
        # Afficher les transformers et leurs colonnes
        if hasattr(transformer, 'transformers_'):
            for name, trans, cols in transformer.transformers_:
                print(f"\nTransformer '{name}':")
                print(f"  Type: {type(trans)}")
                print(f"  Colonnes: {cols}")

elif hasattr(model, 'n_features_in_'):
    print(f"\nNombre de features attendues: {model.n_features_in_}")
    if hasattr(model, 'feature_names_in_'):
        print(f"Noms des features: {model.feature_names_in_}")
else:
    print("\nImpossible de déterminer le nombre de features")

print("\n" + "="*50)
print("\nVos 20 features actuelles:")
features = [
    'hours_studied', 'attendance', 'previous_scores', 'sleep_hours',
    'tutoring_sessions', 'physical_activity', 'distance_home', 'age',
    'parental_involvement', 'parental_education', 'family_income',
    'access_resources', 'internet_access', 'teacher_quality',
    'motivation_level', 'learning_disabilities', 'school_type',
    'peer_influence', 'extracurricular', 'gender'
]

for i, feat in enumerate(features, 1):
    print(f"{i:2d}. {feat}")

print(f"\nTotal: {len(features)} features")
