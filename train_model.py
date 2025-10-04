"""
Script pour entraîner et sauvegarder le modèle de régression linéaire
Utilisez ce script si le fichier .pkl est corrompu ou incompatible
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import joblib

# Générer des données d'entraînement synthétiques
# (Remplacez ceci par vos vraies données si disponibles)
np.random.seed(42)
n_samples = 1000

# Créer un dataset synthétique
data = {
    'hours_studied': np.random.uniform(0, 10, n_samples),
    'attendance': np.random.uniform(50, 100, n_samples),
    'previous_scores': np.random.uniform(40, 100, n_samples),
    'sleep_hours': np.random.uniform(4, 10, n_samples),
    'tutoring_sessions': np.random.randint(0, 8, n_samples),
    'physical_activity': np.random.uniform(0, 5, n_samples),
    'distance_home': np.random.uniform(0, 50, n_samples),
    'age': np.random.randint(15, 25, n_samples),
    'parental_involvement': np.random.randint(0, 3, n_samples),  # Low=0, Medium=1, High=2
    'parental_education': np.random.randint(0, 3, n_samples),    # High School=0, College=1, Postgraduate=2
    'family_income': np.random.randint(0, 3, n_samples),         # Low=0, Medium=1, High=2
    'access_resources': np.random.randint(0, 3, n_samples),      # Low=0, Medium=1, High=2
    'internet_access': np.random.randint(0, 2, n_samples),       # No=0, Yes=1
    'teacher_quality': np.random.randint(0, 3, n_samples),       # Low=0, Medium=1, High=2
    'motivation_level': np.random.randint(0, 3, n_samples),      # Low=0, Medium=1, High=2
    'learning_disabilities': np.random.randint(0, 2, n_samples), # No=0, Yes=1
    'school_type': np.random.randint(0, 2, n_samples),           # Public=0, Private=1
    'peer_influence': np.random.randint(0, 3, n_samples),        # Negative=0, Neutral=1, Positive=2
    'extracurricular': np.random.randint(0, 2, n_samples),       # No=0, Yes=1
    'gender': np.random.randint(0, 2, n_samples)                 # Male=0, Female=1
}

df = pd.DataFrame(data)

# Créer la variable cible (score de réussite) basée sur les features
# Formule simplifiée pour générer des scores réalistes
df['exam_score'] = (
    df['hours_studied'] * 3.5 +
    df['attendance'] * 0.3 +
    df['previous_scores'] * 0.4 +
    df['sleep_hours'] * 1.5 +
    df['tutoring_sessions'] * 2 +
    df['physical_activity'] * 1 +
    df['parental_involvement'] * 3 +
    df['parental_education'] * 2 +
    df['family_income'] * 1.5 +
    df['access_resources'] * 2 +
    df['internet_access'] * 3 +
    df['teacher_quality'] * 2.5 +
    df['motivation_level'] * 4 +
    df['peer_influence'] * 2 -
    df['learning_disabilities'] * 5 +
    df['school_type'] * 2 +
    df['extracurricular'] * 1.5 -
    df['distance_home'] * 0.1 +
    np.random.normal(0, 5, n_samples)  # Ajout de bruit
)

# Normaliser les scores entre 0 et 100
df['exam_score'] = df['exam_score'].clip(0, 100)

# Séparer features et target
feature_order = [
    'hours_studied', 'attendance', 'previous_scores', 'sleep_hours',
    'tutoring_sessions', 'physical_activity', 'distance_home', 'age',
    'parental_involvement', 'parental_education', 'family_income',
    'access_resources', 'internet_access', 'teacher_quality',
    'motivation_level', 'learning_disabilities', 'school_type',
    'peer_influence', 'extracurricular', 'gender'
]

X = df[feature_order]
y = df['exam_score']

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
print("Entraînement du modèle...")
model = LinearRegression()
model.fit(X_train, y_train)

# Évaluer le modèle
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Score d'entraînement (R²): {train_score:.4f}")
print(f"Score de test (R²): {test_score:.4f}")

# Sauvegarder le modèle avec joblib (recommandé pour scikit-learn)
print("\nSauvegarde du modèle...")
joblib.dump(model, 'linear_regression_model.pkl')
print("✓ Modèle sauvegardé avec joblib dans 'linear_regression_model.pkl'")

# Sauvegarder aussi avec pickle pour compatibilité
with open('linear_regression_model_pickle.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
print("✓ Modèle sauvegardé avec pickle dans 'linear_regression_model_pickle.pkl'")

print("\n✅ Modèle créé avec succès!")
print("\nVous pouvez maintenant lancer l'application Flask avec: python app.py")
