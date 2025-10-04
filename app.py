from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Charger le modèle
try:
    # Essayer avec joblib (recommandé pour scikit-learn)
    model = joblib.load('linear_regression_model.pkl')
    print("Modèle chargé avec joblib")
except Exception as e1:
    print(f"Échec avec joblib: {e1}")
    try:
        # Essayer avec pickle et encoding latin1
        with open('linear_regression_model.pkl', 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        print("Modèle chargé avec pickle (encoding latin1)")
    except Exception as e2:
        print(f"Échec avec pickle latin1: {e2}")
        try:
            # Essayer avec pickle standard
            with open('linear_regression_model.pkl', 'rb') as f:
                model = pickle.load(f)
            print("Modèle chargé avec pickle standard")
        except Exception as e3:
            print(f"Échec avec pickle standard: {e3}")
            raise Exception("Impossible de charger le modèle. Veuillez le régénérer.")

# Les noms des colonnes doivent correspondre exactement au modèle
# Le modèle utilise un encodeur catégoriel, donc on garde les valeurs en string

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Créer un DataFrame avec les noms de colonnes exacts attendus par le modèle
        data = pd.DataFrame([{
            'Hours_Studied': float(request.form['hours_studied']),
            'Attendance': float(request.form['attendance']),
            'Sleep_Hours': float(request.form['sleep_hours']),
            'Previous_Scores': float(request.form['previous_scores']),
            'Tutoring_Sessions': int(request.form['tutoring_sessions']),
            'Physical_Activity': float(request.form['physical_activity']),
            'Parental_Involvement': request.form['parental_involvement'],
            'Access_to_Resources': request.form['access_resources'],
            'Extracurricular_Activities': request.form['extracurricular'],
            'Motivation_Level': request.form['motivation_level'],
            'Internet_Access': request.form['internet_access'],
            'Family_Income': request.form['family_income'],
            'Teacher_Quality': request.form['teacher_quality'],
            'School_Type': request.form['school_type'],
            'Peer_Influence': request.form['peer_influence'],
            'Learning_Disabilities': request.form['learning_disabilities'],
            'Parental_Education_Level': request.form['parental_education'],
            'Distance_from_Home': request.form['distance_home'],
            'Gender': request.form['gender']
        }])

        # Prédiction
        prediction = model.predict(data)[0]
        predicted_score = max(0, min(100, round(prediction)))

        return render_template('result.html', score=predicted_score)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Erreur détaillée:\n{error_details}")
        return f"""<html>
        <head><title>Erreur</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h2 style="color: red;">❌ Erreur lors de la prédiction</h2>
            <p><strong>Message:</strong> {str(e)}</p>
            <pre style="background: #f5f5f5; padding: 10px; border-radius: 5px;">{error_details}</pre>
            <a href="/form" style="display: inline-block; margin-top: 20px; padding: 10px 20px; background: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">← Retour au formulaire</a>
        </body>
        </html>""", 500

if __name__ == '__main__':
    app.run(debug=True)