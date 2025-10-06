from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Charger le modèle
model = joblib.load('linear_regression_model.pkl')
print("✅ Modèle chargé avec succès")

# Mappings pour convertir les chaînes en entiers (correspond aux valeurs envoyées par le formulaire)
CATEGORY_MAPPINGS = {
    'Parental_Involvement': {'Faible': 0, 'Moyenne': 1, 'Élevée': 2},
    'Parental_Education_Level': {'Lycée': 0, 'Université': 1, 'Études supérieures': 2},
    'Family_Income': {'Faible': 0, 'Moyen': 1, 'Élevé': 2},
    'Access_to_Resources': {'Faible': 0, 'Moyen': 1, 'Élevé': 2},
    'Internet_Access': {'Non': 0, 'Oui': 1},
    'Teacher_Quality': {'Faible': 0, 'Moyenne': 1, 'Élevée': 2},
    'Motivation_Level': {'Faible': 0, 'Moyenne': 1, 'Élevée': 2},
    'Learning_Disabilities': {'Non': 0, 'Oui': 1},
    'School_Type': {'Publique': 0, 'Privée': 1},
    'Peer_Influence': {'Négative': 0, 'Neutre': 1, 'Positive': 2},
    'Extracurricular_Activities': {'Non': 0, 'Oui': 1},
    'Gender': {'Masculin': 0, 'Féminin': 1},
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        form_data = request.form

        # Construire les données avec les **noms de colonnes exacts** du modèle (en anglais)
        data_dict = {
            'Hours_Studied': float(form_data['hours_studied']),
            'Attendance': float(form_data['attendance']),
            'Previous_Scores': float(form_data['previous_scores']),
            'Sleep_Hours': float(form_data['sleep_hours']),
            'Tutoring_Sessions': int(form_data['tutoring_sessions']),
            'Physical_Activity': float(form_data['physical_activity']),
            'Distance_from_Home': float(form_data['distance_home']),
            'Age': int(form_data.get('age', 18)),
            'Parental_Involvement': CATEGORY_MAPPINGS['Parental_Involvement'][form_data['parental_involvement']],
            'Parental_Education_Level': CATEGORY_MAPPINGS['Parental_Education_Level'][form_data['parental_education']],
            'Family_Income': CATEGORY_MAPPINGS['Family_Income'][form_data['family_income']],
            'Access_to_Resources': CATEGORY_MAPPINGS['Access_to_Resources'][form_data['access_resources']],
            'Internet_Access': CATEGORY_MAPPINGS['Internet_Access'][form_data['internet_access']],
            'Teacher_Quality': CATEGORY_MAPPINGS['Teacher_Quality'][form_data['teacher_quality']],
            'Motivation_Level': CATEGORY_MAPPINGS['Motivation_Level'][form_data['motivation_level']],
            'Learning_Disabilities': CATEGORY_MAPPINGS['Learning_Disabilities'][form_data['learning_disabilities']],
            'School_Type': CATEGORY_MAPPINGS['School_Type'][form_data['school_type']],
            'Peer_Influence': CATEGORY_MAPPINGS['Peer_Influence'][form_data['peer_influence']],
            'Extracurricular_Activities': CATEGORY_MAPPINGS['Extracurricular_Activities'][form_data['extracurricular']],
            'Gender': CATEGORY_MAPPINGS['Gender'][form_data['gender']]
        }

        df = pd.DataFrame([data_dict])
        prediction = model.predict(df)[0]
        predicted_score = max(0, min(100, round(prediction)))

        return render_template('result.html', score=predicted_score)

    except Exception as e:
        import traceback
        print("❌ Erreur :", str(e))
        print(traceback.format_exc())
        return f"""
        <html>
        <head><title>Erreur</title></head>
        <body style="font-family: Inter, sans-serif; padding: 40px; background: #fef2f2;">
            <h2 style="color: #ef4444;">❌ Erreur lors de la prédiction</h2>
            <p><strong>Message :</strong> {str(e)}</p>
            <pre>{traceback.format_exc()}</pre>
            <a href="/form" style="display: inline-block; margin-top: 20px; padding: 12px 24px; 
               background: #3b82f6; color: white; text-decoration: none; border-radius: 8px;">
                ← Retour au formulaire
            </a>
        </body>
        </html>
        """, 500

if __name__ == '__main__':
    app.run(debug=True)
