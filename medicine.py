import streamlit as st
import numpy as np
import pandas as pd
import joblib
from fpdf import FPDF
import tempfile
from PIL import Image

# Load model and data
model_svc1=joblib.load('D:/Users/admin/pyt/medicne_predict.pkl')
desc_df = pd.read_csv("D:/Users/admin/pyt/archive (3)/description.csv")
prec_df = pd.read_csv("D:/Users/admin/pyt/archive (3)/precautions_df.csv")
diet_df = pd.read_csv("D:/Users/admin/pyt/archive (3)/diets.csv")
workout_df = pd.read_csv("D:/Users/admin/pyt/archive (3)/workout_df.csv")
med_df = pd.read_csv("D:/Users/admin/pyt/archive (3)/medications.csv")

# Create dictionaries
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
desc_dict = dict(zip(desc_df['Disease'], desc_df['Description']))
prec_dict = dict(zip(prec_df['Disease'], prec_df[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()))
med_dict = dict(zip(med_df['Disease'], med_df[['Medication']].values.tolist()))
diet_dict = dict(zip(diet_df['Disease'], diet_df[['Diet']].values.tolist()))
workout_dict = dict(zip(workout_df['disease'], workout_df[['workout']].values.tolist()))
# Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))  # Create a zero vector of the correct length
    for item in patient_symptoms:
        item = item.strip()  
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
        else:
            print(f"Warning: '{item}' not found in symptoms_dict.")  # Debugging line

    print("Input Vector:", input_vector)  # Debugging line
    input_vector = input_vector.reshape(1, -1)  # Reshape to 2D array (1, n_features)
    prediction_disease = model_svc1.predict(input_vector)[0]
    print("Predicted Disease:", prediction_disease)  # Debugging line
    return prediction_disease
def clean_symptoms(raw_input):
    symptoms = [s.strip().lower().replace(" ", "_") for s in raw_input.split(",")]
    valid_symptoms = [s for s in symptoms if s in symptoms_dict]
    invalid_symptoms = [s for s in symptoms if s not in symptoms_dict]
    return valid_symptoms, invalid_symptoms


# Helper to get all recommendations
def helper(disease):
    return (
        desc_dict.get(disease, "No description found."),
        prec_dict.get(disease, ["No precautions found."]),
        med_dict.get(disease, ["No medicine found."]),
        diet_dict.get(disease, ["No diet found."]),
        workout_dict.get(disease, ["No workout found."])
    )

# --- UI ---
st.title("🧠 DiaGnoFy AI 🤖Next-Gen AI Doctor")
st.subheader("           Diagnose Fast Live Better")

st.header("👤 Patient Information")
name = st.text_input("Full Name")
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0, max_value=120, value=25)
mobile = st.text_input("Mobile Number")
email = st.text_input("Email Address")

st.header("🩹 Symptoms")
symptom_input = st.text_area("Enter your symptoms separated by commas (e.g., cough, fever, fatigue)")

# Predict Button
if st.button("🔍 DiaogNise"):
    if name and symptom_input and mobile and email:
        user_symptoms, invalid_symptoms = clean_symptoms(symptom_input)
        if not user_symptoms:
            st.warning("⚠️ No valid symptoms detected. Please check your input format.")
            st.stop()

        predicted_disease = get_predicted_value(user_symptoms)
        desc, precautions, meds, diets, workouts = helper(predicted_disease)

        st.success("✅ Report Generated Successfully")
        st.subheader("📋 Health Report Card")

        st.markdown(f"""
        **Name:** {name}  
        **Age:** {age}  
        **Gender:** {gender}  
        **Mobile:** {mobile}  
        **Email:** {email}  
        """)

        st.markdown("### 🩹 Symptoms")
        st.write(", ".join(user_symptoms))

        st.markdown("### 🦠 Disease Prediction")
        st.info(predicted_disease)

        st.markdown("### 📄 Description")
        st.write(desc)

        st.markdown("### 💊 Medications")
        for med in meds:
            st.write(f"- {med}")

        st.markdown("### ✅ Precautions")
        for p in precautions:
            st.write(f"- {p}")

        st.markdown("### 🥗 Diet")
        for d in diets:
            st.write(f"- {d}")

        st.markdown("### 🏃 Workout")
        for w in workouts:
            st.write(f"- {w}")

        # --- Generate PDF ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_fill_color(240, 248, 255)  # light blue background
        pdf.rect(0, 0, 210, 297, 'F')  # fill entire A4

        pdf.set_font("Helvetica", size=12)

        # Add logo
        logo_path = "D:/Users/admin/pyt/diangonfy.png"  # Save logo to working dir
        try:
            pdf.image(logo_path, x=150, y=5, w=50)
        except:
            pass

        pdf.ln(40)
        pdf.set_font("Helvetica", style='B', size=16)
        pdf.cell(0, 10, txt="Smart Health Report Card", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Helvetica", size=12)

        pdf.multi_cell(0, 10, f"Name: {name}\nAge: {age}\nGender: {gender}\nMobile: {mobile}\nEmail: {email}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"Symptoms:\n {', '.join(user_symptoms)}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"Disease: \n{predicted_disease}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"Description: \n{desc}")
        pdf.ln(5)

        pdf.multi_cell(0, 10, "Medications:")
        for med in meds:
            pdf.cell(0, 10, f"- {med}", ln=True)

        pdf.multi_cell(0, 10, "Precautions:")
        for p in precautions:
            pdf.cell(0, 10, f"- {p}", ln=True)

        pdf.multi_cell(0, 10, "Diet:")
        for d in diets:
            pdf.cell(0, 10, f"- {d}", ln=True)

        pdf.multi_cell(0, 10, "Workout:")
        for w in workouts:
            pdf.cell(0, 10, f"- {w}", ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            tmp_file_path = tmp_file.name

        with open(tmp_file_path, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name=f"{name.replace(' ', '_')}_health_report.pdf", mime="application/pdf")

    else:
        st.warning("Please fill in all required fields!")
