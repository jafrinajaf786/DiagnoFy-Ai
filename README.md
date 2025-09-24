🩺 DiaGnoFy AI – AI Health Assistant

DiaGnoFy AI is an AI-powered health assistant that predicts diseases from symptoms and provides:

Disease description

Recommended medicines

Precautions

Diet plans

Workout suggestions

Downloadable personalized health reports

It’s designed to work as a chatbot-like assistant where the user inputs their details (name, age, symptoms), and DiaGnoFy AI predicts the disease and recommends health advice.

🚀 Features

Disease Prediction – Predicts the most likely disease based on symptoms.

Health Recommendations – Provides medicines, precautions, diet, and workouts.

Disease Description – Explains the condition in simple terms.

Chatbot Interaction – Collects user info step by step (Name → Age → Symptoms).

PDF Report Generation – Creates a downloadable health report for the user.

Web-based Interface – Built with Streamlit for easy access.

🛠️ Tech Stack

Programming Language: Python

Frontend: Streamlit

Machine Learning: scikit-learn / TensorFlow

NLP: Custom dataset of symptoms → diseases

Data Handling: Pandas, Numpy

PDF Generation: ReportLab / FPDF

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/diagnoFy-ai.git
cd diagnoFy-ai

2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run application
streamlit run app.py

📊 Dataset

The dataset contains:

Symptoms → List of symptoms

Disease → Target disease prediction

Description → Explanation of the disease

Precautions → Things to follow for better health

Diet → Recommended food habits

Workouts → Exercises for recovery

Medicines → Suggested medications

(Example row)

Symptoms	Disease	Description	Precautions	Diet	Workouts	Medicines
fever, cough, fatigue	Influenza	Viral infection affecting...	Rest, hydration	Soups, fluids, fruits	Light walking	Paracetamol
🏗️ Workflow

User Interaction → Chatbot asks for name, age, and symptoms.

Preprocessing → Symptoms cleaned and vectorized.

Disease Prediction → ML model predicts disease.

Recommendation Engine → Fetches description, medicines, diet, workouts.

Report Generation → Creates downloadable PDF health report.

📝 Future Improvements

Integration with LLMs (LangChain + Hugging Face) for better chatbot experience.

Voice-based interaction.

API integration with real medical databases.

Mobile app version.

🤝 Contributing

Contributions are welcome! Fork the repo and submit a pull request.

📜 License

This project is licensed under the MIT License – see the LICENSE
 file for details.

👨‍💻 Author

Najaf Jafri

AI & Data Science Enthusiast

Developer of DiaGnoFy AI
