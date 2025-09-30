
# Predict Salary

💼 Salary Predictor (Flask + TailwindCSS)

The Salary Predictor is a full-stack web application that predicts salaries based on years of experience.
It uses a Flask backend (Python) with a simple regression formula and a modern TailwindCSS-powered frontend for clean and responsive UI.

This project demonstrates how to:

🔗 Connect a Flask backend to a frontend via API requests.

📊 Process user input and return predictions in real time.

🎨 Build an interactive, responsive UI with TailwindCSS and Vanilla JS.

🚀 Features

✅ Flask-powered backend with a prediction API (/predict).

✅ User-friendly web interface with TailwindCSS.

✅ Salary prediction using a linear regression-like formula:

Salary = (9339.08 × Years of Experience) + 25918.43


✅ Input validation and error handling.

✅ Real-time results with loading indicator.

🛠️ Tech Stack

Backend: Python, Flask

Frontend: HTML, TailwindCSS, Vanilla JavaScript

Prediction Logic: Simulated Linear Regression Formula

📂 Project Structure

Salary-Predictor/

│── app.py              # Flask backend

│── templates/

│    └── index.html     # Frontend (Tailwind UI + JS)

│── static/             # Optional CSS/JS files

│── README.md           # Project documentation

▶️ Getting Started

1️⃣ Clone the repository
git clone https://github.com/your-username/salary-predictor.git

cd salary-predictor

2️⃣ Install dependencies
pip install flask

3️⃣ Run the Flask app
python app.py

4️⃣ Open in browser
http://127.0.0.1:5000

📊 Example API Usage

POST /predict

{

  "experience": 5

}


Response

{

  "predicted_salary": 72614

}

📸 UI Preview

When you open the app in the browser, you’ll see:

A blue background with a centered card.

An input box for years of experience.

A Predict Salary button.

A real-time result display showing the salary in ₹ (Indian Rupees).

📌 Future Enhancements

🔹 Replace formula with a trained Machine Learning model (model.pkl).

🔹 Add multiple input fields (education, job role, location, skills).

🔹 Deploy on Heroku / Render / AWS.

🔹 Add charts/graphs for better visualization.

👨‍💻 Developer

Hitesh Bachale

💡Passionate about Data Science, Machine Learning, Artificial Intelligence, and Web Development — with a keen interest in building predictive models, deploying AI-powered applications, and solving real-world problems using data-driven insights. Enthusiastic about Deep Learning, Natural Language Processing (NLP), and integrating intelligent systems into modern web solutions.💡 

