# 🏎️ F1 Lap Time Prediction

A Machine Learning web application that predicts the next Formula 1 lap time based on driver, team, circuit, tyre compound, track status, and race conditions.

The project uses historical Formula 1 data, an XGBoost regression model, and a Flask web interface for real-time predictions.

---

## 🚀 Features

* Predict Formula 1 lap times using Machine Learning
* Interactive web interface built with Flask
* XGBoost regression model
* Dockerized application
* Automated testing with Pytest
* CI/CD pipeline using GitHub Actions
* Docker Hub deployment support

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* FastF1

### Backend

* Flask

### DevOps

* Docker
* GitHub Actions
* Docker Hub

---

## 📂 Project Structure

```text
f1-lap-time-prediction/
│
├── .github/
│   └── workflows/
│
├── templates/
│   └── index.html
│
├── tests/
│   ├── test_app.py
│   └── test_model.py
│
├── app.py
├── data.py
├── xgboost_model.py
├── requirements.txt
├── Dockerfile
├── f1_xgboost_model.pkl
├── f1_feature_columns.pkl
└── README.md
```

---

## 📊 Dataset

The dataset contains Formula 1 lap information collected using the FastF1 API.

Features include:

* Driver
* Team
* Circuit
* Tyre Compound
* Track Status
* Lap Number
* Position
* Tyre Life

Target:

* Next Lap Time

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/<your-username>/f1-lap-time-prediction.git

cd f1-lap-time-prediction
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

---

## 🧪 Running Tests

```bash
python -m pytest
```

Example Output:

```text
3 passed
```

---

## 🐳 Docker

### Build Image

```bash
docker build -t f1-lap-time-prediction .
```

### Run Container

```bash
docker run -p 5000:5000 f1-lap-time-prediction
```

Open:

```text
http://localhost:5000
```

---

## 🔄 CI/CD Pipeline

The project uses GitHub Actions to:

1. Install dependencies
2. Run automated tests
3. Build Docker image
4. Push image to Docker Hub

Workflow triggers:

* Push to main
* Pull Request to main


---

## 📈 Model

Algorithm Used:

* XGBoost Regressor

Evaluation Metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 🎯 Future Improvements

* Driver-specific prediction analysis
* Weather integration
* Real-time race telemetry
* Race strategy simulation
* Deployment to AWS/Azure

---

## 👨‍💻 Author

Siddardha B

GitHub: https://github.com/<your-github-username>

LinkedIn: https://linkedin.com/in/<your-linkedin-profile>
