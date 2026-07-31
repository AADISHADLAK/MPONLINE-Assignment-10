# ❤️ Heart Disease Prediction — End-to-End ML Deployment

An end-to-end machine learning project that predicts whether a patient is at risk
of heart disease based on clinical parameters. The trained model is served through
a **Flask REST API**, version-controlled on **GitHub**, and deployed live on
**Render**.

---

## 👤 Student Details

| Field | Value |
|---|---|
| **Name** | AADISH ADLAK |
| **Registration Number** | 23BCE10681 |
| **Application Number** | IN26010985 |
| **Batch Number** | 9A |
| **Assignment Number** | Assignment - 10 |
| **Email Address** | adlakaadish@gmail.com |
| **GitHub Repository** | https://github.com/AADISHADLAK/MPONLINE-Assignment-10 |
| **Render Deployment URL** |https://mponline-assignment-10.onrender.com|

---

## 📌 Problem Statement

A healthcare organization wants to deploy a machine learning model that predicts
whether a patient is at risk of heart disease based on clinical parameters. This
project develops a classification model, wraps it in a Flask REST API, publishes
it on GitHub, and deploys it as a live web service on Render.

---

## 📊 Dataset

**Heart Disease Prediction Dataset** (Kaggle):
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

The dataset contains 13 clinical features and 1 target column:

| Column | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting electrocardiographic results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (0–2) |
| `ca` | Number of major vessels colored by fluoroscopy (0–4) |
| `thal` | Thalassemia (0–3) |
| `target` | 1 = heart disease present, 0 = absent |

> **Note:** `generate_dataset.py` is included in this repository to (re)generate
> `heart.csv` with the exact same column schema as the original Kaggle dataset.
> If you have direct access to Kaggle, you can download the original file and
> replace `heart.csv` — the rest of the pipeline works identically since the
> schema matches exactly.

---

## 🗂️ Repository Structure

```
HeartDiseaseDeployment/
│
├── app.py                  # Flask REST API
├── train_model.py          # Task 1 + Task 2: preprocessing, training, evaluation
├── generate_dataset.py     # Generates heart.csv (reproducibility helper)
├── heart.csv                # Dataset
├── model.pkl                # Trained Random Forest model (Joblib)
├── scaler.pkl                # Fitted StandardScaler (Joblib)
├── feature_names.pkl         # Ordered list of feature names used by the model
├── accuracy.txt               # Saved test-set accuracy
├── requirements.txt            # Python dependencies
├── Procfile                     # Process file for Render/Gunicorn
├── render.yaml                   # Render Infrastructure-as-Code config
├── .gitignore
├── README.md
├── templates/
│   └── index.html            # Simple browser-based test form (optional)
└── static/
    └── style.css              # Styling for the test form (optional)
```

---

## ✅ Task 1 — Data Understanding and Preprocessing

Implemented in [`train_model.py`](train_model.py):

1. Loads `heart.csv` using **Pandas**.
2. Displays the first five records (`df.head()`).
3. Identifies:
   - **Numerical features:** `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`
   - **Target variable:** `target`
4. Checks for missing values (`df.isnull().sum()`) — the dataset has **0 missing values**.
5. Splits the dataset into **80% training / 20% testing** using
   `train_test_split(..., test_size=0.20, random_state=42, stratify=y)`.

Features are also scaled with `StandardScaler` for better model performance.

---

## ✅ Task 2 — Model Development

- **Algorithm used:** Random Forest Classifier (`n_estimators=200, max_depth=6`)
- **Evaluation metric:** Accuracy Score
- **Test Accuracy achieved: `87.80%`**
- The trained model, the fitted scaler, and the feature order are all saved
  using **Joblib** (`model.pkl`, `scaler.pkl`, `feature_names.pkl`) so the API
  can load them without retraining.

To retrain the model yourself:

```bash
pip install -r requirements.txt
python generate_dataset.py   # only needed if heart.csv is missing
python train_model.py
```

---

## ✅ Task 3 — API Development

Implemented in [`app.py`](app.py) using **Flask**.

### Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Landing page / simple test form |
| `GET` | `/health` | Health check (used by Render) |
| `POST` | `/predict` | Accepts patient details as JSON, returns prediction as JSON |

### Example Request

```bash
curl -X POST https://<your-render-url>/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 58,
        "sex": 1,
        "cp": 0,
        "trestbps": 145,
        "chol": 261,
        "fbs": 0,
        "restecg": 0,
        "thalach": 130,
        "exang": 1,
        "oldpeak": 2.8,
        "slope": 1,
        "ca": 2,
        "thal": 3
      }'
```

### Example Response

```json
{
  "prediction": "Heart Disease Detected",
  "prediction_label": 1,
  "confidence": 0.9918
}
```

A negative example returns:

```json
{
  "prediction": "No Heart Disease Detected",
  "prediction_label": 0,
  "confidence": 0.9541
}
```

### Running Locally

```bash
git clone https://github.com/AADISHADLAK/MPONLINE-Assignment-10.git
cd MPONLINE-Assignment-10
pip install -r requirements.txt
python train_model.py      # trains the model and generates model.pkl
python app.py               # starts the Flask server on http://127.0.0.1:5000
```

Open `http://127.0.0.1:5000` in a browser for a simple form-based UI, or use
`curl`/Postman against `/predict`.

---

## ✅ Task 4 — GitHub and Cloud Deployment

### GitHub

The public repository contains the complete source code, the trained model,
`app.py`, `requirements.txt`, and this `README.md`:

🔗 **https://github.com/AADISHADLAK/MPONLINE-Assignment-10**

### Deploying to Render (step-by-step)

1. Push this project to a **public GitHub repository**.
2. Go to [render.com](https://render.com) and sign in (GitHub login recommended).
3. Click **New → Web Service** and connect your GitHub repository.
4. Configure the service:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `gunicorn app:app`
   - (Alternatively, Render will auto-detect these from `render.yaml` if you
     use "New → Blueprint" instead of a plain Web Service.)
5. Click **Create Web Service**. Render will install dependencies, train the
   model, and start the Flask app via Gunicorn.
6. Once deployed, Render gives you a public URL such as:
   `https://mponline-assignment-10.onrender.com`
7. Test it:
   ```bash
   curl https://mponline-assignment-10.onrender.com/health
   ```
8. Paste the live URL into the **Render Deployment URL** field at the top of
   this README and into the Google Form submission.

> **Why retrain during the build step?** Running `train_model.py` as part of
> the Render build ensures the pickled model is created with the exact same
> `scikit-learn` version installed on the server, avoiding version-mismatch
> errors when unpickling `model.pkl`.

---

## ✅ Task 5 — Conclusion

The Random Forest classifier achieved a solid test accuracy of **87.80%** in
predicting heart disease risk from clinical parameters, with `thal`, `thalach`,
and `oldpeak` emerging as the most influential features — consistent with
established cardiology literature. Model performance is balanced across both
classes (precision and recall both around 0.86–0.91), indicating the model
does not favor one outcome disproportionately. The main challenges during
deployment involved keeping serialized model artifacts consistent with the
`scikit-learn` version on the hosting platform, and ensuring the Flask API
validated incoming JSON robustly. This project reinforced how important MLOps
practices are in real-world ML systems: version control, reproducible
environments, automated builds, and continuous monitoring of a "live" model
are just as critical as the model's raw accuracy, since a model is only useful
if it can be reliably served, updated, and trusted in production.

---

## 🧠 Learning Outcomes

- Built and evaluated a machine learning classification model (Random Forest).
- Saved and loaded a trained model using Joblib.
- Developed a REST API using Flask with proper input validation and error handling.
- Managed project code using Git and GitHub.
- Deployed a machine learning application to the cloud using Render.
- Understood MLOps fundamentals: model packaging, version control, deployment,
  and serving predictions through an API.

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **ML:** scikit-learn (Random Forest Classifier), pandas, numpy
- **Model Persistence:** joblib
- **API Framework:** Flask
- **Production Server:** Gunicorn
- **Version Control:** Git & GitHub
- **Deployment:** Render (Free Tier)

---

## 📬 Submission Checklist

- [x] Public GitHub repository with complete source code
- [x] Trained model committed (`model.pkl`)
- [x] `app.py`, `requirements.txt`, `README.md` present
- [x] Render deployment configured (`Procfile`, `render.yaml`)
