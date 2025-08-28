# 🩺 Diabetes Prediction Toolkit

A beginner-friendly Machine Learning project that predicts the likelihood of diabetes using real-world health datasets.  
This project demonstrates **data preprocessing, model training, evaluation, and deployment-ready pipelines**, making it suitable for academic submission, portfolio building, and healthcare AI/ML learning.

---

## 🚀 Features
- Preprocessing: handle missing values, encode categorical features, scale numerical features.
- Models implemented:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- Visualizations: Confusion Matrix, ROC Curve.
- Export: Save best model as `best_model.pkl` for later inference.
- Colab Notebook included for easy cloud execution.

---

## 📊 Dataset
- **Pima Indians Diabetes Dataset (UCI / Kaggle)**  
- Target Column: `Outcome`  
  - `1` → Diabetic  
  - `0` → Non-Diabetic  

You can download from Kaggle:  
👉 [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---


## 📈 Results
- Models compared: Logistic Regression, Decision Tree, Random Forest
- Metrics evaluated: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion Matrices and ROC Curves are saved in `outputs/run-YYYYMMDD-HHMMSS/`
- Best model auto-saved as `best_model.pkl`

---

## 🎥 Demo & Slides
- **5-Slide Presentation**: Summarizes dataset, methods, results, and next steps.  
- **30-second Demo Video Script**: Explains how to run the toolkit (included in `demo_video_script.txt`).  

---

## 🔮 Next Steps
- Add more models (XGBoost, SVM, Neural Networks).
- Cross-validation & hyperparameter tuning.
- Interpretability (Feature Importance, SHAP values).
- Deploy model via Flask/Django web app.

---

## 📝 Credits
- Dataset: [UCI / Kaggle Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- Toolkit built using **Python, scikit-learn, matplotlib, pandas, seaborn**.  

