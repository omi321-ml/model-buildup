Linear Regression vs Ridge vs Lasso – Model Comparison

This repository contains a complete comparison of Linear Regression, Ridge Regression, and Lasso Regression using Python (Scikit-Learn).
The project demonstrates preprocessing, training, evaluation, and error analysis across different regression models.

📂 Project Structure
│── Linear_vs_Ridge_vs_Lasso.ipynb     # Jupyter notebook with full analysis
│── README.md                          # Documentation file  
│── data/                              # (Optional) dataset folder  
│── models/                            # (Optional) saved models  

📌 Overview

Regression is one of the most widely used machine-learning techniques.
This project compares:

Model	Description
Linear Regression	Basic regression, no regularization
Ridge Regression	L2 regularization (reduces coefficient size)
Lasso Regression	L1 regularization (performs feature selection)

We evaluate the models using:

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

R² Score (Accuracy)

🚀 Technologies Used

Python

NumPy

Pandas

Scikit-Learn

Matplotlib / Seaborn (Optional)

📊 Model Training & Evaluation Code
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

for model_name, model in models.items():
    print("Model Name:", model_name)

    model1 = model.fit(x, y)

    # Train score
    print("Train Score:", model1.score(x, y))

    # Predictions
    pred = model1.predict(x)

    # Error metrics
    mae = mean_absolute_error(y, pred)
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)

    print("MAE:", mae)
    print("MSE:", mse)
    print("R2 Score:", r2)
    print("<-------------------------------------------->")

🧪 Evaluation Metrics Explained
✅ MAE – Mean Absolute Error

Measures average absolute difference between prediction and actual value.
Lower MAE → Better model.

✅ MSE – Mean Squared Error

Penalizes large errors heavily.
Lower MSE → Better performance.

✅ R² Score

Indicates how well the model fits the data.
1.0 = perfect fit
0.0 = poor fit

📈 Results Summary (Example)
Model	MAE	MSE	R² Score
Linear Regression	1.23	2.45	0.89
Ridge Regression	1.19	2.40	0.90
Lasso Regression	1.25	2.52	0.88

(Your actual results will appear in the notebook)

📥 How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/yourusername/your-repo-name.git

2️⃣ Install required libraries
pip install -r requirements.txt

3️⃣ Run the notebook
jupyter notebook Linear_vs_Ridge_vs_Lasso.ipynb

🏆 Conclusion

Ridge performs well when multicollinearity exists.

Lasso is useful for feature selection (makes coefficients zero).

Linear Regression works best when data is clean & stable.
