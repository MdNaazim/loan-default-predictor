# Loan Default Predictor — Machine Learning Project

**Author:** Mohammed Naazim Pasha Z  
**GitHub:** [github.com/MdNaazim](https://github.com/MdNaazim)  
**Tools:** Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

---

## Project Overview

A machine learning pipeline that predicts whether a loan applicant will **default or repay** their loan — helping banks and financial institutions make smarter, data-driven lending decisions.

This project covers the **full ML lifecycle**:
- Data generation and exploratory analysis
- Feature engineering and preprocessing
- Model training, evaluation, and comparison
- Live prediction on new applicants
- Results visualization dashboard

---

## Business Problem

Banks lose billions every year due to loan defaults. Traditional manual review is slow, inconsistent, and error-prone. This ML model automates risk assessment — flagging high-risk applicants before loans are approved.

---

## Dataset

| Feature | Description |
|---|---|
| Age | Applicant's age |
| Income | Annual income (INR/USD) |
| Loan Amount | Amount requested |
| Loan Tenure | Repayment period in months |
| Credit Score | Credit bureau score (300–850) |
| Existing Loans | Number of active loans |
| Missed Payments | Historical missed payments |
| Employment Type | Salaried / Self-Employed / Business |
| Education | Graduate / Post-Graduate / Under-Graduate |

**Engineered Features:**
- Debt-to-Income Ratio
- Credit Risk Score (normalized)
- Payment History Score

---

## ML Models Used

| Model | Accuracy | AUC Score |
|---|---|---|
| Logistic Regression | ~82% | ~0.88 |
| Random Forest | ~88% | ~0.94 |
| Gradient Boosting | ~87% | ~0.93 |

**Best Model: Random Forest** with highest AUC score

---

## Results

The model successfully:
- Identifies **high-risk borrowers** before loan approval
- Reduces false approvals that lead to defaults
- Provides **feature importance insights** for business decisions
- Generates a complete **results dashboard** with charts

---

## Project Structure

```
loan-default-predictor/
├── loan_default_predictor.py   # Main ML pipeline
├── requirements.txt            # Python dependencies
├── results_dashboard.png       # Generated output charts
└── README.md                   # Project documentation
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/MdNaazim/loan-default-predictor.git
cd loan-default-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the project
python loan_default_predictor.py
```

---

## Sample Output

```
[1/6] Generating loan dataset...
      Dataset created: 2000 records
      Default rate: 28.4%

[2/6] Running Exploratory Data Analysis (EDA)...

[3/6] Preprocessing data...
      Training samples: 1600
      Testing samples:  400

[4/6] Training ML models...
      Logistic Regression       | Accuracy: 82.3% | AUC: 0.881
      Random Forest             | Accuracy: 88.5% | AUC: 0.942
      Gradient Boosting         | Accuracy: 87.2% | AUC: 0.935

[5/6] Evaluating best model...
      Best Model: Random Forest

[6/6] Generating visualizations...
      Saved: results_dashboard.png

LIVE PREDICTION DEMO
      Applicant 1 (Good profile)  → LOW RISK  — Default prob: 8.2%
      Applicant 2 (Risky profile) → HIGH RISK — Default prob: 94.7%
```

---

## Key Learnings

- End-to-end ML pipeline development
- Feature engineering for financial risk modeling
- Model comparison and selection using AUC-ROC
- Confusion matrix interpretation for business decisions
- Real-world application of Scikit-Learn in fintech

---

## Skills Demonstrated

`Python` `Scikit-Learn` `Pandas` `NumPy` `Matplotlib` `Seaborn`  
`Machine Learning` `EDA` `Feature Engineering` `Random Forest`  
`Gradient Boosting` `Logistic Regression` `Data Visualization`

---

## Connect

- LinkedIn: [linkedin.com/in/mohammed-naazim-pasha](https://linkedin.com/in/mohammed-naazim-pasha)
- Email: mohammednaazim77@gmail.com

---

*This project demonstrates practical ML skills for Data Science and AI roles in the Finance and Insurance (I&F) domain.*
