# 📡 Customer Churn Predictor

**Live app:** https://your-app.streamlit.app

Predicts which telecom customers are likely to leave, and suggests actions to keep them.
Built for the PSEB Artificial Intelligence Bootcamp (Business ML Activity).

## The problem
A telecom company loses customers every month, and keeping a customer costs far less than finding a new one.
This project finds **why** customers leave, **predicts who is next**, and gives the team a **live tool**.

## Key findings
- Month-to-month customers churn at 43%, compared with 3% on two-year contracts
- Most churn happens in the first year
- Electronic check users churn at 45%
- Fiber optic customers (42%) and customers without tech support (42%) also churn far more

## Model
Random Forest (balanced, tuned with GridSearchCV), catching 79% of customers who leave.

| Metric (20% unseen test data) | Score |
|---|---|
| Recall (leavers caught) | 79% |
| Precision | 50% |
| Accuracy | 74% |

Recall is the main score: a missed churner is a lost customer, while a wrongly flagged customer only costs a retention offer.

## Business recommendations
1. Move month-to-month customers to 1 or 2 year contracts with a discount.
2. Run onboarding and check-ins (30/60/90 days) in the first year.
3. Give a bill credit for automatic payment instead of electronic check.
4. Review fiber pricing and bundle tech support and online security.

## What the app does
- **Predict:** enter a customer's details, get a High / Medium / Low risk, the reasons and suggested actions
- **Insights:** interactive charts of churn by contract, tenure, payment method and monthly bill
- **Model:** recall, precision, confusion matrix and the top factors the model uses

## Run it locally
```bash
git clone https://github.com/<your-username>/customer-churn-predictor.git
cd customer-churn-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `Churn_Project.ipynb` | Full analysis: cleaning, EDA, models, tuning, evaluation |
| `telco_churn.csv` | Dataset |
| `requirements.txt` | Python packages |

## Built with
Python, pandas, scikit-learn, Plotly, Streamlit. Data from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
