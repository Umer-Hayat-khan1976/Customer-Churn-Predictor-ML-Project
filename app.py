"""
Customer Churn Predictor - Streamlit app
PSEB Artificial Intelligence Bootcamp | Business ML Activity

The app trains the final model (balanced Random Forest) when it starts,
using telco_churn.csv, so there is no pickle file to keep in sync.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Page setup and styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📡", layout="wide")

STAY, LEAVE, PRIMARY = "#0d9488", "#e11d48", "#4f46e5"

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1200px;}
    .hero {background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 60%, #0d9488 120%);
           padding: 28px 32px; border-radius: 16px; color: white; margin-bottom: 18px;}
    .hero h1 {margin: 0; font-size: 2rem; color: white;}
    .hero p {margin: 6px 0 0 0; opacity: .92; font-size: 1.02rem;}
    .card {background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px;
           padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,.05); height: 100%;}
    .card h4 {margin: 0 0 6px 0; color: #374151; font-size: .9rem; font-weight: 600;
              text-transform: uppercase; letter-spacing: .04em;}
    .big {font-size: 2rem; font-weight: 700; margin: 0;}
    .badge {display: inline-block; padding: 6px 16px; border-radius: 999px;
            color: white; font-weight: 700; font-size: 1.1rem;}
    .factor {background: #fef2f2; border-left: 4px solid #e11d48; padding: 8px 12px;
             border-radius: 6px; margin-bottom: 8px; color: #7f1d1d; font-size: .93rem;}
    .action {background: #f0fdfa; border-left: 4px solid #0d9488; padding: 8px 12px;
             border-radius: 6px; margin-bottom: 8px; color: #134e4a; font-size: .93rem;}
    .good {background: #f0fdf4; border-left: 4px solid #16a34a; padding: 8px 12px;
           border-radius: 6px; color: #14532d; font-size: .93rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Data and model (cached so they run only once)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("telco_churn.csv")
    # TotalCharges is text with 11 blank values (customers with tenure 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df.drop(columns="customerID")


NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]


@st.cache_resource
def train_model():
    df = load_data()
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns="Churn")
    categorical = [c for c in X.columns if c not in NUMERIC]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    # Best settings found with GridSearchCV in the notebook
    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=10,
        class_weight="balanced", random_state=42,
    )
    pipe = Pipeline([("prep", preprocess), ("model", model)]).fit(X_train, y_train)

    pred = pipe.predict(X_test)
    metrics = {
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "accuracy": accuracy_score(y_test, pred),
        "cm": confusion_matrix(y_test, pred),
    }
    names = pipe.named_steps["prep"].get_feature_names_out()
    importance = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
    return pipe, metrics, importance


def risk_level(p):
    """Turn a probability into High / Medium / Low (bands checked on the test set)."""
    if p >= 0.60:
        return "High", LEAVE
    if p >= 0.35:
        return "Medium", "#f59e0b"
    return "Low", STAY


# ----------------------------------------------------------------------------
# Sidebar: customer details
# ----------------------------------------------------------------------------
def customer_form():
    st.sidebar.markdown("## 👤 Customer details")

    with st.sidebar.expander("Account", expanded=True):
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        tenure = st.slider("Tenure (months)", 0, 72, 6)
        monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 75.0, 0.5)
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)

    with st.sidebar.expander("Services", expanded=True):
        phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
        lines = "No phone service"
        if phone == "Yes":
            lines = st.radio("Multiple lines", ["No", "Yes"], horizontal=True)

        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        extras = {}
        for col, label in [
            ("OnlineSecurity", "Online security"), ("OnlineBackup", "Online backup"),
            ("DeviceProtection", "Device protection"), ("TechSupport", "Tech support"),
            ("StreamingTV", "Streaming TV"), ("StreamingMovies", "Streaming movies"),
        ]:
            if internet == "No":
                extras[col] = "No internet service"
            else:
                extras[col] = st.radio(label, ["No", "Yes"], horizontal=True)

    with st.sidebar.expander("Profile"):
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
        partner = st.radio("Has partner", ["No", "Yes"], horizontal=True)
        dependents = st.radio("Has dependents", ["No", "Yes"], horizontal=True)

    # TotalCharges is not asked: we estimate it as tenure x monthly charges
    return pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
        **extras, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly,
    }])


# ----------------------------------------------------------------------------
# Simple, explainable rules: risk factors -> retention actions
# ----------------------------------------------------------------------------
def explain(c):
    r = c.iloc[0]
    items = []  # (risk factor, suggested action)
    if r.Contract == "Month-to-month":
        items.append(("Month-to-month contract (43% of these customers churn, vs 3% on two-year)",
                      "Offer a discount or free upgrade for moving to a 1 or 2 year contract."))
    if r.tenure < 12:
        items.append(("New customer, still in the first year when most churn happens",
                      "Start an onboarding programme: welcome call, 30/60/90-day check-ins."))
    if r.PaymentMethod == "Electronic check":
        items.append(("Pays by electronic check (45% churn, the highest of all methods)",
                      "Encourage automatic bank or card payment with a small bill credit."))
    if r.InternetService == "Fiber optic" and r.MonthlyCharges > 70:
        items.append(("High-priced fiber plan (fiber users churn at 42%)",
                      "Review price and service quality; offer a loyalty price or speed guarantee."))
    if r.InternetService != "No" and r.TechSupport == "No":
        items.append(("No tech support",
                      "Offer a free tech support trial for 3 months."))
    if r.InternetService != "No" and r.OnlineSecurity == "No":
        items.append(("No online security add-on",
                      "Bundle online security at a discount to increase stickiness."))
    return items


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def predict_tab(pipe, customer):
    prob = pipe.predict_proba(customer)[0, 1]
    level, color = risk_level(prob)

    c1, c2 = st.columns([1, 1.2])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100,
            number={"suffix": "%", "font": {"size": 44}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                   "steps": [{"range": [0, 35], "color": "#ccfbf1"},
                             {"range": [35, 60], "color": "#fef3c7"},
                             {"range": [60, 100], "color": "#fee2e2"}]},
        ))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=0))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown(
            f"""<div class="card"><h4>Churn risk</h4>
            <span class="badge" style="background:{color}">{level.upper()}</span>
            <p style="margin-top:12px;color:#4b5563;">
            The model scores this customer at <b>{prob:.0%}</b>. The model is tuned to
            catch leavers, so treat the score as a <b>risk ranking</b>, not an exact chance.
            In our test data about 5% of Low, 24% of Medium and 56% of High-risk customers
            really left.</p></div>""",
            unsafe_allow_html=True,
        )

    items = explain(customer)
    st.markdown("### 🔎 Why this customer may leave")
    if items:
        for factor, _ in items:
            st.markdown(f'<div class="factor">⚠️ {factor}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="good">✅ No major risk factors found for this profile.</div>',
                    unsafe_allow_html=True)

    st.markdown("### 💡 Suggested actions to keep them")
    if level == "Low" and not items:
        st.markdown('<div class="good">Keep the customer happy with regular service. '
                    'No special action needed.</div>', unsafe_allow_html=True)
    elif items:
        for _, action in items:
            st.markdown(f'<div class="action">✔ {action}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="action">✔ Monitor the account and check in occasionally.</div>',
                    unsafe_allow_html=True)


def churn_rate_from(data, col):
    return (data.assign(churn=(data.Churn == "Yes") * 100)
            .groupby(col, observed=True)["churn"].mean().reset_index())


def insights_tab(df):
    st.markdown("### Why do customers leave?")
    st.caption(f"Based on {len(df):,} customers. Overall churn rate: "
               f"{(df.Churn == 'Yes').mean():.1%}")

    def bar(data, x, title):
        fig = px.bar(data, x=x, y="churn", text=data["churn"].round(1).astype(str) + "%",
                     title=title, color_discrete_sequence=[PRIMARY])
        fig.update_layout(yaxis_title="Churn rate (%)", xaxis_title="", height=340,
                          margin=dict(t=50, b=10), plot_bgcolor="white")
        fig.update_traces(textposition="outside")
        return fig

    a, b = st.columns(2)
    a.plotly_chart(bar(churn_rate_from(df, "Contract"), "Contract", "Churn by contract type"),
                   width="stretch")
    b.plotly_chart(bar(churn_rate_from(df, "PaymentMethod"), "PaymentMethod", "Churn by payment method"),
                   width="stretch")

    tenure_band = pd.cut(df.tenure, [-1, 12, 24, 48, 72],
                         labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"])
    t = df.assign(Tenure=tenure_band)
    c, d = st.columns(2)
    c.plotly_chart(bar(churn_rate_from(t, "Tenure"), "Tenure", "Churn by tenure"),
                   width="stretch")
    fig = px.histogram(df, x="MonthlyCharges", color="Churn", barmode="overlay", opacity=.65,
                       nbins=40, title="Monthly bill: stayed vs left",
                       color_discrete_map={"No": STAY, "Yes": LEAVE})
    fig.update_layout(height=340, margin=dict(t=50, b=10), plot_bgcolor="white")
    d.plotly_chart(fig, width="stretch")


def model_tab(metrics, importance):
    st.markdown("### Model performance (20% unseen test data)")
    cols = st.columns(3)
    for col, (title, key, note) in zip(cols, [
        ("Recall", "recall", "Leavers we caught"),
        ("Precision", "precision", "Flagged who really leave"),
        ("Accuracy", "accuracy", "All correct predictions"),
    ]):
        col.markdown(f'<div class="card"><h4>{title}</h4><p class="big" style="color:{PRIMARY}">'
                     f'{metrics[key]:.0%}</p><span style="color:#6b7280">{note}</span></div>',
                     unsafe_allow_html=True)

    st.write("")
    a, b = st.columns(2)
    cm = metrics["cm"]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                    x=["Predicted stay", "Predicted leave"], y=["Actually stayed", "Actually left"],
                    title="Confusion matrix")
    fig.update_layout(height=360, margin=dict(t=50, b=10), coloraxis_showscale=False)
    a.plotly_chart(fig, width="stretch")

    top = importance.sort_values().tail(10)
    top.index = [i.replace("num__", "").replace("cat__", "").replace("_", " = ") for i in top.index]
    fig = px.bar(top, orientation="h", title="Top 10 factors the model uses",
                 color_discrete_sequence=[PRIMARY])
    fig.update_layout(height=360, showlegend=False, xaxis_title="Importance", yaxis_title="",
                      margin=dict(t=50, b=10), plot_bgcolor="white")
    b.plotly_chart(fig, width="stretch")

    st.info("We chose **recall** as the main score because a missed churner is a lost customer. "
            "The trade-off: some loyal customers get flagged too, which costs only a retention offer.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.markdown(
    """<div class="hero"><h1>📡 Customer Churn Predictor</h1>
    <p>Enter a customer's details in the sidebar to see their churn risk and the best
    actions to keep them.</p></div>""",
    unsafe_allow_html=True,
)

data = load_data()
pipeline, model_metrics, feature_importance = train_model()
customer_row = customer_form()

tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📊 Insights", "🤖 Model"])
with tab1:
    predict_tab(pipeline, customer_row)
with tab2:
    insights_tab(data)
with tab3:
    model_tab(model_metrics, feature_importance)

st.caption("Built with Python, pandas, scikit-learn and Streamlit. Data: Telco Customer Churn (Kaggle).")
