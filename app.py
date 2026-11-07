"""
Customer Churn Predictor - Streamlit app
PSEB Artificial Intelligence Bootcamp | Business ML Activity

The app trains the final model (balanced Random Forest) when it starts,
using telco_churn.csv, so there is no pickle file to keep in sync.
"""

import re

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
# Page setup and look
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer churn predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colour palette: one ink colour, one brand blue, and three risk colours
INK, MUTED, LINE = "#14213D", "#5B6B82", "#DCE3EC"
PRIMARY, SLATE = "#2457D6", "#B7C3D6"
STAY, AMBER, LEAVE = "#1F8A70", "#D9902F", "#C8372D"
FONT = "Public Sans, system-ui, sans-serif"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=Public+Sans:wght@400;500;600;700&display=swap');

    /* Type: bold display headings, calm body text */
    .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input,
    .stApp [data-baseweb="select"], .stApp [data-baseweb="tab"] {
        font-family: 'Public Sans', system-ui, sans-serif;
    }
    h1, h2, h3, h4, .display {
        font-family: 'Bricolage Grotesque', 'Public Sans', sans-serif !important;
        font-weight: 700 !important; color: #14213D; letter-spacing: -0.02em;
    }
    .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1120px;}
    #MainMenu, footer {visibility: hidden;}

    /* Sidebar */
    [data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #DCE3EC;}
    [data-testid="stSidebar"] > div:first-child {padding-top: 1rem;}
    .brand {font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.35rem;
            color: #14213D; letter-spacing: -0.02em; border-left: 4px solid #2457D6; padding-left: 10px;}
    .side-note {color: #5B6B82; font-size: .86rem; margin: 6px 0 4px 0;}
    .side-label {font-weight: 700; font-size: .95rem; color: #14213D; margin: 18px 0 2px 0;
                 padding-bottom: 6px; border-bottom: 1px solid #DCE3EC;}
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid #DCE3EC; border-radius: 10px; background: #F6F8FB; margin-top: 14px;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {font-weight: 600;}

    /* Tabs */
    [data-baseweb="tab-list"] {gap: 30px; border-bottom: 1px solid #DCE3EC;}
    [data-baseweb="tab"] {padding: 12px 0; font-weight: 600; color: #5B6B82;}
    [aria-selected="true"][data-baseweb="tab"] {color: #14213D; font-weight: 700;}

    /* Page header */
    .page-head h1 {font-size: 2.5rem; font-weight: 800 !important; margin: 0 0 8px 0; line-height: 1.08;}
    .page-head p {color: #5B6B82; font-size: 1.06rem; line-height: 1.55; max-width: 660px; margin: 0;}
    .kpis {display: flex; flex-wrap: wrap; gap: 14px 44px; margin: 22px 0 8px 0;
           padding-top: 16px; border-top: 1px solid #DCE3EC;}
    .kpis b {display: block; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.7rem;
             font-weight: 800; letter-spacing: -0.02em; color: #14213D; line-height: 1.1;}
    .kpis span {color: #5B6B82; font-size: .88rem;}

    /* Section headings and descriptions */
    .section-title {font-size: 1.45rem; font-weight: 800 !important; margin: 30px 0 4px 0;}
    .lede {color: #5B6B82; line-height: 1.55; max-width: 720px; margin: 0 0 16px 0;}
    .chart-title {font-size: 1.05rem; font-weight: 700 !important; line-height: 1.3;
                  margin: 4px 0 4px 0; min-height: 2.7em;}
    .desc {color: #5B6B82; font-size: .9rem; line-height: 1.5; margin: 0 0 6px 0; min-height: 4.3em;}

    /* Panels: result, charts, stats */
    .panel {background: #FFFFFF; border: 1px solid #DCE3EC; border-radius: 14px;
            padding: 26px 30px 24px 30px; box-shadow: 0 1px 2px rgba(20,33,61,.04);}
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #DCE3EC !important; border-radius: 14px !important; background: #FFFFFF;
        box-shadow: 0 1px 2px rgba(20,33,61,.04); padding: 6px 10px 2px 10px;
    }
    .stat {background: #FFFFFF; border: 1px solid #DCE3EC; border-left: 4px solid #2457D6;
           border-radius: 12px; padding: 16px 20px; box-shadow: 0 1px 2px rgba(20,33,61,.04);}
    .stat .num {font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.6rem; font-weight: 800;
                line-height: 1; letter-spacing: -0.03em;}
    .stat .lab {font-weight: 700; margin-top: 8px;}
    .stat .sub {color: #5B6B82; font-size: .88rem;}

    /* Result: big score, plain words, and a meter */
    .profile {color: #5B6B82; margin: 0 0 18px 0; font-size: .96rem;}
    .verdict {display: flex; gap: 40px; flex-wrap: wrap; align-items: flex-end; margin-bottom: 26px;}
    .score {font-family: 'Bricolage Grotesque', sans-serif; font-size: 5rem; font-weight: 800;
            line-height: .95; letter-spacing: -0.04em;}
    .level {font-size: 1.15rem; font-weight: 700; margin-top: 8px;}
    .blurb {flex: 1; min-width: 260px; color: #5B6B82; font-size: .94rem; line-height: 1.6; max-width: 470px;}
    .meter-wrap {position: relative; padding-top: 6px;}
    .meter {display: flex; height: 12px; border-radius: 6px; overflow: hidden;}
    .meter span {height: 100%;}
    .pin {position: absolute; top: 0; width: 4px; height: 24px; background: #14213D;
          border: 2px solid #fff; border-radius: 3px; transform: translateX(-50%);}
    .meter-labels {display: flex; font-size: .82rem; color: #5B6B82; margin-top: 8px;}

    /* Two-column lists (risk signals and actions, findings and recommendations) */
    .rows {background: #FFFFFF; border: 1px solid #DCE3EC; border-radius: 14px; overflow: hidden;
           box-shadow: 0 1px 2px rgba(20,33,61,.04);}
    .rows .row {display: grid; grid-template-columns: 1fr 1fr; gap: 28px; padding: 14px 24px;
                border-top: 1px solid #DCE3EC; font-size: .96rem; line-height: 1.5;}
    .rows .head {background: #F6F8FB; color: #14213D; font-size: .86rem; font-weight: 700;
                 padding: 10px 24px; border-top: none;}
    .plain {background: #FFFFFF; border: 1px solid #DCE3EC; border-radius: 14px; color: #5B6B82;
            padding: 16px 24px; margin: 0;}
    @media (max-width: 700px) {.rows .row {grid-template-columns: 1fr; gap: 4px;}}

    .note {color: #5B6B82; max-width: 740px; line-height: 1.6; margin-top: 8px;}
    .foot {color: #8A97AB; font-size: .85rem; margin-top: 44px; padding-top: 14px; border-top: 1px solid #DCE3EC;}
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
        return "Medium", AMBER
    return "Low", STAY


# ----------------------------------------------------------------------------
# Sidebar: customer details (only the essentials are always visible)
# ----------------------------------------------------------------------------
def customer_form():
    sb = st.sidebar
    sb.markdown(
        '<div class="brand">Churn Predictor</div>'
        '<p class="side-note">Describe a customer. The result updates as you edit.</p>',
        unsafe_allow_html=True,
    )

    sb.markdown('<p class="side-label">Account</p>', unsafe_allow_html=True)
    contract = sb.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tenure = sb.slider("Tenure (months)", 0, 72, 6)
    monthly = sb.slider("Monthly charges ($)", 18.0, 120.0, 75.0, 0.5)
    payment = sb.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    paperless = sb.radio("Paperless billing", ["Yes", "No"], horizontal=True)

    sb.markdown('<p class="side-label">Services</p>', unsafe_allow_html=True)
    internet = sb.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    phone = sb.radio("Phone service", ["Yes", "No"], horizontal=True)
    lines = "No phone service"
    if phone == "Yes":
        lines = sb.radio("Multiple lines", ["No", "Yes"], horizontal=True)

    extras = {}
    for col, label in [("OnlineSecurity", "Online security"), ("TechSupport", "Tech support")]:
        if internet == "No":
            extras[col] = "No internet service"
        else:
            extras[col] = sb.radio(label, ["No", "Yes"], horizontal=True)

    # Less important inputs are tucked away so the sidebar stays short
    with sb.expander("More services"):
        if internet == "No":
            st.caption("This customer has no internet service.")
        for col, label in [
            ("OnlineBackup", "Online backup"), ("DeviceProtection", "Device protection"),
            ("StreamingTV", "Streaming TV"), ("StreamingMovies", "Streaming movies"),
        ]:
            if internet == "No":
                extras[col] = "No internet service"
            else:
                extras[col] = st.radio(label, ["No", "Yes"], horizontal=True)

    with sb.expander("About the customer"):
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
# Chart helpers (one consistent look for every chart)
# ----------------------------------------------------------------------------
def style(fig, height=300, legend=False):
    fig.update_layout(
        height=height, margin=dict(l=0, r=10, t=10, b=0), showlegend=legend,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=13),
        legend=dict(orientation="h", y=1.12, x=0, title=None),
    )
    return fig


def churn_rate_from(data, col):
    return (data.assign(churn=(data.Churn == "Yes") * 100)
            .groupby(col, observed=True)["churn"].mean().reset_index())


def rate_bar(data, col, horizontal=False):
    """Bar chart of churn rate. The highest bar is red, the rest are grey-blue."""
    rate = churn_rate_from(data, col)
    if horizontal:
        rate = rate.sort_values("churn", ascending=False)
    colors = [LEAVE if v == rate["churn"].max() else SLATE for v in rate["churn"]]
    labels = rate["churn"].round(0).astype(int).astype(str) + "%"
    if horizontal:
        fig = go.Figure(go.Bar(
            y=rate[col], x=rate["churn"], orientation="h", marker_color=colors,
            text=labels, textposition="outside",
            hovertemplate="%{y}<br>Churn rate: %{x:.1f}%<extra></extra>"))
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_xaxes(visible=False, range=[0, rate["churn"].max() * 1.25])
    else:
        fig = go.Figure(go.Bar(
            x=rate[col].astype(str), y=rate["churn"], marker_color=colors,
            text=labels, textposition="outside",
            hovertemplate="%{x}<br>Churn rate: %{y:.1f}%<extra></extra>"))
        fig.update_xaxes(title=None)
        fig.update_yaxes(visible=False, range=[0, rate["churn"].max() * 1.25])
    return style(fig, height=280)


def chart_panel(title, desc, fig):
    """A white panel with a headline finding, a short explanation and the chart."""
    with st.container(border=True):
        st.markdown(f'<h4 class="chart-title">{title}</h4><p class="desc">{desc}</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def section(title, text=None, first=False):
    margin = "margin-top:22px" if first else ""
    st.markdown(f'<h3 class="section-title" style="{margin}">{title}</h3>', unsafe_allow_html=True)
    if text:
        st.markdown(f'<p class="lede">{text}</p>', unsafe_allow_html=True)


def pretty(name):
    """num__tenure -> Tenure, cat__PaymentMethod_Mailed check -> Payment Method: Mailed check"""
    name = name.replace("num__", "").replace("cat__", "")
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).replace("_", ": ", 1)
    return name[0].upper() + name[1:]


def two_column_list(head_left, head_right, rows):
    body = "".join(f'<div class="row"><div>{a}</div><div>{b}</div></div>' for a, b in rows)
    st.markdown(
        f'<div class="rows"><div class="row head"><div>{head_left}</div><div>{head_right}</div></div>{body}</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Page header
# ----------------------------------------------------------------------------
def header(df, metrics):
    st.markdown(
        f"""<div class="page-head">
        <h1>Customer churn predictor</h1>
        <p>Find out which telecom customers are likely to leave, understand why, and see what
        the retention team can do about it. Start by describing a customer in the sidebar.</p>
        <div class="kpis">
            <div><b>{len(df):,}</b><span>customers analysed</span></div>
            <div><b>{(df.Churn == "Yes").mean():.1%}</b><span>have left the company</span></div>
            <div><b>{metrics["recall"]:.0%}</b><span>of leavers caught by the model</span></div>
        </div></div>""",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Tab 1: check a customer
# ----------------------------------------------------------------------------
def predict_tab(pipe, customer):
    prob = pipe.predict_proba(customer)[0, 1]
    level, color = risk_level(prob)
    r = customer.iloc[0]

    # Meter: the band the customer falls in is shown in full colour
    bands = [("Low", 35, STAY, "#CFE9E0"), ("Medium", 25, AMBER, "#F5E3C3"), ("High", 40, LEAVE, "#F0CBC7")]
    segments = "".join(
        f'<span style="width:{w}%;background:{full if name == level else tint}"></span>'
        for name, w, full, tint in bands
    )
    labels = "".join(f'<span style="width:{w}%">{name}</span>' for name, w, _, _ in bands)

    section("Churn risk for this customer",
            "The score below is the model's estimate for the customer described in the sidebar.",
            first=True)
    st.markdown(
        f'''<div class="panel">
        <p class="profile">{r.Contract} contract, {r.tenure} months with us,
        ${r.MonthlyCharges:.2f} a month, pays by {r.PaymentMethod.lower()}.</p>
        <div class="verdict">
            <div><div class="score" style="color:{color}">{prob:.0%}</div>
                 <div class="level" style="color:{color}">{level} risk of leaving</div></div>
            <div class="blurb">The model is tuned to catch customers who leave, so read this
            score as a ranking of risk, not an exact chance. In our test data about 5% of
            Low, 24% of Medium and 56% of High-risk customers really left.</div>
        </div>
        <div class="meter-wrap">
            <div class="meter">{segments}</div>
            <div class="pin" style="left:{prob * 100:.1f}%"></div>
            <div class="meter-labels">{labels}</div>
        </div></div>''',
        unsafe_allow_html=True,
    )

    items = explain(customer)
    section("Risk drivers and retention actions",
            "Each row pairs a risk signal found in this customer's profile with an action the "
            "retention team can take. The signals come from the patterns in the historical data "
            "shown in the next tab.")
    if items:
        two_column_list("What we see", "What to try", items)
    elif level == "Low":
        st.markdown('<p class="plain">Nothing in this profile stands out. Regular good service is enough.</p>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<p class="plain">No single factor stands out. Keep an eye on the account and check in now and then.</p>',
                    unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Tab 2: why customers leave
# ----------------------------------------------------------------------------
def insights_tab(df):
    overall = (df.Churn == "Yes").mean()
    section("Why customers leave",
            f"A look at all {len(df):,} customers in the data, of whom {overall:.1%} have left. "
            "Each chart shows the churn rate, which is the share of customers in a group who left. "
            "The red bar marks the group that leaves most often.", first=True)

    contract = churn_rate_from(df, "Contract").set_index("Contract")["churn"]
    pay = churn_rate_from(df, "PaymentMethod").set_index("PaymentMethod")["churn"]
    auto = pay[["Bank transfer (automatic)", "Credit card (automatic)"]].mean()
    tenure_band = pd.cut(df.tenure, [-1, 12, 24, 48, 72],
                         labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"])
    by_tenure = churn_rate_from(df.assign(Tenure=tenure_band), "Tenure").set_index("Tenure")["churn"]
    bill = df.groupby("Churn")["MonthlyCharges"].mean()

    service = df.assign(Service=df.InternetService.map(
        {"DSL": "DSL", "Fiber optic": "Fiber optic", "No": "No internet"}))
    by_service = churn_rate_from(service, "Service").set_index("Service")["churn"]
    net = df[df.InternetService != "No"]
    support = net.assign(Support=net.TechSupport.map(
        {"Yes": "Has tech support", "No": "No tech support"}))
    by_support = churn_rate_from(support, "Support").set_index("Support")["churn"]

    a, b = st.columns(2, gap="medium")
    with a:
        chart_panel(
            f'Month-to-month customers leave {contract["Month-to-month"] / contract["Two year"]:.0f} '
            "times as often as customers on a two-year contract.",
            "Customers on a short contract are free to leave at any time, while a longer contract "
            "keeps them for longer. Contract type is the strongest single signal in the data.",
            rate_bar(df, "Contract"))
    with b:
        chart_panel(
            f'Customers paying by electronic check leave at {pay["Electronic check"]:.0f}%, '
            f"against {auto:.0f}% on automatic payments.",
            "Customers who pay by hand are reminded of the cost every month, and they have to act "
            "each time to keep paying. The data shows the pattern but cannot prove the reason.",
            rate_bar(df, "PaymentMethod", horizontal=True))

    c, d = st.columns(2, gap="medium")
    with c:
        chart_panel(
            f'{by_tenure["0-12 months"]:.0f}% of customers leave in their first year. '
            f'After four years it falls to {by_tenure["49-72 months"]:.0f}%.',
            "Tenure is how many months a customer has been with the company. Churn is highest soon "
            "after sign-up and falls steadily, so the first year is the best time to invest in retention.",
            rate_bar(df.assign(Tenure=tenure_band), "Tenure"))
    with d:
        fig = px.histogram(
            df.assign(Status=df.Churn.map({"No": "Stayed", "Yes": "Left"})),
            x="MonthlyCharges", color="Status", barmode="overlay", opacity=0.7, nbins=36,
            histnorm="percent", color_discrete_map={"Stayed": SLATE, "Left": LEAVE})
        fig.update_xaxes(title="Monthly charges ($)", showgrid=False)
        fig.update_yaxes(title="Share of group (%)", gridcolor=LINE)
        chart_panel(
            f'Customers who leave pay about ${bill["Yes"] - bill["No"]:.0f} more per month '
            "than those who stay.",
            "Both groups are shown as a share of their own size, so the shapes are easy to compare. "
            "A larger red area on the right means leavers are concentrated among higher bills.",
            style(fig, height=280, legend=True))

    e, f = st.columns(2, gap="medium")
    with e:
        chart_panel(
            f'Fiber optic customers leave at {by_service["Fiber optic"]:.0f}%, against '
            f'{by_service["DSL"]:.0f}% for DSL customers.',
            "Fiber is the premium plan and comes with higher bills. The gap suggests price or "
            "service quality deserves a closer look rather than the technology itself.",
            rate_bar(service, "Service"))
    with f:
        chart_panel(
            f'Internet customers without tech support leave at {by_support["No tech support"]:.0f}%, '
            f'against {by_support["Has tech support"]:.0f}% for those who have it.',
            "Support and security add-ons seem to keep customers engaged. This is an association "
            "in the data, so test an offer with a small group before rolling it out widely.",
            rate_bar(support, "Support"))

    section("Where to focus",
            "The findings above point to four practical steps for the retention team.")
    two_column_list("Finding", "Recommended action", [
        ("Month-to-month contracts have the highest churn.",
         "Offer a discount or free upgrade for moving to a 1 or 2 year plan."),
        ("Churn is highest in the first year.",
         "Add a welcome call and check-ins at 30, 60 and 90 days."),
        ("Electronic check users leave most among payment methods.",
         "Give a small bill credit for switching to automatic payment."),
        ("Fiber customers and customers without tech support leave more.",
         "Review fiber pricing, and bundle tech support and security at a discount."),
    ])


# ----------------------------------------------------------------------------
# Tab 3: how the model performs
# ----------------------------------------------------------------------------
def model_tab(metrics, importance):
    tn, fp, fn, tp = metrics["cm"].ravel()
    section("How well the model works",
            "These scores come from the 20% of customers that the model never saw while it was "
            "learning, so they show how it behaves on new customers.", first=True)

    cols = st.columns(3, gap="medium")
    for col, (title, key, note) in zip(cols, [
        ("Recall", "recall", "of customers who left were caught"),
        ("Precision", "precision", "of flagged customers really left"),
        ("Accuracy", "accuracy", "of all predictions were right"),
    ]):
        col.markdown(f'<div class="stat"><div class="num">{metrics[key]:.0%}</div>'
                     f'<div class="lab">{title}</div><div class="sub">{note}</div></div>',
                     unsafe_allow_html=True)

    st.markdown(
        f'<p class="note" style="margin-top:16px">Recall is the main score here. Of every 100 customers '
        f'who really leave, the model catches about {metrics["recall"] * 100:.0f}. Precision is lower: '
        f'of every 100 customers it flags, about {metrics["precision"] * 100:.0f} really leave. '
        "We accept that trade-off because a missed churner is a lost customer, while a loyal customer "
        "flagged by mistake only costs a retention offer.</p>",
        unsafe_allow_html=True,
    )

    st.write("")
    a, b = st.columns(2, gap="medium")
    with a:
        fig = px.imshow(metrics["cm"], text_auto=True, aspect="auto",
                        color_continuous_scale=["#EEF2F8", PRIMARY],
                        x=["Predicted to stay", "Predicted to leave"],
                        y=["Actually stayed", "Actually left"])
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(textfont_size=18)
        chart_panel(
            "Right and wrong calls on the test customers",
            f"The diagonal cells are correct calls. The bottom-left cell holds the {fn} leavers the "
            f"model missed, which are the costly errors. The top-right cell holds {fp} loyal "
            "customers flagged by mistake.",
            style(fig, height=300))
    with b:
        top = importance.sort_values().tail(10)
        fig = go.Figure(go.Bar(
            x=top.values, y=[pretty(i) for i in top.index], orientation="h", marker_color=PRIMARY,
            hovertemplate="%{y}<br>Importance: %{x:.3f}<extra></extra>"))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(title=None)
        chart_panel(
            "The ten things the model relies on most",
            "A longer bar means the model leans on that factor more when it separates leavers from "
            "stayers. It shows what is linked to churn, not what causes it.",
            style(fig, height=300))

    section("How the model was built")
    two_column_list("Step", "What we did", [
        ("Data", "7,043 customers. The 11 blank TotalCharges values belong to brand-new customers and were set to 0."),
        ("Split", "80% of customers for training and 20% for testing, keeping the same share of leavers in both."),
        ("Preparation", "Numbers were scaled and categories were one-hot encoded inside one pipeline, so the app "
                        "prepares new customers in exactly the same way."),
        ("Model", "A Random Forest with balanced class weights, tuned with 5-fold GridSearchCV to maximise recall."),
    ])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
data = load_data()
pipeline, model_metrics, feature_importance = train_model()
customer_row = customer_form()

header(data, model_metrics)

tab1, tab2, tab3 = st.tabs(["Check a customer", "Why customers leave", "How the model performs"])
with tab1:
    predict_tab(pipeline, customer_row)
with tab2:
    insights_tab(data)
with tab3:
    model_tab(model_metrics, feature_importance)

st.markdown('<p class="foot">Built with Python, pandas, scikit-learn and Streamlit. '
            'Data: Telco Customer Churn (Kaggle).</p>', unsafe_allow_html=True)
