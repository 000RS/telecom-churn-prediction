import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from html import escape
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_models():
    return (joblib.load("model/churn_model.pkl"),
            joblib.load("model/improved_churn_model.pkl"))

baseline_model, improved_model = load_models()
model = improved_model

# Load dataset for dashboard analysis
@st.cache_data
def load_data():
    return pd.read_csv("data/telco_churn_clean.csv")

data = load_data()

# --------------------------------------------------

# --------------------------------------------------
# VISUAL STYLE ONLY: same analytics, softer palette
# --------------------------------------------------
SAGE = "#77A999"
BLUE = "#86ADC6"
INK = "#304E59"
MUTED = "#56727D"
GRID = "#DCE7E3"
PASTEL_CMAP = LinearSegmentedColormap.from_list(
    "pastel_palette", ["#F5FAF8", "#D9ECE5", "#93C4B4"]
)

st.markdown("""
<style>
/* Soft, readable and consistent light theme */
:root { color-scheme: light; }
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
 background: #F7FAF8 !important; color: #304E59 !important;
}
[data-testid="stHeader"] { background: rgba(247,250,248,.94) !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
 background: #EDF3F0 !important; border-right: 1px solid #DCE7E3 !important;
}
[data-testid="stSidebarUserContent"] { padding-top: clamp(100px, 22vh, 210px) !important; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 10px !important; }
[data-testid="stSidebar"] [role="radiogroup"] label {
 padding: 10px 12px !important; border-radius: 12px !important;
 background: #FAFCFB !important; border: 1px solid #D4E1DC !important;
 color: #304E59 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
 background: #CFE4DC !important; border-color: #8CB6A7 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #E0EFE8 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #304E59 !important; }
.block-container { padding-top: 1.4rem !important; padding-bottom: 2rem !important; max-width: 1230px !important; }
h1, h2, h3, h4 { color: #304E59 !important; font-weight: 680 !important; }
h1 {font-size: 1.95rem !important;}
h2 {font-size: 1.48rem !important;}
h3 {font-size: 1.12rem !important;}
[data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] p { color: #4E6973 !important; }
/* Crucial: form label visibility. Do not blanket-override all spans/buttons. */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stSelectbox"] label p,
[data-testid="stNumberInput"] label p {
 color: #304E59 !important; opacity: 1 !important;
 font-weight: 600 !important; font-size: .91rem !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
 background-color: #FFFFFF !important; color: #304E59 !important;
 -webkit-text-fill-color: #304E59 !important;
 border: 1px solid #C7D9D3 !important; border-radius: 9px !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] div { color: #304E59 !important; }
[data-testid="stNumberInput"] button { background: #EDF3F0 !important; color: #304E59 !important; }
[data-testid="stNumberInput"] button svg { fill: #304E59 !important; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #DCE7E3;
 border-radius: 13px; padding: 13px 15px; box-shadow: 0 3px 13px rgba(69,104,91,.055); }
[data-testid="stMetricLabel"] p { color: #56727D !important; }
[data-testid="stMetricValue"] { color: #304E59 !important; }
.stButton > button[kind="primary"] { background: #648E96 !important; color: #FFFFFF !important;
 border: none !important; border-radius: 10px !important; font-weight: 650 !important; }
.stButton > button[kind="primary"]:hover { background: #527B84 !important; color: #FFFFFF !important; }
[data-testid="stAlert"] { border-radius: 11px !important; }
hr {border-color: #DCE7E3 !important;}
/* Custom HTML tables, avoiding dataframe grid's black header defaults. */
.churn-table-wrap {overflow-x:auto; border:1px solid #DCE7E3; border-radius:12px;
 background:white; margin:8px 0 14px;}
.churn-table {width:100%; border-collapse:collapse; font-size: .91rem; color:#304E59;}
.churn-table th {background:#D8E8E5; color:#304E59; font-weight:700;
 padding:12px 10px; border-bottom:1px solid #C5D9D2; text-align:center;}
.churn-table td {background:#FFFFFF; color:#304E59;
 padding:12px 10px; text-align:center; border-bottom:1px solid #EBF0ED;}
.churn-table tr:last-child td {border-bottom:none;}
.churn-table td:first-child {font-weight:600; text-align:left;}
.churn-table th:first-child {text-align:left;}
.churn-table .shade-strong {background:#B8DCCE !important;}
.churn-table .shade-mid {background:#DDEDE7 !important;}
.churn-table .shade-light {background:#F1F7F4 !important;}


/* Unified light controls including open dropdown overlays and selected states */
.stApp, [data-testid="stAppViewContainer"] { background:#F7FAF8 !important; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display:none !important;}
[data-testid="stHeader"] {background:rgba(247,250,248,.96) !important;}
[data-testid="stWidgetLabel"] p, .stMarkdown p {color:#304E59 !important;}
[data-testid="stSelectbox"] > div, [data-testid="stNumberInput"] > div {color:#304E59 !important;}
[data-baseweb="popover"], [data-baseweb="popover"] > div,
[data-baseweb="menu"], [role="listbox"], [role="option"],
[data-baseweb="select"] [role="listbox"], [data-baseweb="select"] [role="option"] {
 background:#FFFFFF !important; color:#304E59 !important;
}
[data-baseweb="popover"] li, [role="option"] * {color:#304E59 !important;}
[role="option"]:hover, [role="option"][aria-selected="true"], [data-highlighted="true"] {
 background:#DDEDE7 !important; color:#304E59 !important;
}
[data-baseweb="select"] svg, [data-testid="stExpander"] svg {color:#50777B !important; fill:none !important;}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {
 background:#FFFFFF !important; color:#304E59 !important; border-color:#DCE7E3 !important;
}
[data-testid="stExpander"] summary:hover {background:#EFF5F2 !important;}
[data-testid="stExpander"] summary p { color:#304E59 !important; }
[data-testid="stNumberInput"] button, [data-testid="stNumberInput"] button:hover {
 background:#E5EFEA !important; color:#304E59 !important; border-color:#C7D9D3 !important;
}
[data-testid="stNumberInput"] button svg {fill:none !important; stroke:#304E59 !important;}
[data-testid="stElementToolbar"], [data-testid="stElementToolbarButtonContainer"],
[data-testid="stImageToolbar"], button[title="Fullscreen"] {display:none !important;}
[data-testid="stButton"] > button {border-radius:10px !important; border:1px solid #C7D9D3 !important;
 background:#EAF2EE !important; color:#304E59 !important; font-weight:650 !important;}
[data-testid="stButton"] > button:hover {background:#D7E8E1 !important; color:#304E59 !important;}
[data-testid="stButton"] > button[kind="primary"], [data-testid="stButton"] > button[data-testid="baseButton-primary"] {
 background:#547F89 !important; color:#FFFFFF !important; border-color:#547F89 !important;
}
[data-testid="stButton"] > button[kind="primary"] *, [data-testid="stButton"] > button[data-testid="baseButton-primary"] * {color:#FFFFFF !important;}
[data-testid="stButton"] > button[kind="primary"]:hover {background:#436C77 !important;}
[data-testid="stButton"] button:focus-visible {outline:2px solid #77A999 !important; outline-offset:2px;}
/* Navigation row remains immediately beneath KPIs */
.nav-label {font-size:.86rem; color:#56727D; font-weight:650; padding:.1rem 0 .25rem;}
.kpi-caption {color:#56727D; font-size:.85rem;}


/* Universal soft controls, including the plus/minus stepper and drop-down overlay */
[data-testid="stNumberInput"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] input {
    background: #FFFFFF !important;
    border-color: #C7D9D3 !important;
    color: #304E59 !important;
    box-shadow: none !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
    outline: 1px solid #77A999 !important;
    box-shadow: 0 0 0 2px rgba(119,169,153,.13) !important;
}
[data-testid="stNumberInput"] button,
[data-testid="stNumberInput"] button:hover {
    background: #E8F0ED !important;
    border-color: #C7D9D3 !important;
    color: #304E59 !important;
}
[data-testid="stNumberInput"] button svg {
    fill: none !important;
    stroke: #304E59 !important;
    stroke-width: 1.7px !important;
}
[role="listbox"], [data-baseweb="menu"], [data-baseweb="popover"],
[role="option"], [role="option"] * {
    background-color: #FFFFFF !important;
    color: #304E59 !important;
}
[role="option"]:hover, [role="option"][aria-selected="true"] {
    background-color: #DDEDE7 !important;
}
/* compact three-button navigation without decorative radio icons */
[data-testid="stButton"] button p {font-weight: 600 !important;}

</style>
""", unsafe_allow_html=True)


def chart_theme(ax):
    ax.set_facecolor("#FFFFFF")
    ax.figure.patch.set_facecolor("#FFFFFF")
    ax.tick_params(colors=INK, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def pastel_bar_chart(series, xlabel, ylabel, horizontal=False, format_kind="count", maximum=None):
    values = series.astype(float)
    h = max(3.0, len(values) * .42 + 1.1) if horizontal else 3.2
    fig, ax = plt.subplots(figsize=(9.6, h), dpi=115)
    chart_theme(ax)
    palette = [SAGE if i % 2 == 0 else BLUE for i in range(len(values))]
    if horizontal:
        bars = ax.barh([str(x) for x in values.index], values.values, color=palette, height=.63, zorder=3)
        ax.set_xlabel(ylabel, labelpad=10)
        ax.set_ylabel(xlabel, labelpad=10)
        ax.grid(axis="x", color=GRID, linewidth=.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(values.max()*1.24, .01))
        for b,v in zip(bars,values):
            ax.text(v + max(values.max()*.012,.001), b.get_y()+b.get_height()/2,
                    f"{v:.3f}", va="center", color=INK, fontsize=9)
    else:
        bars = ax.bar([str(x) for x in values.index], values.values, color=palette, width=.55, zorder=3)
        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_ylim(0, maximum if maximum else max(values.max()*1.2, 1))
        for b,v in zip(bars, values):
            txt = f"{v:.1f}%" if format_kind=="percent" else f"{int(v):,}"
            ax.annotate(txt, (b.get_x()+b.get_width()/2, v), xytext=(0, 6),
                        textcoords="offset points", ha="center", color=INK, fontsize=9)
    fig.tight_layout(pad=1.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_light_table(df, number_format=None, heatmap=False):
    """Render a static, readable light table regardless of Streamlit dataframe theme."""
    columns = [str(col) for col in df.columns]
    header = "".join(f"<th>{escape(c)}</th>" for c in columns)
    body = []
    numeric_values = pd.to_numeric(df.select_dtypes(include="number").stack(), errors="coerce")
    maximum = float(numeric_values.max()) if not numeric_values.empty else 0.0
    for _, row in df.iterrows():
        cells = []
        for col, val in zip(df.columns, row):
            shade = ""
            if heatmap and isinstance(val, (int, float)):
                ratio = (float(val) / maximum) if maximum else 0
                shade = "shade-strong" if ratio >= .7 else "shade-mid" if ratio >= .3 else "shade-light"
            if isinstance(val, (int, float)) and number_format:
                display = format(float(val), number_format)
            else:
                display = str(val)
            cells.append(f'<td class="{shade}">{escape(display)}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    markup = ('<div class="churn-table-wrap"><table class="churn-table">'
              '<thead><tr>' + header + '</tr></thead><tbody>'
              + ''.join(body) + '</tbody></table></div>')
    st.markdown(markup, unsafe_allow_html=True)


# MODEL PERFORMANCE EVALUATION

# Prepare the same test dataset used during training
X = data.drop(columns=["Churn"])
y = data["Churn"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load trained models
# Models already loaded and cached above.

# Calculate evaluation results automatically
def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    return metrics, matrix


baseline_metrics, baseline_cm = evaluate_model(
    baseline_model, X_test, y_test
)

improved_metrics, improved_cm = evaluate_model(
    improved_model, X_test, y_test
)

# --------------------------------------------------

# Single-frame dashboard: shared heading, live KPIs, then three page buttons.
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Business Intelligence Dashboard"

st.title("Telecom Customer Churn Prediction")
st.caption("Business Intelligence Dashboard · Customer trends, model evaluation and individual predictions")

total_customers = len(data)
total_churned = int(data["Churn"].sum())
churn_rate = (total_churned / total_customers) * 100
average_monthly_charge = data["MonthlyCharges"].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="small")
kpi1.metric("Total Customers", f"{total_customers:,}")
kpi2.metric("Customers Churned", f"{total_churned:,}")
kpi3.metric("Churn Rate", f"{churn_rate:.1f}%")
kpi4.metric("Average Monthly Charge", f"${average_monthly_charge:.2f}")

st.markdown('<div class="nav-label">EXPLORE THE DASHBOARD</div>', unsafe_allow_html=True)
nav_a, nav_b, nav_c = st.columns(3, gap="small")
page_names = ["Business Intelligence Dashboard", "Model Analysis", "Customer Prediction"]
for col, label, title in zip([nav_a,nav_b,nav_c], page_names,
                              ["Overview & Charts", "Model Analysis", "Customer Prediction"]):
    with col:
        selected = st.session_state["active_page"] == label
        if st.button(title,
                     key="go_"+label, use_container_width=True,
                     type="primary" if selected else "secondary"):
            st.session_state["active_page"] = label
            st.rerun()
page = st.session_state["active_page"]
st.divider()

if page == "Business Intelligence Dashboard":
    st.subheader("Customer Churn Overview")
    st.caption("The figures above describe the full cleaned dataset. The charts below show customer outcomes and contract patterns.")

    st.subheader("Churn Distribution")

    churn_counts = (
        data["Churn"]
        .map({0: "Stayed", 1: "Churned"})
        .value_counts()
    )

    pastel_bar_chart(churn_counts, "Customer status", "Number of customers", horizontal=False, format_kind="count")

    st.subheader("Churn Rate by Contract Type")

    contract_churn = (
        data.groupby("Contract")["Churn"]
        .mean()
        .mul(100)
        .round(1)
    )

    pastel_bar_chart(contract_churn, "Contract type", "Churn rate (%)", horizontal=False, format_kind="percent", maximum=100)

    st.divider()


elif page == "Model Analysis":
    # PERFORMANCE COMPARISON TABLE
    # --------------------------------------------------

    st.subheader("Model performance")

    performance = pd.DataFrame({
        "Metric": list(baseline_metrics.keys()),
        "Baseline Random Forest": list(baseline_metrics.values()),
        "Improved Random Forest": list(improved_metrics.values())
    })

    display_performance = performance.copy()
    display_performance["Baseline Random Forest"] = display_performance["Baseline Random Forest"].map(lambda x: f"{x:.2%}")
    display_performance["Improved Random Forest"] = display_performance["Improved Random Forest"].map(lambda x: f"{x:.2%}")
    show_light_table(display_performance)

    st.caption("Five evaluation metrics calculated from the same held-out test customers.")
    st.divider()

    st.subheader("From customer records to model outcomes")
    n_total = len(data)
    n_test = len(y_test)
    n_train = n_total - n_test
    actual_stayed = int((y_test == 0).sum())
    actual_churned = int((y_test == 1).sum())

    a,b,c = st.columns(3, gap="small")
    a.metric("Cleaned customers", f"{n_total:,}")
    b.metric("Training · {0:.1%}".format(n_train/n_total), f"{n_train:,}")
    c.metric("Testing · {0:.1%}".format(n_test/n_total), f"{n_test:,}")
    st.caption(
        f"Of {n_test:,} test customers, {actual_stayed:,} ({actual_stayed/n_test:.1%}) actually stayed "
        f"and {actual_churned:,} ({actual_churned/n_test:.1%}) actually churned. "
        "Both models were tested on these same customers."
    )

    st.subheader("What each model predicted")
    outcomes = pd.DataFrame([
        ["Churners correctly detected", f"{baseline_cm[1,1]:,} ({baseline_cm[1,1]/actual_churned:.1%})", f"{improved_cm[1,1]:,} ({improved_cm[1,1]/actual_churned:.1%})"],
        ["Churners missed", f"{baseline_cm[1,0]:,} ({baseline_cm[1,0]/actual_churned:.1%})", f"{improved_cm[1,0]:,} ({improved_cm[1,0]/actual_churned:.1%})"],
        ["Stayed but falsely flagged", f"{baseline_cm[0,1]:,} ({baseline_cm[0,1]/actual_stayed:.1%})", f"{improved_cm[0,1]:,} ({improved_cm[0,1]/actual_stayed:.1%})"],
        ["Correctly predicted to stay", f"{baseline_cm[0,0]:,} ({baseline_cm[0,0]/actual_stayed:.1%})", f"{improved_cm[0,0]:,} ({improved_cm[0,0]/actual_stayed:.1%})"],
    ], columns=["Actual test outcome", "Baseline model", "Improved model"])
    show_light_table(outcomes)
    st.caption(
        f"Churner percentages are out of {actual_churned:,} actual churners. "
        f"Stayer percentages are out of {actual_stayed:,} actual stayers."
    )
    recall_change = (improved_metrics["Recall"] - baseline_metrics["Recall"]) * 100
    st.info(
        f"The improved model identified {int(improved_cm[1,1]):,} of {actual_churned:,} churners "
        f"(vs {int(baseline_cm[1,1]):,} for baseline). "
        f"Recall improved by {recall_change:.2f} percentage points, "
        "but false alerts increased. This trade-off matters when selecting customers for retention outreach."
    )
    st.divider()

    # --------------------------------------------------
    # FEATURE IMPORTANCE ANALYSIS
    # --------------------------------------------------

    st.subheader("Feature Importance Analysis")

    preprocessor = improved_model.named_steps["preprocessor"]
    classifier = improved_model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importance_scores = classifier.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_scores
    })

    feature_importance["Feature"] = (
        feature_importance["Feature"]
        .str.replace(r"^(cat__|remainder__)", "", regex=True)
    )

    top_features = (
        feature_importance
        .sort_values("Importance", ascending=False)
        .head(10)
    )

    st.caption("Top 10 predictive features from the improved model, sorted by importance. Bigger scores indicate more model influence, not a customer churn probability.")
    pastel_bar_chart(top_features.set_index("Feature")["Importance"].sort_values(), "Feature", "Relative importance score", horizontal=True, format_kind="score")

    st.caption(
        "Importance scores are calculated automatically from "
        "the trained Random Forest model. They show predictive "
        "importance, not causal relationships."
    )

    # --------------------------------------------------

else:  # Customer Prediction
    st.subheader("Customer Churn Prediction")
    st.caption("Adjust customer services to explore a churn prediction. Billing estimates are illustrative, not official Telco pricing.")

    # Example tariffs are used ONLY to make the customer form interactive.
    # They are not learned prices, contractual prices, or statements about the IBM dataset.
    PRICE = {
        "phone": 18.0, "multiple": 7.0, "DSL": 26.0, "Fiber optic": 45.0,
        "OnlineSecurity": 6.0, "OnlineBackup": 5.0,
        "DeviceProtection": 6.0, "TechSupport": 7.0,
        "StreamingTV": 9.0, "StreamingMovies": 9.0,
    }
    def estimated_bill():
        v = 0.0
        if st.session_state.get("phone_service", "Yes") == "Yes":
            v += PRICE["phone"]
            if st.session_state.get("multiple_lines") == "Yes":
                v += PRICE["multiple"]
        internet = st.session_state.get("internet_service", "DSL")
        v += PRICE.get(internet, 0.0)
        if internet != "No":
            for field in ("OnlineSecurity", "OnlineBackup", "DeviceProtection",
                          "TechSupport", "StreamingTV", "StreamingMovies"):
                if st.session_state.get(field) == "Yes":
                    v += PRICE[field]
        return round(v, 2)

    def refresh_bill():
        # Streamlit on_change callbacks execute before the next app rerun.
        # Match dependent services automatically, so invalid combinations are avoided.
        if st.session_state.get("phone_service") == "No":
            st.session_state["multiple_lines"] = "No phone service"
        elif st.session_state.get("multiple_lines") == "No phone service":
            st.session_state["multiple_lines"] = "No"
        if st.session_state.get("internet_service") == "No":
            for field in ("OnlineSecurity", "OnlineBackup", "DeviceProtection",
                          "TechSupport", "StreamingTV", "StreamingMovies"):
                st.session_state[field] = "No internet service"
        else:
            for field in ("OnlineSecurity", "OnlineBackup", "DeviceProtection",
                          "TechSupport", "StreamingTV", "StreamingMovies"):
                if st.session_state.get(field) == "No internet service":
                    st.session_state[field] = "No"
        st.session_state["monthly_input"] = estimated_bill()
        if st.session_state.get("auto_total", True):
            st.session_state["total_input"] = round(
                st.session_state["monthly_input"] * st.session_state.get("tenure", 12), 2
            )

    def sync_total():
        if st.session_state.get("auto_total", True):
            st.session_state["total_input"] = round(
                float(st.session_state.get("monthly_input", 0)) * int(st.session_state.get("tenure", 12)), 2
            )

    def set_manual_monthly():
        sync_total()

    defaults = {
        "gender": "Male", "senior": "No", "partner": "No", "dependents": "No",
        "tenure": 12, "phone_service": "Yes", "multiple_lines": "No",
        "internet_service": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "contract": "Month-to-month", "paperless_billing": "Yes",
        "payment_method": "Electronic check", "auto_total": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "monthly_input" not in st.session_state:
        st.session_state["monthly_input"] = estimated_bill()
    if "total_input" not in st.session_state:
        st.session_state["total_input"] = round(st.session_state["monthly_input"] * st.session_state["tenure"], 2)

    col1,col2,col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown("#### Customer profile")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        senior = st.selectbox("Senior citizen", ["No", "Yes"], key="senior")
        partner = st.selectbox("Partner", ["No", "Yes"], key="partner")
        dependents = st.selectbox("Dependents", ["No", "Yes"], key="dependents")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72,
                                 step=1, key="tenure", on_change=sync_total)
        phone_service = st.selectbox("Phone service", ["Yes", "No"], key="phone_service", on_change=refresh_bill)
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"],
                                      key="multiple_lines", on_change=refresh_bill)
    with col2:
        st.markdown("#### Service information")
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"],
                                        key="internet_service", on_change=refresh_bill)
        online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"],
                                       key="OnlineSecurity", on_change=refresh_bill)
        online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"],
                                     key="OnlineBackup", on_change=refresh_bill)
        device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"],
                                         key="DeviceProtection", on_change=refresh_bill)
        tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"],
                                    key="TechSupport", on_change=refresh_bill)
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"],
                                    key="StreamingTV", on_change=refresh_bill)
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"],
                                        key="StreamingMovies", on_change=refresh_bill)
    with col3:
        st.markdown("#### Contract & billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"], key="paperless_billing")
        payment_method = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                "Bank transfer (automatic)", "Credit card (automatic)"], key="payment_method")
        monthly_charges = st.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=10000.0, step=5.0,
            format="%.2f", key="monthly_input", on_change=set_manual_monthly,
            help="Editable: type any value, or use +/- in increments of $5. Service changes recalculate the illustrative estimate."
        )
        st.checkbox("Update total charges automatically", key="auto_total", on_change=sync_total)
        total_charges = st.number_input(
            "Total charges ($)", min_value=0.0, max_value=1000000.0, step=25.0,
            format="%.2f", key="total_input",
            help="Editable: type any value, or use +/- in increments of $25. Turn off automatic updates to keep a custom historical total."
        )
        st.caption("Illustrative monthly estimate: phone $18, DSL $26, fibre $45, and selected add-ons $5–$9. These are demo assumptions, not dataset tariffs.")

    st.divider()

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------



    st.subheader("Churn Risk Assessment")
    st.caption("Your prediction will appear below after you enter valid customer information.")

    predict_clicked = st.button("Predict Churn Risk", type="primary", use_container_width=True)
    if not predict_clicked:
        st.info("Select **Predict Churn Risk** to see the estimated risk level and retention recommendation.")

    if predict_clicked:

        customer = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [1 if senior == "Yes" else 0],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        })

            # Validate customer service combinations
        internet_features = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        if internet_service == "No":
            if any(
                customer[feature].iloc[0] != "No internet service"
                for feature in internet_features
            ):
                st.error(
                    "Please select 'No internet service' for all "
                    "internet-related features."
                )
                st.stop()

        elif any(
            customer[feature].iloc[0] == "No internet service"
            for feature in internet_features
        ):
            st.error(
                "Customers with internet service must select "
                "'Yes' or 'No' for internet-related features."
            )
            st.stop()

        if phone_service == "No" and multiple_lines != "No phone service":
            st.error(
                "Select 'No phone service' under Multiple Lines."
            )
            st.stop()

        if phone_service == "Yes" and multiple_lines == "No phone service":
            st.error(
                "Select 'Yes' or 'No' under Multiple Lines."
            )
            st.stop()

        probability = model.predict_proba(customer)[0][1]

        st.subheader("Prediction Result")

        st.metric(
            "Churn Probability",
            f"{probability * 100:.1f}%"
        )

        if probability >= 0.60:
            st.error("🔴 High Churn Risk")

            st.write(
                "This customer should be prioritised for retention action."
            )

        elif probability >= 0.30:
            st.warning("🟠 Medium Churn Risk")

            st.write(
                "This customer may require monitoring or a targeted retention offer."
            )

        else:
            st.success("🟢 Low Churn Risk")

            st.write(
                "This customer currently shows a relatively low risk of churn."
            )

        st.subheader("Recommended Retention Actions")

        if probability >= 0.60:
            st.write(
                "Priority: High. Consider contacting this customer "
                "to understand their concerns and reviewing suitable "
                "retention offers."
            )

        elif probability >= 0.30:
            st.write(
                "Priority: Medium. Consider reviewing customer satisfaction, "
                "service usage, and available loyalty benefits."
            )

        else:
            st.write(
                "Priority: Low. Continue normal customer engagement "
                "and monitor for changes in service needs."
            )

        st.caption(
            "Recommendations are illustrative business rules based on "
            "predicted risk, not independently validated retention strategies."
        )    
