"""
app.py  —  Telco Customer Churn Predictor  |  5-Model Streamlit Dashboard
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a40);
        border-radius: 12px; padding: 20px; margin: 8px 0;
        border-left: 4px solid #7c3aed;
        box-shadow: 0 4px 15px rgba(124,58,237,0.2);
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #a78bfa; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; margin-top: 4px; }
    .risk-high   { color: #ef4444; font-weight: 700; font-size: 1.6rem; }
    .risk-medium { color: #f59e0b; font-weight: 700; font-size: 1.6rem; }
    .risk-low    { color: #10b981; font-weight: 700; font-size: 1.6rem; }
    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #c4b5fd;
        border-bottom: 2px solid #7c3aed; padding-bottom: 6px; margin: 20px 0 12px;
    }
    .model-badge {
        display: inline-block; padding: 3px 12px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; margin: 2px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed, #5b21b6);
        color: white; border-radius: 8px; border: none;
        padding: 10px 28px; font-weight: 600; font-size: 1rem;
    }
    div[data-testid="stSidebar"] { background: #161b2e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    return {
        "scaler":       joblib.load("models/scaler.pkl"),
        "imputer":      joblib.load("models/imputer.pkl"),
        "encoders":     joblib.load("models/encoders.pkl"),
        "best_model":   joblib.load("models/best_model.pkl"),
        "feature_names":joblib.load("models/feature_names.pkl"),
        "results":      joblib.load("models/results.pkl"),
        "fi":           joblib.load("models/feature_importance.pkl"),
        "best_name":    joblib.load("models/best_model_name.pkl"),
        "model_fi":     joblib.load("models/model_feature_importance.pkl"),
        # individual model pkls
        "models": {
            "XGBoost":              joblib.load("models/model_xgboost.pkl"),
            "Random Forest":        joblib.load("models/model_random_forest.pkl"),
            "Gradient Boosting":    joblib.load("models/model_gradient_boosting.pkl"),
            "Logistic Regression":  joblib.load("models/model_logistic_regression.pkl"),
            "SVM (RBF)":            joblib.load("models/model_svm_rbf.pkl"),
        }
    }

@st.cache_data
def load_data():
    df = pd.read_csv("TelcoCustomerChurn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"]  = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
    return df

art = load_artifacts()
scaler, imputer, encoders  = art["scaler"], art["imputer"], art["encoders"]
best_model, feature_names  = art["best_model"], art["feature_names"]
results, fi, best_name     = art["results"], art["fi"], art["best_name"]
model_fi, all_models       = art["model_fi"], art["models"]
df = load_data()

MODEL_COLORS = {
    "XGBoost":             "#f59e0b",
    "Random Forest":       "#10b981",
    "Gradient Boosting":   "#3b82f6",
    "Logistic Regression": "#a78bfa",
    "SVM (RBF)":           "#f472b6",
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Telco Churn App")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔍 Predict Churn", "📊 Model Performance", "🤖 Model Deep Dive", "📈 Data Explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**5 Models Trained**")
    for name, color in MODEL_COLORS.items():
        auc = results[name]["auc"]
        star = " ⭐" if name == best_name else ""
        st.markdown(
            f'<span style="color:{color}">●</span> `{name}` — AUC {auc:.4f}{star}',
            unsafe_allow_html=True
        )
    st.markdown("---")
    st.markdown(f"**Dataset:** `{len(df):,}` rows")
    st.markdown(f"**Churn Rate:** `{df['Churn_bin'].mean()*100:.1f}%`")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📡 Telco Customer Churn Intelligence")
    st.markdown("**5-model ensemble pipeline · Real-time inference · Joblib-persisted artifacts**")
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (f"{len(df):,}",                       "Total Customers"),
        (f"{df['Churn_bin'].sum():,}",          "Churned"),
        (f"{df['Churn_bin'].mean()*100:.1f}%",  "Churn Rate"),
        (f"{results[best_name]['auc']:.4f}",    f"Best AUC ({best_name[:4]}.)"),
        ("5",                                   "Models Trained"),
    ]
    for col, (val, label) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Model AUC comparison bar
    st.markdown('<div class="section-header">Model AUC-ROC Comparison</div>', unsafe_allow_html=True)
    auc_df = pd.DataFrame([
        {"Model": n, "AUC": results[n]["auc"], "AP": results[n]["ap"], "CV": results[n]["cv_mean"]}
        for n in results
    ]).sort_values("AUC", ascending=False)
    fig = go.Figure()
    for _, row in auc_df.iterrows():
        fig.add_trace(go.Bar(
            name=row["Model"], x=[row["Model"]],
            y=[row["AUC"]], marker_color=MODEL_COLORS[row["Model"]],
            text=[f'{row["AUC"]:.4f}'], textposition="outside",
        ))
    fig.add_hline(y=0.8, line_dash="dash", line_color="#6b7280",
                  annotation_text="AUC=0.8 threshold")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,20,35,0.8)",
        showlegend=False, height=320, yaxis=dict(range=[0.75, 0.9]),
        font=dict(color="#d1d5db"), margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Churn by Contract Type</div>', unsafe_allow_html=True)
        contract_churn = df.groupby("Contract")["Churn_bin"].mean().reset_index()
        contract_churn["ChurnRate"] = contract_churn["Churn_bin"] * 100
        fig = px.bar(contract_churn, x="Contract", y="ChurnRate",
                     color="ChurnRate", color_continuous_scale="Purples",
                     template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=300, showlegend=False, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Tenure Distribution by Churn</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="tenure", color="Churn", nbins=40, barmode="overlay",
                           opacity=0.75, color_discrete_map={"Yes":"#ef4444","No":"#10b981"},
                           template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Monthly Charges vs Churn</div>', unsafe_allow_html=True)
        fig = px.box(df, x="Churn", y="MonthlyCharges", color="Churn",
                     color_discrete_map={"Yes":"#ef4444","No":"#10b981"}, template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          showlegend=False, height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Churn by Internet Service</div>', unsafe_allow_html=True)
        inet_churn = df.groupby("InternetService")["Churn_bin"].mean().reset_index()
        inet_churn["ChurnRate"] = inet_churn["Churn_bin"] * 100
        fig = px.pie(inet_churn, values="ChurnRate", names="InternetService",
                     color_discrete_sequence=px.colors.sequential.Purples_r, template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Top 10 Churn Drivers (Best Model)</div>', unsafe_allow_html=True)
    top_fi = fi.head(10).reset_index()
    top_fi.columns = ["Feature", "Importance"]
    fig = px.bar(top_fi, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Purples", template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      yaxis=dict(autorange="reversed"), height=380, showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT CHURN
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict Churn":
    st.title("🔍 Real-Time Churn Risk Predictor")
    st.markdown("Score a customer against **all 5 models** simultaneously.")
    st.markdown("---")

    def preprocess_input(inputs):
        row = inputs.copy()
        total = row["tenure"] * row["MonthlyCharges"]
        row["TotalCharges"]      = total
        row["AvgMonthlyCharge"]  = total / (row["tenure"] + 1)
        tenure_bins = [0, 12, 24, 48, 72]
        row["TenureGroup"]       = int(np.digitize(row["tenure"], tenure_bins[1:]))
        svc_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
        row["MultipleServices"]  = sum(1 for c in svc_cols if row.get(c) == "Yes")
        for col, le in encoders.items():
            if col in row:
                val = str(row[col])
                row[col] = int(le.transform([val])[0]) if val in le.classes_ else 0
        arr = np.array([[row.get(f, 0) for f in feature_names]], dtype=float)
        arr = imputer.transform(arr)
        arr = scaler.transform(arr)
        return arr

    with st.form("predict_form"):
        st.markdown('<div class="section-header">📋 Customer Profile</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender          = st.selectbox("Gender",           ["Male","Female"])
            senior          = st.selectbox("Senior Citizen",   ["No","Yes"])
            partner         = st.selectbox("Partner",          ["Yes","No"])
            dependents      = st.selectbox("Dependents",       ["No","Yes"])
            tenure          = st.slider("Tenure (months)",     0, 72, 12)
        with c2:
            phone_service   = st.selectbox("Phone Service",    ["Yes","No"])
            multiple_lines  = st.selectbox("Multiple Lines",   ["No","Yes","No phone service"])
            internet_svc    = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
            online_security = st.selectbox("Online Security",  ["No","Yes","No internet service"])
            online_backup   = st.selectbox("Online Backup",    ["No","Yes","No internet service"])
        with c3:
            device_prot     = st.selectbox("Device Protection",["No","Yes","No internet service"])
            tech_support    = st.selectbox("Tech Support",     ["No","Yes","No internet service"])
            streaming_tv    = st.selectbox("Streaming TV",     ["No","Yes","No internet service"])
            streaming_movies= st.selectbox("Streaming Movies", ["No","Yes","No internet service"])

        st.markdown('<div class="section-header">💳 Billing & Contract</div>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1:
            contract        = st.selectbox("Contract",         ["Month-to-month","One year","Two year"])
        with b2:
            paperless       = st.selectbox("Paperless Billing",["Yes","No"])
            payment         = st.selectbox("Payment Method",   [
                "Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        with b3:
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)

        submitted = st.form_submit_button("⚡ Run All 5 Models", use_container_width=True)

    if submitted:
        inputs = {
            "gender":gender, "SeniorCitizen":1 if senior=="Yes" else 0,
            "Partner":partner, "Dependents":dependents, "tenure":tenure,
            "PhoneService":phone_service, "MultipleLines":multiple_lines,
            "InternetService":internet_svc, "OnlineSecurity":online_security,
            "OnlineBackup":online_backup, "DeviceProtection":device_prot,
            "TechSupport":tech_support, "StreamingTV":streaming_tv,
            "StreamingMovies":streaming_movies, "Contract":contract,
            "PaperlessBilling":paperless, "PaymentMethod":payment,
            "MonthlyCharges":monthly_charges,
        }
        X_input = preprocess_input(inputs)

        st.markdown("---")
        st.markdown('<div class="section-header">🎯 Predictions from All 5 Models</div>', unsafe_allow_html=True)

        probas = {}
        cols = st.columns(5)
        for (name, model), col in zip(all_models.items(), cols):
            p = model.predict_proba(X_input)[0][1]
            probas[name] = p
            color = "#ef4444" if p >= 0.7 else "#f59e0b" if p >= 0.4 else "#10b981"
            with col:
                st.markdown(f"""<div class="metric-card" style="border-left-color:{color}; text-align:center;">
                    <div class="metric-label">{name}</div>
                    <div style="font-size:1.8rem; font-weight:700; color:{color};">{p*100:.1f}%</div>
                    <div class="metric-label">{'⚠ CHURN' if p>=0.5 else '✅ STAY'}</div>
                </div>""", unsafe_allow_html=True)

        # Ensemble average
        ensemble_proba = np.mean(list(probas.values()))
        st.markdown("---")
        st.markdown('<div class="section-header">🧠 Ensemble Consensus</div>', unsafe_allow_html=True)

        ec1, ec2, ec3 = st.columns([1,2,1])
        with ec2:
            color = "#ef4444" if ensemble_proba>=0.7 else "#f59e0b" if ensemble_proba>=0.4 else "#10b981"
            risk  = "🔴 HIGH RISK" if ensemble_proba>=0.7 else "🟡 MEDIUM RISK" if ensemble_proba>=0.4 else "🟢 LOW RISK"
            st.markdown(f"""<div class="metric-card" style="text-align:center; border-left-color:{color};">
                <div class="metric-label">5-Model Average Probability</div>
                <div style="font-size:2.5rem; font-weight:700; color:{color};">{ensemble_proba*100:.1f}%</div>
                <div class="metric-label">{risk}</div>
            </div>""", unsafe_allow_html=True)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ensemble_proba * 100,
            number={"suffix":"%", "font":{"size":36,"color":"#a78bfa"}},
            gauge={
                "axis":  {"range":[0,100],"tickcolor":"#6b7280"},
                "bar":   {"color":"#7c3aed"},
                "steps": [{"range":[0,40],"color":"#064e3b"},
                           {"range":[40,70],"color":"#78350f"},
                           {"range":[70,100],"color":"#7f1d1d"}],
                "threshold":{"line":{"color":"white","width":3},"value":ensemble_proba*100},
            },
            title={"text":"Ensemble Churn Risk","font":{"color":"#c4b5fd","size":18}},
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color":"#d1d5db"},
                          height=320, margin=dict(t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Model agreement radar
        st.markdown('<div class="section-header">📡 Model Agreement Radar</div>', unsafe_allow_html=True)
        cats = list(probas.keys()) + [list(probas.keys())[0]]
        vals = list(probas.values()) + [list(probas.values())[0]]
        fig = go.Figure(go.Scatterpolar(
            r=[v*100 for v in vals], theta=cats, fill="toself",
            fillcolor="rgba(124,58,237,0.2)", line=dict(color="#7c3aed", width=2),
        ))
        fig.add_trace(go.Scatterpolar(
            r=[50]*len(cats), theta=cats, mode="lines",
            line=dict(color="#ef4444", dash="dash", width=1.5),
            name="50% threshold"
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#d1d5db"),
            height=400, showlegend=True, margin=dict(t=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Retention tips
        st.markdown('<div class="section-header">💡 Retention Recommendations</div>', unsafe_allow_html=True)
        tips = []
        if contract == "Month-to-month":
            tips.append("📄 Offer a discounted **annual or two-year contract**.")
        if internet_svc == "Fiber optic" and online_security == "No":
            tips.append("🔒 Bundle **Online Security** — Fiber users without it churn far more.")
        if tenure < 12:
            tips.append("🎁 Apply **new-customer loyalty discount** for year 1.")
        if monthly_charges > 80:
            tips.append("💰 Review billing — consider a **custom pricing plan**.")
        if payment == "Electronic check":
            tips.append("💳 Encourage switch to **auto-payment** — correlated with lower churn.")
        if not tips:
            tips.append("✅ Low-risk profile. Maintain regular engagement.")
        for tip in tips:
            st.markdown(f"- {tip}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Dashboard")
    st.markdown("All 5 models — ROC, PR, confusion matrix, score distributions.")
    st.markdown("---")

    # Leaderboard
    st.markdown('<div class="section-header">📋 Model Leaderboard</div>', unsafe_allow_html=True)
    rows = []
    for name, r in results.items():
        rep = r["report"]
        rows.append({
            "Model": name,
            "AUC-ROC":       f"{r['auc']:.4f}",
            "Avg Precision": f"{r['ap']:.4f}",
            "CV AUC":        f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}",
            "Accuracy":      f"{rep['accuracy']:.4f}",
            "F1 (Churn)":    f"{rep['1']['f1-score']:.4f}",
            "Recall (Churn)":f"{rep['1']['recall']:.4f}",
            "Best?":         "⭐" if name == best_name else "",
        })
    st.dataframe(pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False),
                 use_container_width=True, hide_index=True)

    # All ROC curves on one chart
    st.markdown('<div class="section-header">ROC Curves — All 5 Models</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for name, r in results.items():
        fig.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines",
            name=f'{name} ({r["auc"]:.4f})',
            line=dict(color=MODEL_COLORS[name], width=2.5)
        ))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                             line=dict(color="#4b5563", dash="dash")))
    fig.update_layout(xaxis_title="FPR", yaxis_title="TPR",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,20,35,0.8)",
                      height=420, font=dict(color="#d1d5db"), margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    # Per-model deep dive
    sel = st.selectbox("Select model for detailed analysis:", list(results.keys()))
    r   = results[sel]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = np.array(r["conf_matrix"])
        fig = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale="Purples",
                        labels=dict(x="Predicted", y="Actual"),
                        x=["No Churn","Churn"], y=["No Churn","Churn"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=350,
                          font=dict(color="#d1d5db"), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Score Distribution</div>', unsafe_allow_html=True)
        y_proba_arr = np.array(r["y_proba"])
        y_test_arr  = np.array(r["y_test"])
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=y_proba_arr[y_test_arr==0], name="No Churn",
                                   opacity=0.7, nbinsx=40, marker_color="#10b981"))
        fig.add_trace(go.Histogram(x=y_proba_arr[y_test_arr==1], name="Churn",
                                   opacity=0.7, nbinsx=40, marker_color="#ef4444"))
        fig.update_layout(barmode="overlay", xaxis_title="Predicted Probability",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,20,35,0.8)",
                          height=350, font=dict(color="#d1d5db"), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Precision-Recall Curve</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r["recall"], y=r["precision"], mode="lines",
                                 name=f'AP={r["ap"]:.4f}',
                                 line=dict(color=MODEL_COLORS[sel], width=2.5)))
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,20,35,0.8)",
                          height=350, font=dict(color="#d1d5db"), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">CV Score Distribution</div>', unsafe_allow_html=True)
        cv_data = pd.DataFrame({
            "Model": list(results.keys()),
            "CV AUC": [results[n]["cv_mean"] for n in results],
            "CV Std": [results[n]["cv_std"]  for n in results],
        })
        fig = go.Figure()
        for _, row in cv_data.iterrows():
            fig.add_trace(go.Bar(
                x=[row["Model"]], y=[row["CV AUC"]],
                error_y=dict(type="data", array=[row["CV Std"]], visible=True),
                marker_color=MODEL_COLORS[row["Model"]],
                name=row["Model"],
            ))
        fig.update_layout(showlegend=False, xaxis_tickangle=-20,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,20,35,0.8)",
                          height=350, yaxis=dict(range=[0.8, 0.96]),
                          font=dict(color="#d1d5db"), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL DEEP DIVE (SVM + Feature Importance per model)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Deep Dive":
    st.title("🤖 Model Deep Dive")
    st.markdown("Feature importance per model · SVM explained · Joblib artifact map")
    st.markdown("---")

    # ── SVM info panel ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🆕 5th Model: SVM (RBF Kernel)</div>', unsafe_allow_html=True)
    svm_r = results["SVM (RBF)"]
    s1, s2, s3, s4 = st.columns(4)
    for col, (label, val) in zip([s1,s2,s3,s4], [
        ("AUC-ROC",       f"{svm_r['auc']:.4f}"),
        ("Avg Precision", f"{svm_r['ap']:.4f}"),
        ("CV AUC",        f"{svm_r['cv_mean']:.4f}"),
        ("F1 (Churn)",    f"{svm_r['report']['1']['f1-score']:.4f}"),
    ]):
        with col:
            st.markdown(f"""<div class="metric-card" style="border-left-color:#f472b6;">
                <div class="metric-value" style="color:#f472b6;">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    with st.expander("📖 How SVM works in this pipeline", expanded=True):
        st.markdown("""
**Support Vector Machine (RBF kernel)** finds the optimal hyperplane separating churners from non-churners
in a high-dimensional feature space.

| Setting | Value | Why |
|---------|-------|-----|
| `kernel='rbf'` | Radial Basis Function | Captures non-linear churn boundaries |
| `C=1.0` | Regularisation | Balanced bias-variance trade-off |
| `gamma='scale'` | Auto = 1/(n_features × X.var()) | Robust default for scaled data |
| `probability=True` | Platt scaling enabled | Required for `predict_proba` & AUC-ROC |
| **Saved via** | `joblib.dump(svm_model, 'models/model_svm_rbf.pkl')` | Persisted independently |

**Why `probability=True` matters:**
SVM natively outputs a decision boundary distance, not a probability.
`probability=True` applies **Platt scaling** (logistic regression on the SVM scores)
so we get calibrated probabilities — essential for the gauge chart and ensemble averaging.

**Why SVM ranked 5th here:**
RBF SVM is slow on large balanced datasets (SMOTE doubled our training set to ~8,278 rows).
Its AUC of **0.8164** is still competitive — 5-fold CV shows it generalises well (0.8862).
        """)

    st.markdown("---")

    # ── Feature importance per model ─────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Importance — All Models Side by Side</div>',
                unsafe_allow_html=True)

    top_n = st.slider("Top N features to show", 5, 23, 10)
    fig = go.Figure()
    for name, fi_series in model_fi.items():
        top = fi_series.head(top_n)
        fig.add_trace(go.Bar(
            name=name, x=top.index.tolist(), y=top.values,
            marker_color=MODEL_COLORS[name], opacity=0.85,
        ))
    fig.update_layout(
        barmode="group", xaxis_tickangle=-35,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,20,35,0.8)",
        height=450, font=dict(color="#d1d5db"), margin=dict(t=10),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("*Note: SVM (RBF) uses Logistic Regression coefficients as a linear proxy for feature importance — RBF kernel has no native feature importance.*", unsafe_allow_html=False)

    # ── Joblib artifact map ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🗄️ Joblib Artifact Map</div>', unsafe_allow_html=True)
    import os
    artifact_data = []
    for f in sorted(os.listdir("models")):
        size_kb = os.path.getsize(f"models/{f}") / 1024
        role = {
            "best_model.pkl":              "Winning model (Logistic Regression) — used for single prediction",
            "best_model_name.pkl":         "String name of the best model",
            "scaler.pkl":                  "StandardScaler — applied to all input before inference",
            "imputer.pkl":                 "SimpleImputer (median) — handles missing values",
            "encoders.pkl":                "Dict of LabelEncoders for all categorical columns",
            "feature_names.pkl":           "Ordered list of 23 feature names",
            "results.pkl":                 "Full metrics dict for all 5 models (ROC, PR, CM, etc.)",
            "feature_importance.pkl":      "Top feature importances from best model",
            "model_feature_importance.pkl":"Per-model feature importance dict",
            "model_xgboost.pkl":           "XGBoost model — AUC 0.8369",
            "model_random_forest.pkl":     "Random Forest model — AUC 0.8409",
            "model_gradient_boosting.pkl": "Gradient Boosting model — AUC 0.8371",
            "model_logistic_regression.pkl":"Logistic Regression model — AUC 0.8443 ⭐",
            "model_svm_rbf.pkl":           "SVM RBF model — AUC 0.8164 (5th model)",
        }.get(f, "—")
        artifact_data.append({"File": f, "Size (KB)": f"{size_kb:.1f}", "Role": role})
    st.dataframe(pd.DataFrame(artifact_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Load Any Model (Code Example)</div>', unsafe_allow_html=True)
    st.code("""
import joblib
import numpy as np

# Load any individual model
svm_model = joblib.load("models/model_svm_rbf.pkl")
scaler    = joblib.load("models/scaler.pkl")
imputer   = joblib.load("models/imputer.pkl")

# Preprocess new input
X_new_scaled = scaler.transform(imputer.transform(X_new))

# Predict
proba = svm_model.predict_proba(X_new_scaled)[:, 1]
print(f"Churn probability: {proba[0]:.4f}")
    """, language="python")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Data Explorer":
    st.title("📈 Data Explorer")
    st.markdown("Explore the raw dataset with filters and interactive charts.")
    st.markdown("---")

    with st.expander("🔧 Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            churn_filter    = st.multiselect("Churn",            ["Yes","No"],               default=["Yes","No"])
        with fc2:
            contract_filter = st.multiselect("Contract",         df["Contract"].unique().tolist(), default=df["Contract"].unique().tolist())
        with fc3:
            inet_filter     = st.multiselect("Internet Service", df["InternetService"].unique().tolist(), default=df["InternetService"].unique().tolist())

    fdf = df[df["Churn"].isin(churn_filter) & df["Contract"].isin(contract_filter) & df["InternetService"].isin(inet_filter)]
    st.markdown(f"**Filtered rows:** `{len(fdf):,}`")

    st.markdown('<div class="section-header">Sample Records</div>', unsafe_allow_html=True)
    st.dataframe(fdf.drop(columns=["Churn_bin"]).head(200), use_container_width=True)

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    num_df = fdf.select_dtypes(include=np.number).drop(columns=["Churn_bin"], errors="ignore")
    fig = px.imshow(num_df.corr(), text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=500, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Tenure vs Monthly Charges</div>', unsafe_allow_html=True)
    fig = px.scatter(fdf, x="tenure", y="MonthlyCharges", color="Churn",
                     color_discrete_map={"Yes":"#ef4444","No":"#10b981"},
                     opacity=0.5, template="plotly_dark",
                     hover_data=["Contract","InternetService"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,20,35,0.8)",
                      height=400, margin=dict(t=10), font=dict(color="#d1d5db"))
    st.plotly_chart(fig, use_container_width=True)
