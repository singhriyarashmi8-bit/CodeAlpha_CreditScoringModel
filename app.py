import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Credit Scoring & Creditworthiness Prediction",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

if os.path.exists("credit_model.pkl"):
    model = joblib.load("credit_model.pkl")
elif os.path.exists("credit_model (2).pkl"):
    model = joblib.load("credit_model (2).pkl")
else:
    st.error(
        "❌ Model file not found. Please keep credit_model.pkl "
        "or credit_model (2).pkl in the same folder as app.py."
    )
    st.stop()


# =========================================================
# ROYAL PURPLE DESIGN
# =========================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #090512 0%, #170b2d 45%, #2a0d45 100%);
    color: white;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(135deg, #261044, #54208a, #7b2cbf);
    padding: 32px;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 15px 50px rgba(0,0,0,0.45);
    margin-bottom: 35px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    font-size: 17px;
    color: #dfd0ef;
    margin-top: 8px;
}

.section-title {
    font-size: 26px;
    font-weight: 750;
    color: white;
    margin-bottom: 5px;
}

.section-text {
    color: #aa9ab9;
    margin-bottom: 22px;
}

.info-card {
    background: rgba(30,17,49,0.85);
    border: 1px solid rgba(183,135,255,0.20);
    border-radius: 18px;
    padding: 22px;
    margin-top: 20px;
}

.metric-card {
    background: rgba(18,10,30,0.9);
    border: 1px solid rgba(183,135,255,0.18);
    border-radius: 15px;
    padding: 18px;
    text-align: center;
}

.metric-label {
    color: #aa9ab9;
    font-size: 13px;
}

.metric-value {
    color: white;
    font-size: 25px;
    font-weight: 800;
    margin-top: 5px;
}

.approved {
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(52,211,153,0.35);
    color: #6ee7b7;
    padding: 18px;
    border-radius: 15px;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 15px;
}

.risk {
    background: rgba(168,85,247,0.15);
    border: 1px solid rgba(192,132,252,0.35);
    color: #d8b4fe;
    padding: 18px;
    border-radius: 15px;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 15px;
}

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    padding: 13px;
    font-size: 16px;
    font-weight: 700;
    color: white;
    background: linear-gradient(135deg, #6d28d9, #9333ea, #c026d3);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #7c3aed, #a855f7, #d946ef);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">💳 Credit Scoring & Creditworthiness Prediction</div>
    <div class="hero-subtitle">Machine Learning Based Credit Risk Assessment</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# TWO COLUMNS
# =========================================================

left, right = st.columns([1, 1], gap="large")


# =========================================================
# LEFT SIDE
# =========================================================

with left:
    st.markdown('<div class="section-title">👤 Applicant Financial Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-text">Enter the applicant information to evaluate creditworthiness.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input(
            "Annual Income (₹)",
            min_value=15000.0,
            max_value=200000.0,
            value=60000.0,
            step=1000.0
        )

        total_debt = st.number_input(
            "Total Debt (₹)",
            min_value=1000.0,
            max_value=120000.0,
            value=30000.0,
            step=1000.0
        )

    with col2:
        age = st.number_input(
            "Applicant Age",
            min_value=21,
            max_value=65,
            value=35,
            step=1
        )

        num_accounts = st.number_input(
            "Open Credit Accounts",
            min_value=1,
            max_value=15,
            value=6,
            step=1
        )

    payment_score = st.slider(
        "Payment History Score",
        min_value=300,
        max_value=850,
        value=680,
        step=1
    )


    # =====================================================
    # ENGINEERED FEATURES
    # =====================================================

    debt_to_income = total_debt / income
    score_to_debt = payment_score / (total_debt + 1)

    st.markdown('<div class="info-card"><b>⚡ Engineered Features (Auto-computed)</b></div>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)

    with f1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Debt-to-Income (DTI)</div>
                <div class="metric-value">{debt_to_income:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Score-to-Debt Ratio</div>
                <div class="metric-value">{score_to_debt:.6f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    evaluate = st.button("🚀 Evaluate Creditworthiness")


# =========================================================
# RIGHT SIDE
# =========================================================

with right:
    st.markdown('<div class="section-title">📊 Decision & Risk Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-text">AI-powered assessment generated from the trained ML model.</div>', unsafe_allow_html=True)

    # =====================================================
    # BEFORE PREDICTION
    # =====================================================

    if not evaluate:
        st.markdown(
            """
            <div class="info-card" style="text-align:center; padding:70px 20px;">
                <div style="font-size:60px;">🔮</div>
                <h2 style="color:white;">Ready for Assessment</h2>
                <p style="color:#aa9ab9;">Enter applicant information and click <b>Evaluate Creditworthiness</b>.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    if evaluate:
        applicant_data = pd.DataFrame({
            "Income": [income],
            "Total_Debt": [total_debt],
            "Payment_History_Score": [payment_score],
            "Age": [age],
            "Num_Open_Accounts": [num_accounts],
            "Debt_to_Income": [debt_to_income],
            "Score_to_Debt": [score_to_debt]
        })

        prediction = model.predict(applicant_data)[0]
        probabilities = model.predict_proba(applicant_data)[0]

        higher_risk_probability = probabilities[0] * 100
        creditworthy_probability = probabilities[1] * 100

        if creditworthy_probability >= 70:
            risk_level = "Low Financial Credit Risk"
        elif creditworthy_probability >= 50:
            risk_level = "Moderate Financial Credit Risk"
        else:
            risk_level = "Higher Financial Credit Risk"

        if prediction == 1:
            st.markdown(
                f"""
                <div class="approved">
                    🎉 STATUS: CREDITWORTHY — {risk_level}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="risk">
                    ⚠️️ STATUS: HIGHER RISK — {risk_level}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Gauge chart
        meter = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=creditworthy_probability,
                title={"text": "Creditworthiness Confidence", "font": {"size": 20, "color": "white"}},
                number={"suffix": "%", "font": {"size": 42, "color": "white"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#bda8d5"},
                    "bar": {"color": "#a855f7", "thickness": 0.35},
                    "bgcolor": "#21152f",
                    "borderwidth": 2,
                    "bordercolor": "#68458a",
                    "steps": [
                        {"range": [0, 40], "color": "#321d42"},
                        {"range": [40, 70], "color": "#492460"},
                        {"range": [70, 100], "color": "#244b3c"}
                    ],
                    "threshold": {
                        "line": {"color": "#f0abfc", "width": 5},
                        "thickness": 0.8,
                        "value": creditworthy_probability
                    }
                }
            )
        )

        meter.update_layout(
            height=350,
            margin={"l": 20, "r": 20, "t": 60, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(meter, use_container_width=True, config={"displayModeBar": False})

        p1, p2 = st.columns(2)

        with p1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Creditworthy Probability</div>
                    <div class="metric-value">{creditworthy_probability:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with p2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Higher Risk Probability</div>
                    <div class="metric-value">{higher_risk_probability:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# FINANCIAL SNAPSHOT
# =========================================================

if evaluate:
    st.write("")
    st.write("")

    st.markdown('<div class="section-title">💰 Financial Snapshot</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Annual Income</div>
                <div class="metric-value">₹{income:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Debt</div>
                <div class="metric-value">₹{total_debt:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Debt-to-Income</div>
                <div class="metric-value">{debt_to_income:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Payment Score</div>
                <div class="metric-value">{payment_score}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="text-align:center; margin-top:50px; padding:20px; color:#75677f; border-top:1px solid rgba(183,135,255,0.15);">
        💳 Credit Scoring & Creditworthiness Prediction<br>
        Machine Learning Based Credit Risk Assessment<br><br>
        Built with Python • Scikit-learn • Streamlit • Plotly
    </div>
    """,
    unsafe_allow_html=True
)