import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path

from src.data_loader import clean_data
from src.scorecard import CreditScorecardEngine
from src.explainability import explain_applicant_risk, get_global_feature_importance

# Page Config
st.set_page_config(
    page_title="AI Credit Risk Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .status-approve {
        background-color: #d4edda;
        color: #155724;
        padding: 10px 15px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    .status-review {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px 15px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    .status-decline {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px 15px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_artifact():
    artifact_path = Path(__file__).parent / "models" / "credit_risk_model.joblib"
    if not artifact_path.exists():
        st.warning("Model artifact not found. Training model now...")
        from train_model import main as train_main
        train_main()
    return joblib.load(artifact_path)

def create_score_gauge(score: int):
    """Generates Plotly gauge chart for Credit Score visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Calculated Credit Score", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [300, 620], 'color': '#ff4d4d'},
                {'range': [620, 720], 'color': '#ffcc00'},
                {'range': [720, 850], 'color': '#2ed573'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Main Header
st.title("💳 AI-Based Credit Risk Prediction & Underwriting System")
st.markdown("Automated loan default risk assessment, scorecards, and Explainable AI (XAI) risk insights.")

# Load Model Artifact
try:
    artifact = load_model_artifact()
    model_wrapper = artifact['wrapper']
    scorecard = artifact['scorecard_engine']
    model_name = artifact['model_name']
    evaluations = artifact['evaluations']
except Exception as e:
    st.error(f"Failed to load credit risk model: {e}")
    st.stop()

# Sidebar
st.sidebar.header("⚡ System Parameters")
st.sidebar.info(
    f"**Active Model**: {model_name}\n\n"
    f"**ROC-AUC**: {evaluations[model_name]['roc_auc']:.4f}\n\n"
    f"**Gini Index**: {evaluations[model_name]['gini']:.4f}\n\n"
    f"**KS Statistic**: {evaluations[model_name]['ks_statistic']:.4f}"
)

tabs = st.tabs(["🎯 Single Applicant Assessment", "📁 Batch Portfolio Processing", "📊 Model Analytics & Explainability"])

# TAB 1: Single Applicant Assessment
with tabs[0]:
    st.subheader("📋 Loan Applicant Financial Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Applicant Age", min_value=18, max_value=100, value=34)
        income = st.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=65000, step=5000)
        emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=50.0, value=4.5, step=0.5)
        home_ownership = st.selectbox("Homeownership Status", ["RENT", "MORTGAGE", "OWN", "OTHER"])

    with col2:
        loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=200000, value=15000, step=1000)
        loan_intent = st.selectbox("Loan Purpose / Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        int_rate = st.number_input("Interest Rate (%)", min_value=3.0, max_value=30.0, value=11.5, step=0.25)
        loan_grade = st.selectbox("Assigned Grade", ["A", "B", "C", "D", "E", "F"])

    with col3:
        cred_hist_len = st.number_input("Credit History Length (Years)", min_value=1, max_value=50, value=8)
        prev_default = st.selectbox("Prior Default on File?", ["N", "Y"])
        credit_score_input = st.number_input("Bureau Credit Score (FICO)", min_value=300, max_value=850, value=680)
        
    loan_pct_income = round(loan_amnt / max(income, 1.0), 2)
    st.caption(f"💡 Calculated Loan-to-Income Ratio: **{loan_pct_income:.0%}**")

    if st.button("🚀 Evaluate Credit Risk", type="primary", use_container_width=True):
        applicant_df = pd.DataFrame([{
            'person_age': age,
            'person_income': income,
            'person_home_ownership': home_ownership,
            'person_emp_length': emp_length,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt,
            'loan_int_rate': int_rate,
            'loan_percent_income': loan_pct_income,
            'cb_person_default_on_file': prev_default,
            'cb_person_cred_hist_length': cred_hist_len,
            'credit_score': credit_score_input
        }])
        
        # Predict
        prob_default = model_wrapper.predict_proba(applicant_df)[0, 1]
        eval_result = scorecard.evaluate_applicant(prob_default)
        
        st.markdown("---")
        st.subheader("📈 Underwriting Risk Decision Output")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.plotly_chart(create_score_gauge(eval_result['credit_score']), use_container_width=True)
            
            rec = eval_result['recommendation']
            if rec == 'APPROVE':
                st.markdown(f"<div class='status-approve'>✅ RECOMMENDATION: {rec}</div>", unsafe_allow_html=True)
            elif rec == 'MANUAL REVIEW':
                st.markdown(f"<div class='status-review'>⚠️ RECOMMENDATION: {rec}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='status-decline'>❌ RECOMMENDATION: {rec}</div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("### Metrics Summary")
            st.metric("Predicted Default Probability", eval_result['default_probability_percent'])
            st.metric("Risk Classification Tier", eval_result['risk_tier'])
            st.metric("Recommended Interest Pricing", eval_result['suggested_pricing_tier'])
            st.info(f"**Underwriting Rationale**: {eval_result['explanation']}")

        # Local Risk Factors Explanation
        st.subheader("🔍 Explainable AI (XAI) Risk Factors")
        explanations = explain_applicant_risk(applicant_df)
        
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("#### 🚨 Key Risk Drivers")
            for item in explanations['risk_drivers']:
                st.write(f"- {item}")
                
        with exp_col2:
            st.markdown("#### 🛡️ Risk Mitigating Factors")
            for item in explanations['mitigating_factors']:
                st.write(f"- {item}")

# TAB 2: Batch Portfolio Processing
with tabs[1]:
    st.subheader("📁 Bulk Credit Applicant Evaluation")
    st.markdown("Upload a CSV file containing multiple loan applicants to calculate scores and batch underwriting decisions.")
    
    uploaded_file = st.file_uploader("Upload Applicants CSV", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(batch_df)}** records from CSV.")
        
        if st.button("⚡ Process Batch Portfolio", type="primary"):
            cleaned_batch = clean_data(batch_df)
            probas = model_wrapper.predict_proba(cleaned_batch)[:, 1]
            
            scores = scorecard.probability_to_score(probas)
            results = [scorecard.get_risk_tier_and_decision(s, p) for s, p in zip(scores, probas)]
            
            res_df = batch_df.copy()
            res_df['predicted_default_prob'] = [r['default_probability'] for r in results]
            res_df['credit_score'] = scores
            res_df['risk_tier'] = [r['risk_tier'] for r in results]
            res_df['recommendation'] = [r['recommendation'] for r in results]
            
            st.success("Batch processing complete!")
            st.dataframe(res_df.head(20), use_container_width=True)
            
            # Risk Breakdown Chart
            fig_pie = px.pie(res_df, names='recommendation', title='Portfolio Recommendation Breakdown',
                             color='recommendation',
                             color_discrete_map={'APPROVE': '#2ed573', 'MANUAL REVIEW': '#ffcc00', 'DECLINE': '#ff4d4d'})
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Download CSV
            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Underwriting Decisions CSV",
                data=csv_data,
                file_name="credit_risk_underwriting_decisions.csv",
                mime="text/csv"
            )

# TAB 3: Model Analytics & Explainability
with tabs[2]:
    st.subheader("📊 Model Performance & Feature Importance Benchmarking")
    
    eval_df = pd.DataFrame(evaluations).T.reset_index()
    eval_df = eval_df.rename(columns={'index': 'Model Name'})
    st.dataframe(eval_df[['Model Name', 'roc_auc', 'gini', 'ks_statistic', 'accuracy', 'f1_score']], use_container_width=True)
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.subheader("🏆 Global Feature Importances")
        df_imp = get_global_feature_importance(model_wrapper)
        fig_bar = px.bar(df_imp.head(12), x='importance', y='feature', orientation='h',
                         title=f"Top 12 Features ({model_name})",
                         labels={'importance': 'Relative Importance / Weight', 'feature': 'Feature Name'})
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_feat2:
        st.subheader("📐 Model Metrics Comparison")
        fig_metrics = px.bar(eval_df, x='Model Name', y=['roc_auc', 'gini', 'ks_statistic'], barmode='group',
                             title="ROC-AUC vs Gini vs KS-Statistic")
        st.plotly_chart(fig_metrics, use_container_width=True)
