import numpy as np
import pandas as pd
from typing import Dict, List, Any

def get_global_feature_importance(wrapper) -> pd.DataFrame:
    """Returns sorted DataFrame of global feature importances for trained model wrapper."""
    pipeline = wrapper.pipeline
    classifier = wrapper.classifier
    feature_names = pipeline.feature_names
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'weights'):
        importances = np.abs(classifier.weights)
    else:
        importances = np.ones(len(feature_names)) / max(len(feature_names), 1)
        
    if len(importances) != len(feature_names):
        importances = np.resize(importances, len(feature_names))
        
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    return df_importance

def explain_applicant_risk(applicant_data: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Generates human-readable localized risk driver explanations for an individual applicant.
    Compares applicant parameters against standard credit underwriting benchmark thresholds.
    """
    row = applicant_data.iloc[0] if isinstance(applicant_data, pd.DataFrame) else applicant_data
    
    risk_drivers = []
    mitigating_factors = []
    
    loan_pct = float(row.get('loan_percent_income', 0.0))
    if loan_pct > 0.35:
        risk_drivers.append(f"High Loan-to-Income Ratio ({loan_pct:.0%}): Requested loan is a large proportion of annual income.")
    elif loan_pct <= 0.15:
        mitigating_factors.append(f"Conservative Loan-to-Income Ratio ({loan_pct:.0%}): Requested loan is easily manageable.")

    credit_score = int(row.get('credit_score', 650))
    if credit_score < 620:
        risk_drivers.append(f"Low Bureau Credit Score ({credit_score}): Bureau history shows elevated credit risk.")
    elif credit_score >= 740:
        mitigating_factors.append(f"Excellent Bureau Credit Score ({credit_score}): Strong credit history record.")

    prev_default = str(row.get('cb_person_default_on_file', 'N'))
    if prev_default == 'Y':
        risk_drivers.append("Prior Default History Recorded: Historical default/delinquency on file.")
    else:
        mitigating_factors.append("Clean Credit File: No prior defaults or delinquencies on record.")

    emp_length = float(row.get('person_emp_length', 0))
    if emp_length < 2:
        risk_drivers.append(f"Short Employment History ({emp_length} years): Limited job stability track record.")
    elif emp_length >= 5:
        mitigating_factors.append(f"Stable Employment History ({emp_length} years): Strong career stability.")

    int_rate = float(row.get('loan_int_rate', 10.0))
    if int_rate > 16.0:
        risk_drivers.append(f"High Interest Rate ({int_rate}%): Increased monthly debt service burden.")

    home_ownership = str(row.get('person_home_ownership', 'RENT'))
    if home_ownership == 'OWN':
        mitigating_factors.append("Homeownership Status: Owns property outright, lowering financial fragility.")
    elif home_ownership == 'RENT':
        risk_drivers.append("Renter Status: Ongoing rent payments increase net monthly expenditures.")

    return {
        'risk_drivers': risk_drivers if risk_drivers else ["No major high-risk indicators flagged."],
        'mitigating_factors': mitigating_factors if mitigating_factors else ["Standard risk profile."]
    }
