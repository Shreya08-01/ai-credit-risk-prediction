import numpy as np
import pandas as pd
from pathlib import Path

def generate_synthetic_credit_data(num_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic credit risk dataset with financial attributes
    and non-linear relationships to loan default targets.
    """
    np.random.seed(random_state)
    
    # Demographics & Income
    age = np.random.randint(21, 68, size=num_samples)
    income = np.random.lognormal(mean=10.8, sigma=0.6, size=num_samples).astype(int)
    income = np.clip(income, 18000, 250000)
    
    emp_length = np.clip((age - 18) * np.random.uniform(0.1, 0.8, size=num_samples), 0, 40).round(1)
    
    home_ownership_choices = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
    home_ownership_probs = [0.45, 0.45, 0.08, 0.02]
    home_ownership = np.random.choice(home_ownership_choices, size=num_samples, p=home_ownership_probs)
    
    # Loan parameters
    loan_intents = ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
    loan_intent = np.random.choice(loan_intents, size=num_samples)
    
    # Loan Amount dependent on income
    max_loan_capacity = income * np.random.uniform(0.1, 0.65, size=num_samples)
    loan_amnt = np.clip(max_loan_capacity * np.random.uniform(0.3, 1.0, size=num_samples), 1000, 40000).astype(int)
    loan_percent_income = np.round(loan_amnt / income, 2)
    
    # Credit History & Bureau Score
    cred_hist_len = np.clip((age - 18) * np.random.uniform(0.2, 0.7, size=num_samples), 1, 30).astype(int)
    
    # Historical default flag
    prev_default_prob = 0.15 + (0.1 if 'RENT' in home_ownership else 0.0)
    cb_default = np.random.choice(['N', 'Y'], size=num_samples, p=[0.82, 0.18])
    
    # Credit score generation (300 to 850)
    base_score = 680 + (income / 10000) * 4 + (emp_length * 2) - (cb_default == 'Y') * 70 - (loan_percent_income * 80)
    credit_score = np.clip(base_score + np.random.normal(0, 35, size=num_samples), 350, 850).astype(int)
    
    # Loan Interest Rate & Grade based on credit score & risk
    base_int_rate = 22.0 - (credit_score - 300) * (17.0 / 550)
    loan_int_rate = np.clip(base_int_rate + np.random.normal(0, 1.5, size=num_samples), 5.0, 26.0).round(2)
    
    # Assign Loan Grade
    def assign_grade(score):
        if score >= 750: return 'A'
        elif score >= 700: return 'B'
        elif score >= 650: return 'C'
        elif score >= 600: return 'D'
        elif score >= 550: return 'E'
        else: return 'F'
        
    loan_grade = np.array([assign_grade(s) for s in credit_score])
    
    # Ground Truth Default Probability calculation (log-odds model)
    log_odds = (
        -1.8
        + 3.2 * loan_percent_income
        + 0.12 * (loan_int_rate - 10.0)
        - 0.008 * (credit_score - 600)
        - 0.04 * emp_length
        + 0.8 * (cb_default == 'Y').astype(int)
        + 0.5 * (home_ownership == 'RENT').astype(int)
        - 0.4 * (home_ownership == 'OWN').astype(int)
        + np.random.normal(0, 0.5, size=num_samples)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    loan_status = (prob_default > 0.48).astype(int)
    
    df = pd.DataFrame({
        'person_age': age,
        'person_income': income,
        'person_home_ownership': home_ownership,
        'person_emp_length': emp_length,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': cb_default,
        'cb_person_cred_hist_length': cred_hist_len,
        'credit_score': credit_score,
        'loan_status': loan_status
    })
    
    return df

if __name__ == "__main__":
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_credit_data(num_samples=5000)
    file_path = out_dir / "credit_risk_dataset.csv"
    df.to_csv(file_path, index=False)
    print(f"Synthetic credit dataset generated successfully: {file_path}")
    print(f"Shape: {df.shape}, Default rate: {df['loan_status'].mean():.2%}")
