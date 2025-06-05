import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_rfp_features(df):
    df = df.copy()

    df['funded_flag'] = (df['deal_status'] == 'funded').astype(int)

    df['company_age'] = df['submission_date'].dt.year - df['company_founding_year']

    selected_cols = [
        'rfp_id', 'funded_flag', 'deal_size_usd', 'company_revenue_last_fy_usd',
        'company_age', 'industry_sector', 'region', 'loan_type_requested'
    ]
    df = df[selected_cols]

    # Normalize numeric columns
    numeric_cols = ['deal_size_usd', 'company_revenue_last_fy_usd', 'company_age']
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # One-hot encode categorical columns
    categorical_cols = ['industry_sector', 'region', 'loan_type_requested']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    df_encoded.set_index('rfp_id', drop=True, inplace=True)

    return df_encoded

def preprocess_lenders_features(df):
    df = df.copy()

    selected_cols = [
        'lender_id', 'lender_type', 'risk_appetite',
        'preferred_industries', 'preferred_regions', 'preferred_loan_types'
    ]
    df = df[selected_cols]

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['lender_type', 'risk_appetite'], prefix=['lender_type', 'risk'])

    # Multi-hot encode semicolon-separated fields
    multi_hot_cols = ['preferred_industries', 'preferred_regions', 'preferred_loan_types']

    for col in multi_hot_cols:
        # Split and explode the column
        exploded = df[[col]].dropna().copy()
        exploded[col] = exploded[col].str.split(';')
        exploded = exploded.explode(col)
        exploded[col] = exploded[col].str.strip().str.lower()

        # Create multi-hot encoded DataFrame
        multi_hot = pd.get_dummies(exploded[col], prefix=col)
        multi_hot[col + '_index'] = exploded.index

        # Back to original index and merge
        multi_hot = multi_hot.groupby(col + '_index').max()
        df = df.join(multi_hot, how='left')

    # Fill NaN in multi-hot columns with 0
    df.fillna(0, inplace=True)

    df = df.drop(columns=multi_hot_cols)
    df.set_index('lender_id', drop=True, inplace=True)

    return df

def match_preference(preferred: str, rfp_value: str) -> bool:
    """
    Check if a single RFP value matches any of the lender's preferred values in a semicolon-separated string.
    """
    if pd.isna(preferred) or pd.isna(rfp_value):
        return False
    preferred_list = [x.strip().lower() for x in preferred.split(';')]
    return rfp_value.strip().lower() in preferred_list

def is_deal_size_in_range(deal_size, min_size, max_size) -> bool:
    """
    Check if the deal size is within the lender's preferred deal size range.
    """
    if pd.notna(deal_size) and pd.notna(min_size) and pd.notna(max_size):
        return min_size <= deal_size <= max_size
    return False

def preprocess_preference_alignment_flags(df):
    """
    Adds binary columns indicating whether the RFP aligns with lender preferences:
      - region
      - industry
      - loan type
      - deal size range

    Assumes the DataFrame includes:
    - 'preferred_regions', 'region'
    - 'preferred_industries', 'industry_sector'
    - 'preferred_loan_types', 'loan_type_requested'
    - 'deal_size_usd', 'preferred_deal_size_min_usd', 'preferred_deal_size_max_usd'
    """
    df = df.copy()
    df['funded_flag'] = (df['deal_status'] == 'funded').astype(int)

    df['region_match'] = df.apply(lambda row: match_preference(row['preferred_regions'], row['region']), axis=1)
    df['industry_match'] = df.apply(lambda row: match_preference(row['preferred_industries'], row['industry_sector']), axis=1)
    df['loan_type_match'] = df.apply(lambda row: match_preference(row['preferred_loan_types'], row['loan_type_requested']), axis=1)
    df['deal_size_in_range'] = df.apply(lambda row: is_deal_size_in_range(row['deal_size_usd'], row['preferred_deal_size_min_usd'], row['preferred_deal_size_max_usd']), axis=1)

    selected_cols = [
        'rfp_id', 'lender_id', 'funded_flag', 'region_match',
        'industry_match', 'loan_type_match', 'deal_size_in_range'
    ]
    df = df[selected_cols]

    return df