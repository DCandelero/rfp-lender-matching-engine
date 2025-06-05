import streamlit as st
import pandas as pd
import os
import config
from recommendation_solutions import similar_rfp_profile_recommendation, similar_lenders_profile_recommendation
from datetime import datetime

st.set_page_config(layout="wide")

# Load datasets
if 'rfps_cleaned_dataset' not in st.session_state:
    cleaned_rfp_file_path = os.path.join(config.DATA_PATH_WRANGLE, 'cleaned_historical_rfps.parquet')
    df_cleaned_historical_rfps = pd.read_parquet(cleaned_rfp_file_path)
    st.session_state.rfps_cleaned_dataset = df_cleaned_historical_rfps
df_rfps = st.session_state.rfps_cleaned_dataset

if 'lenders_cleaned_dataset' not in st.session_state:
    cleaned_lenders_file_path = os.path.join(config.DATA_PATH_WRANGLE, 'cleaned_lender_preferences.parquet')
    df_cleaned_lender_preferences = pd.read_parquet(cleaned_lenders_file_path)
    st.session_state.lenders_cleaned_dataset = df_cleaned_lender_preferences
df_lenders = st.session_state.lenders_cleaned_dataset

# --- Sidebar ---
st.sidebar.title("Recommendation Settings")
model_choice = st.sidebar.selectbox("Choose recommendation model", ["similar-rfps", "similar-lenders", "SIMS (AWS)"])
if model_choice == "SIMS (AWS)":
    st.sidebar.write('Still in development. Choosing default method [similar-rfps]')
    model_choice = 'similar-rfps'
max_recommendations = st.sidebar.number_input(
    "Max number of recommendations",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)

# --- Main Interface ---
st.title("RFP → Lender Recommendation")

# --- Step 1: RFP ID Input ---
if 'show_form' not in st.session_state:
    st.session_state.show_form = False

rfp_id_selected = st.selectbox("Select RFP ID  (or choose 'Other')", 
                             options=["Other"] + df_rfps['rfp_id'].unique().tolist())
if rfp_id_selected == "Other":
    rfp_id_input = st.text_input("Enter RFP ID", help='Some examples of rfp_id: [rfp00001, rfp00002, ... rfp20000]')
else:
    rfp_id_input = rfp_id_selected
check_button = st.button("Get Lenders Recommendations")

if check_button:
    rfp_id_selected = ''
    if rfp_id_input in df_rfps['rfp_id'].values:
        st.session_state.rfp_known = True
        st.session_state.show_form = False
    else:
        st.session_state.rfp_known = False
        st.session_state.show_form = True

# --- Step 2: Handle Known or New RFP ---
rfp_row = None

if st.session_state.get("rfp_known", False) and rfp_id_selected != 'Other':
    rfp_row = df_rfps[df_rfps['rfp_id'] == rfp_id_input].iloc[0]
    st.success("RFP found in the dataset.")
elif st.session_state.get("show_form", False):
    st.warning("RFP not found. Please fill in the details.")
    with st.form("new_rfp_form"):
        company_stage = st.selectbox("Company Stage", 
                                     options=df_rfps['company_stage'].unique().tolist())
        company_founding_year = st.number_input("Company Founding Year",
                                                min_value=1900, max_value=datetime.now().year,)
        company_revenue_last_fy_usd = st.number_input("Company Revenue Last FY (USD)", step=100000, min_value=1)
        industry = st.selectbox("Industry Sector",
                                options=df_rfps['industry_sector'].unique().tolist())
        region = st.selectbox("Region",
                              options=df_rfps['region'].unique().tolist())
        loan_type = st.selectbox("Loan Type Requested",
                                 options=df_rfps['loan_type_requested'].unique().tolist())
        deal_size = st.number_input("Deal Size (USD)", step=1000, min_value=1)
        purpose_of_funds = st.selectbox("Purpose of Funds", 
                                        options=df_rfps['purpose_of_funds'].unique().tolist())
        submit_new = st.form_submit_button("Get Recommendations")

        if submit_new:
            if all([industry, region, loan_type, deal_size > 0]):
                rfp_row = pd.Series({
                    'rfp_id': rfp_id_input,
                    'company_stage': company_stage,
                    'company_founding_year': company_founding_year,
                    'company_revenue_last_fy_usd': company_revenue_last_fy_usd,
                    'purpose_of_funds': purpose_of_funds,
                    'submission_date': datetime.now(),
                    'industry_sector': industry,
                    'region': region,
                    'loan_type_requested': loan_type,
                    'deal_size_usd': deal_size
                })
                st.success("Custom RFP created.")
                # Append new rfp in the dataset (For now without cleaning)
                rfp_df = pd.DataFrame([rfp_row])
                st.session_state.rfps_cleaned_dataset = pd.concat([
                    st.session_state.rfps_cleaned_dataset,
                    rfp_df
                ], ignore_index=True)
                df_rfps = st.session_state.rfps_cleaned_dataset
            else:
                st.error("Please complete all fields.")

# --- Step 3: Recommendation ---
if rfp_row is not None:
    st.subheader("Input RFP")
    st.dataframe(pd.DataFrame([rfp_row]))

    st.subheader("Recommended Lenders")
    with st.spinner("Recommending lenders..."):
        if model_choice == 'similar-lenders':
            suggested_lenders = similar_lenders_profile_recommendation(
                rfp_id_input, 
                df_rfps, 
                df_lenders, 
                top_n=max_recommendations)
        else:
            suggested_lenders = similar_rfp_profile_recommendation(
                rfp_id_input, 
                df_rfps, 
                df_lenders, 
                top_n=max_recommendations)
        st.dataframe(suggested_lenders.reset_index(drop=True))
