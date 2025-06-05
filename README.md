# 🔍 RFP-Lender Matching System

## 🧠 Project Goal

This project aims to **recommend suitable lenders** for a given RFP (Request for Funding Proposal) using both:

- **Content-based methods**, such as cosine similarity between engineered RFP and lender profiles.
- **Collaborative filtering**, using **AWS Personalize**'s SIMS and User-Personalization recipes. [Still in Development]

It supports:
- Real-time recommendation for existing or new RFPs.
- Interactive exploration through a **Streamlit web app**.
- Hybrid approaches combining similarity scores and profile matches.

---

## Main code - Notebook
Location: src/main.ipynb

## Initial data location
- data\01-raw\historical_rfps.csv
- data\01-raw\lender_preferences.csv

### Needed to run streamlit 
-> If you run the main notebook, the clened data will be saved in:
- data\02-cleaned\historical_rfps.csv
- data\02-cleaned\lender_preferences.csv

## 🚀 How to Run the Streamlit App

### 1. Install dependencies and run Streamlit
Inside the src folder:
```bash
# Create venv
python -m venv venv
# Install packages
pip install -r requirements.txt
# Run streamlit
streamlit run .\app.py