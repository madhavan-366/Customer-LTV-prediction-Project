# Customer Lifetime Value (LTV) Prediction App

A comprehensive, end-to-end data science project to **predict customer lifetime value** and **segment customers** using machine learning.

---

## Live Application

You can access and interact with the live application here:  
https://customer-ltv-prediction-project-bpqvpimbborsioqfrskaxb.streamlit.app/

---

## Project Overview

This project addresses a key business challenge — **identifying and understanding the value of customers**.  
Using historical transaction data, the application predicts the **future 3-month spending (LTV)** of each customer.

The core engine is an **XGBoost Regressor model** trained on **RFM (Recency, Frequency, Monetary)** features.  
Predictions are then used to classify customers into three actionable tiers:

- High Value  
- Mid Value  
- Low Value  

These segments enable targeted marketing, retention, and personalized offers.

The entire pipeline — from **data cleaning to prediction and visualization** — is wrapped into a user-friendly **Streamlit web application**.

---

## Key Features

- **Single Customer Prediction:**  
  Enter RFM values manually and instantly predict a customer's 3-month LTV.

- **Batch Prediction:**  
  Upload a raw transaction CSV to generate LTV predictions for thousands of customers.

- **Intelligent Currency Conversion:**
  The app automatically detects the country for each transaction and applies an economic adjustment factor, converting all prices to a common baseline (GBP) before prediction. This ensures the model's accuracy regardless of the data's origin.

- **Transparent Data Validation:**
  When a file is uploaded, the app provides a full summary of the data quality, showing exactly which rows were excluded and why (e.g., "Missing CustomerID," "Non-positive Quantity"). This builds trust and provides clarity to the user.
  
- **Automated Feature Engineering:**  
  The app automatically computes Recency, Frequency, Monetary Value, and Average Order Value (AOV).

- **Interactive Dashboard:**  
  Visual KPIs, customer segment distribution (pie chart), and average LTV per segment (bar chart).

- **Color-Coded Results:**  
  Green (High), Orange (Mid), and Red (Low) for immediate clarity.

- **Dynamic Local Currency Display:**
  The final results table shows not only the official prediction in GBP but also intelligently converts it back to each customer's original local currency.

- **Data Export:**  
  Download predictions and customer segments as a CSV file for further business use.

---

## Tech Stack

| Category | Tools/Frameworks |
|-----------|------------------|
| Language | Python 3.11 |
| App Framework | Streamlit |
| Machine Learning | Scikit-learn, XGBoost |
| Data Manipulation | Pandas |
| Visualization | Plotly Express |
| Model Serialization | Joblib |

---

## How to Use the App

### Single Prediction
1. Go to the **Single Prediction** tab.  
2. Enter values for **Recency, Frequency, and Monetary**.  
3. Click **Predict LTV**.  
4. View the predicted 3-month LTV and customer segment.

### Batch Prediction
1. Go to the **Batch Prediction** tab.  
2. Download the sample CSV to check the required format.  
3. Upload your own transaction data.  
4. Click **Run Prediction on Uploaded File**.  
5. View:
   - KPIs  
   - Customer Segment Distribution  
   - Average LTV per Segment (bar chart)  
   - Color-coded results table  

---

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/LTV-Project.git
   cd LTV-Project
   
2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On Mac/Linux

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py

## Purpose
This project was developed as a portfolio piece to demonstrate practical skills in:

- Data Science and Feature Engineering  
- Machine Learning (Regression and Segmentation)  
- Streamlit App Development  
- Data Visualization and Reporting  

---

## Deliverables
| File Name | Description |
|------------|--------------|
| `app.py` | Streamlit application script |
| `model.pkl` | Trained XGBoost model file |
| `rfm_feature_engineering.py` | Feature extraction module |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## Example Output
**Input:**  
Customer with Recency = 12 days, Frequency = 5, Monetary = 4500  

**Predicted LTV (3-month):** ₹12,500  
**Segment:** High Value  

---

## Tech Stack
- Python  
- Streamlit  
- XGBoost  
- Pandas  
- Plotly  

---

## Description
This project demonstrates the full lifecycle of a machine learning solution — from raw data preprocessing and feature engineering to model training, evaluation, and deployment as an interactive web application using Streamlit.
