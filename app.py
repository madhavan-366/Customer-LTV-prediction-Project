import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="LTV Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('ltv_xgboost_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'ltv_xgboost_model.pkl' is in the same directory.")
        return None

model = load_model()

# --- Economic Adjustment Factors ---
COUNTRY_FACTORS = {
    'United Kingdom': 1.0, 'USA': 0.80, 'Germany': 0.85, 'France': 0.85,
    'Japan': 0.0050, 'China': 0.11, 'India': 0.01, 'Canada': 0.60,
    'Australia': 0.55, 'Brazil': 0.15, 'Italy': 0.85, 'Spain': 0.85,
    'Russia': 0.009, 'South Korea': 0.0006, 'Netherlands': 0.85, 'Switzerland': 0.90,
    'Sweden': 0.075, 'Norway': 0.073, 'Denmark': 0.11, 'Ireland': 0.85,
    'Belgium': 0.85, 'Austria': 0.85, 'Finland': 0.85, 'Portugal': 0.85,
    'Greece': 0.85, 'Poland': 0.20, 'Singapore': 0.60, 'Hong Kong': 0.10,
    'United Arab Emirates': 0.22, 'Saudi Arabia': 0.21, 'Israel': 0.22, 'Turkey': 0.025,
    'Malaysia': 0.17, 'Thailand': 0.022, 'Indonesia': 0.00005, 'Philippines': 0.014,
    'Mexico': 0.045, 'Argentina': 0.0009, 'Chile': 0.0008, 'Colombia': 0.0002,
    'South Africa': 0.045, 'Nigeria': 0.0007, 'Egypt': 0.016, 'New Zealand': 0.50,
    'Default': 1.0 
}
REVERSE_COUNTRY_FACTORS = {country: 1 / rate for country, rate in COUNTRY_FACTORS.items() if rate != 0}
CURRENCY_SYMBOLS = {
    'United Kingdom': '£', 'USA': '$', 'Germany': '€', 'France': '€', 'Japan': '¥',
    'China': '¥', 'India': '₹', 'Canada': '$', 'Australia': '$', 'Brazil': 'R$',
    'Italy': '€', 'Spain': '€', 'Russia': '₽', 'South Korea': '₩', 'Netherlands': '€',
    'Switzerland': 'Fr', 'Default': ''
}

# --- Data Processing and Feature Engineering Function ---
@st.cache_data
def process_and_validate_data(df):
    """
    Performs a comprehensive validation on the raw data, separating valid and invalid rows.
    Returns the cleaned DataFrame and a DataFrame of excluded rows with reasons.
    """
    df_processed = df.copy()
    
    # Define all conditions for exclusion
    conditions = [
        df_processed['CustomerID'].isnull(),
        df_processed['Country'].isnull(),
        (df_processed['Quantity'].notnull()) & (df_processed['Quantity'] <= 0),
        (df_processed['UnitPrice'].notnull()) & (df_processed['UnitPrice'] <= 0)
    ]
    # Define the reason for exclusion for each condition
    reasons = [
        "Missing CustomerID",
        "Missing Country",
        "Non-positive Quantity",
        "Non-positive UnitPrice"
    ]
    
    # Use np.select to assign the first reason that matches
    df_processed['Reason_for_Exclusion'] = np.select(conditions, reasons, default='')
    
    # Separate the valid and excluded data
    invalid_mask = df_processed['Reason_for_Exclusion'] != ''
    excluded_df = df_processed[invalid_mask].copy()
    valid_df = df_processed[~invalid_mask].copy()

    return valid_df, excluded_df

@st.cache_data
def perform_feature_engineering(df):
    """Takes a validated DataFrame and calculates RFM features."""
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['AdjustmentFactor'] = df['Country'].map(COUNTRY_FACTORS).fillna(COUNTRY_FACTORS['Default'])
    df['UnitPrice_Adjusted'] = df['UnitPrice'] * df['AdjustmentFactor']
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice_Adjusted']
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm_df = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda date: (snapshot_date - date.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        MonetaryValue=('TotalPrice', 'sum')
    )
    rfm_df['AOV'] = rfm_df['MonetaryValue'] / rfm_df['Frequency']
    return rfm_df

def segment_customer(predicted_ltv):
    if predicted_ltv < 500: return 'Low Value'
    elif predicted_ltv <= 2000: return 'Mid Value'
    else: return 'High Value'

# --- App UI ---
st.title("🌍 Global Customer Lifetime Value (LTV) Predictor")
st.markdown("This intelligent app predicts **3-Month LTV** by automatically validating data and adjusting for the economic context of different countries.")

tab1, tab2 = st.tabs(["👤 Single Prediction (GBP-based)", "📂 Batch Prediction (Global)"])

with tab1:
    st.header("Predict for an Individual Customer")
    st.info("Note: The inputs below should be in **GBP (£)**, as this reflects the model's training data.", icon="ℹ️")
    col1, col2, col3 = st.columns(3)
    with col1: recency = st.number_input('Recency (Days)', 1, 1000, 50, 1)
    with col2: frequency = st.number_input('Frequency (Transactions)', 1, 1000, 5, 1)
    with col3: monetary_value = st.number_input('Monetary Value (Total Spend £)', 1, 200000, 1500, 10)
    
    aov = monetary_value / frequency if frequency > 0 else 0
    st.metric(label="Calculated Average Order Value (AOV)", value=f"£{aov:,.2f}")

    if model is not None and st.button('Predict LTV', type="primary", use_container_width=True, key='single_predict'):
        input_data = pd.DataFrame({'Recency': [recency], 'Frequency': [frequency], 'MonetaryValue': [monetary_value], 'AOV': [aov]})
        prediction = model.predict(input_data)
        segment = segment_customer(prediction[0])
        st.subheader("✨ Prediction Results")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Predicted 3-Month LTV", f"£{prediction[0]:,.2f}")
        with res_col2:
            st.write("Customer Segment")
            if segment == 'High Value': st.success(segment)
            elif segment == 'Mid Value': st.warning(segment)
            else: st.error(segment)

with tab2:
    st.header("Predict from a Raw Transaction File")
    st.subheader("1. Download Sample Data Format")
    st.markdown("Your CSV must contain the columns below. The app will automatically validate all records.")
    
    @st.cache_data
    def get_raw_sample_df():
        return pd.DataFrame({
            'InvoiceNo': [536365, 536367, 536370, 'C536379', 536380, 536381],
            'CustomerID': [17850, 13047, 12583, 14527, np.nan, 15311],
            'InvoiceDate': ['2010-12-01 08:26:00', '2010-12-01 08:34:00', '2010-12-01 08:45:00', '2010-12-01 09:41:00', '2010-12-01 09:42:00', '2010-12-01 09:43:00'],
            'Quantity': [6, 32, 20, -1, 10, 5],
            'UnitPrice': [2.55, 6.00, 500.0, 2.75, 0.0, 3.50],
            'Country': ['United Kingdom', 'USA', 'India', 'France', 'United Kingdom', np.nan]
        })
    st.download_button("Download Sample CSV", get_raw_sample_df().to_csv(index=False).encode('utf-8'), 'sample_global_transactions.csv', 'text/csv')
    st.markdown("---")

    st.subheader("2. Upload Your Transaction File")
    uploaded_file = st.file_uploader("CSV must contain 'CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice', and 'Country'", type="csv")

    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            st.dataframe(raw_df.head())
            if st.button('Run Global Prediction', type="primary", use_container_width=True, key='batch_predict'):
                required_cols = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice', 'Country']
                if all(col in raw_df.columns for col in required_cols):
                    with st.spinner('Validating data, applying adjustments, and running predictions...'):
                        valid_df, excluded_df = process_and_validate_data(raw_df.copy())
                        
                        st.success('Processing complete!')

                        st.subheader("📝 Data Processing Summary")
                        total_customers_found = valid_df['CustomerID'].nunique() if not valid_df.empty else 0
                        
                        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                        summary_col1.metric("Total Rows Uploaded", len(raw_df))
                        summary_col2.metric("Rows Excluded", len(excluded_df), help="Rows with invalid or missing data.")
                        summary_col3.metric("Rows Processed", len(valid_df), help="Number of valid transactions after cleaning.")
                        summary_col4.metric("Unique Customers Found", total_customers_found, help="The number of unique customers these transactions belong to.")
                        
                        if not excluded_df.empty:
                            with st.expander("View Excluded Rows and Reasons"):
                                st.dataframe(excluded_df[['CustomerID', 'InvoiceNo', 'Quantity', 'UnitPrice', 'Country', 'Reason_for_Exclusion']])

                        if valid_df.empty:
                            st.error("No valid customer data was found to process after cleaning. Please check your file.")
                            st.stop()
                        
                        rfm_results = perform_feature_engineering(valid_df)
                        predictions = model.predict(rfm_results)
                        rfm_results['Predicted_LTV_GBP'] = predictions
                        
                        customer_countries = valid_df.drop_duplicates(subset='CustomerID').set_index('CustomerID')['Country']
                        rfm_results = rfm_results.join(customer_countries)
                        rfm_results.rename(columns={'Country': 'Customer_Country'}, inplace=True)
                        
                        rfm_results['ReverseFactor'] = rfm_results['Customer_Country'].map(REVERSE_COUNTRY_FACTORS).fillna(REVERSE_COUNTRY_FACTORS['Default'])
                        rfm_results['Predicted_LTV_Local'] = rfm_results['Predicted_LTV_GBP'] * rfm_results['ReverseFactor']
                        rfm_results['Segment'] = rfm_results['Predicted_LTV_GBP'].apply(segment_customer)
                        
                        st.subheader("📊 Results Dashboard")
                        kpi1, kpi2, kpi3 = st.columns(3)
                        kpi1.metric("Total Customers Predicted", len(rfm_results))
                        kpi2.metric("Avg. Predicted LTV (GBP)", f"£{rfm_results['Predicted_LTV_GBP'].mean():,.2f}")
                        kpi3.metric("High-Value Customers", rfm_results[rfm_results['Segment'] == 'High Value'].shape[0])
                        
                        viz_col1, viz_col2 = st.columns(2)
                        fig_pie = px.pie(rfm_results, names='Segment', title='Customer Segmentation', color='Segment', color_discrete_map={'High Value':'green', 'Mid Value':'orange', 'Low Value':'red'})
                        viz_col1.plotly_chart(fig_pie, use_container_width=True)
                        avg_ltv_by_segment = rfm_results.groupby('Segment')['Predicted_LTV_GBP'].mean().reset_index()
                        fig_bar = px.bar(avg_ltv_by_segment, x='Segment', y='Predicted_LTV_GBP', title='Avg. Predicted LTV (GBP) by Segment', color='Segment', color_discrete_map={'High Value':'green', 'Mid Value':'orange', 'Low Value':'red'})
                        viz_col2.plotly_chart(fig_bar, use_container_width=True)
                        
                        st.subheader("📋 Detailed Results")
                        display_df = rfm_results[['Customer_Country', 'Predicted_LTV_GBP', 'Predicted_LTV_Local', 'Segment']].copy()
                        display_df['Currency_Symbol'] = display_df['Customer_Country'].map(CURRENCY_SYMBOLS).fillna('')
                        display_df['Predicted_LTV_Local_Formatted'] = display_df.apply(lambda row: f"{row['Currency_Symbol']}{row['Predicted_LTV_Local']:,.2f}", axis=1)
                        def segment_color(segment):
                            if segment == 'High Value': return 'background-color: #90EE90'
                            elif segment == 'Mid Value': return 'background-color: #FFFFE0'
                            else: return 'background-color: #F08080'
                        final_df_to_display = display_df[['Customer_Country', 'Predicted_LTV_GBP', 'Predicted_LTV_Local_Formatted', 'Segment']].rename(columns={'Predicted_LTV_Local_Formatted': 'Predicted LTV (Local Currency)'})
                        styled_df = final_df_to_display.style.format({'Predicted_LTV_GBP': '£{:.2f}'}).applymap(segment_color, subset=['Segment'])
                        st.dataframe(styled_df)
                        
                        results_csv = display_df.to_csv().encode('utf-8')
                        st.download_button("Download Results as CSV", results_csv, 'batch_ltv_predictions.csv', 'text/csv')
                else:
                    st.error(f"Error: Your CSV must have these columns: {', '.join(required_cols)}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

