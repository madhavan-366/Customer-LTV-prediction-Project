import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="LTV Predictor",
    page_icon="💰",
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

# --- Feature Engineering Function ---
@st.cache_data
def perform_feature_engineering(df):
    """Takes a raw transaction DataFrame and returns a customer-centric RFM DataFrame."""
    # Robust Data Validation
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm_df = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm_df.rename(columns={'InvoiceDate': 'Recency',
                           'InvoiceNo': 'Frequency',
                           'TotalPrice': 'MonetaryValue'}, inplace=True)
    rfm_df['AOV'] = rfm_df['MonetaryValue'] / rfm_df['Frequency']
    return rfm_df

# --- Customer Segmentation Function ---
def segment_customer(predicted_ltv):
    """Assigns a segment based on the predicted LTV value."""
    if predicted_ltv < 500:
        return 'Low Value'
    elif predicted_ltv <= 2000:
        return 'Mid Value'
    else:
        return 'High Value'

# --- App Title and Description ---
st.title("📈 Customer Lifetime Value (LTV) Prediction")
st.markdown("""
This app predicts the **3-Month LTV** of a customer and assigns a value segment.
""")

# --- Tabbed Interface ---
tab1, tab2 = st.tabs(["👤 Single Prediction", "📂 Batch Prediction"])

# --- Single Prediction Tab ---
with tab1:
    st.header("Predict for an Individual Customer")
    st.markdown("Use this tab for a 'what-if' analysis on a single customer's RFM profile.")
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input('Recency (Days)', min_value=1, max_value=1000, value=50, step=1)
    with col2:
        frequency = st.number_input('Frequency (Transactions)', min_value=1, max_value=1000, value=5, step=1)
    with col3:
        monetary_value = st.number_input('Monetary Value (Total Spend ₹)', min_value=1, max_value=200000, value=1500, step=10) # Changed $ to ₹
    
    aov = monetary_value / frequency if frequency > 0 else 0
    st.metric(label="Calculated Average Order Value (AOV)", value=f"₹{aov:,.2f}") # Changed $ to ₹

    if model is not None and st.button('Predict LTV', type="primary", use_container_width=True, key='single_predict'):
        input_data = pd.DataFrame({
            'Recency': [recency], 'Frequency': [frequency],
            'MonetaryValue': [monetary_value], 'AOV': [aov]
        })
        prediction = model.predict(input_data)
        segment = segment_customer(prediction[0])
        
        st.subheader("✨ Prediction Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Predicted 3-Month LTV", value=f"₹{prediction[0]:,.2f}") # Changed $ to ₹
        with res_col2:
            st.write("Customer Segment")
            if segment == 'High Value':
                st.success(segment)
            elif segment == 'Mid Value':
                st.warning(segment)
            else: # Low Value
                st.error(segment)

# --- Batch Prediction from Raw Data Tab ---
with tab2:
    st.header("Predict from a Raw Transaction File")

    st.subheader("1. Download Sample Transaction Data")
    st.markdown("Download the sample CSV, replace it with your data (keeping the column names), and upload it below.")
    
    @st.cache_data
    def get_raw_sample_df():
        sample_data = {
            'InvoiceNo': [536365, 536365, 536366, 536367, 536367],
            'CustomerID': [17850, 17850, 17850, 13047, 13047],
            'InvoiceDate': ['2010-12-01 08:26:00', '2010-12-01 08:26:00', '2010-12-01 08:28:00', '2010-12-01 08:34:00', '2010-12-01 08:34:00'],
            'Quantity': [6, 8, 2, 6, 32],
            'UnitPrice': [2.55, 2.75, 7.65, 4.25, 1.69]
        }
        return pd.DataFrame(sample_data)

    csv_template = get_raw_sample_df().to_csv(index=False).encode('utf-8')
    st.download_button("Download Sample CSV", csv_template, 'sample_transaction_data.csv', 'text/csv')
    st.markdown("---")

    st.subheader("2. Upload Your Transaction File")
    uploaded_file = st.file_uploader("The CSV must contain 'CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice'", type="csv")

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            st.dataframe(raw_df.head())

            if st.button('Run Prediction on Uploaded File', type="primary", use_container_width=True, key='batch_predict'):
                required_columns = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice']
                if all(col in raw_df.columns for col in required_columns):
                    with st.spinner('Cleaning data, performing feature engineering, and making predictions...'):
                        rfm_results = perform_feature_engineering(raw_df)
                        predictions = model.predict(rfm_results)
                        rfm_results['Predicted_LTV'] = predictions
                        rfm_results['Segment'] = rfm_results['Predicted_LTV'].apply(segment_customer)
                        
                        st.success('Prediction complete!')
                        
                        st.subheader("📊 Results Dashboard")
                        
                        # KPIs
                        kpi1, kpi2, kpi3 = st.columns(3)
                        kpi1.metric("Total Customers Processed", len(rfm_results))
                        kpi2.metric("Average Predicted LTV", f"₹{rfm_results['Predicted_LTV'].mean():,.2f}") # Changed $ to ₹
                        high_value_count = rfm_results[rfm_results['Segment'] == 'High Value'].shape[0]
                        kpi3.metric("High-Value Customers", high_value_count)
                        
                        # Visualizations
                        viz_col1, viz_col2 = st.columns(2)
                        with viz_col1:
                            fig_pie = px.pie(rfm_results, names='Segment', title='Customer Segmentation Distribution',
                                             color='Segment', color_discrete_map={'High Value':'green', 'Mid Value':'orange', 'Low Value':'red'})
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with viz_col2:
                            avg_ltv_by_segment = rfm_results.groupby('Segment')['Predicted_LTV'].mean().reset_index()
                            fig_bar = px.bar(avg_ltv_by_segment, x='Segment', y='Predicted_LTV', title='Average Predicted LTV by Segment',
                                             color='Segment', color_discrete_map={'High Value':'green', 'Mid Value':'orange', 'Low Value':'red'})
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        st.subheader("📋 Detailed Results")
                        def segment_color(segment):
                            if segment == 'High Value': return 'background-color: #90EE90'
                            elif segment == 'Mid Value': return 'background-color: #FFFFE0'
                            else: return 'background-color: #F08080'
                        
                        styled_df = rfm_results.style.applymap(segment_color, subset=['Segment'])
                        st.dataframe(styled_df)
                        
                        results_csv = rfm_results.to_csv().encode('utf-8')
                        st.download_button("Download Results as CSV", results_csv, 'batch_ltv_predictions.csv', 'text/csv')
                else:
                    st.error(f"Error: The uploaded CSV must have these columns: {', '.join(required_columns)}")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

