import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px

# Set the URL for the Flask API
API_URL = "http://localhost:5000"

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return False, None

def predict_transaction(transaction_data):
    try:
        response = requests.post(f"{API_URL}/predict", json=transaction_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get('error', 'Unknown error')}
    except requests.exceptions.RequestException as e:
        return {"error": f"API connection failed: {str(e)}"}

def predict_batch(transactions):
    try:
        # Batch inference can take longer on larger CSV uploads.
        response = requests.post(f"{API_URL}/batch_predict", json={"transactions": transactions}, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get('error', 'Unknown error')}
    except requests.exceptions.RequestException as e:
        return {"error": f"API connection failed: {str(e)}"}

def main():
    st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🔍", layout="wide")
    st.title("🔍 Fraud Detection Application")
    st.markdown("Real-time fraud detection using ensemble ML models. Ensure the Flask API is running at `http://localhost:5000`.")

    # Sidebar for inputs
    st.sidebar.header("Transaction Input")
    transaction_id = st.sidebar.text_input("Transaction ID", placeholder="e.g., TXN-001")
    time = st.sidebar.number_input("Time (seconds since epoch)", min_value=0, value=1000)
    amount = st.sidebar.number_input("Amount ($)", min_value=0.0, value=50.0, step=0.01)
    
    # V features in expandable section
    with st.sidebar.expander("PCA Features (V1-V28)"):
        v_features = {}
        cols = st.columns(2)
        for i in range(1, 29):
            col = cols[(i-1) % 2]
            v_features[f'V{i}'] = col.number_input(f"V{i}", value=0.0, step=0.01, format="%.2f")

    # API Health Check
    st.sidebar.subheader("API Status")
    api_ok, health_data = check_api_health()
    if api_ok:
        st.sidebar.success("✅ API Connected")
        st.sidebar.json(health_data)
    else:
        st.sidebar.error("❌ API Not Reachable - Start Flask app first")

    # Main content
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    with tab1:
        st.header("Single Transaction Prediction")
        if st.button("Predict Fraud", type="primary"):
            if not transaction_id:
                st.error("Please enter a Transaction ID.")
            elif not api_ok:
                st.error("API is not connected. Please start the Flask server.")
            else:
                with st.spinner("Analyzing transaction..."):
                    transaction_data = {
                        "transaction_id": transaction_id,
                        "Time": time,
                        "Amount": amount,
                        **v_features
                    }
                    result = predict_transaction(transaction_data)
                
                if "error" in result:
                    st.error(f"Prediction failed: {result['error']}")
                else:
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Prediction Summary")
                        risk_score = result['risk_score']
                        is_fraud = result['is_fraud']
                        risk_level = result['risk_level']
                        
                        # Color-coded risk level
                        if risk_level == "LOW":
                            st.success(f"🟢 Risk Level: {risk_level}")
                        elif risk_level == "MEDIUM":
                            st.warning(f"🟡 Risk Level: {risk_level}")
                        else:
                            st.error(f"🔴 Risk Level: {risk_level}")
                        
                        st.metric("Risk Score", f"{risk_score:.4f}")
                        st.metric("Is Fraud", "Yes" if is_fraud else "No")
                        st.metric("Recommended Action", result['recommended_action'])
                    
                    with col2:
                        st.subheader("Model Contributions")
                        if 'model_contributions' in result:
                            contrib_df = pd.DataFrame(list(result['model_contributions'].items()), columns=["Model", "Score"])
                            fig = px.bar(contrib_df, x="Model", y="Score", title="Model Breakdown", color="Score", color_continuous_scale="RdYlGn_r")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Model breakdown not available.")

    with tab2:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with columns: transaction_id, Time, Amount, V1-V28")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            required_cols = ['transaction_id', 'Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
            if all(col in df.columns for col in required_cols):
                st.dataframe(df.head())
                if st.button("Predict Batch", type="primary"):
                    if not api_ok:
                        st.error("API is not connected.")
                    else:
                        with st.spinner("Processing batch..."):
                            transactions = df.to_dict('records')
                            result = predict_batch(transactions)
                        
                        if "error" in result:
                            st.error(f"Batch prediction failed: {result['error']}")
                        else:
                            results_df = pd.DataFrame(result['results'])
                            st.subheader("Batch Results")
                            st.dataframe(results_df)
                            
                            # Summary stats
                            fraud_count = results_df['is_fraud'].sum()
                            st.metric("Total Transactions", len(results_df))
                            st.metric("Detected Frauds", fraud_count)
                            st.metric("Fraud Rate", f"{fraud_count / len(results_df) * 100:.2f}%")
            else:
                st.error("CSV must contain columns: transaction_id, Time, Amount, V1-V28")

if __name__ == "__main__":
    main()