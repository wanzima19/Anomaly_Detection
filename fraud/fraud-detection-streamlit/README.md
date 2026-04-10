# Fraud Detection Streamlit Application

This project provides a user-friendly interface for fraud detection using a Streamlit application that communicates with a Flask API. The application allows users to input transaction data and receive predictions regarding potential fraud.

## Project Structure

```
fraud-detection-streamlit
├── app.py                # Main entry point for the Streamlit application
├── requirements.txt      # Lists dependencies for the application
└── README.md             # Documentation for the project
```

## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd fraud-detection-streamlit
   ```

2. **Install the required dependencies**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.

   For `pip`, run:
   ```
   pip install -r requirements.txt
   ```

3. **Run the Flask API**:
   Ensure that the Flask API is running before starting the Streamlit application. Navigate to the directory containing the Flask API and run:
   ```
   python fraud_api.py
   ```

4. **Run the Streamlit application**:
   In a new terminal, navigate to the `fraud-detection-streamlit` directory and run:
   ```
   streamlit run app.py
   ```

## Usage

- Open your web browser and go to `http://localhost:8501` to access the Streamlit application.
- Input the required transaction details, including `transaction_id`, `Time`, `Amount`, and the feature variables `V1` to `V28`.
- Click on the "Submit" button to get the fraud detection results.
- The application will display the risk score, recommended action, and whether the transaction is flagged as fraud.

## Features

- User-friendly interface for inputting transaction data.
- Real-time communication with the Flask API for fraud detection predictions.
- Display of risk scores and recommended actions based on the predictions.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.