import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf # Ensure TensorFlow is imported
from tensorflow.keras.models import load_model

# Updated model path
MODEL_PATH = 'optimized_keras_model.h5'

def preprocess_data(df):
    """Applies preprocessing steps to the input DataFrame."""
    st.write("Original data columns:", df.columns.tolist())

    # Drop unnecessary columns - based on 'Automated EDA .ipynb' and 'DL-Experiments (1).ipynb'
    columns_to_drop = ['msno', 'bd', 'msno_R', 'msno_R1', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'msno_R2', 'payment_method_id', 'date']
    # Filter out columns that are not present in the uploaded df
    columns_present_in_df = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_present_in_df, axis=1)
    st.write(f"Dropped columns: {columns_present_in_df}")
    st.write("Data columns after dropping:", df.columns.tolist())

    # Encode 'gender' if it exists
    if 'gender' in df.columns:
        st.write("Encoding 'gender' column...")
        labelencoder = LabelEncoder()
        df['gender'] = labelencoder.fit_transform(df['gender'])
        st.write("'gender' column encoded.")
    else:
        st.write("'gender' column not found in the uploaded data.")

    # Ensure all remaining columns are numeric for scaling
    # This is a simplification; in a real scenario, more robust type checking and handling would be needed.
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                st.error(f"Could not convert column '{col}' to numeric. Please ensure data is clean or handle non-numeric columns appropriately before scaling.")
                return None


    # Normalize data using MinMaxScaler if there are features to scale
    if not df.empty:
        st.write("Normalizing data using MinMaxScaler...")
        # Exclude target variable 'is_churn' if present, before scaling
        features_to_scale = df.columns.tolist()
        target_column = 'is_churn' # Define target column
        if target_column in features_to_scale:
            features_to_scale.remove(target_column)

        if features_to_scale: # Proceed only if there are features to scale
            scaler = MinMaxScaler()
            df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
            st.write("Data normalized.")
        else:
            st.write("No features to scale after excluding target variable or if dataframe became empty.")
    else:
        st.write("DataFrame is empty after preprocessing steps before scaling.")

    return df

def load_keras_model(model_path):
    """Loads the trained Keras model."""
    # Placeholder for model loading
    # try:
    #     model = load_model(model_path)
    #     st.success("Keras model loaded successfully.")
    #     return model
    # except Exception as e:
    try:
        model = load_model(model_path)
        st.success("Keras model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading Keras model from '{model_path}': {e}")
        st.info("Please ensure 'optimized_keras_model.h5' is in the same directory as app.py or provide the correct path.")
        return None

st.title("KKBox Customer Churn Prediction")

st.header("Upload Customer Data (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully!")
        st.write("Raw Data:")
        st.dataframe(data.head())

        st.header("1. Data Preprocessing")
        processed_data = preprocess_data(data.copy()) # Use a copy to avoid altering original uploaded data

        if processed_data is not None:
            st.write("Preprocessed Data:")
            st.dataframe(processed_data.head())

            st.header("2. Churn Prediction")
            model = load_keras_model(MODEL_PATH)

            if model is not None:
                if st.button("Predict Churn"):
                    features_for_prediction = processed_data.copy()
                    if 'is_churn' in features_for_prediction.columns:
                        features_for_prediction = features_for_prediction.drop('is_churn', axis=1)

                    st.write("Features for prediction (ensure order and number match model training):", features_for_prediction.columns.tolist())
                    st.write(f"Number of features for prediction: {features_for_prediction.shape[1]}")


                    if not features_for_prediction.empty:
                        try:
                            # Model expects numpy array
                            predictions_proba = model.predict(features_for_prediction.to_numpy())
                            # Convert probabilities to class labels (0 or 1)
                            # Since the last layer is sigmoid, threshold is 0.5
                            predictions_label = (predictions_proba > 0.5).astype(int)

                            predictions_df = pd.DataFrame(predictions_label, columns=["Prediction_Label"])
                            predictions_df["Prediction"] = predictions_df["Prediction_Label"].apply(lambda x: "Churn" if x == 1 else "Not Churn")

                            st.write("Prediction Results:")
                            st.dataframe(predictions_df[['Prediction']].head())

                            churn_count = predictions_df[predictions_df["Prediction"] == "Churn"].shape[0]
                            not_churn_count = predictions_df[predictions_df["Prediction"] == "Not Churn"].shape[0]
                            st.write(f"Predicted Churn: {churn_count}")
                            st.write(f"Predicted Not Churn: {not_churn_count}")

                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                            st.info("Ensure the input data has the correct number of features (expected 12 after preprocessing, excluding target) and format.")
                    else:
                        st.warning("No data available for prediction after preprocessing.")
            else:
                st.error("Model could not be loaded. Prediction unavailable.")
        else:
            st.error("Preprocessing failed. Cannot proceed to prediction.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.sidebar.header("Model Information")
st.sidebar.markdown("""
This app uses a Keras-based Deep Learning model to predict customer churn.

**Note:** The model loading and prediction parts are currently placeholders.
A pre-trained model (`keras_model.h5`) needs to be available for full functionality.
""")

st.sidebar.header("Model Comparison")
# Attempt to read and display model comparison from README or directly
try:
    with open("README.md", "r") as f:
        readme_content = f.read()
    # Extract model comparison section (this is a bit fragile)
    comparison_section = readme_content[readme_content.find("## Model Comparison"):]
    st.sidebar.markdown(comparison_section)
except Exception as e:
    st.sidebar.warning(f"Could not load model comparison from README: {e}")
    st.sidebar.markdown("""
    Detailed model comparison can be found in `model comparison.xlsx`.
    Key accuracy scores from notebooks:
    - DNN Classifier: 0.509
    - Keras NN (Training): 0.5001
    - Keras NN (Validation): 0.4994
    """)

st.sidebar.info("Project: KKBox Music Streaming Customer Churn Rate Prediction")
