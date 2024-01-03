import pandas as pd 
import numpy as np 
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from fastapi import HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler





BASE_DIR = Path(__file__).resolve(strict=True).parent
# model = tf.saved_model.load(str(BASE_DIR))
model = keras.models.load_model(str(BASE_DIR ))



def load_data():
    script_dir = Path(__file__).resolve(strict=True).parent
    csv_path = script_dir / "telecomData.csv"
    df1 = pd.read_csv(csv_path)
    df1.drop('customerID',axis='columns',inplace=True)
    df1.drop('Partner',axis='columns',inplace=True)
    df1.drop('Dependents',axis='columns',inplace=True)
    df1.drop('MultipleLines',axis='columns',inplace=True)
    df1.drop('DeviceProtection',axis='columns',inplace=True)
    df1.drop('TechSupport',axis='columns',inplace=True)
    df1.drop('StreamingTV',axis='columns',inplace=True)
    df1.drop('StreamingMovies',axis='columns',inplace=True)
    df1.drop('gender',axis='columns',inplace=True)
    df1['TotalCharges'] = pd.to_numeric(df1['TotalCharges'], errors='coerce')
    return df1

# print(df1.sample(5))
# print(df1.shape)
# new_row = {'tenure': 12, 'MonthlyCharges': 50, 'TotalCharges':600.0}
# df1 = pd.concat([df1, pd.DataFrame([new_row])], ignore_index=True)

# cols_to_scale=['tenure','MonthlyCharges','TotalCharges']
# #to scale these values we are going to use a MinMaxScaler from sklearn

# scaler = MinMaxScaler()
# df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])

# print(df1.iloc[7043].tenure)

# class TextIn(BaseModel):
#     tenure: float
#     PhoneService: str
#     OnlineSecurity: str
#     OnlineBackup: str
#     PaperlessBilling: str
#     MonthlyCharges: float
#     TotalCharges: float
#     InternetService: str
#     Contract: str
#     PaymentMethod: str


# payload = TextIn(
#     tenure=12,
#     PhoneService="Yes",
#     OnlineSecurity="No",
#     OnlineBackup="Yes",
#     PaperlessBilling="Yes",
#     MonthlyCharges=50.0,
#     TotalCharges=600.0,
#     InternetService="DSL",
#     Contract="Month-to-month",
#     PaymentMethod="Credit card (automatic)"
# )

# print(payload)


# new_row = {'tenure': payload.tenure,'MonthlyCharges': payload.MonthlyCharges, 'TotalCharges': payload.TotalCharges}

# df1 = df1.append(new_row, ignore_index=True)

def preprocess_input(payload):
    # Perform any necessary preprocessing on the payload
    # Convert categorical variables to one-hot encoding, handle missing values, etc.
    # Ensure that the preprocessing steps align with what was done during model training
    binary_mapping = {'Yes': 1, 'No': 0}
    payload.PhoneService = binary_mapping.get(payload.PhoneService, payload.PhoneService)
    payload.OnlineSecurity = binary_mapping.get(payload.OnlineSecurity, payload.OnlineSecurity)
    payload.OnlineBackup = binary_mapping.get(payload.OnlineBackup, payload.OnlineBackup)
    payload.PaperlessBilling = binary_mapping.get(payload.PaperlessBilling, payload.PaperlessBilling)

    # Example: Assuming InternetService is a categorical variable
    internet_service_categories = ["DSL", "Fiber optic", "No"]
    internet_service_encoded = np.zeros(len(internet_service_categories))
    internet_service_encoded[internet_service_categories.index(payload.InternetService)] = 1

    # Add additional preprocessing steps for other features

    # Stack all features into a NumPy array

    contract_categories = ["Month-to-month", "One year", "Two year"]
    contract_encoded = np.zeros(len(contract_categories))
    contract_encoded[contract_categories.index(payload.Contract)] = 1

    # Perform one-hot encoding for "PaymentMethod"
    payment_method_categories = [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    payment_method_encoded = np.zeros(len(payment_method_categories))
    payment_method_encoded[payment_method_categories.index(payload.PaymentMethod)] = 1

    
    payload.OnlineSecurity = binary_mapping.get(payload.OnlineSecurity, payload.OnlineSecurity)
    payload.OnlineBackup = binary_mapping.get(payload.OnlineBackup, payload.OnlineBackup)
    payload.PaperlessBilling = binary_mapping.get(payload.PaperlessBilling, payload.PaperlessBilling)

    

    processed_input = np.array([
        payload.tenure,
        payload.PhoneService,
        payload.OnlineSecurity,
        payload.OnlineBackup,
        payload.PaperlessBilling,
        payload.MonthlyCharges,
        payload.TotalCharges,
        *internet_service_encoded,
        *contract_encoded,
        *payment_method_encoded
    ])


    return processed_input

# processed_input = preprocess_input(payload)

def scaleFun(processed_input,df1):
    new_row = {'tenure': processed_input[0], 'MonthlyCharges': processed_input[5], 'TotalCharges':processed_input[6]}
    df1 = pd.concat([df1, pd.DataFrame([new_row])], ignore_index=True)

    cols_to_scale=['tenure','MonthlyCharges','TotalCharges']
    #to scale these values we are going to use a MinMaxScaler from sklearn

    scaler = MinMaxScaler()
    df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])

    # print(df1.iloc[7043].tenure)
    processed_input[0] = df1.iloc[7043].tenure
    processed_input[5] = df1.iloc[7043].MonthlyCharges
    processed_input[6] = df1.iloc[7043].TotalCharges

    return processed_input


# print(preprocess_input(payload))

# input_data = np.array([0.16901408, 1, 1, 0, 1, 0.33681592, 0.07521925, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
# input_data_reshaped = input_data.reshape(1, -1)
# print(input_data_reshaped.shape)
# payload = input_data_reshaped
# print(payload)

def predict_pipeline(processed_input):
        # processed_input = preprocess_input(payload)
        # processed_input = preprocess_input(payload)
        # processed_input = scaleFun(processed_input,df1)
        prediction = model.predict(np.array([processed_input]))
        # prediction = model.predict(processed_input)
        # Assuming your model returns probabilities, you might want to threshold them
        binary_prediction = 1 if prediction[0] > 0.5 else 0
        return binary_prediction
    
# df1 = load_data()
# processed_input = preprocess_input(payload)
# scaled_input = scaleFun(processed_input, df1)
# result = predict_pipeline(scaled_input)

# print(result)