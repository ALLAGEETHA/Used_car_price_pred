import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./car_data.csv")  # Replace this with the correct path to your dataset
    return df

# Data Preprocessing
def preprocess_data(df):
    # Encode categorical variables
    df = pd.get_dummies(df, columns=["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"])

    # Split data into features and target variable
    X = df.drop(columns=["Selling_Price"])
    y = df["Selling_Price"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Model Training
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Prediction
def predict_price(model, input_features):
    # Predict price based on preprocessed input features
    prediction = model.predict(input_features)
    return prediction

# Main app function
def main():
    st.sidebar.title("Car Price Prediction App")
    st.sidebar.header("Description")
    st.sidebar.write("""
        This application predicts the car selling price based on various input parameters such as car name, year, present price, kilometers driven, fuel type, seller type, transmission, and owner.
        The model is trained using Random Forest Regressor.
    """)

    st.title("Car Price Prediction")
    st.header("Predict the Selling Price")

    # Load data
    df = load_data()

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    mse, r2 = evaluate_model(model, X_test, y_test)
    #st.subheader("Model Evaluation Metrics")
    #st.write(f"- Mean Squared Error: {mse}")
    #st.write(f"- R-squared: {r2}")

    st.subheader("Select Input Parameters")

    # Dropdown menu for selecting car name
    selected_car_name = st.selectbox("Select Car Name", df["Car_Name"].unique())

    # Input for selecting year
    year = st.number_input("Year", min_value=int(df["Year"].min()), max_value=int(df["Year"].max()), value=int(df["Year"].min()))

    # Slider for selecting present price
    present_price = st.slider("Select Present Price", min_value=float(df['Present_Price'].min()), max_value=float(df['Present_Price'].max()), value=float(df['Present_Price'].min()))

    # Slider for selecting kilometers driven
    kms_driven = st.slider("Select Kilometers Driven", min_value=int(df['Kms_Driven'].min()), max_value=int(df['Kms_Driven'].max()), value=int(df['Kms_Driven'].min()))

    # Dropdown menu for selecting fuel type
    selected_fuel_type = st.selectbox("Select Fuel Type", df["Fuel_Type"].unique())

    # Dropdown menu for selecting seller type
    selected_seller_type = st.selectbox("Select Seller Type", df["Seller_Type"].unique())

    # Dropdown menu for selecting transmission
    selected_transmission = st.selectbox("Select Transmission", df["Transmission"].unique())

    # Input for selecting owner
    owner = st.number_input("Owner", min_value=int(df["Owner"].min()), max_value=int(df["Owner"].max()), value=int(df["Owner"].min()))

    # Predict price based on user input
    if st.button("Predict Price"):
        input_features = pd.DataFrame({
            'Car_Name': [selected_car_name],
            'Year': [year],
            'Present_Price': [present_price],
            'Kms_Driven': [kms_driven],
            'Fuel_Type': [selected_fuel_type],
            'Seller_Type': [selected_seller_type],
            'Transmission': [selected_transmission],
            'Owner': [owner]
        })

        # One-hot encode categorical variables
        input_features = pd.get_dummies(input_features, columns=["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"])

        # Ensure input features contain all columns present during training
        for col in X_train.columns:
            if col not in input_features.columns:
                input_features[col] = 0

        # Reorder columns to match training data
        input_features = input_features[X_train.columns]

        # Predict price based on preprocessed input features
        prediction = predict_price(model, input_features)
        st.subheader(f"Predicted Selling Price: {prediction[0]}")

if __name__ == "__main__":
    main()
