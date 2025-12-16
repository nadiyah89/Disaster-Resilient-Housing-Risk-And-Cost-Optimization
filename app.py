#User Interface
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error


@st.cache_data
def load_data():
    data = pd.read_csv(r"D:\ml_project\datasets\disaster_dataset.csv")

   
    data.columns = data.columns.str.strip()

   
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data['Disaster Date'] = pd.to_datetime(data['Disaster Date'], dayfirst=True)
    data['Income Level (annual)'] = data['Income Level (annual)'].replace({',': ''}, regex=True)
    data['Income Level (annual)'] = pd.to_numeric(data['Income Level (annual)'], errors='coerce')

   
    categorical_columns = [
        'Disaster Type','Risk level','Disaster Location',
        'Climate Zone','Recommended Material',
        'Material Durability','Soil Type','Structural Features',
        'Foundation Type'
    ]
    encoders = {}
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

    
    scaler = MinMaxScaler()
    numerical_columns = [
        'Disaster Frequency','Temperature (C)','Precipitation (mm)',
        'Wind Speed (km/h)','Humidity (%)',
        'Material Cost (per sqft)','Income Level (annual)'
    ]
    available = [col for col in numerical_columns if col in data.columns]
    data[available] = scaler.fit_transform(data[available])

    return data, encoders

data, encoders = load_data()


X_risk = data[['Temperature (C)', 'Precipitation (mm)', 'Wind Speed (km/h)', 'Humidity (%)']]
y_risk = data['Risk level']
X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)
risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X_train_risk, y_train_risk)


X_mat = data[['Temperature (C)', 'Precipitation (mm)', 'Wind Speed (km/h)',
              'Humidity (%)','Climate Zone','Disaster Type','Risk level']]
y_mat = data['Recommended Material']
X_train, X_test, y_train, y_test = train_test_split(X_mat, y_mat, test_size=0.2, random_state=42)
material_model = RandomForestClassifier(n_estimators=100, random_state=42)
material_model.fit(X_train, y_train)


X_cost = X_mat.copy()
y_cost = data['Material Cost (per sqft)']
X_train_cost, X_test_cost, y_train_cost, y_test_cost = train_test_split(X_cost, y_cost, test_size=0.2, random_state=42)
cost_model = LinearRegression()
cost_model.fit(X_train_cost, y_train_cost)


st.set_page_config(page_title="Disaster-Resilient Housing", layout="wide")

st.title("🏠 Smart Disaster-Resilient Housing System")
st.write("Predicting risks, recommending sustainable materials, and optimizing construction costs.")

st.sidebar.header("🌍 Input Features")
temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
precipitation = st.sidebar.slider("Precipitation (mm)", 0, 500, 100)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 300, 50)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)

climate_zone_choice = st.sidebar.selectbox("Climate Zone", encoders["Climate Zone"].classes_)
disaster_type_choice = st.sidebar.selectbox("Disaster Type", encoders["Disaster Type"].classes_)


climate_zone = encoders["Climate Zone"].transform([climate_zone_choice])[0]
disaster_type = encoders["Disaster Type"].transform([disaster_type_choice])[0]

if st.sidebar.button("🔍 Predict"):
   
    climate_inputs = pd.DataFrame({
        "Temperature (C)": [temperature/50],
        "Precipitation (mm)": [precipitation/500],
        "Wind Speed (km/h)": [wind_speed/300],
        "Humidity (%)": [humidity/100]
    })
    risk_pred = risk_model.predict(climate_inputs)[0]
    risk_label = encoders["Risk level"].inverse_transform([risk_pred])[0]

   
    input_features = pd.DataFrame({
        "Temperature (C)": [temperature/50],
        "Precipitation (mm)": [precipitation/500],
        "Wind Speed (km/h)": [wind_speed/300],
        "Humidity (%)": [humidity/100],
        "Climate Zone": [climate_zone],
        "Disaster Type": [disaster_type],
        "Risk level": [risk_pred]
    })

    material_pred = material_model.predict(input_features)[0]
    material_label = encoders["Recommended Material"].inverse_transform([material_pred])[0]

    
    cost_pred = cost_model.predict(input_features)[0]

  
    st.subheader("⚠️ Predicted Risk Level")
    st.warning(risk_label)

    st.subheader("🏗️ Recommended Material")
    st.success(material_label)

    st.subheader("💰 Estimated Material Cost (per sqft)")
    st.info(f"₹ {cost_pred:.2f}")


st.sidebar.header("📊 Model Performance")
mat_acc = accuracy_score(y_test, material_model.predict(X_test))
risk_acc = accuracy_score(y_test_risk, risk_model.predict(X_test_risk))
mse = mean_squared_error(y_test_cost, cost_model.predict(X_test_cost))
st.sidebar.write(f"Material Model Accuracy: **{mat_acc:.2f}**")
st.sidebar.write(f"Risk Model Accuracy: **{risk_acc:.2f}**")
st.sidebar.write(f"Cost Model MSE: **{mse:.4f}**")

