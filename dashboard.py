# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import requests
from shapely.geometry import Polygon
from pyproj import Geod
from satellite_ndvi import get_real_ndvi
from cnn_feature_extractor import extract_image_features
from satellite_image import download_satellite_image

# Sentinel Hub Credentials
CLIENT_ID = "414e76a5-4a7b-4fbb-899e-9302afb91e3f"
CLIENT_SECRET = "3c6NGoRYDpkwVzf7H3F0jnoj8z7CS5jG"

# --------------------------
# Area Calculation
# --------------------------
geod = Geod(ellps="WGS84")

def calculate_area(coords):

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    area, _ = geod.polygon_area_perimeter(lons, lats)
    area = abs(area)

    hectares = area / 10000

    return hectares


# --------------------------
# Weather Function
# --------------------------
def get_weather(lat, lon):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True
    }

    response = requests.get(url, params=params)
    data = response.json()

    temperature = data["current_weather"]["temperature"]

    # Stable rainfall for demo
    rainfall = 60

    return temperature, rainfall


# --------------------------
# Load Model
# --------------------------
model = joblib.load("crop_yield_model.pkl")
scaler = joblib.load("crop_yield_scaler.pkl")


# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv("crop_yield_climate_soil_data_2019_2023.csv")
df.columns = df.columns.str.strip()

features = ['NDVI','Rainfall','Temperature','Soil_PH','Sunlight','month','year','CNN_Feature']
target = 'Combined_Crop_Yield'

low_thresh = df[target].quantile(0.33)
med_thresh = df[target].quantile(0.66)


# --------------------------
# Session State
# --------------------------
if "pred_yield" not in st.session_state:
    st.session_state.pred_yield = None

if "img_path" not in st.session_state:
    st.session_state.img_path = None

if "cnn_feature" not in st.session_state:
    st.session_state.cnn_feature = None


# --------------------------
# Title
# --------------------------
st.title("🌾 Farm-Level Crop Yield Prediction System")


# --------------------------
# Farm Area Selection
# --------------------------
st.header("🗺 Select Farm Land Area")

m = folium.Map(
    location=[20.5937,78.9629],
    zoom_start=5,
    tiles=None
)

# Satellite tiles
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google",
    name="Google Satellite"
).add_to(m)

# Labels
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Labels"
).add_to(m)

draw = Draw(
    draw_options={
        "polyline": False,
        "rectangle": True,
        "polygon": True,
        "circle": False,
        "marker": False
    }
)

draw.add_to(m)

map_data = st_folium(m,width=700,height=500)

latitude = None
longitude = None
farm_area_hectares = None
polygon_coords = None


if map_data and map_data.get("last_active_drawing"):

    polygon_coords = map_data["last_active_drawing"]["geometry"]["coordinates"][0]

    st.success("Farm Area Selected!")

    poly = Polygon(polygon_coords)

    centroid = poly.centroid
    latitude = centroid.y
    longitude = centroid.x

    st.write(f"Farm Center Location: {latitude:.4f}, {longitude:.4f}")

    farm_area_hectares = calculate_area(polygon_coords)

    st.write(f"Farm Area: {farm_area_hectares:.2f} hectares")
    st.write(f"Farm Area: {farm_area_hectares*2.471:.2f} acres")


# --------------------------
# NDVI
# --------------------------
st.header("Satellite NDVI Estimation")

ndvi = None

if latitude and longitude:

    try:

        ndvi = get_real_ndvi(polygon_coords, CLIENT_ID, CLIENT_SECRET)

        st.success(f"🌿 Satellite NDVI: {ndvi:.3f}")

    except Exception as e:

        st.error(f"Satellite NDVI Error: {e}")

        red = st.number_input("RED Band Value",0.0,1.0,0.2)
        nir = st.number_input("NIR Band Value",0.0,1.0,0.8)

        ndvi = (nir-red)/(nir+red) if (nir+red)!=0 else 0

        st.write("NDVI:",round(ndvi,3))

else:

    red = st.number_input("RED Band Value",0.0,1.0,0.2)
    nir = st.number_input("NIR Band Value",0.0,1.0,0.8)

    ndvi = (nir-red)/(nir+red) if (nir+red)!=0 else 0

    st.write("NDVI:",round(ndvi,3))


# --------------------------
# NDVI Health
# --------------------------
if ndvi is not None:

    if ndvi > 0.6:
        st.markdown("Crop Health: 🟢 Healthy")

    elif ndvi > 0.3:
        st.markdown("Crop Health: 🟡 Moderate")

    else:
        st.markdown("Crop Health: 🔴 Poor")


# --------------------------
# Crop Selection
# --------------------------
st.header("🌱 Crop Selection")

crop = st.selectbox(
    "Select Crop",
    ["Rice","Wheat","Maize","Cotton"]
)

crop_multiplier = {
    "Rice":1.3,
    "Wheat":1.0,
    "Maize":1.5,
    "Cotton":0.8
}

multiplier = crop_multiplier[crop]


# --------------------------
# Climate Inputs
# --------------------------
st.header("Climate / Soil Inputs")

if latitude and longitude:

    temperature,rainfall = get_weather(latitude,longitude)

    st.write("🌡 Temperature:",temperature)
    st.write("🌧 Rainfall:",rainfall)

else:

    rainfall = st.number_input("Rainfall",value=50.0)
    temperature = st.number_input("Temperature",value=28.0)


soil_ph = st.number_input("Soil PH",value=6.5)
sunlight = st.number_input("Sunlight",value=10.0)
month = st.number_input("Month",1,12,6)
year = st.number_input("Year",2024)


# --------------------------
# Prediction
# --------------------------
if st.button("Predict Crop Yield"):

    if polygon_coords is None:

        st.warning("Please select farm area first!")
        st.stop()

    img_path = download_satellite_image(
        polygon_coords,
        CLIENT_ID,
        CLIENT_SECRET
    )

    cnn_feature = extract_image_features(img_path)

    input_df = pd.DataFrame(
        [[ndvi,rainfall,temperature,soil_ph,sunlight,month,year,cnn_feature]],
        columns=features
    )

    input_scaled = scaler.transform(input_df)

    base_yield = model.predict(input_scaled)[0]

    adjusted_yield = base_yield * multiplier

    st.session_state.pred_yield = adjusted_yield
    st.session_state.img_path = img_path
    st.session_state.cnn_feature = cnn_feature


# --------------------------
# Show Satellite Image
# --------------------------
if st.session_state.pred_yield is not None:

    if st.session_state.img_path:

        st.image(
            st.session_state.img_path,
            caption="Satellite Image of Farm",
            width=500
        )

        st.write(
            "🧠 CNN Vegetation Score:",
            round(st.session_state.cnn_feature,3)
        )


# --------------------------
# Prediction Result
# --------------------------
if st.session_state.pred_yield is not None:

    pred_yield = st.session_state.pred_yield

    st.header("🌾 Predicted Crop Yield")

    st.write(f"Crop: {crop}")

    st.write(f"Yield per hectare: {pred_yield:.3f} tons")

    if farm_area_hectares:

        total_yield = pred_yield * farm_area_hectares

        st.write(f"Estimated Total Yield: {total_yield:.2f} tons")

    if pred_yield >= med_thresh:
        st.write("Yield Category: 🟢 High")

    elif pred_yield >= low_thresh:
        st.write("Yield Category: 🟡 Medium")

    else:
        st.write("Yield Category: 🔴 Low")


# --------------------------
# Visualizations
# --------------------------
if st.session_state.pred_yield is not None:

    if st.checkbox("Show Visualizations"):

        fig,ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df[features+[target]].corr(),annot=True,cmap="coolwarm",ax=ax)
        st.pyplot(fig)

        fig2,ax2 = plt.subplots()
        sns.scatterplot(x="NDVI",y=target,data=df,ax=ax2)
        st.pyplot(fig2)

        fig3,ax3 = plt.subplots()
        sns.histplot([pred_yield],bins=5)
        st.pyplot(fig3)