###########################################
# app.py
# Integrated App with:
#   - MongoDB integration for authentication & inventory management
#   - Weather, NDVI, and Satellite view for Irrigation Recommendations
#   - Fertilizer & Pesticide Recommendations (using AI model)
#   - Inventory Management for crops and pesticides
#   - Leaf Health Classification (Healthy vs. Not Healthy) using improved synthetic data
#   - Yield Prediction using an AI model on synthetic data (improved to avoid negative/flat predictions)
#   - Language translation and inline predictive city input in sidebar
#   - Logout option and display of current username in sidebar
###########################################

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import streamlit as st
import streamlit.components.v1 as components
import requests
import numpy as np
import pandas as pd
import ee
import datetime
import pymongo
import bcrypt
from urllib.parse import quote_plus
import threading
from concurrent.futures import ThreadPoolExecutor
from googletrans import Translator
from PIL import Image
import random
import glob
import joblib

# ---------------------------
# TensorFlow imports for CNN models
# ---------------------------
import tensorflow as tf
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
RandomFlip = tf.keras.layers.RandomFlip
RandomRotation = tf.keras.layers.RandomRotation
RandomZoom = tf.keras.layers.RandomZoom

# ---------------------------
# Paths for saving pre-trained models
# ---------------------------
LEAF_MODEL_PATH = "leaf_health_model.h5"
YIELD_MODEL_PATH = "yield_model.pkl"

# ---------------------------
# WATCHDOG SETUP (optional)
# ---------------------------
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        print(f"[Watchdog] File modified: {event.src_path}")

def start_watchdog(path='.'):
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=True)
    observer_thread = threading.Thread(target=observer.start)
    observer_thread.daemon = True
    observer_thread.start()
    return observer

start_watchdog()

# ---------------------------
# CUSTOM CSS & TRANSLATION SETUP
# ---------------------------
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

html, body {
    height: 100%;
    margin: 0;
    padding: 0;
    font-family: 'Inter', sans-serif;
    background: url('https://images.unsplash.com/photo-1518837695005-2083093ee35b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80') no-repeat center center fixed;
    background-size: cover;
}

.overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.75);
    z-index: -1;
}

header {
    text-align: center;
    padding: 2rem 0;
    color: #f0f0f0;
}

header h1 {
    font-size: 3.5rem;
    margin: 0;
}

header p {
    font-size: 1.3rem;
    margin: 0.5rem 0 0;
    color: #aaa;
}

.card {
    background: rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 2rem;
    margin: 2rem auto;
    max-width: 400px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.5);
    backdrop-filter: blur(5px);
    -webkit-backdrop-filter: blur(5px);
    border: 1px solid rgba(255,255,255,0.3);
}

.stTextInput>div>div>input {
    background-color: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    color: #f0f0f0;
}

.stButton>button {
    background-color: #4a90e2;
    color: #fff;
    border: none;
    padding: 0.8rem 1.6rem;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.stButton>button:hover {
    background-color: #357ABD;
}

.sidebar .css-1d391kg {
    background: rgba(0,0,0,0.85);
    padding: 1rem;
    border-radius: 12px;
}

hr {
    border: 1px solid #444;
}

a {
    color: #4a90e2;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

.icon {
    margin-right: 0.5rem;
    color: #4a90e2;
}

.section-title {
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
}

.section-title i {
    font-size: 1.5rem;
    margin-right: 0.5rem;
}

.prediction-text {
    color: gray;
    font-size: 0.9rem;
    margin-top: -8px;
}

/* --- Custom styling for tabs --- */
/* This selector targets tab buttons in recent versions of Streamlit */
div.stTabs > div > button {
    background: #ffffff;
    border-radius: 50%;
    padding: 10px 16px;
    margin: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    transition: background 0.3s ease, transform 0.3s ease;
}
div.stTabs > div > button:hover {
    transform: scale(1.05);
}
div.stTabs > div > button[aria-selected="true"] {
    background: #4a90e2;
    color: #fff;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown("<div class='overlay'></div>", unsafe_allow_html=True)

translator = Translator()

@st.cache_data(show_spinner=False)
def translate_text(text, dest_language):
    try:
        return translator.translate(text, dest=dest_language).text
    except Exception:
        return text

def tr(text):
    lang = st.session_state.get("lang", "English")
    return text if lang == "English" else translate_text(text, dest_language=lang)

# ---------------------------
# HEADER FUNCTION
# ---------------------------
def show_header():
    st.markdown(f"""
    <header>
        <h1><i class="fa-solid fa-tractor"></i> {tr("Agrinfo")}</h1>
        <p>{tr("Unleash AI-Driven Insights for Your Farm")}</p>
    </header>
    """, unsafe_allow_html=True)

# ---------------------------
# EARTH ENGINE INITIALIZATION
# ---------------------------
try:
    ee.Initialize(project='ee-soveetprusty')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='ee-soveetprusty')

# ---------------------------
# MONGODB CONNECTION (for authentication & inventory)
# ---------------------------
username_db = quote_plus("soveetprusty")
password_db = quote_plus("@Noobdamaster69")
connection_string = f"mongodb+srv://{username_db}:{password_db}@cluster0.bjzstq0.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(connection_string, tls=True, tlsAllowInvalidCertificates=True)
db = client["agri_app"]
farmers_col = db["farmers"]
crop_inventory_col = db["crop_inventory"]
pesticide_inventory_col = db["pesticide_inventory"]

# ---------------------------
# DEFAULTS & HELPER VARIABLES
# ---------------------------
GOOGLE_MAPS_EMBED_API_KEY = "AIzaSyAWHIWaKtmhnRfXL8_FO7KXyuWq79MKCvs"
default_crop_prices = {"Wheat": 20, "Rice": 25, "Maize": 18, "Sugarcane": 30, "Cotton": 40}
soil_types = ["Sandy", "Loamy", "Clay", "Silty"]

# ---------------------------
# Session State Initialization (if not set)
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "lang" not in st.session_state:
    st.session_state.lang = "English"
if "username" not in st.session_state:
    st.session_state.username = ""
if "city_input" not in st.session_state:
    st.session_state.city_input = ""

# ---------------------------
# HELPER FUNCTIONS FOR WEATHER, NDVI, & SHOPS
# ---------------------------
@st.cache_data(show_spinner=False)
def get_weather_data(city_name):
    geo_url = "https://nominatim.openstreetmap.org/search"
    params_geo = {"city": city_name, "country": "India", "format": "json"}
    r_geo = requests.get(geo_url, params=params_geo, headers={"User-Agent": "Mozilla/5.0"})
    if r_geo.status_code != 200 or not r_geo.json():
        return None, None, None, None, None, None
    geo_data = r_geo.json()[0]
    lat = float(geo_data["lat"])
    lon = float(geo_data["lon"])
    weather_url = "https://api.open-meteo.com/v1/forecast"
    params_weather = {"latitude": lat, "longitude": lon,
                      "current_weather": "true", "hourly": "precipitation",
                      "timezone": "Asia/Kolkata"}
    r_wth = requests.get(weather_url, params=params_weather)
    if r_wth.status_code != 200:
        return None, None, lat, lon, None, None
    wdata = r_wth.json()
    current_temp = wdata["current_weather"]["temperature"]
    current_time = wdata["current_weather"]["time"]
    hourly_times = wdata["hourly"]["time"]
    hourly_precip = wdata["hourly"]["precipitation"]
    current_precip = hourly_precip[hourly_times.index(current_time)] if current_time in hourly_times else 0
    return current_temp, current_precip, lat, lon, hourly_precip, hourly_times

@st.cache_data(show_spinner=False)
def get_real_ndvi(lat, lon):
    point = ee.Geometry.Point(lon, lat)
    region = point.buffer(5000)
    today = datetime.date.today()
    start_date = str(today - datetime.timedelta(days=30))
    end_date = str(today)
    s2 = ee.ImageCollection('COPERNICUS/S2') \
            .filterBounds(region).filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    def add_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    s2 = s2.map(add_ndvi)
    ndvi_image = s2.select('NDVI').median()
    ndvi_dict = ndvi_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=30)
    ndvi_value = ee.Number(ndvi_dict.get('NDVI')).getInfo()
    return ndvi_value

@st.cache_data(show_spinner=False)
def get_live_shop_list(lat, lon, radius=7000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node(around:{radius}, {lat}, {lon})["shop"];
    out body;
    """
    r = requests.post(overpass_url, data=query)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    elements = data.get("elements", [])
    keywords = ["agro", "farm", "agr", "hort", "garden", "agriculture"]
    exclusions = ["clothes", "apparel", "fashion", "footwear"]
    shops = []
    for elem in elements:
        tags = elem.get("tags", {})
        name = tags.get("name", "").strip()
        shop_tag = tags.get("shop", "").strip()
        if not name:
            continue
        if any(exc in name.lower() for exc in exclusions):
            continue
        if not (any(k in name.lower() for k in keywords) or any(k in shop_tag.lower() for k in keywords)):
            continue
        addr_full = tags.get("addr:full", "").strip()
        address = addr_full if addr_full else "Address not available"
        shops.append({"Name": name, "Type": shop_tag, "Address": address})
    df = pd.DataFrame(shops)
    if not df.empty:
        df.index = np.arange(1, len(df) + 1)
        df.index.name = "No."
    return df

def style_shops_dataframe(shops_df):
    shops_df_renamed = shops_df.rename(columns={
        "Name": tr("Shop Name"),
        "Type": tr("Category"),
        "Address": tr("Full Address")
    })
    styled_df = shops_df_renamed.style.set_properties(**{"border": "1px solid #444", "padding": "6px"})\
                           .set_table_styles([
                               {"selector": "th", "props": [
                                   ("background-color", "#2c2c2c"),
                                   ("font-weight", "bold"),
                                   ("text-align", "center"),
                                   ("color", "#e0e0e0")
                               ]},
                               {"selector": "td", "props": [
                                   ("text-align", "left"),
                                   ("vertical-align", "top"),
                                   ("color", "#e0e0e0")
                               ]}
                           ])
    return styled_df

# ---------------------------
# SYNTHETIC LEAF HEALTH CLASSIFICATION MODEL
# (Healthy vs. Not Healthy)
# ---------------------------
IMG_HEIGHT = 64
IMG_WIDTH = 64
NUM_CHANNELS = 3
NUM_CLASSES = 2

def train_leaf_model_fn(epochs=20):
    num_samples_per_class = 800
    X = []
    y = []
    for _ in range(num_samples_per_class):
        img = np.random.randint(100, 256, size=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=np.uint8)
        img[:, :, 1] = np.random.randint(180, 256, (IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        X.append(img / 255.0)
        y.append(0)
    for _ in range(num_samples_per_class):
        img = np.random.randint(80, 256, size=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=np.uint8)
        img[:, :, 1] = np.random.randint(50, 140, (IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        patch_count = np.random.randint(5, 12)
        for _ in range(patch_count):
            patch_size = np.random.randint(8, 20)
            x_start = np.random.randint(0, IMG_HEIGHT - patch_size)
            y_start = np.random.randint(0, IMG_WIDTH - patch_size)
            color = [150, 75, 0] if random.random() < 0.5 else [200, 200, 50]
            img[x_start:x_start+patch_size, y_start:y_start+patch_size] = color
        X.append(img / 255.0)
        y.append(1)
    X = np.array(X)
    y = np.array(y)
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2),
    ], name="data_augmentation")
    model = Sequential([
        data_augmentation,
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)
    return model

if os.path.exists(LEAF_MODEL_PATH):
    leaf_health_model = tf.keras.models.load_model(LEAF_MODEL_PATH)
else:
    leaf_health_model = train_leaf_model_fn(epochs=20)
    leaf_health_model.save(LEAF_MODEL_PATH)

def classify_leaf(image: Image.Image):
    image = image.resize((IMG_WIDTH, IMG_HEIGHT)).convert("RGB")
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = leaf_health_model.predict(arr)
    class_idx = np.argmax(preds, axis=1)[0]
    return "Healthy" if class_idx == 0 else "Not Healthy"

# ---------------------------
# SYNTHETIC YIELD PREDICTION MODEL
# ---------------------------
def train_yield_model_fn(num_samples=500):
    np.random.seed(42)
    ndvi = np.random.uniform(0.3, 0.9, num_samples)
    temperature = np.random.uniform(10, 40, num_samples)
    precipitation = np.random.uniform(0, 50, num_samples)
    soil_type = np.random.randint(0, 4, num_samples)
    soil_map = {0: 0.9, 1: 1.0, 2: 0.8, 3: 0.95}
    y = []
    for i in range(num_samples):
        stype = soil_map[soil_type[i]]
        val = 20 * ndvi[i] + 0.5 * (temperature[i] - 20) - 0.2 * (precipitation[i] - 25)**2
        val *= stype
        val += 50
        val += np.random.normal(0, 3)
        y.append(val)
    y = np.array(y)
    y = np.clip(y, 0, None)
    X = np.column_stack((ndvi, temperature, precipitation, soil_type))
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=80, random_state=42)
    rf.fit(X, y)
    return rf

if os.path.exists(YIELD_MODEL_PATH):
    yield_model = joblib.load(YIELD_MODEL_PATH)
else:
    yield_model = train_yield_model_fn(num_samples=500)
    joblib.dump(yield_model, YIELD_MODEL_PATH)

def predict_yield(ndvi, temperature, precipitation, soil_type_str):
    soil_map = {"Sandy": 0, "Loamy": 1, "Clay": 2, "Silty": 3}
    soil_val = soil_map.get(soil_type_str, 1)
    X = np.array([[ndvi, temperature, precipitation, soil_val]])
    y_pred = yield_model.predict(X)
    return round(y_pred[0], 2)

# ---------------------------
# AUTHENTICATION FUNCTIONS
# ---------------------------
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def register_farmer(username, password):
    if farmers_col.find_one({"username": username}):
        return False, tr("Username already exists.")
    hashed_pw = hash_password(password)
    farmers_col.insert_one({"username": username, "password": hashed_pw})
    st.session_state.logged_in = False
    st.session_state.page = "login"
    return True, tr("Registration successful.")

def login_farmer(username, password):
    user = farmers_col.find_one({"username": username})
    if user and check_password(password, user["password"]):
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.page = "main"
        return True, tr("Login successful.")
    return False, tr("Invalid username or password.")

# ---------------------------
# PAGE FUNCTIONS
# ---------------------------
def show_login():
    show_header()
    st.markdown(f"""
    <div style='text-align:center; font-size:1.2rem; margin-bottom:1rem;'>
      <i class='fa-solid fa-right-to-bracket icon'></i>{tr("Log in to access your personalized insights.")}
    </div>
    """, unsafe_allow_html=True)
    username = st.text_input(tr("Username"))
    password = st.text_input(tr("Password"), type="password")
    if st.button(tr("Login")):
        success, msg = login_farmer(username, password)
        if not success:
            st.error(msg)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align:center;'>
      <i class='fa-solid fa-user-plus icon'></i>{tr("Don't have an account? Register here")}
    </div>
    """, unsafe_allow_html=True)
    if st.button(tr("Go to Registration")):
        st.session_state.page = "register"

def show_register():
    show_header()
    st.markdown(f"""
    <div style='text-align:center; font-size:1.2rem; margin-bottom:1rem;'>
      <i class='fa-solid fa-user-plus icon'></i>{tr("Create your account to start exploring.")}
    </div>
    """, unsafe_allow_html=True)
    username = st.text_input(tr("Choose a Username"))
    password = st.text_input(tr("Choose a Password"), type="password")
    if st.button(tr("Register")):
        success, msg = register_farmer(username, password)
        if not success:
            st.error(msg)
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button(tr("Back to Login")):
        st.session_state.page = "login"

def show_main_app():
    show_header()
    with st.sidebar:
        current_user = st.session_state.get("username", "")
        if not current_user:
            current_user = "Guest"
        st.write(f"Logged in as: **{current_user}**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.page = "login"
        st.markdown("---")
        lang_options = ["English", "Hindi", "Tamil", "Telugu", "Marathi", "Bengali"]
        st.session_state.lang = st.selectbox(tr("Select Language for Translation:"), lang_options, index=0)
        typed = st.text_input(tr("Enter your City:"), placeholder=tr("Type city..."), value=st.session_state.city_input)
        st.session_state.city_input = typed
        recommended_cities = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Ahmedabad", "Chennai", "Kolkata", "Pune", "Jaipur"]
        typed_stripped = typed.strip()
        best_match = None
        if typed_stripped:
            matches = [c for c in recommended_cities if c.lower().startswith(typed_stripped.lower())]
            if matches:
                best_match = matches[0]
        if best_match and best_match.lower() != typed_stripped.lower():
            remainder = best_match[len(typed_stripped):]
            st.markdown(
                f"""
                <div style="margin-top:-5px;">
                  <strong>{tr("Prediction")}:</strong>
                  <span>{typed_stripped}</span>
                  <span class="prediction-text">{remainder}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    city_name = st.session_state.city_input.strip()
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        tr("Irrigation & Satellite"),
        tr("Fertilizer & Pesticide"),
        tr("Inventory Management"),
        tr("Leaf Health Classification"),
        tr("Yield Prediction")
    ])
    with tab1:
        if not city_name:
            st.warning(tr("Please enter a city name above."))
        else:
            temp, current_precip, lat, lon, hourly_precip, hourly_times = get_weather_data(city_name)
            if temp is None:
                st.error(tr("Could not fetch weather data. Check city name."))
            else:
                avg_rain = np.mean(hourly_precip[-3:]) if hourly_precip and len(hourly_precip) >= 3 else current_precip
                st.markdown(f"<div class='section-title'><i class='fa-solid fa-cloud-sun'></i><strong>{tr('Weather in')} {city_name}</strong></div>", unsafe_allow_html=True)
                st.write(f"**{tr('Temperature')}:** {temp} °C")
                st.write(f"**{tr('Current Rain')}:** {current_precip} mm")
                st.write(f"**{tr('Avg Forecast Rain (next 3 hrs)')}:** {avg_rain:.2f} mm")
                irrigation_req = max(0, 25 + (temp - 20) - avg_rain)
                st.markdown(f"<div class='section-title'><i class='fa-solid fa-tint'></i><strong>{tr('Irrigation Recommendation')}</strong></div>", unsafe_allow_html=True)
                st.write(f"**{tr('Recommended Irrigation')}:** {irrigation_req:.2f} mm")
                if irrigation_req > 40:
                    st.warning(tr("High water requirement! Your crop is stressed."))
                elif irrigation_req > 10:
                    st.info(tr("Moderate water requirement."))
                else:
                    st.success(tr("Low water requirement."))
                st.markdown(f"<div class='section-title'><i class='fa-solid fa-satellite'></i><strong>{tr('Satellite View')}</strong></div>", unsafe_allow_html=True)
                if GOOGLE_MAPS_EMBED_API_KEY and lat is not None and lon is not None:
                    maps_url = (f"https://www.google.com/maps/embed/v1/view?"
                                f"key={GOOGLE_MAPS_EMBED_API_KEY}&center={lat},{lon}"
                                f"&zoom=18&maptype=satellite")
                    components.html(f'<iframe width="100%" height="450" src="{maps_url}" frameborder="0" allowfullscreen></iframe>', height=450)
                else:
                    st.info(tr("Google Maps Embed API key not provided or invalid lat/lon."))
    with tab2:
        if not city_name:
            st.warning(tr("Please enter a city name above."))
        else:
            temp, current_precip, lat, lon, _, _ = get_weather_data(city_name)
            if temp is None:
                st.error(tr("Could not fetch weather data. Check city name."))
            else:
                try:
                    with ThreadPoolExecutor() as executor:
                        future_ndvi = executor.submit(get_real_ndvi, lat, lon)
                        future_shops = executor.submit(get_live_shop_list, lat, lon)
                        ndvi_val = future_ndvi.result()
                        shops_df = future_shops.result()
                except Exception:
                    st.error(tr("Error fetching data concurrently."))
                    ndvi_val, shops_df = None, pd.DataFrame()
                if ndvi_val is not None:
                    soil_selected = st.selectbox(tr("Select Soil Type:"), soil_types, key='soil_for_fert')
                    st.markdown(f"<div class='section-title'><i class='fa-solid fa-seedling'></i><strong>{tr('Fertilizer & Pesticide Recommendations')}</strong></div>", unsafe_allow_html=True)
                    st.write(f"**{tr('Soil Type')}:** {soil_selected}")
                    st.write(f"**{tr('Real NDVI')}:** {ndvi_val:.2f}")
                    st.write(f"**{tr('Fertilizer Recommendation')}:** {tr('Moderate NPK mix (Balanced fertilizer)')}")
                    st.write(f"**{tr('Pesticide Recommendation')}:** {tr('Targeted pesticide (e.g., Imidacloprid)')}")
                else:
                    st.error(tr("NDVI data unavailable."))
                st.markdown(f"<div class='section-title'><i class='fa-solid fa-shop'></i><strong>{tr('Nearby Agro-Shops')}</strong></div>", unsafe_allow_html=True)
                if shops_df.empty:
                    st.info(tr("No nearby agro-shops found."))
                else:
                    styled_df = style_shops_dataframe(shops_df)
                    st.dataframe(styled_df, use_container_width=True)
    with tab3:
        st.markdown(f"<div class='section-title'><i class='fa-solid fa-leaf'></i><strong>{tr('Crop Inventory Management')}</strong></div>", unsafe_allow_html=True)
        crop_selected = st.selectbox(tr("Select a Crop:"), list(default_crop_prices.keys()))
        quantity = st.number_input(tr("Enter Quantity (in kg):"), min_value=0, value=0, step=1)
        price = st.number_input(tr("Enter Market Price (per kg):"), min_value=0, value=default_crop_prices[crop_selected], step=1)
        if st.button(tr("Add Crop"), key='crop_add'):
            crop_inventory_col.insert_one({
                "username": st.session_state.username,
                "crop": crop_selected,
                "quantity": quantity,
                "price": price
            })
            st.success(tr("Crop inventory added."))
        with ThreadPoolExecutor() as executor:
            future_crop = executor.submit(list, crop_inventory_col.find({"username": st.session_state.username}, {"_id": 0}))
            future_pest = executor.submit(list, pesticide_inventory_col.find({"username": st.session_state.username}, {"_id": 0}))
            user_crops = future_crop.result()
            user_pesticides = future_pest.result()
        if user_crops:
            st.write(tr("### Current Crop Inventory"))
            df_crop = pd.DataFrame(user_crops)
            df_crop.index = range(1, len(df_crop) + 1)
            st.dataframe(df_crop)
            total_price = (df_crop["quantity"] * df_crop["price"]).sum()
            st.write(f"*{tr('Total Inventory Price')}:* {total_price}")
        st.markdown(f"<div class='section-title'><i class='fa-solid fa-box'></i><strong>{tr('Pesticide Inventory Management')}</strong></div>", unsafe_allow_html=True)
        pesticide_name = st.text_input(tr("Enter Pesticide Name:"), key='pest_name')
        pesticide_qty = st.number_input(tr("Enter Quantity (liters/kg):"), min_value=0, value=0, step=1, key='pest_qty')
        if st.button(tr("Add Pesticide"), key='pest_add'):
            pesticide_inventory_col.insert_one({
                "username": st.session_state.username,
                "pesticide": pesticide_name,
                "quantity": pesticide_qty
            })
            st.success(tr("Pesticide inventory added."))
        if user_pesticides:
            st.write(tr("### Current Pesticide Inventory"))
            df_pest = pd.DataFrame(user_pesticides)
            df_pest.index = range(1, len(df_pest) + 1)
            st.dataframe(df_pest)
    with tab4:
        st.markdown(f"<div class='section-title'><i class='fa-solid fa-leaf'></i><strong>{tr('Leaf Health Classification')}</strong></div>", unsafe_allow_html=True)
        st.write(tr("Upload an image of a leaf to classify its health (Healthy vs. Not Healthy)."))
        uploaded_file = st.file_uploader(tr("Upload Leaf Image"), type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=tr("Uploaded Leaf Image"), use_container_width=True)
            prediction = classify_leaf(image)
            st.markdown(f"<h4>{tr('Prediction')}: {prediction}</h4>", unsafe_allow_html=True)
            if prediction != "Healthy":
                st.warning(tr("Alert: The leaf appears to be Not Healthy. Consider further inspection."))
            else:
                st.success(tr("The leaf appears to be Healthy."))
    with tab5:
        st.markdown(f"<div class='section-title'><i class='fa-solid fa-chart-line'></i><strong>{tr('Yield Prediction')}</strong></div>", unsafe_allow_html=True)
        st.write(tr("Estimate your crop yield (tons per hectare) using current NDVI, temperature, precipitation, and soil type."))
        if not city_name:
            st.warning(tr("Please enter a city name above."))
        else:
            temp, current_precip, lat, lon, _, _ = get_weather_data(city_name)
            if temp is None:
                st.error(tr("Could not fetch weather data. Check city name."))
            else:
                try:
                    ndvi_val = get_real_ndvi(lat, lon)
                except Exception:
                    st.error(tr("Error fetching NDVI data."))
                    ndvi_val = None
                soil_selected = st.selectbox(tr("Select Soil Type:"), soil_types, key='yield_soil')
                if ndvi_val is not None:
                    predicted_yield = predict_yield(ndvi_val, temp, current_precip, soil_selected)
                    st.markdown(f"<h4>{tr('Predicted Yield (tons/ha)')}: {predicted_yield}</h4>", unsafe_allow_html=True)
                else:
                    st.error(tr("Yield prediction unavailable due to NDVI data issue."))

def main():
    if st.session_state.page == "main":
        show_main_app()
    elif st.session_state.page == "register":
        show_register()
    else:
        show_login()

if __name__ == "__main__":
    main()
