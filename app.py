# AI Micro-Decision Optimizer — Streamlit Cloud Safe Version
# ---------------------------------------------------------
# Features:
# - Daily energy prediction (1–10)
# - Multi-factor inputs: sleep, coffee, sugar, screen time, steps, water, stress, mood, previous day energy
# - Baseline fallback if scikit-learn not available
# - Related questions engine
# - Recommendations
# - What-if simulation
# - Trend charts (energy over time, factor importance)

import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Tuple

# -----------------------------
# Check for scikit-learn
# -----------------------------
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    sklearn_available = True
except ModuleNotFoundError:
    sklearn_available = False
    st.warning("scikit-learn not available — baseline prediction will be used.")

# -----------------------------
# Constants
# -----------------------------
DATA_PATH = os.environ.get("MDO_DATA_PATH", "user_data.csv")
FEATURES = [
    "sleep_hours",
    "sleep_quality",
    "coffee_cups",
    "sugar_servings",
    "screen_time_hours",
    "steps",
    "water_liters",
    "stress_level",
    "mood",
    "prev_day_energy",
]
TARGET = "energy_today"

# -----------------------------
# Utility functions
# -----------------------------
def ensure_csv(path: str) -> None:
    if not os.path.exists(path):
        df = pd.DataFrame(columns=["date"] + FEATURES + [TARGET])
        df.to_csv(path, index=False)

def load_data(path: str) -> pd.DataFrame:
    ensure_csv(path)
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame(columns=["date"] + FEATURES + [TARGET])
    for col in FEATURES + [TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def append_row(path: str, row: Dict[str, Any]) -> None:
    df = load_data(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)

# -----------------------------
# Prediction functions
# -----------------------------
def train_model(df: pd.DataFrame) -> Tuple[Any, Dict[str, float]]:
    if not sklearn_available:
        return None, {f: 0.0 for f in FEATURES}
    train_df = df.dropna(subset=[TARGET]).copy()
    if len(train_df) < 20:
        return None, {f: 0.0 for f in FEATURES}
    X = train_df[FEATURES].fillna(train_df[FEATURES].median())
    y = train_df[TARGET].clip(1, 10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.sidebar.success(f"Model trained on {len(train_df)} days — R²: {r2:.2f}, MAE: {mae:.2f}")
    importances = {}
    if hasattr(model, "feature_importances_"):
        for i, f in enumerate(FEATURES):
            importances[f] = float(model.feature_importances_[i])
    return model, importances

def baseline_predict(row: Dict[str, Any]) -> float:
    energy = row.get("prev_day_energy", 6.0)
    sleep = row.get("sleep_hours", 7.0)
    energy += (sleep - 7.0) * 0.4
    coffee = row.get("coffee_cups", 1.0)
    if coffee > 2:
        energy -= 0.3 * (coffee - 2)
    sugar = row.get("sugar_servings", 1.0)
    if sugar > 2:
        energy -= 0.2 * (sugar - 2)
    stress = row.get("stress_level", 5.0)
    energy -= max(0, stress - 5) * 0.2
    steps = row.get("steps", 5000)
    energy += (min(steps, 10000) - 5000) / 5000 * 0.4
    screen = row.get("screen_time_hours", 4.0)
    energy -= max(0, screen - 5) * 0.2
    water = row.get("water_liters", 2.0)
    energy += max(0, water - 2) * 0.2
    mood = row.get("mood", 6.0)
    energy += (mood - 6) * 0.3
    return float(np.clip(energy, 1.0, 10.0))

def simulate_prediction(test_row: Dict[str, Any], df: pd.DataFrame) -> float:
    if sklearn_available:
        model, _ = train_model(df)
        if model is not None:
            X = pd.DataFrame([test_row])[FEATURES].fillna(0)
            return float(np.clip(model.predict(X)[0], 1.0, 10.0))
    return baseline_predict(test_row)

# -----------------------------
# Recommendations
# -----------------------------
def make_recommendations(row: Dict[str, Any], pred: float) -> List[str]:
    recs = []
    if row.get("sleep_hours", 7.0) < 6.5:
        recs.append("Add 30–60 mins of sleep tonight; short sleep reduces energy.")
    if row.get("coffee_cups", 1.0) >= 3:
        recs.append("Reduce caffeine after 2 PM; too much can hurt sleep quality.")
    if row.get("sugar_servings", 1.0) >= 3:
        recs.append("Cut back on sugary snacks; energy spikes then dips.")
    if row.get("screen_time_hours", 4.0) > 6:
        recs.append("Take short off-screen breaks every 60–90 mins.")
    if row.get("steps", 5000) < 4000:
        recs.append("A 10–15 min walk can boost alertness.")
    if row.get("water_liters", 2.0) < 1.5:
        recs.append("Increase hydration; mild dehydration reduces energy.")
    if row.get("stress_level", 5.0) >= 7:
        recs.append("Try a 3–5 min breathing break; high stress reduces energy.")
    if row.get("mood", 6.0) <= 4:
        recs.append("Lift your mood: music, sunlight, or a short chat.")
    if pred < 6.0:
        recs.append("Plan deep-focus work early; energy may dip later.")
    if not recs:
        recs.append("Nice balance today—keep routines steady.")
    return recs[:5]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Micro-Decision Optimizer", page_icon="⚡", layout="wide")
st.title("⚡ AI Micro-Decision Optimizer (Safe Cloud Version)")
st.caption("Predict your energy based on small daily habits. Fully free and private.")

# Load data
ensure_csv(DATA_PATH)
df = load_data(DATA_PATH)

# User inputs
st.subheader("1) Enter Today's Inputs")
col1, col2, col3 = st.columns(3)
with col1:
    sleep_hours = st.number_input("Sleep hours", 0.0, 14.0, 7.0, 0.5)
    sleep_quality = st.slider("Sleep quality (1–10)", 1, 10, 7)
    coffee_cups = st.number_input("Coffee cups", 0.0, 10.0, 1.0)
    sugar_servings = st.number_input("Sugar servings", 0.0, 10.0, 1.0)
with col2:
    screen_time_hours = st.number_input("Screen time hours", 0.0, 18.0, 4.0)
    steps = st.number_input("Steps today", 0, 50000, 5000)
    water_liters = st.number_input("Water liters", 0.0, 10.0, 2.0)
with col3:
    stress_level = st.slider("Stress (1–10)", 1, 10, 5)
    mood = st.slider("Mood (1–10)", 1, 10, 6)
    prev_day_energy = st.slider("Yesterday's energy (1–10)", 1, 10, 6)
    energy_today = st.slider("Today's energy (optional, 1–10)", 1, 10, 6)

current_row = {
    "date": dt.date.today().isoformat(),
    "sleep_hours": float(sleep_hours),
    "sleep_quality": float(sleep_quality),
    "coffee_cups": float(coffee_cups),
    "sugar_servings": float(sugar_servings),
    "screen_time_hours": float(screen_time_hours),
    "steps": int(steps),
    "water_liters": float(water_liters),
    "stress_level": float(stress_level),
    "mood": float(mood),
    "prev_day_energy": float(prev_day_energy),
    "energy_today": float(energy_today) if energy_today else np.nan,
}

# Predict & save
st.subheader("2) Predict & Log")
c1, c2 = st.columns(2)
pred = None
with c1:
    if st.button("🔮 Predict Energy"):
        pred = simulate_prediction(current_row, df)
        st.success(f"Predicted energy today: {pred:.1f}/10")
        st.session_state["latest_pred"] = pred
with c2:
    if st.button("📝 Save Today's Row"):
        append_row(DATA_PATH, current_row)
        st.success("Saved today’s data.")
        df = load_data(DATA_PATH)

pred = st.session_state.get("latest_pred", None)

# Recommendations
st.subheader("3) Recommendations")
if pred:
    recs = make_recommendations(current_row, pred)
    for r in recs:
        st.write(f"- {r}")
else:
    st.info("Predict first to see recommendations.")

# What-if simulation
st.subheader("4) What-if Simulation")
if pred:
    adj_sleep = st.slider("Adjust sleep (hours)", -2.0, 2.0, 0.0)
    adj_coffee = st.slider("Adjust coffee (cups)", -2.0, 2.0, 0.0)
    adj_screen = st.slider("Adjust screen time (hours)", -3.0, 3.0, 0.0)
    test_row = current_row.copy()
    test_row["sleep_hours"] = max(0.0, test_row["sleep_hours"] + adj_sleep)
    test_row["coffee_cups"] = max(0.0, test_row["coffee_cups"] + adj_coffee)
    test_row["screen_time_hours"] = max(0.0, test_row["screen_time_hours"] + adj_screen)
    sim_pred = simulate_prediction(test_row, df)
    st.write(f"Simulated energy with changes: {sim_pred:.1f}/10")
else:
    st.info("Predict first to use simulation.")

# Trends
st.subheader("5) Energy Trends")
sub = df.dropna(subset=[TARGET])
if len(sub) >= 2:
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.sort_values("date")
    st.line_chart(sub.set_index("date")[[TARGET]])
else:
    st.info("Log 2+ days with actual energy to see trends.")
