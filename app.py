# AI Micro-Decision Optimizer — Free, Local, Streamlit MVP
# --------------------------------------------------------
# Features:
# - Multi-factor inputs (sleep, coffee, sugar, screen time, steps, water, stress, mood, prior energy)
# - Predict today's energy (1–10) using a local ML model (RandomForestRegressor fallback to baseline)
# - Log data to CSV; continuous learning on your personal data
# - Related Questions engine (contextual suggestions + template answers)
# - What-if simulation (change sleep/coffee/etc and re-predict)
# - Recommendations (rule-based + data-driven hints)
# - Charts: energy over time + factor correlation/importance
#
# Run: streamlit run app.py
# No paid APIs. All local.

import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Tuple

# ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH = os.environ.get("MDO_DATA_PATH", "user_data.csv")

st.set_page_config(
    page_title="AI Micro-Decision Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Utilities
# ----------------------------
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
    # Clean types
    for col in FEATURES + [TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def append_row(path: str, row: Dict[str, Any]) -> None:
    df = load_data(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)

def train_model(df: pd.DataFrame) -> Tuple[Any, Dict[str, float]]:
    # only rows with target
    train_df = df.dropna(subset=[TARGET]).copy()
    if len(train_df) < 20:
        return None, {f: 0.0 for f in FEATURES}
    X = train_df[FEATURES].fillna(train_df[FEATURES].median())
    y = train_df[TARGET].clip(1, 10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
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
    # Simple heuristic baseline if not enough data to train ML
    energy = row.get("prev_day_energy", 6.0)
    # sleep
    sleep = row.get("sleep_hours", 7.0)
    energy += (sleep - 7.0) * 0.4
    # coffee
    coffee = row.get("coffee_cups", 1.0)
    if coffee > 2:
        energy -= 0.3 * (coffee - 2)
    # sugar
    sugar = row.get("sugar_servings", 1.0)
    if sugar > 2:
        energy -= 0.2 * (sugar - 2)
    # stress
    stress = row.get("stress_level", 5.0)
    energy -= max(0, stress - 5) * 0.2
    # steps
    steps = row.get("steps", 5000)
    energy += (min(steps, 10000) - 5000) / 5000 * 0.4
    # screen time
    screen = row.get("screen_time_hours", 4.0)
    energy -= max(0, screen - 5) * 0.2
    # water
    water = row.get("water_liters", 2.0)
    energy += max(0, water - 2) * 0.2
    # mood
    mood = row.get("mood", 6.0)
    energy += (mood - 6) * 0.3
    return float(np.clip(energy, 1.0, 10.0))

def make_recommendations(row: Dict[str, Any], pred: float) -> List[str]:
    recs = []
    if row.get("sleep_hours", 7.0) < 6.5:
        recs.append("Try to add 30–60 mins of sleep tonight; short sleep is dragging energy.")
    if row.get("coffee_cups", 1.0) >= 3:
        recs.append("Reduce caffeine after 2 PM; high intake can reduce evening energy and sleep quality.")
    if row.get("sugar_servings", 1.0) >= 3:
        recs.append("Cut back on sugary snacks; quick spikes may lead to an energy dip.")
    if row.get("screen_time_hours", 4.0) > 6:
        recs.append("Insert short off-screen breaks every 60–90 mins to protect mental energy.")
    if row.get("steps", 5000) < 4000:
        recs.append("A 10–15 min walk can raise alertness and stabilize energy.")
    if row.get("water_liters", 2.0) < 1.5:
        recs.append("Increase hydration; mild dehydration often feels like fatigue.")
    if row.get("stress_level", 5.0) >= 7:
        recs.append("Try a 3–5 min breathing break; high stress correlates with lower energy.")
    if row.get("mood", 6.0) <= 4:
        recs.append("Do a quick mood lift: music, sunlight, or a short chat with a friend.")
    if pred < 6.0:
        recs.append("Plan deep-focus work earlier today; energy may dip later.")
    if not recs:
        recs.append("Nice balance today—keep routines steady and review again tomorrow.")
    return recs[:5]

def related_questions(row: Dict[str, Any], pred: float) -> List[Tuple[str, str]]:
    # Return (question, key) pairs; key used to render answer
    qs = []
    if pred < 6:
        qs += [
            ("Why is my energy predicted to be low today?", "why_low"),
            ("What can I do right now to boost energy?", "boost_now"),
            ("Which habit affected today's energy the most?", "top_factor"),
            ("What if I sleep 1 more hour tonight?", "what_if_sleep_plus1"),
        ]
    else:
        qs += [
            ("Which habits are helping my energy today?", "whats_helping"),
            ("How can I maintain this level through the week?", "maintain_week"),
            ("What if I reduce coffee by 1 cup?", "what_if_coffee_minus1"),
        ]
    qs += [
        ("Over the last 14 days, what correlates most with my energy?", "corr_14d"),
        ("When during the day do I tend to feel low?", "time_pattern"),
    ]
    return qs[:5]

def simulate_prediction(test_row: Dict[str, Any], df: pd.DataFrame) -> float:
    model, _ = train_model(df)
    if model is not None:
        X = pd.DataFrame([test_row])[FEATURES].fillna(0)
        p = float(model.predict(X)[0])
        return float(np.clip(p, 1.0, 10.0))
    return baseline_predict(test_row)

def answer_template(key: str, row: Dict[str, Any], pred: float, df: pd.DataFrame, importances: Dict[str, float]) -> str:
    def top_factor_name() -> str:
        if importances and sum(importances.values()) > 0:
            return max(importances, key=importances.get)
        # fallback heuristic
        weights = {
            "sleep_hours": 0.25,
            "stress_level": 0.2,
            "coffee_cups": 0.15,
            "screen_time_hours": 0.15,
            "steps": 0.1,
            "water_liters": 0.05,
            "mood": 0.05,
            "sleep_quality": 0.03,
            "sugar_servings": 0.015,
            "prev_day_energy": 0.01,
        }
        return max(weights, key=weights.get)

    if key == "why_low":
        return (
            f"Energy is predicted at {pred:.1f}/10. Contributing factors likely include: "
            f"sleep {row.get('sleep_hours', 'NA')}h, caffeine {row.get('coffee_cups','NA')} cups, "
            f"screen time {row.get('screen_time_hours','NA')}h, stress {row.get('stress_level','NA')}/10. "
            f"Improving sleep and managing caffeine timing usually move this up."
        )
    if key == "boost_now":
        return (
            "Quick boosts: 10–15 min brisk walk, 300–500 ml water, 2–3 min breathing reset, "
            "short sunlight exposure, and delay caffeine if you've already had 2+ cups."
        )
    if key == "top_factor":
        topf = top_factor_name().replace("_", " ")
        return f"The most influential factor in your pattern appears to be: {topf}. Focus small changes there first."
    if key == "what_if_sleep_plus1":
        test = row.copy()
        test["sleep_hours"] = row.get("sleep_hours", 7.0) + 1.0
        sim = simulate_prediction(test, df)
        return f"If you sleep 1 more hour tonight, predicted energy could be around {sim:.1f}/10."
    if key == "whats_helping":
        return "Consistent sleep, moderate caffeine, lower stress, and movement seem to support your energy today."
    if key == "maintain_week":
        return "Keep sleep within a 60–90 min window, schedule key tasks in your peak hours, and plan light movement daily."
    if key == "what_if_coffee_minus1":
        test = row.copy()
        test["coffee_cups"] = max(0, row.get("coffee_cups", 1.0) - 1.0)
        sim = simulate_prediction(test, df)
        return f"Reducing coffee by 1 cup could shift predicted energy to about {sim:.1f}/10."
    if key == "corr_14d":
        sub = df.dropna(subset=[TARGET]).tail(14)
        if len(sub) < 5:
            return "Not enough recent data to analyze 14-day correlations yet—log a few more days."
        corrs = sub[FEATURES+[TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
        top = corrs.head(3).round(2).to_dict()
        bottom = corrs.tail(3).round(2).to_dict()
        return f"Top positive correlations (14d): {top}. Notable negatives: {bottom}."
    if key == "time_pattern":
        return "This MVP doesn't track intraday times yet. Tip: add a quick midday energy check to learn your daily rhythm."
    return "Coming soon."

# ----------------------------
# UI
# ----------------------------
st.title("⚡ AI Micro-Decision Optimizer (Free & Local)")
st.caption("Predict energy from small daily choices. Log data, get predictions, and see smart follow-up questions.")

with st.sidebar:
    st.header("Data")
    st.write("Your data is stored locally in a CSV file.")
    st.code(f"CSV path: {os.path.abspath(DATA_PATH)}", language="bash")
    manage = st.radio("Data actions", ["Use existing CSV", "Create/Reset CSV"], index=0)
    if manage == "Create/Reset CSV":
        ensure_csv(DATA_PATH)
        st.warning("CSV initialized/reset. (Existing data cleared.)")
    df = load_data(DATA_PATH)
    st.write(f"Rows in dataset: {len(df)}")

st.subheader("1) Enter Today's Inputs")
col1, col2, col3 = st.columns(3)
with col1:
    sleep_hours = st.number_input("Sleep hours (last night)", min_value=0.0, max_value=14.0, value=7.0, step=0.5)
    sleep_quality = st.slider("Sleep quality (1–10)", 1.0, 10.0, 7.0, step=1.0)
    coffee_cups = st.number_input("Coffee cups today", min_value=0.0, max_value=10.0, value=1.0, step=1.0)
    sugar_servings = st.number_input("Sugar servings today", min_value=0.0, max_value=10.0, value=1.0, step=1.0)
with col2:
    screen_time_hours = st.number_input("Screen time hours (today)", min_value=0.0, max_value=18.0, value=4.0, step=0.5)
    steps = st.number_input("Steps (today)", min_value=0, max_value=50000, value=5000, step=500)
    water_liters = st.number_input("Water (liters)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
with col3:
    stress_level = st.slider("Stress (1–10)", 1, 10, 5, step=1)
    mood = st.slider("Mood (1–10)", 1, 10, 6, step=1)
    prev_day_energy = st.slider("Yesterday's energy (1–10)", 1, 10, 6, step=1)
    energy_today = st.slider("Today's energy (actual, 1–10) — optional", 1, 10, 6, step=1)

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

st.subheader("2) Predict & Log")
c1, c2, c3 = st.columns([1,1,2])
with c1:
    if st.button("🔮 Predict Energy"):
        # Train model (if enough data) then predict
        model, importances = train_model(df)
        if model is not None:
            X = pd.DataFrame([current_row])[FEATURES].fillna(0)
            pred = float(model.predict(X)[0])
            pred = float(np.clip(pred, 1.0, 10.0))
            st.success(f"Predicted energy today: {pred:.1f} / 10")
        else:
            pred = baseline_predict(current_row)
            st.info(f"Predicted energy today (baseline): {pred:.1f} / 10 — log more days to train ML.")
        st.session_state["latest_pred"] = pred
        st.session_state["latest_importances"] = importances if model is not None else {}

with c2:
    if st.button("📝 Save Today's Row"):
        append_row(DATA_PATH, current_row)
        st.success("Saved. (Tip: Log daily to improve predictions.)")
        # refresh df
        df = load_data(DATA_PATH)

with c3:
    import datetime as dtm
    if st.button("🧹 Remove today's existing entries"):
        df2 = df[df["date"] != dtm.date.today().isoformat()].copy()
        df2.to_csv(DATA_PATH, index=False)
        st.warning("Any existing entries for today have been removed.")

st.markdown("---")

# Show latest prediction, related questions, and recommendations
pred = st.session_state.get("latest_pred", None)
importances = st.session_state.get("latest_importances", {})

st.subheader("3) Related Questions")
if pred is None:
    st.info("Make a prediction to see context-aware questions.")
else:
    qs = related_questions(current_row, pred)
    cols = st.columns(len(qs))
    answers = {}
    for i, (q, key) in enumerate(qs):
        with cols[i]:
            if st.button(q, key=f"qbtn_{key}"):
                answers[key] = True
    for key in [k for _, k in qs]:
        if answers.get(key):
            st.write(answer_template(key, current_row, pred, df, importances))

st.subheader("4) Recommendations")
if pred is not None:
    recs = make_recommendations(current_row, pred)
    for r in recs:
        st.write(f"- {r}")
else:
    st.info("After predicting, recommendations will appear here.")

st.markdown("---")
st.subheader("5) What-if Simulation")
if pred is not None:
    adj_sleep = st.slider("If I change sleep by (hours)", -2.0, 2.0, 0.0, step=0.5)
    adj_coffee = st.slider("If I change coffee by (cups)", -2.0, 2.0, 0.0, step=1.0)
    adj_screen = st.slider("If I change screen time by (hours)", -3.0, 3.0, 0.0, step=0.5)
    test_row = current_row.copy()
    test_row["sleep_hours"] = max(0.0, test_row["sleep_hours"] + adj_sleep)
    test_row["coffee_cups"] = max(0.0, test_row["coffee_cups"] + adj_coffee)
    test_row["screen_time_hours"] = max(0.0, test_row["screen_time_hours"] + adj_screen)
    sim_pred = simulate_prediction(test_row, df)
    st.write(f"**Simulated energy** with these changes: {sim_pred:.1f} / 10")
else:
    st.info("Predict first, then try simulations.")

st.markdown("---")
st.subheader("6) Trends & Insights")

tab1, tab2 = st.tabs(["Energy Over Time", "Factor Importance & Correlation"])

with tab1:
    if len(df.dropna(subset=[TARGET])) >= 2:
        chart_df = df.dropna(subset=[TARGET]).copy()
        chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
        chart_df = chart_df.sort_values("date")
        st.line_chart(chart_df.set_index("date")[[TARGET]])
    else:
        st.info("Log at least 2 days with actual energy to see trends.")

with tab2:
    model, importances = train_model(df)
    if model is not None and importances and sum(importances.values()) > 0:
        imp_series = pd.Series(importances).sort_values(ascending=False)
        st.bar_chart(imp_series)
        st.caption("Feature importances from the current model.")
    else:
        sub = df.dropna(subset=[TARGET])
        if len(sub) >= 5:
            corrs = sub[FEATURES+[TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
            st.bar_chart(corrs)
            st.caption("Pearson correlations (requires at least 5 days of data).")
        else:
            st.info("Not enough data for importances or correlations yet. Log more days.")

st.markdown("---")
st.caption("Built to be free, private, and local. Improve accuracy by logging daily.")
