# AI Micro-Decision Optimizer (Free & Local)

Predict your daily energy (1–10) from small decisions (sleep, coffee, screen time, steps, water, stress, mood, etc.). 
Local ML (RandomForest) + baseline fallback, related questions, what-if simulation, and charts — all free, no paid APIs.

## Run
1) Python 3.10+
2) pip install -r requirements.txt
3) streamlit run app.py

Data saves to user_data.csv next to the app. To change:
export MDO_DATA_PATH=/full/path/to/your_data.csv

Tips:
- Accuracy improves after ~20 labeled days (enter today's actual energy).
- Everything runs locally/private.
