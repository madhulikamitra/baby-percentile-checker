import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import requests
from scipy.stats import norm
from datetime import date

# ===== Streamlit Page Config =====
st.set_page_config(page_title="Baby Tracker AI", page_icon="🍼", layout="centered")
st.title("👶 Baby Weight Tracker")

# ===== Helper Functions =====
@st.cache_data
def load_history():
    df = pd.read_csv("baby_weights.csv", parse_dates=["date"])
    return df.sort_values("date")

def calculate_age_in_months(dob, today=None):
    today = today or date.today()
    days = (today - dob).days
    return round(days / 30.4375, 2)

def calculate_percentile(weight, L, M, S):
    if L == 0:
        z = (np.log(weight / M)) / S
    else:
        z = (((weight / M) ** L) - 1) / (L * S)
    return norm.cdf(z) * 100

def calculate_weight_from_percentile(p, L, M, S):
    z = norm.ppf(p / 100)
    if L == 0:
        return M * np.exp(S * z)
    else:
        return M * ((L * S * z + 1) ** (1 / L))

def ask_openrouter(prompt):
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['openrouter_key']}",
        "HTTP-Referer": "https://baby-percentile-checker.streamlit.app",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [
            {"role": "system", "content": "You are a friendly pediatric AI assistant that helps parents understand baby growth and feeding."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"OpenRouter API error: {response.text}")
    return response.json()["choices"][0]["message"]["content"]

# ===== User Inputs =====
sex = st.radio("Select baby's sex:", ["Girl", "Boy"])
dob = st.date_input("Baby's Date of Birth", value=datetime.date(2025, 5, 8))
weight_input = st.text_input("Current weight (kg)", value="")
percentile_input = st.text_input("Target percentile (1–99)", value="50")
feeding_type = st.radio("Feeding Type", ["Breastfeeding only", "Breastfeeding + Formula", "Formula only", "Weaning / solids"])

# ===== Load Weight History & Append New =====
current_date = datetime.date.today()
history = load_history()
history["date"] = pd.to_datetime(history["date"])
try:
    weight_val = float(weight_input)
    if not ((history["date"].dt.date == current_date) & (np.isclose(history["weight_kg"], weight_val))).any():
        new_row = pd.DataFrame([{"date": pd.to_datetime(current_date), "weight_kg": weight_val}])
        history = pd.concat([history, new_row], ignore_index=True)
        history["date"] = pd.to_datetime(history["date"])
        history = history.sort_values("date")
except ValueError:
    pass

# ===== Display Weight History Chart =====
st.subheader("📊 Baby Weight History")
history["date"] = pd.to_datetime(history["date"]).dt.date
chart = alt.Chart(history).mark_line().encode(x='date', y='weight_kg').properties(width=500, height=200)
st.altair_chart(chart, use_container_width=True)
with st.expander("📋 Show Full Weight Table"):
    st.dataframe(history, hide_index=True)

# ===== LMS Curve Calculations =====
try:
    weight = float(weight_input)
    target_percentile = int(percentile_input)
    if not 1 <= target_percentile <= 99:
        st.error("❌ Please enter a target percentile between 1 and 99.")
        st.stop()
except ValueError:
    st.warning("Please enter valid numeric values for weight and percentile.")
    st.stop()

age_months = calculate_age_in_months(dob)
st.write(f"🕒 Baby is **{age_months:.2f} months** old")

lms_file = "who_lms_girls.csv" if sex == "Girl" else "who_lms_boys.csv"
df = pd.read_csv(lms_file)
L = np.interp(age_months, df["Age"], df["L"])
M = np.interp(age_months, df["Age"], df["M"])
S = np.interp(age_months, df["Age"], df["S"])

percentile = calculate_percentile(weight, L, M, S)
expected_weight = calculate_weight_from_percentile(target_percentile, L, M, S)

# ===== Metrics and Health Info =====
st.metric(label="Current Percentile", value=f"{percentile:.1f}th")
st.metric(label=f"Expected Weight @ {target_percentile}th", value=f"{expected_weight:.2f} kg")

diff = weight - expected_weight
if abs(diff) < 0.2:
    st.success("✅ Baby's weight is close to the target percentile.")
elif diff > 0:
    st.info(f"📈 Baby is above the {target_percentile}th percentile by {diff:.2f} kg.")
else:
    st.info(f"📉 Baby is below the {target_percentile}th percentile by {abs(diff):.2f} kg.")

if percentile < 5:
    st.error("⚠️ Baby is under the 5th percentile. Please consult your pediatrician.")
elif percentile > 95:
    st.error("⚠️ Baby is above the 95th percentile. Monitor growth with doctor.")

# ===== Growth Curve Plot =====
ages = np.linspace(0, 60, 121)
percentiles = [5, 50, 95]
colors = {5: 'blue', 50: 'green', 95: 'red'}
plt.figure(figsize=(8, 5))
for p in percentiles:
    weights = [calculate_weight_from_percentile(
        p,
        np.interp(a, df["Age"], df["L"]),
        np.interp(a, df["Age"], df["M"]),
        np.interp(a, df["Age"], df["S"])
    ) for a in ages]
    plt.plot(ages, weights, label=f'{p}th percentile', color=colors[p])

plt.scatter(age_months, weight, color='black', label="Your Baby")
plt.xlabel("Age (months)")
plt.ylabel("Weight (kg)")
plt.title("Baby Weight vs WHO Growth Standards")
plt.legend()
plt.grid(True)
st.pyplot(plt)

# ===== AI Assistant Triggered by Weight Entry =====
if weight_input:
    st.subheader("🤖 AI Advice Based on Current Entry")
    with st.spinner("Thinking..."):
        recent_trend = history.tail(5).to_string(index=False)
        full_prompt = f"""
You are a pediatric AI assistant helping parents understand baby growth using weight trends and WHO percentile charts.

Baby Overview:
- Age: {age_months:.2f} months
- Current Weight: {weight} kg
- Weight Percentile: {percentile:.1f}
- Sex: {sex}
- Feeding Type: {feeding_type}

Recent weight trend:
{recent_trend}

Give a general update or concern based on this info.
"""
        try:
            auto_reply = ask_openrouter(full_prompt)
            st.markdown(auto_reply)
        except Exception as e:
            st.error(str(e))

# ===== Manual Gen-AI Assistant =====
st.subheader("🤖 Ask the AI Assistant")
user_question = st.text_area("Ask something like: *Is this weight normal for her age?*", key="ai_input")
if st.button("Get Advice") and user_question:
    with st.spinner("Thinking..."):
        recent_trend = history.tail(5).to_string(index=False)
        full_prompt = f"""
You are a pediatric AI assistant helping parents understand baby growth using weight trends and WHO percentile charts.

Baby Overview:
- Age: {age_months:.2f} months
- Current Weight: {weight} kg
- Weight Percentile: {percentile:.1f}
- Sex: {sex}
- Feeding Type: {feeding_type}

Recent weight trend:
{recent_trend}

Answer the user's question clearly and warmly:
\"{user_question}\"
"""
        try:
            reply = ask_openrouter(full_prompt)
            st.success("AI Assistant says:")
            st.markdown(reply)
        except Exception as e:
            st.error(str(e))
