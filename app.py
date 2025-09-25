import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
from scipy.stats import norm
from datetime import date
import requests

# ========== CONFIG ==========
API_URL = "https://baby-api-ffut.onrender.com/"
st.set_page_config(page_title="Baby Tracker AI", page_icon="🍼", layout="centered")
st.title("👶 Baby Weight Tracker")

# ========== UTILITIES ==========

def calculate_age_in_months(dob, today=None):
    today = today or date.today()
    return round((today - dob).days / 30.4375, 2)

def calculate_percentile(weight, L, M, S):
    z = ((np.log(weight / M)) / S) if L == 0 else (((weight / M) ** L) - 1) / (L * S)
    return norm.cdf(z) * 100

def calculate_weight_from_percentile(p, L, M, S):
    z = norm.ppf(p / 100)
    return M * np.exp(S * z) if L == 0 else M * ((L * S * z + 1) ** (1 / L))

@st.cache_data
def load_history():
    try:
        res = requests.get(API_URL + "/weights")
        if res.status_code == 200:
            df = pd.DataFrame(res.json())
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df.sort_values("date")
    except Exception as e:
        st.error(f"Error loading history: {e}")
    return pd.DataFrame(columns=["date", "weight"])

def save_new_weight(date_str, weight_val):
    try:
        res = requests.post(API_URL + "/weights", json={"date": date_str, "weight": weight_val})
        return res.status_code == 200
    except Exception as e:
        st.error(f"❌ Failed to save weight: {e}")
        return False

def ask_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {st.secrets['openrouter_key']}",
        "HTTP-Referer": "https://baby-percentile-checker.streamlit.app",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [
            {"role": "system", "content": "You are a friendly pediatric AI assistant..."},
            {"role": "user", "content": prompt}
        ]
    }
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if res.status_code != 200:
        raise Exception(f"OpenRouter API error: {res.text}")
    return res.json()["choices"][0]["message"]["content"]

@st.cache_data(ttl=86400)
def get_daily_tip(age_bucket, sex, feeding_type):
    prompt = f"""
    You are a helpful pediatric AI assistant. The baby is approximately {age_bucket} old,
    assigned sex at birth: {sex.lower()}, and is currently being fed: {feeding_type.lower()}.
    Give one warm and practical tip (max 2 lines) for the parent.
    """
    return ask_openrouter(prompt)

# ========== USER INPUTS ==========
sex = st.radio("Select baby's sex:", ["Girl", "Boy"])
dob = st.date_input("Baby's Date of Birth", value=datetime.date(2025, 5, 8))
feeding_type = st.radio("Feeding Type", ["Breastfeeding only", "Breastfeeding + Formula", "Formula only", "Weaning / solids"])
age_months = calculate_age_in_months(dob)

st.write(f"🕒 Baby is **{age_months:.2f} months** old")
with st.spinner("💬 Fetching today's baby tip..."):
    try:
        tip = get_daily_tip(f"{int(age_months)} months", sex, feeding_type)
        st.info(f"💡 **Tip of the Day**\n\n{tip}")
    except:
        st.error("Couldn't fetch today's tip.")

# ========== WEIGHT INPUT & STORAGE ==========
weight_input = st.text_input("Current weight (kg)", value="")
percentile_input = st.text_input("Target percentile (1–99)", value="50")
current_date = datetime.date.today()
history = load_history()
history["date"] = pd.to_datetime(history["date"])

try:
    weight_val = float(weight_input)
    if not ((history["date"].dt.date == current_date) & (np.isclose(history["weight"], weight_val))).any():
        if save_new_weight(str(current_date), weight_val):
            st.success("✅ Saved weight to API")
            st.cache_data.clear()
            history = load_history()
except ValueError:
    st.warning("Please enter a valid weight.")

# ========== WEIGHT HISTORY CHART ==========
st.subheader("📊 Baby Weight History")
history["date"] = pd.to_datetime(history["date"]).dt.date
chart = alt.Chart(history).mark_line().encode(x='date', y='weight').properties(width=500, height=200)
st.altair_chart(chart, use_container_width=True)
with st.expander("📋 Show Full Weight Table"):
    st.dataframe(history, hide_index=True)

# ========== LMS + METRICS ==========
try:
    weight = float(weight_input)
    target_percentile = int(percentile_input)
    if not 1 <= target_percentile <= 99:
        st.error("❌ Enter a percentile between 1 and 99.")
        st.stop()
except ValueError:
    st.warning("Enter valid numeric values for weight and percentile.")
    st.stop()

lms_file = "who_lms_girls.csv" if sex == "Girl" else "who_lms_boys.csv"
df = pd.read_csv(lms_file)
L = np.interp(age_months, df["Age"], df["L"])
M = np.interp(age_months, df["Age"], df["M"])
S = np.interp(age_months, df["Age"], df["S"])

percentile = calculate_percentile(weight, L, M, S)
expected_weight = calculate_weight_from_percentile(target_percentile, L, M, S)

st.metric("Current Percentile", f"{percentile:.1f}th")
st.metric(f"Expected Weight @ {target_percentile}th", f"{expected_weight:.2f} kg")

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

# ========== GROWTH CURVE ==========
ages = np.linspace(0, 60, 121)
percentiles = [5, 50, 95]
colors = {5: 'blue', 50: 'green', 95: 'red'}
plt.figure(figsize=(8, 5))
for p in percentiles:
    weights = [calculate_weight_from_percentile(p,
        np.interp(a, df["Age"], df["L"]),
        np.interp(a, df["Age"], df["M"]),
        np.interp(a, df["Age"], df["S"])) for a in ages]
    plt.plot(ages, weights, label=f'{p}th percentile', color=colors[p])

plt.scatter(age_months, weight, color='black', label="Your Baby")
plt.xlabel("Age (months)")
plt.ylabel("Weight (kg)")
plt.title("Baby Weight vs WHO Growth Standards")
plt.legend()
plt.grid(True)
st.pyplot(plt)

# ========== AUTO AI RESPONSE ==========
if weight_input:
    st.subheader("🧐 AI Advice Based on Current Entry")
    with st.spinner("Thinking..."):
        trend = history.tail(5).to_string(index=False)
        prompt = f"""
Baby Overview:
- Age: {age_months:.2f} months
- Weight: {weight} kg
- Percentile: {percentile:.1f}
- Sex: {sex}
- Feeding: {feeding_type}
Recent Trend:
{trend}
Respond warmly and helpfully to summarize any concerns.
"""
        try:
            st.markdown(ask_openrouter(prompt))
        except Exception as e:
            st.error(str(e))

# ========== MANUAL AI CHAT ==========
st.subheader("🧐 Ask the AI Assistant")
user_q = st.text_area("Ask something like: *Is this weight normal for her age?*", key="ai_input")
if st.button("Get Advice") and user_q:
    with st.spinner("Thinking..."):
        trend = history.tail(5).to_string(index=False)
        prompt = f"""
Baby Overview:
- Age: {age_months:.2f} months
- Weight: {weight} kg
- Percentile: {percentile:.1f}
- Sex: {sex}
- Feeding: {feeding_type}
Recent Trend:
{trend}
Answer this:
\"{user_q}\"
"""
        try:
            st.success("AI Assistant says:")
            st.markdown(ask_openrouter(prompt))
        except Exception as e:
            st.error(str(e))
