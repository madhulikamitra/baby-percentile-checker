import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from scipy.stats import norm

# ===== Utility Functions =====
def calculate_age_in_months(dob, today=None):
    today = today or date.today()
    days = (today - dob).days
    return round(days / 30.4375, 2)  # Average month length

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

# ===== Streamlit App UI =====
st.set_page_config(page_title="Baby Weight Checker", page_icon="👶", layout="centered")
st.title("👶 Baby Weight Tracker")

sex = st.radio("Select baby's sex:", ["Girl", "Boy"])
dob = st.date_input("Baby's Date of Birth")
weight_input = st.text_input("Current weight (kg)", value="")
percentile_input = st.text_input("Target percentile (1–99)", value="50")

# Basic validation
try:
    weight = float(weight_input)
    target_percentile = int(percentile_input)
    if not 1 <= target_percentile <= 99:
        st.error("❌ Please enter a target percentile between 1 and 99.")
        st.stop()
except ValueError:
    st.warning("Please enter valid numeric values for weight and percentile.")
    st.stop()

# NEW: Feeding Type
feeding_type = st.radio(
    "Feeding Type",
    ["Breastfeeding only", "Breastfeeding + Formula", "Formula only", "Weaning / solids"]
)

# Calculate age
age_months = calculate_age_in_months(dob)
st.write(f"🕒 Baby is **{age_months:.2f} months** old")

# Load LMS data
df = pd.read_csv("who_lms_girls.csv") if sex == "Girl" else pd.read_csv("who_lms_boys.csv")

# Interpolate LMS values
L = np.interp(age_months, df["Age"], df["L"])
M = np.interp(age_months, df["Age"], df["M"])
S = np.interp(age_months, df["Age"], df["S"])

# Calculate percentile and expected weight
percentile = calculate_percentile(weight, L, M, S)
expected_weight = calculate_weight_from_percentile(target_percentile, L, M, S)

# Output metrics
st.metric(label="Current Percentile", value=f"{percentile:.1f}th")
st.metric(label=f"Expected Weight @ {target_percentile}th", value=f"{expected_weight:.2f} kg")

diff = weight - expected_weight
if abs(diff) < 0.2:
    st.success("✅ Baby's weight is close to the target percentile.")
elif diff > 0:
    st.info(f"📈 Baby is above the {target_percentile}th percentile by {diff:.2f} kg.")
else:
    st.info(f"📉 Baby is below the {target_percentile}th percentile by {abs(diff):.2f} kg.")

# Health concern flags
if percentile < 5:
    st.error("⚠️ Baby is under the 5th percentile. Please consult your pediatrician.")
elif percentile > 95:
    st.error("⚠️ Baby is above the 95th percentile. Monitor growth with doctor.")

# Plot growth curve
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
