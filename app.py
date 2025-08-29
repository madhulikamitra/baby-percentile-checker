import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def calculate_percentile(weight, L, M, S):
    if L == 0:
        z = (np.log(weight / M)) / S
    else:
        z = (((weight / M) ** L) - 1) / (L * S)
    return norm.cdf(z) * 100

def calculate_weight_from_percentile(p, L, M, S):
    z = norm.ppf(p / 100)
    if L == 0:
        weight = M * np.exp(S * z)
    else:
        weight = M * ((L * S * z + 1) ** (1 / L))
    return weight

st.title("👶 Baby Weight Tracker")

sex = st.radio("Select baby's sex:", ["Girl", "Boy"])
age = st.number_input("Age (in months)", min_value=0.0, max_value=60.0, step=0.1)
weight = st.number_input("Current weight (kg)", min_value=1.0, step=0.1)
target_percentile = st.number_input("Target percentile", min_value=1, max_value=99, value=50)

# Load LMS data
if sex == "Girl":
    df = pd.read_csv("who_lms_girls.csv")
else:
    df = pd.read_csv("who_lms_boys.csv")

# Interpolate if age is non-integer
L = np.interp(age, df["Age"], df["L"])
M = np.interp(age, df["Age"], df["M"])
S = np.interp(age, df["Age"], df["S"])

percentile = calculate_percentile(weight, L, M, S)
expected_weight = calculate_weight_from_percentile(target_percentile, L, M, S)

# Output results
st.metric(label="Current Percentile", value=f"{percentile:.1f}th")
st.metric(label=f"Expected Weight @ {target_percentile}th", value=f"{expected_weight:.2f} kg")

diff = weight - expected_weight
if abs(diff) < 0.2:
    st.success("✅ Your baby's weight is very close to the target percentile!")
elif diff > 0:
    st.info(f"📈 Baby is above the {target_percentile}th percentile by {diff:.2f} kg")
else:
    st.info(f"📉 Baby is below the {target_percentile}th percentile by {abs(diff):.2f} kg")



    # Generate growth curves
    ages = np.linspace(0, 60, 121)  # 0 to 60 months, step=0.5
    percentiles = [5, 50, 95]
    colors = {5: 'blue', 50: 'green', 95: 'red'}

    plt.figure(figsize=(8, 5))
    for p in percentiles:
        weights = [calculate_weight_from_percentile(p,
                                                    np.interp(a, df["Age"], df["L"]),
                                                    np.interp(a, df["Age"], df["M"]),
                                                    np.interp(a, df["Age"], df["S"])) for a in ages]
        plt.plot(ages, weights, label=f'{p}th percentile', color=colors[p])

    # Plot baby's weight
    plt.scatter(age, weight, color='black', label="Your Baby")
    plt.xlabel("Age (months)")
    plt.ylabel("Weight (kg)")
    plt.title("Baby Weight vs WHO Growth Standards")
    plt.legend()
    plt.grid(True)

    st.pyplot(plt)
