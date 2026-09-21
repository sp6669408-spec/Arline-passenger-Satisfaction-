"""
DEMO DATA GENERATOR — NOT part of the actual mini project.

This script creates a synthetic dataset with the EXACT same column names
and structure as the real, publicly available Kaggle dataset:
  "Airline Passenger Satisfaction"
  https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

It exists only so the analysis script (project.py) can be demonstrated
end-to-end without internet access in this environment.

TO SUBMIT A GENUINE PROJECT: download the real CSV from Kaggle (free
account, one click), rename it to airline_passenger_satisfaction.csv,
and drop it in this folder in place of this synthetic file. Because
project.py uses the same column names as the real dataset, no code
changes are needed — you will simply be analysing real passenger data
instead of a simulation.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 4000

gender = rng.choice(["Male", "Female"], N)
cust_type = rng.choice(["Loyal Customer", "disloyal Customer"], N, p=[0.82, 0.18])
age = rng.integers(7, 85, N)
travel_type = rng.choice(["Business travel", "Personal Travel"], N, p=[0.69, 0.31])
travel_class = rng.choice(["Business", "Eco", "Eco Plus"], N, p=[0.48, 0.45, 0.07])
flight_distance = rng.integers(50, 5000, N)

# service ratings 0-5 (0 = not applicable / very poor, 5 = excellent)
def rating(bias=0.0, n=N):
    r = rng.normal(3 + bias, 1.1, n)
    return np.clip(np.round(r), 0, 5).astype(int)

wifi = rating()
time_convenient = rating()
online_booking = rating()
gate_location = rating()
food_drink = rating()
online_boarding = rating(bias=0.2)
seat_comfort = rating(bias=0.2)
entertainment = rating(bias=0.2)
onboard_service = rating(bias=0.2)
legroom = rating()
baggage_handling = rating(bias=0.2)
checkin_service = rating()
inflight_service = rating(bias=0.2)
cleanliness = rating(bias=0.2)

dep_delay = np.clip(rng.exponential(12, N).astype(int), 0, 600)
arr_delay = np.clip(dep_delay + rng.normal(0, 8, N).astype(int), 0, 600)

# business travellers in Business class with good service tend to be satisfied
score = (
    0.9 * (travel_class == "Business").astype(int)
    + 0.5 * (travel_type == "Business travel").astype(int)
    + 0.35 * online_boarding
    + 0.30 * seat_comfort
    + 0.30 * entertainment
    + 0.25 * cleanliness
    + 0.20 * onboard_service
    + 0.15 * (cust_type == "Loyal Customer").astype(int)
    - 0.02 * dep_delay
    - 0.015 * arr_delay
    - 0.10 * (age < 15).astype(int)
    + rng.normal(0, 1.4, N)
)
prob_satisfied = 1 / (1 + np.exp(-(score - score.mean()) / score.std()))
satisfaction = np.where(rng.random(N) < prob_satisfied, "satisfied", "neutral or dissatisfied")

df = pd.DataFrame({
    "id": np.arange(1, N + 1),
    "Gender": gender,
    "Customer Type": cust_type,
    "Age": age,
    "Type of Travel": travel_type,
    "Class": travel_class,
    "Flight Distance": flight_distance,
    "Inflight wifi service": wifi,
    "Departure/Arrival time convenient": time_convenient,
    "Ease of Online booking": online_booking,
    "Gate location": gate_location,
    "Food and drink": food_drink,
    "Online boarding": online_boarding,
    "Seat comfort": seat_comfort,
    "Inflight entertainment": entertainment,
    "On-board service": onboard_service,
    "Leg room service": legroom,
    "Baggage handling": baggage_handling,
    "Checkin service": checkin_service,
    "Inflight service": inflight_service,
    "Cleanliness": cleanliness,
    "Departure Delay in Minutes": dep_delay,
    "Arrival Delay in Minutes": arr_delay.astype(float),
    "satisfaction": satisfaction,
})

# inject realistic messiness: missing values + a few extreme outliers
miss_idx = rng.choice(N, size=int(N * 0.02), replace=False)
df.loc[miss_idx, "Arrival Delay in Minutes"] = np.nan
out_idx = rng.choice(N, size=8, replace=False)
df.loc[out_idx, "Arrival Delay in Minutes"] = rng.integers(700, 1400, 8)

df.to_csv("airline_passenger_satisfaction.csv", index=False)
print("Synthetic demo dataset written:", df.shape)
print(df["satisfaction"].value_counts(normalize=True))
