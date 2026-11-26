"""Shared loaders for the HEEW Mini-Dataset analysis."""
import os, hashlib
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "HEEW_Mini-Dataset")
OUT = os.path.join(ROOT, "outputs")
IMG = os.path.join(ROOT, "report", "images")
os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)

BUILDINGS = [f"BN{i:03d}" for i in range(1, 11)]
VARS = {"Electricity [kW]": "E", "Heat [mmBTU]": "H", "Cooling Energy [Ton]": "C",
        "PV Power Generation [kW]": "PV", "Greenhouse Gas Emission [Ton]": "GHG"}
WVARS = {"Temperature [°F]": "Temp", "Dew Point [°F]": "DewPt", "Humidity [%]": "RH",
         "Wind Speed [mph]": "Wind", "Wind Gust [mph]": "Gust", "Pressure [in]": "Press",
         "Precipitation [in]": "Precip"}

def load_energy(node):
    df = pd.read_csv(os.path.join(DATA, f"{node}_energy.csv"))
    idx = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns=["year", "month", "day", "hour"]).rename(columns=VARS)
    df.index = idx; df.index.name = "time"
    assert len(df) == 8760 and (np.diff(df.index.values).astype("timedelta64[h]") == 1).all()
    return df

def load_weather():
    df = pd.read_csv(os.path.join(DATA, "Total_weather.csv"), parse_dates=["datetime"]).set_index("datetime")
    df = df.rename(columns=WVARS); df.index.name = "time"
    assert len(df) == 8760 and (np.diff(df.index.values).astype("timedelta64[h]") == 1).all()
    return df

def load_all():
    return {n: load_energy(n) for n in BUILDINGS + ["CN01", "Total"]}

def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
