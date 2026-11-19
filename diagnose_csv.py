import pandas as pd

ais = pd.read_csv("AIS-15.csv", nrows=3)
print("AIS-15.csv")
print("Shape:", ais.shape)
print("Columns:", list(ais.columns))
print("\nFirst 3 rows:")
print(ais)

print("\n" + "="*50)

radar = pd.read_csv("Radar-15.csv", nrows=3)
print("Radar-15.csv")
print("Shape:", radar.shape)
print("Columns:", list(radar.columns))
print("\nFirst 3 rows:")
print(radar)