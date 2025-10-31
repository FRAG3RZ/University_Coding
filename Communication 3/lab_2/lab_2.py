import pandas as pd
import matplotlib.pyplot as plt

# === 1. Load CSV file ===
# Change this path to your actual CSV file name
csv_file = "csv_files/scope_1.csv"

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(csv_file)

# === 2. Display basic info ===
print("Columns in CSV:", data.columns.tolist())
print(data.head())

# === 3. Plot ===
# If your CSV has columns like "Time" and "Value"
# You can replace them with your actual column names below
x_col = data.columns[0]  # First column for X-axis
y_col = data.columns[1]  # Second column for Y-axis

plt.figure(figsize=(8, 5))
plt.plot(data[x_col], data[y_col], marker='o', linestyle='-', color='b')

# === 4. Style the plot ===
plt.title(f"{y_col} vs {x_col}")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude [dBV]")
plt.grid(True)
plt.tight_layout()

# === 5. Show the plot ===
plt.show()
