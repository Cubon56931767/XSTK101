import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==========================================
# PART 0: Data Preparation (Requirement 1)
# ==========================================
# Note: In a real scenario, you would download this data from the IMF or World Bank website.
# Here, we generate synthetic data for demonstration purposes.
# We will simulate 24 years of economic data (2000-2023).

np.random.seed(42)  # For reproducibility

years = np.arange(2000, 2024)
n = len(years)

# Simulating Interval Data
# GDP Growth (%): Normal distribution around 3% with some variance
gdp_growth = np.random.normal(loc=3.0, scale=1.5, size=n)
gdp_growth = np.round(gdp_growth, 2)

# Inflation Rate (%): Gamma distribution (skewed) to make it interesting
inflation_rate = np.random.gamma(shape=2, scale=1.5, size=n)
inflation_rate = np.round(inflation_rate, 2)

# Create a DataFrame
df = pd.DataFrame({
    'Year': years,
    'GDP_Growth': gdp_growth,
    'Inflation_Rate': inflation_rate
})

print("First 5 rows of the dataset:")
print(df.head())
print("-" * 30)

# ==========================================
# PART 1: Histogram, Stem-and-Leaf, Boxplot (Requirement 2)
# ==========================================

# Setting the visual style for "Beautiful" plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12

# 1.1 Histogram for GDP Growth
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='GDP_Growth', kde=True, bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of GDP Growth (2000-2023)', fontsize=16, fontweight='bold')
plt.xlabel('GDP Growth (%)')
plt.ylabel('Frequency')
# Adding mean line
plt.axvline(df['GDP_Growth'].mean(), color='red', linestyle='--', label=f"Mean: {df['GDP_Growth'].mean():.2f}%")
plt.legend()
plt.tight_layout()
plt.show()

# 1.2 Stem-and-Leaf Plot for Inflation Rate
# Python doesn't have a built-in graphical stem-and-leaf, so we create a text-based one
# and a "Lollipop" chart which is the modern graphical equivalent.

def print_stem_and_leaf(data):
    """
    Prints a simple text-based stem-and-leaf plot.
    """
    print("\nStem-and-Leaf Plot for Inflation Rate:")
    sorted_data = np.sort(data)
    stems = np.floor(sorted_data).astype(int)
    leafs = np.round((sorted_data - stems) * 10).astype(int)
    
    current_stem = None
    for s, l in zip(stems, leafs):
        if s == current_stem:
            print(f" {l}", end="")
        else:
            if current_stem is not None:
                print()
            current_stem = s
            print(f"{s} | {l}", end="")
    print("\n" + "-"*30)

print_stem_and_leaf(df['Inflation_Rate'].values)

# Graphical "Stem" plot (Lollipop chart)
plt.figure(figsize=(10, 6))
plt.stem(df['Year'], df['Inflation_Rate'], basefmt=" ")
plt.title('Stem Plot (Lollipop Chart) of Inflation Rate Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Inflation Rate (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 1.3 Boxplot for GDP Growth and Inflation
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['GDP_Growth', 'Inflation_Rate']], palette="Set2")
plt.title('Boxplot Comparison: GDP Growth vs Inflation Rate', fontsize=16, fontweight='bold')
plt.ylabel('Percentage (%)')
plt.tight_layout()
plt.show()

# ==========================================
# PART 2: Time-Series Plot (Requirement 3)
# ==========================================

plt.figure(figsize=(12, 6))
plt.plot(df['Year'], df['GDP_Growth'], marker='o', linestyle='-', linewidth=2, label='GDP Growth', color='#1f77b4')
plt.plot(df['Year'], df['Inflation_Rate'], marker='s', linestyle='--', linewidth=2, label='Inflation Rate', color='#ff7f0e')

plt.title('Time Series: Economic Indicators (2000-2023)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.axhline(0, color='black', linewidth=1) # Zero line for reference
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Highlight max points
max_gdp_idx = df['GDP_Growth'].idxmax()
plt.annotate(f'Max GDP: {df.iloc[max_gdp_idx]["GDP_Growth"]}%', 
             xy=(df.iloc[max_gdp_idx]['Year'], df.iloc[max_gdp_idx]['GDP_Growth']),
             xytext=(5, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.show()

# ==========================================
# PART 3: Scatter Plot (Requirement 4)
# ==========================================

plt.figure(figsize=(10, 8))
# Scatter plot with regression line using regplot
sns.regplot(x='Inflation_Rate', y='GDP_Growth', data=df, 
            scatter_kws={'s': 100, 'alpha': 0.7, 'edgecolor': 'w'}, 
            line_kws={'color': 'red', 'linestyle': '--'})

plt.title('Scatter Plot: Inflation vs GDP Growth', fontsize=16, fontweight='bold')
plt.xlabel('Inflation Rate (%)', fontsize=12)
plt.ylabel('GDP Growth (%)', fontsize=12)

# Calculate Correlation
corr_coef = df['Inflation_Rate'].corr(df['GDP_Growth'])
plt.text(0.05, 0.95, f'Correlation Coefficient (r) = {corr_coef:.2f}', 
         transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.show()

# ==========================================
# PART 4: Interpretation & Principles (Requirement 5 & 6)
# ==========================================

print("\n=== INTERPRETATION OF RESULTS ===")
print("1. Distribution: The histogram shows the spread of GDP growth. A bell curve suggests a normal distribution.")
print("2. Outliers: The boxplot helps identify any extreme values (outliers) in the economic data.")
print("3. Trends: The time-series plot reveals cyclical patterns or trends over the last 24 years.")
print(f"4. Relationship: The scatter plot shows a correlation of r = {corr_coef:.2f} between inflation and GDP.")

print("\n=== 3 PRINCIPLES FROM 'ART & SCIENCE OF DATA VISUALIZATION' ===")
print("1. Keep it Simple: Avoid 'chart junk' (unnecessary grid lines, 3D effects, or distracting colors) that clutter the message.")
print("2. Label Explicitly: Always include clear titles, axis labels, and legends so the chart can stand alone without verbal explanation.")
print("3. Use the Right Tool: Choose the correct chart type for the data (e.g., Line charts for time series, Scatter plots for relationships).")