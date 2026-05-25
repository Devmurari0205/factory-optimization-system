

# 📊 2. EDA FILE (eda.py)

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/Nassau Candy Distributor.csv")

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df['Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days

# Sales by Region
sns.barplot(data=df, x='Region', y='Sales')
plt.title("Sales by Region")
plt.show()

# Lead Time by Ship Mode
sns.boxplot(data=df, x='Ship Mode', y='Lead Time')
plt.title("Lead Time by Ship Mode")
plt.show()
