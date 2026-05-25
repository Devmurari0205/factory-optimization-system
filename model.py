import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def load_and_train():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Auto find CSV
    files = os.listdir(BASE_DIR)
    csv_file = [f for f in files if f.endswith(".csv")][0]

    path = os.path.join(BASE_DIR, csv_file)
    df = pd.read_csv(path)

    # 🔥 FIX DATE ERROR
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce', dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce', dayfirst=True)

    # Remove bad rows
    df = df.dropna(subset=['Order Date', 'Ship Date'])

    # Create Lead Time
    df['Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days

    # Encoding
    le_region = LabelEncoder()
    le_ship = LabelEncoder()
    le_product = LabelEncoder()

    df['Region_enc'] = le_region.fit_transform(df['Region'])
    df['Ship Mode_enc'] = le_ship.fit_transform(df['Ship Mode'])
    df['Product_enc'] = le_product.fit_transform(df['Product Name'])

    X = df[['Region_enc','Ship Mode_enc','Product_enc','Units','Cost']]
    y = df['Lead Time']

    model = RandomForestRegressor()
    model.fit(X, y)

    return df, model, {
        "region": le_region,
        "ship": le_ship,
        "product": le_product
    }
