import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def load_and_train():
    # Get current directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 🔥 Automatically find CSV file
    files = os.listdir(BASE_DIR)

    csv_file = None
    for file in files:
        if file.endswith(".csv"):
            csv_file = file
            break

    if csv_file is None:
        raise FileNotFoundError("❌ No CSV file found in project folder")

    file_path = os.path.join(BASE_DIR, csv_file)

    # Load dataset
    df = pd.read_csv(file_path)

    # Convert dates
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])

    # Create Lead Time
    df['Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days

    # Encoding
    le_region = LabelEncoder()
    le_ship = LabelEncoder()
    le_product = LabelEncoder()

    df['Region_enc'] = le_region.fit_transform(df['Region'])
    df['Ship Mode_enc'] = le_ship.fit_transform(df['Ship Mode'])
    df['Product_enc'] = le_product.fit_transform(df['Product Name'])

    # Features
    X = df[['Region_enc','Ship Mode_enc','Product_enc','Units','Cost']]
    y = df['Lead Time']

    # Train model
    model = RandomForestRegressor()
    model.fit(X, y)

    encoders = {
        "region": le_region,
        "ship": le_ship,
        "product": le_product
    }

    return df, model, encoders
