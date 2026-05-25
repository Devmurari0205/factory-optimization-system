import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def load_and_train():
    df = pd.read_csv("data.csv")

    # Date conversion
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

    model = RandomForestRegressor()
    model.fit(X, y)

    encoders = {
        "region": le_region,
        "ship": le_ship,
        "product": le_product
    }

    return df, model, encoders
