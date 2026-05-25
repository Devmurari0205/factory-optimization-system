import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

def load_and_train():

    df = pd.read_csv("Nassau Candy Distributor.csv")

    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df['Shipping Duration'] = (df['Ship Date'] - df['Order Date']).dt.days

    df = df.dropna()
    df = df[df['Shipping Duration'] < 30]

    le = LabelEncoder()

    categorical_cols = ['Region','Ship Mode','Division','Product Name']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()
    num_cols = ['Sales','Cost','Units']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df[['Region','Ship Mode','Division','Product Name','Sales','Cost','Units']]
    y = df['Lead Time']

    model = RandomForestRegressor()
    model.fit(X, y)

    return df, model
