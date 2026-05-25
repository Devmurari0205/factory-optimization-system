import pandas as pd

factories = [
    "Lot's O' Nuts",
    "Wicked Choccy's",
    "Sugar Shack",
    "Secret Factory",
    "The Other Factory"
]

def simulate_factory(df, model, encoders, product, region, ship_mode):
    results = []

    sample = df[df['Product Name'] == product].iloc[0]

    for factory in factories:
        X = pd.DataFrame({
            'Region_enc': [encoders['region'].transform([region])[0]],
            'Ship Mode_enc': [encoders['ship'].transform([ship_mode])[0]],
            'Product_enc': [encoders['product'].transform([product])[0]],
            'Units': [sample['Units']],
            'Cost': [sample['Cost']]
        })

        pred = model.predict(X)[0]

        results.append({
            "Factory": factory,
            "Predicted Lead Time": round(pred,2)
        })

    return pd.DataFrame(results).sort_values("Predicted Lead Time")


def recommend_top(df_sim):
    return df_sim.iloc[0]
