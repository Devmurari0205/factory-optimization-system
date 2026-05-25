import pandas as pd

def simulate(df, model, X_cols, product, region, ship_mode):

    factories = df['Division'].unique()
    results = []

    for f in factories:

        sample = df.sample(1).copy()

        sample['Product Name'] = product
        sample['Region'] = region
        sample['Ship Mode'] = ship_mode
        sample['Division'] = f

        pred = model.predict(sample[X_cols])[0]

        results.append({
            "Factory": f,
            "Predicted Lead Time": pred
        })

    return pd.DataFrame(results)


def recommend(sim_df):

    current_time = sim_df['Predicted Lead Time'].mean()

    sim_df['Improvement %'] = ((current_time - sim_df['Predicted Lead Time']) / current_time) * 100

    sim_df = sim_df.sort_values(by='Improvement %', ascending=False)

    return sim_df.head(3)
