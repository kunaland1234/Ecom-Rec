import pandas as pd


def load_and_clean_events(path,cat_df):
    df=pd.read_csv(path)


    # Time features
    df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms')
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    # Label
    df["label"] = (df["event"] == "transaction").astype(int)

    df=df.merge(cat_df, on="itemid", how="left")
    df["category_id"] = df["category_id"].fillna(-1)

    df=df[["visitorid", "itemid", "hour", "dayofweek", "label","category_id"]]

    return df


def load_item_categories(path1, path2):
    p1 = pd.read_csv(path1)
    p2 = pd.read_csv(path2)

    items = pd.concat([p1, p2], ignore_index=True)
    cats = items[items["property"] == "categoryid"]

    cats = cats[["itemid", "value"]].drop_duplicates()
    cats = cats.rename(columns={"value": "category_id"})

    return cats



