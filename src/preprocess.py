import pandas as pd


def loadandcleanevent(path):
    df=pd.read_csv(path)

    # Time features
    df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms')
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    # Label
    df["label"] = (df["event"] == "transaction").astype(int)

    df=df[["visitorid", "itemid", "hour", "dayofweek", "label"]]

    return df



