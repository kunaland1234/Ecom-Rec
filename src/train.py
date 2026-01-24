from src.preprocess import load_and_clean_events, load_item_categories
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


path='data/events.csv'

def train():

    cat_df = load_item_categories(
        "data/item_properties_part1.csv",
        "data/item_properties_part2.csv"
    )



    df=load_and_clean_events(path,cat_df)
    FEATURES = ["category_id", "hour", "dayofweek"]

    X = df[FEATURES]
    y = df["label"]
    #Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,stratify=y)

    #Model Training
    model=LogisticRegression(class_weight="balanced")
    model.fit(X_train,y_train)

    
    joblib.dump(model, "models/model_v2.pkl")

    return model,X_test,y_test


if __name__ == "__main__":
    train()
