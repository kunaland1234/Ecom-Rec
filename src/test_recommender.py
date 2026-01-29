from src.preprocess import load_and_clean_events, load_item_categories
from src.candidate import build_popular_items
from src.recommender import Recommender
from src.build_feature import build_features

cat_df = load_item_categories(
    "data/item_properties_part1.csv",
    "data/item_properties_part2.csv"
)

user_id = int(input("Enter user_id: "))
k = int(input("Enter K (e.g. 5): "))

events,train_stata = load_and_clean_events("data/events.csv", cat_df,is_train=True)

popular_items = build_popular_items(events, user_id)

recommender = Recommender("models/model_v4.pkl")

df = build_features(events, user_id, popular_items)
scored = recommender.score_candidates(df)
top_k = recommender.top_n(scored, n=k)

print(f"\nTop-{k} Recommendations for user {user_id}:")
print(top_k)