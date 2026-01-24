import datetime
from preprocess import loadandcleanevent, load_item_categories
from candidate import build_popular_items
from recommender import Recommender


cat_df = load_item_categories(
    "data/item_properties_part1.csv",
    "data/item_properties_part2.csv"
)


events = loadandcleanevent("data/events.csv", cat_df)

popular_items = build_popular_items(events, top_k=50)


user_id = events["visitorid"].iloc[0]

user_events = events[events["visitorid"] == user_id]


last_event = user_events.iloc[-1]
user_category = last_event["category_id"]

candidates = popular_items[
    popular_items["category_id"] == user_category
].copy()

now = datetime.datetime.now()
candidates["hour"] = now.hour
candidates["dayofweek"] = now.weekday()

recommender = Recommender("models/model_v2.pkl")
scored = recommender.score_candidates(candidates)

top_5 = recommender.top_n(scored, n=5)
print("Top-5 Recommendations:")
print(top_5)
