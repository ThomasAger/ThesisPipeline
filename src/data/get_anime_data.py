import pandas as pd
# Recommendation data, not review data
raw_path = "../../data/raw/anime-recommendations-database/"
anime_path = "anime.csv"
rating_path = "rating.csv"

chunksize = 10 ** 8
for chunk in pd.read_csv(raw_path + rating_path, chunksize=chunksize):
    print(chunk)