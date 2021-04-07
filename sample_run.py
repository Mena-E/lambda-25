import pandas as pd

from lambdata.helper_functions import CleanFrame, TimeSeriesSplit, RandomSplit

df = pd.read_csv("lambdata/market_data.csv")
myframe = RandomSplit(df, 0.8)
print(myframe.train_test_split())