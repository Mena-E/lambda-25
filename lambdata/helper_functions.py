""" lambdata_mena2 - A collection of Data Science helper functions"""


def null_count(df):
    """
    Checks a dataframe for nulls and returns
    the number of missing values.
    """
    return df.isnull().sum().sum() 
    

def train_test_split_time(df, frac):
    """
    Splits a time series dataframe into a training
    and a testing set based on frac (a fraction) of
    the percent of the size of the training data,
    entered in the TimeSeriesSplit class.
    :return: tuple of training and testing dataframes
    """
    val = int(round(len(df) * frac, 0))
    df_train = df.iloc[0:val, :]
    df_test = df.iloc[val + 1:, :]
    return df_train, df_test


def train_test_split_rand(df, frac):
    """
    Randomly splits a dataframe into a training and
    a testing set based on frac value entered in the
    RandomSplit class.
    :return: tuple of training and testing dataframes.
    """
    df_split1 = df.sample(frac=frac)
    df_split2 = df.drop(df_split1.index)
    return df_split1, df_split2