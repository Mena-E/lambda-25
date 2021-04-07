""" lambdata_mena2 - A collection of Data Science helper functions"""


class CleanFrame:
    def __init__(self, df):
        self.df = df

    def null_count(self):
        """Checks a dataframe for nulls and returns
           the number of missing values.
        """
        return self.df.isnull().sum().sum()


class TimeSeriesSplit:
    def __init__(self, df, frac):
        self.df = df
        self.frac = float(frac)

    def train_test_split(self):
        """
        Splits a time series dataframe into a training
        and a testing set based on frac (a fraction) of
        the percent of the size of the training data,
        entered in the TimeSeriesSplit class.
        :return: tuple of training and testing dataframes
        """
        val = int(round(len(self.df) * self.frac, 0))
        df_train = self.df.iloc[0:val, :]
        df_test = self.df.iloc[val + 1:, :]
        return (df_train.shape, df_test.shape)


class RandomSplit:
    def __init__(self, df, frac):
        self.df = df
        self.frac = float(frac)

    def train_test_split(self):
        """
        Randomly splits a dataframe into a training and
        a testing set based on frac value entered in the
        RandomSplit class.
        :return: tuple of training and testing dataframes.
        """
        df_split1 = self.df.sample(frac=self.frac)
        df_split2 = self.df.drop(df_split1.index)
        return df_split1.shape, df_split2.shape
