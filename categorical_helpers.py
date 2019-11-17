from sklearn.base import TransformerMixin, BaseEstimator
from pdb import set_trace
from sklearn.impute import SimpleImputer
import pandas
import numpy as np
from itertools import permutations


def custom_sort(x):
    set_trace()
    pass


# inherit from baseestimator
# 72 Hands-on_machine-learning

class ConcatFeature(BaseEstimator, TransformerMixin):
    def __init__(self, feature=None, concat=None):
        self.feature = feature
        self.concat = concat

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            # @TODO needs implementation form numpy data struct
            return X
        else:
            """
            should concat if one of the values are NaN ?
            """
            data = pandas.DataFrame()
            combo_feature = "%s + %s" % (self.concat, self.feature)
            data[combo_feature] = X[[self.concat, self.feature]].apply(
                lambda x: ' + '.join([str(z) for z in x]), axis=1)
            return data


class ExtractOneFeature(BaseEstimator, TransformerMixin):
    def __init__(self, feature=None, index=None):
        self.feature = feature

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            # @TODO needs implementation form numpy data struct
            return X
        else:
            data = X[[self.feature]]
            return data



class RareCategories(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=25):
        self.replace_series = {}
        self.replace_values = {}
        self.threshold = threshold

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            if len(X) < 1:
                return self
            column_count = len(X[0])
            for _column_index in range(0, column_count):
                counts_dict = dict(zip(*np.unique(X[:, [_column_index]], return_counts=True)))
                replace_values = [x for x in counts_dict.keys() if counts_dict[x] < self.threshold]
                if len(replace_values) > 0:
                    self.replace_values[_column_index] = replace_values
        else:
            for col in X.columns:
                self.replace_series[col] = X[col].value_counts()[X[col].value_counts() < self.threshold]
        return self

    def transform(self, X, y=None):
        transformed = X
        if isinstance(X, np.ndarray):
            for _index in self.replace_values.keys():
                for value in self.replace_values[_index]:
                    column_values = transformed[:, _index]
                    column_values[column_values == value] = "OTHER"
                    transformed[:, _index] = column_values
        else:
            for col in self.replace_series.keys():
                subject = dict.fromkeys(self.replace_series[col].index, "OTHER")
                transformed = X.replace(subject)
        return transformed
