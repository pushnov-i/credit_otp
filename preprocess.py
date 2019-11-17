#!/usr/bin/env python

from categorical_helpers import RareCategories, ExtractOneFeature, custom_sort
from categorical_helpers import ConcatFeature

from category_encoders import WOEEncoder, SumEncoder
from IPython.terminal.debugger import set_trace

import pandas as pd

import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import LeaveOneOutEncoder

from itertools import combinations

from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV
import operator

data_filename = open('./Credit_OTP.csv')

data = pd.read_csv(data_filename, sep=";")
data.drop_duplicates(subset=None, keep="first", inplace=True)

extracted_range = data["FAMILY_INCOME"].str.extract(r'^от (?P<FAMILY_INCOME_FROM>\d+) до (?P<FAMILY_INCOME_TO>\d+)')

data["FAMILY_INCOME_FROM"] = extracted_range["FAMILY_INCOME_FROM"].astype(float)
data["FAMILY_INCOME_TO"] = extracted_range["FAMILY_INCOME_TO"].astype(float)

data = data.drop(columns=["AGREEMENT_RK", "FAMILY_INCOME"])

aim_param = "TARGET"
cat_columns = data.dtypes[data.dtypes == "object"].index
cat_to_num = []

for col_name in cat_columns:
    try:
        # find the first non zero value's index and try to convert its value to a float
        # if succeed, consider the value transformable to float
        # otherwise ValueError will be thrown silently
        first_non_zero_value_index = data[data[col_name] != "0"].first_valid_index()
        test_value = data[col_name][first_non_zero_value_index]
        # just in case, convert comma to dot
        if isinstance(test_value, str) and test_value.find(','):
            test_value = test_value.replace(",", ".")
        to_float = float(test_value)
        cat_to_num.append(col_name)
    except ValueError:
        continue

for col_name in cat_to_num:
    data[col_name] = data[col_name].apply(lambda x: float(x.replace(',', '.')) if isinstance(x[0], str) else float(x))

X_train, X_test, Y_train, Y_test = train_test_split(data.drop(aim_param, axis=1), data[aim_param], test_size=0.3,
                                                    stratify=data[aim_param], random_state=23)

cat_columns = X_train.dtypes[X_train.dtypes == "object"].index
num_columns = X_train.dtypes[X_train.dtypes != "object"].index

cat_scores = {}
num_scores = {}

cat_pipe = Pipeline([
    ('extract_feature', ExtractOneFeature()),
    ("imp", SimpleImputer(strategy='most_frequent')),
    # use rare before encoding, rare categories can cause noise
    ('rare', RareCategories()),
    ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore")),
    ("regression", LogisticRegression(solver="lbfgs", max_iter=200))
])

num_pipe = Pipeline([
    ("extract_feature", ExtractOneFeature()),
    ("imp", SimpleImputer()),
    # what else methods of scaling exist ?
    # roboscaler, module, etc
    ("scaler", StandardScaler()),
    ("regression", LogisticRegression(solver="lbfgs", max_iter=200))
])

for feature in data.drop(columns=[aim_param]).columns:
    if feature in cat_columns:
        pipe = cat_pipe
        res = cat_scores
    else:
        pipe = num_pipe
        res = num_scores

    pipe.set_params(extract_feature__feature=feature)
    model = pipe.fit(X_train, Y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    roc_auc = metrics.roc_auc_score(Y_test, predictions)
    res[feature] = roc_auc

# SelectKBest does the same or not ?
# tried to play with it but got *** ValueError: could not convert string to float: 'Высшее'
# top scored features by roc auc

top_cat_features = sorted(cat_scores.items(), key=operator.itemgetter(1), reverse=True)[:10]
top_num_features = sorted(num_scores.items(), key=operator.itemgetter(1), reverse=True)[:10]

combo_scores = {}

concat_pipeline = Pipeline([
    ("concat", ConcatFeature()),
    ("imp", SimpleImputer(strategy='most_frequent')),
    ('rare', RareCategories()),
    ("ohe", OneHotEncoder(sparse=False, handle_unknown="ignore")),
    ("regression", LogisticRegression(solver="lbfgs", max_iter=200))
])


for combo in set(combinations([x[0] for x in top_cat_features], r=2)):
    combination = "%s + %s" % (combo[0], combo[1])
    print("[~] testing combo %s " % combination)
    concat_pipeline.set_params(concat__feature=combo[0], concat__concat=combo[1])
    model = concat_pipeline.fit(X_train, Y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    roc_auc = metrics.roc_auc_score(Y_test, predictions)
    combo_scores[combination] = roc_auc

top_combos = sorted(combo_scores.items(), key=operator.itemgetter(1), reverse=True)

"""
 Create binary feature to leverage an impact of the strong features education, age, and gen_title  
 age 27 -> 50
 GEN_TITLE ["Руководитель среднего звена", "Высококвалифиц. специалист", "Руководитель высшего звена", "Индивидуальный предприниматель", "Военнослужащий по контракту" ]
 EDUCATION ["Высшее", "Два и более высших образования", "Ученая степень"]
 
"""

age_from = 27
age_to = 50

top_titles = ["Руководитель среднего звена", "Высококвалифиц. специалист", "Руководитель высшего звена",
              "Индивидуальный предприниматель", "Военнослужащий по контракту"]
top_education = ["Высшее", "Два и более высших образования", "Ученая степень"]

data["EDU_TITLE_PRIORITY"] = (data["GEN_TITLE"].isin(top_titles) &
                              data["EDUCATION"].isin(top_education) &
                              data["AGE"].between(age_from, age_to))

"""
create top combo feature which is GEN_TITLE + EDUCATION 
"""

data["GEN_TITLE+EDUCATION"] = data[["GEN_TITLE", "EDUCATION"]].apply(lambda x: ' + '.join([str(z) for z in x]), axis=1)


data.reset_index()
X_train, X_test, Y_train, Y_test = train_test_split(data.drop(aim_param, axis=1), data[aim_param], test_size=0.3,
                                                    stratify=data[aim_param], random_state=23)

hyperparam_cat_pipe = Pipeline([
    ("imp", SimpleImputer()),
    ("rare", RareCategories()),
    ("ohe", OneHotEncoder(sparse=False, handle_unknown='ignore')),
    # ("woe", WOEEncoder(return_df=False)),
    ("sum", SumEncoder(return_df=False))
])

hyperparam_num_pipe = Pipeline([
    ("imp", SimpleImputer()),
    ("scaler", StandardScaler()),
])


cat_columns = cat_columns.tolist()
cat_columns.append("EDU_TITLE_PRIORITY")
# cat_columns.append("GEN_TITLE+EDUCATION")
set_trace()

gridsearch_pipe = Pipeline([
    ("tf", ColumnTransformer(transformers=[
        ("cat", hyperparam_cat_pipe, cat_columns),
        ("num", hyperparam_num_pipe, num_columns.tolist())]),
     ),
    ("regression", LogisticRegression(solver="lbfgs", max_iter=200))
])

param_grid = [
    {
        "tf__num__imp__strategy": ["mean", "median", "most_frequent"],
        "tf__cat__imp__strategy": ["most_frequent"],
        # "tf__cat__woe": [None],
        "tf__cat__ohe": [None],
        # "tf__cat__sum": [None],
        "tf__cat__rare__threshold": [10],
        "regression__C": [0.1, 1, 10]
    },
    {
        "tf__num__imp__strategy": ["mean", "median", "most_frequent"],
        "tf__cat__imp__strategy": ["most_frequent"],
        # "tf__cat__woe": [None],
        # "tf__cat__ohe": [None],
        "tf__cat__sum": [None],
        "tf__cat__rare__threshold": [10],
        "regression__C": [0.1, 1, 10]
    }
    # , {
    #     "tf__num__imp__strategy": ["mean", "median", "most_frequent"],
    #     "tf__cat__imp__strategy": ["most_frequent"],
    #     # "tf__cat__woe": [None],
    #     # "tf__cat__ohe": [None],
    #     "tf__cat__sum": [None],
    #     "tf__cat__rare__threshold": [10, 20],
    #     "regression__C": [0.1, 1, 10]
    # }

]

gs = GridSearchCV(gridsearch_pipe,
                  param_grid,
                  cv=3, scoring='roc_auc', return_train_score=False, verbose=10
                  )

gs.fit(X_train, Y_train)
print(gs.best_score_)
set_trace()


