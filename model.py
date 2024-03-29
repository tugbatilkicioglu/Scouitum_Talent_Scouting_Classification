
import numpy as np
import pandas as pd
import feature_engineering as fe
import variable_evaluations as ve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, accuracy_score


def base_models(X, y, scorings=["roc_auc", "f1", "precision", "recall", "accuracy"]):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for scoring in scorings:
        print(f"Evaluation Metric: {scoring}")
        for name, classifier in classifiers:
            cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
            print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

def evaluate_models(X, y):
    models = [('LR', LogisticRegression()),
              ('KNN', KNeighborsClassifier()),
              ('RF', RandomForestClassifier()),
              ('GBM', GradientBoostingClassifier()),
              ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
              ('LightGBM', LGBMClassifier())]

    for name, model in models:
        print(name)
        for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
            cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
            print(score + " score:" + str(cvs))

def evaluate_lgbm(X, y, random_state=46):

    lgbm_model = LGBMClassifier(random_state=random_state)
    rmse_base = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))
    lgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [500, 1500],
                   "colsample_bytree": [0.5, 0.7, 1]
                  }
    lgbm_gs_best = GridSearchCV(lgbm_model,
                                lgbm_params,
                                cv=3,
                                n_jobs=-1,
                                verbose=True).fit(X, y)

    final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)
    rmse_optimized = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
    print("RMSE before hyperparameter tuning:", rmse_base)
    print("RMSE after hyperparameter tuning:", rmse_optimized)


