import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder



def base_model_results(df, cat_cols, target_col="Churn", drop_first=True, random_state=12345):
    # Create a copy of the DataFrame
    dff = df.copy()

    # Perform one-hot encoding on categorical columns
    dff = pd.get_dummies(dff, columns=[col for col in cat_cols if col != target_col], drop_first=drop_first)

    # Separate features and target variable
    y = dff[target_col]
    X = dff.drop([target_col, "customerID"], axis=1)

    # Define models
    models = [('LR', LogisticRegression(random_state=random_state)),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier(random_state=random_state)),
              ('RF', RandomForestClassifier(random_state=random_state)),
              ('XGB', XGBClassifier(random_state=random_state)),
              ("LightGBM", LGBMClassifier(random_state=random_state)),
              ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

    # Iterate over models and perform cross-validation
    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
        print(f"########## {name} ##########")
        print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"AUC: {round(cv_results['test_roc_auc'].mean(), 4)}")
        print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
        print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
        print("\n")


def feature_engineering(data):
    num_cols = data.columns[3:]
    # Calculate minimum, maximum, sum, mean, median values
    data["min"] = data[num_cols].min(axis=1)
    data["max"] = data[num_cols].max(axis=1)
    data["sum"] = data[num_cols].sum(axis=1)
    data["mean"] = data[num_cols].mean(axis=1)
    data["median"] = data[num_cols].median(axis=1)

    # Label as "defender" or "attacker" based on position
    data["mentality"] = data["position_id"].apply(lambda x: "defender" if x in [2, 5, 3, 4] else "attacker")

    # Create a list containing FLAG columns
    flag_cols = [col for col in data.columns if "_FLAG" in col]

    # Calculate the total of FLAG columns
    data["counts"] = data[flag_cols].sum(axis=1)

    # Calculate the ratio of FLAG columns
    if len(flag_cols) > 0:  # Payda sıfır olmamasını kontrol et
        data["countRatio"] = data["counts"] / len(flag_cols)
    else:
        data["countRatio"] = 0  # Payda sıfır ise, countRatio'yu sıfır olarak ayarla

    return data


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


