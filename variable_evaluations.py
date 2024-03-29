import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_importance(model, X, features, num=None, save=False):
    if num is None:
        num = len(X)
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features})

    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(num))
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")












