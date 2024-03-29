import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
import data_reading_and_understanding as dr
import feature_engineering as fe
import variable_evaluations as ve
import model as ml
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier



warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df1 = pd.read_csv("datasets/scoutium_attributes.csv", sep=';')
df2 = pd.read_csv("datasets/scoutium_potential_labels.csv", sep=';')

dff = pd.merge(df1, df2, how='left', on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])

dff = dff.loc[~(dff["position_id"] == 1)] #Goalkeepers


dff.nunique()
dff.groupby("potential_label").count()
dff["potential_label"].value_counts()
dff = dff.loc[~(dff["potential_label"] == "below_average")]

pt = pd.pivot_table(dff, values="attribute_value", columns="attribute_id", index=["player_id","position_id","potential_label"])

pt = pt.reset_index(drop=False)
pt.columns = pt.columns.map(str)

num_cols = pt.columns[3:]

dr.check_data(pt)

for col in ["position_id","potential_label"]:
    dr.cat_summary(pt, col)


for col in num_cols:
    dr.num_summary(pt, col, plot=True)

for col in num_cols:
    dr.target_summary_with_num(pt, "potential_label", col)


pt[num_cols].corr()

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(pt[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

#It is observed that the monthly charges of TotalChargers are highly correlated with tenure

pt = fe.feature_engineering(pt)

labelEncoderCols = ["potential_label", "mentality"]
for col in labelEncoderCols:
    pt = fe.label_encoder(pt, col)


lst = ["counts", "countRatio", "min", "max", "sum", "mean", "median"]
num_cols = list(num_cols)

for i in lst:
    num_cols.append(i)

scaler = StandardScaler()
pt[num_cols] = scaler.fit_transform(pt[num_cols])

y = pt["potential_label"]
X = pt.drop(["potential_label", "player_id"], axis=1)

ml.evaluate_models(X, y)

ml.evaluate_lgbm(X,y)


model = LGBMClassifier()
model.fit(X, y)
ve.plot_importance(model, X, features = X.columns, save=True)



