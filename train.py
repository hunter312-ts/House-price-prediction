#Import libraries
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV ,cross_val_score
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns 

# Load preprocessor

preprocessor=joblib.load("preprocessor.joblib")

# 1) load and preprocess the data
df=pd.read_csv("Housing.csv")
x=df.drop('price',axis=1)
y=df['price']
x_pre=preprocessor.transform(x)

# 2) Hyperparameter with random search cv

rf=RandomForestRegressor(random_state=42)
param_dist = {
    'n_estimators': [50,100,200],
    'max_depth': [None,10,20,30],
    'min_samples_split': [2,5,10]
}
rnd_search=RandomizedSearchCV(rf,param_dist,n_iter=10,cv=3,random_state=42,n_jobs=-1)
rnd_search.fit(x_pre,y)
print("RandomForestSearchCV best param: ",rnd_search.best_estimator_)

#3) Bayesian tunning with optuna
def objective(trial):
    params={
        'n_estimators':trial.suggest_int('n_estimators',50,300),
        'max_depth': trial.suggest_int('max_depth',5,10),
        'min_samples_split':trial.suggest_int('min_samples_split',2,20)
    }
    model=RandomForestRegressor(**params,random_state=42)
    scores=cross_val_score(model,x_pre,y,cv=3)
    return float(-np.mean(scores))

study=optuna.create_study(direction='minimize')
study.optimize(objective,n_trials=15)
print("Optuna best params:", study.best_params)

# 4) Train the final model
best_param=study.best_params
model=RandomForestRegressor(**best_param,random_state=42)
model.fit(x_pre,y)
joblib.dump(model,'model.joblib')
print("Trained model saved to model.joblib")

# 5) Explain with shap
explainer=shap.TreeExplainer(model)
sample=x_pre[np.random.choice( len(x_pre),100,replace=False)]
shap_values=explainer.shap_values(sample)
shap.summary_plot(shap_values,sample,feature_names=preprocessor.get_feature_names_out())

# 6)  Save a correlation heat map
feature_names = preprocessor.get_feature_names_out()
corr = pd.DataFrame(x_pre, columns=feature_names).corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm", center=0, xticklabels=feature_names, yticklabels=feature_names)
plt.title("Feature Correlation Heatmap")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('feature_correlation_heatmap.png')
print("Heatmap saved to feature_correlation_heatmap.png")