# Importing libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
import joblib

# loading the data set
df=pd.read_csv("Housing.csv")
df.head()

# defining columns
cat_cols=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
num_cols=['area','bedrooms','bathrooms','stories','parking']

# Building pre type pipeline
num_pipe=Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])
cat_pipe=Pipeline([
    ('imputer',SimpleImputer(strategy='constant',fill_value='missing')),
    ('OHE',OneHotEncoder(handle_unknown='ignore'))
])

# combine into sigle column transformer
preprocessor=ColumnTransformer([
    ('num',num_pipe,num_cols),
    ('cat',cat_pipe,cat_cols)
])


def build_and_save_pipeline(df:pd.DataFrame , target_col:str , out_path:str = "preprocessor.joblib"):
    """Fit the into the df and save it """
    x=df.drop(target_col,axis=1)
    y=df[target_col]
    preprocessor.fit_transform(x,y)
    joblib.dump(preprocessor,out_path)
    print(f"Preprocessor save to  {out_path}")

if __name__ =="__main__":
    # Example usage: python preprocessor.py
    df=pd.read_csv("Housing.csv")
    build_and_save_pipeline(df=df , target_col='price')

