import pandas as pd
import ast  # Module for literal_eval
import urllib.request
from PIL import Image
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree  import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def function_template(values):
    """
    Description

    Arguments:

        `values` (type): Variable dataframe de Pandas.


    Returns:
        variable: Type
    """
    pass

def convert_to_list(value:str):
    """
    Custom converter function to convert string representation of list to actual list

    Arguments:

        `value` (str): Text to convert to a list.


    Returns:
        value: List
    """
    # Custom converter function to convert string representation of list to actual list
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

def onehotencoder(df:pd.DataFrame,features_cat:list):
    """
    Turns categorical columns in binary columns

    Arguments:

        `df` (pd.DataFrame): Dataframe de Pandas.
        `features_cat` (list): List of categorical columns.

    Returns:
        df: pd.DataFrame
    """
    #Turns categorical columns in binary columns
    onehot = OneHotEncoder(sparse_output=False, drop='first') 
    data = onehot.fit_transform(df[features_cat])
    new_features = onehot.get_feature_names_out()
    df[new_features] = data
    df.drop(columns= features_cat, axis = 1, inplace = True)
    return df

def scaler_of_x_train_and_x_test(X_train:pd.DataFrame,X_test:pd.DataFrame,minmax:bool=True):
        
        """
        Función que escala valores de un dataframe con MinMaxScaler o StandardScaler

        Args:
                `df` (pandas.DataFrame): DataFrame que contiene los datos.
                `minmax` (bool): Aplica MinMaxScaler por defecto, de lo contrario aplica StandardScaler
        
        Devuelve:
                `df` (pandas.DataFrame) normalizado
        """
        if minmax:
                minmax=MinMaxScaler()
                X_train_scaled=minmax.fit_transform(X_train)
                X_test_scaled=minmax.transform(X_test)
        else:
                standardscaler=StandardScaler()
                X_train_scaled=standardscaler.fit_transform(X_train)
                X_test_scaled=standardscaler.transform(X_test)
        return X_train_scaled,X_test_scaled

def to_lowercase_and_remove_blanks_on_columns_df(df:pd.DataFrame):
    """
    Change to lowercase and remove blanks from column names

    Arguments:

        `df` (pd.DataFrame): Dataframe de Pandas.


    Returns:
        df: pd.DataFrame
    """
    #Change to lowercase and remove blanks from column names
    keys=df.columns.to_list()
    new_columns={}
    for column in keys:
        value=column.lower()
        value=value.replace(" ","_")
        new_columns[column]=value
    #print(new_columns)
    df.rename(columns=new_columns,inplace=True)
    return df
