from smolagents.tools import Tool
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class PreprocessDataTool(Tool):
    name = "preprocess_data"
    description = "Preprocess dataset for ML training."
    inputs = {
        "df": {"type": "object", "description": "The input dataset as a pandas DataFrame."},
        "feature_columns": {"type": "array", "description": "List of column names to use as features (independent variables)."},
        "target_column": {"type": "string", "description": "Name of the column to use as the target (dependent variable)."},
        "task_type": {"type": "string", "description": "Type of ML task, either 'Regression' or 'Classification'."}
    }
    output_type = "array"  # X and y are returned as NumPy arrays in a tuple

    def forward(self, df: pd.DataFrame, feature_columns: list, target_column: str, task_type: str) -> tuple:
        """Preprocess dataset for ML training.

        Args:
            df (pd.DataFrame): The input dataset as a pandas DataFrame.
            feature_columns (list): List of column names to use as features (independent variables).
            target_column (str): Name of the column to use as the target (dependent variable).
            task_type (str): Type of ML task, either 'Regression' or 'Classification'.

        Returns:
            tuple: A tuple containing the preprocessed features (X) and target (y).
        """
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        le = LabelEncoder()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col].astype(str))
        if task_type == "Classification" and y.dtype == 'object':
            y = le.fit_transform(y.astype(str))
        elif task_type == "Regression" and y.dtype == 'object':
            y = le.fit_transform(y.astype(str))
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        y = pd.to_numeric(y, errors='coerce').fillna(0)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y