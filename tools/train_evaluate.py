from smolagents.tools import Tool
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff

class TrainAndEvaluateTool(Tool):
    name = "train_and_evaluate"
    description = "Train and evaluate an ML model."
    inputs = {
        "X": {"type": "array", "description": "Preprocessed feature matrix."},
        "y": {"type": "array", "description": "Preprocessed target array."},
        "task_type": {"type": "string", "description": "Type of ML task, either 'Regression' or 'Classification'."},
        "model_type": {"type": "string", "description": "Name of the model to train, e.g., 'Linear Regression', 'Random Forest Classifier'."}
    }
    output_type = "object"  # Tuple containing a dict and a Plotly figure

    def forward(self, X: np.ndarray, y: np.ndarray, task_type: str, model_type: str) -> tuple:
        """Train and evaluate an ML model.

        Args:
            X (np.ndarray): Preprocessed feature matrix.
            y (np.ndarray): Preprocessed target array.
            task_type (str): Type of ML task, either 'Regression' or 'Classification'.
            model_type (str): Name of the model to train, e.g., 'Linear Regression', 'Random Forest Classifier'.

        Returns:
            tuple: A tuple containing a dictionary of metrics and a Plotly figure for visualization.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_dict = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100)
        }
        model = model_dict[model_type]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            residuals = y_test - y_pred
            fig = px.scatter(x=y_test, y=residuals, title="Residual Analysis",
                             labels={'x': 'Actual Values', 'y': 'Residuals'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            return {"mse": mse, "r2": r2}, fig
        else:
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            fig = ff.create_annotated_heatmap(z=cm, x=list(range(cm.shape[1])), y=list(range(cm.shape[0])),
                                              annotation_text=cm.astype(str), colorscale='Blues')
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            return {"accuracy": accuracy, "report": report}, fig
        
        