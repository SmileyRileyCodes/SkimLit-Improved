"""
To experiment with different tensorflow models, this class will store the results.

TODO: Finish this docstring when I understand the class more.
"""
# imports
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score

class Experimentation:
    def __init__(self):
        self.results = pd.DataFrame()

        self.model_dict = {}
    
    def add_model(self, model_name: str, model: tf.keras.Model):
        """
        Add a TensorFlow model to the experimentation class.

        Args:
            model_name (str): The name of the model.
            model (tf.keras.Model): The TensorFlow model instance.
        """
        self.model_dict[model_name] = model
    
    def pred_timer(self, model: tf.keras.Model, X: np.ndarray) -> tuple[float, float]:
        """
        Times how long it takes a model to make predictions.
        
        Parameters:
            model (tf.keras.Model): The TensorFlow model to make predictions with.
            X (np.ndarray): The input data for predictions.
        
        Returns:
            total_time (float): Time taken to make predictions in seconds.
            time_per_prediction (float): Average time taken per prediction in seconds.
        """
        start_time = tf.timestamp()
        model.predict(X)
        end_time = tf.timestamp()
        total_time = end_time - start_time
        time_per_prediction = total_time / X.shape[0]

        return total_time, time_per_prediction

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, average="weighted") -> dict:
        """
        Evaluate the model predictions against the true values using various metrics.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            average (str): The averaging method for multi-class classification metrics.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        f1_score_value = f1_score(y_true, y_pred, average=average)
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)

        metrics = {
            'MSE': mse,
            'MAE': mae,
            'Accuracy': accuracy,
            'F1 Score': f1_score_value,
            'Precision': precision,
            'Recall': recall,
        }

        return metrics
    
    def predict_and_evaluate(self, model_name: str, X: np.ndarray, y_true: np.ndarray, average="weighted") -> dict:
        """
        Make predictions with the specified model and evaluate the results.

        Args:
            model_name (str): The name of the model to use for predictions.
            X (np.ndarray): Input data for predictions.
            y_true (np.ndarray): True labels for evaluation.
            average (str): The averaging method for multi-class classification metrics.

        Returns:
            dict: A dictionary containing evaluation metrics and prediction times.
        """
        model = self.model_dict.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found in the model dictionary.")

        total_time, time_per_prediction = self.pred_timer(model, X)
        y_pred = model.predict(X)
        metrics = self.evaluate_model(y_true, y_pred, average)

        results = {
            'Metrics': metrics,
            'Total Prediction Time': total_time,
            'Time per Prediction': time_per_prediction
        }

        self.results.loc[model_name] = results

        return results
