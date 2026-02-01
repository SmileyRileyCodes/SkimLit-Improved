"""
To experiment with different tensorflow models, this class will store the results.

TODO: Finish this docstring when I understand the class more.
"""
# imports
import pandas as pd
import numpy as np
from pyparsing import Literal
import tensorflow as tf

from loguru import logger

from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score

logger = logger.bind(module="experimentation")
class ExperimentationClassification:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray = None,
                 Xy_train_dataset: tf.data.Dataset = None, Xy_val_dataset: tf.data.Dataset = None, Xy_test_dataset: tf.data.Dataset = None,
                 classification_type: Literal['binary', 'multi-class'] = 'binary') -> None:
        self.results = pd.DataFrame()
        self.classification_type = classification_type

        self.model_dict = {}

        # input data placeholders
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.Xy_train_dataset: tf.data.Dataset = Xy_train_dataset if Xy_train_dataset is not None else tf.data.Dataset.from_tensor_slices((X_train, y_train))
        self.Xy_val_dataset: tf.data.Dataset = Xy_val_dataset if Xy_val_dataset is not None else tf.data.Dataset.from_tensor_slices((X_val, y_val))
        self.Xy_test_dataset: tf.data.Dataset = Xy_test_dataset if Xy_test_dataset is not None else tf.data.Dataset.from_tensor_slices((X_test, y_test)) if y_test is not None else None

        # default parameters- compile
        self.loss: str = 'binary_crossentropy' if classification_type == 'binary' else 'categorical_crossentropy'
        self.optimizer: str | tf.keras.Optimizer = 'adam'
        self.metrics: list = ['accuracy']

        # default parameters- fit
        self.batch_size: int = 32
        self.epochs: int = 5
        self.steps_per_epoch: float = 0.1 # % of data
        self.validation_steps: float = 0.1 # % of data

        # default parameters- evaluate
        self.average: str = 'binary' if classification_type == 'binary' else 'weighted

    def __str__(self) -> str:
        return f"List of all experimented models and their information: {self.model_dict}"
    
    def __repr__(self) -> str:
        return f"ExperimentationClassification(classification_type={self.classification_type}, models={list(self.model_dict.keys())})"
    
    def add_model(self, model_name: str, model: tf.keras.Model, loss: str = None, optimizer: str | tf.keras.Optimizer = None, 
                  metrics: list = None, batch_size: int = None, epochs: int = None, steps_per_epoch: float = None,
                  validation_steps: float = None, average: str = None, use_default_input_data: bool = True) -> None:
        """
        Add a TensorFlow model to the experimentation class.

        Parameters:
            model_name (str): The name of the model.
            model (tf.keras.Model): The TensorFlow model instance.
            loss (str, optional): Loss function for model compilation. If None, uses defaults. Defaults to None.
            optimizer (str | tf.keras.Optimizer, optional): Optimizer for model compilation. If None, uses defaults.
                Defaults to None.
            metrics (list, optional): List of metrics for model compilation. If None, uses defaults. Defaults to None.
            batch_size (int, optional): Batch size for model training. If None, uses defaults. Defaults to None.
            epochs (int, optional): Number of epochs for model training. If None, uses defaults. Defaults to None.
            steps_per_epoch (float, optional): Steps per epoch for model training. If None, uses defaults. Defaults to None.
            validation_steps (float, optional): Validation steps for model training. If None, uses defaults. Defaults to None.
            average (str, optional): Averaging method for multi-class classification metrics. If None, uses defaults. Defaults to None.
            use_default_input_data (bool, optional): Whether to use the default input data set in the experimentation class. This can
                be overridden later for each model individually. Defaults to True.
        """
        # raise error if model name already exists
        if model_name in self.model_dict:
            raise ValueError(f"Model with name '{model_name}' already exists in the experimentation class.")
        
        # raise error if model is not a tf.keras.Model
        if not isinstance(model, tf.keras.Model):
            raise TypeError("The 'model' parameter must be an instance of tf.keras.Model.")
        
        # grab defaults parameters if none provided
        loss = loss if loss is not None else self.loss
        optimizer = optimizer if optimizer is not None else self.optimizer
        metrics = metrics if metrics is not None else self.metrics
        batch_size = batch_size if batch_size is not None else self.batch_size
        epochs = epochs if epochs is not None else self.epochs
        steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else self.steps_per_epoch
        validation_steps = validation_steps if validation_steps is not None else self.validation_steps
        average = average if average is not None else self.average

        if use_default_input_data:
            if self.X_train is None or self.y_train is None or self.X_val is None or self.y_val is None:
                raise ValueError("Default input data is not set in the experimentation class. Please set the data before adding a model with 'use_default_input_data=True'.")
            # set input data for model
            self.set_input_data_for_model(model_name, self.Xy_train_dataset, self.Xy_val_dataset)
        
         # add model to model dictionary

        self.model_dict[model_name] = {"model": model,
                                       "y_pred": None,
                                       "results": None,
                                       "y_pred_probs": None,
                                       "wrong_predictions_df": None,
                                       "loss": loss,
                                       "optimizer": optimizer,
                                       "metrics": metrics,
                                       "batch_size": batch_size,
                                       "epochs": epochs,
                                       "steps_per_epoch": steps_per_epoch,
                                       "validation_steps": validation_steps,
                                       "average": average}
        
        logger.info(f"Model '{model_name}' added to experimentation class.")
        logger.info(f"Model parameters: loss={loss}, optimizer={optimizer}, metrics={metrics}, "
                    f"batch_size={batch_size}, epochs={epochs}, steps_per_epoch={steps_per_epoch*100:.2f}% of training data, "
                    f"validation_steps={validation_steps*100:.2f}% of validation data, average={average}")
        
    def update_input_data(self, X_train: np.ndarray, y_train: np.ndarray, 
                 X_val: np.ndarray, y_val: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray = None,
                 Xy_train_dataset: tf.data.Dataset = None,
                 Xy_val_dataset: tf.data.Dataset = None,
                 Xy_test_dataset: tf.data.Dataset = None) -> None:
        """
        Set the training, validation, and test data for the experimentation class.

        Parameters:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation features.
            y_val (np.ndarray): Validation labels.
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.
            Xy_train_dataset (tf.data.Dataset, optional): TensorFlow dataset for training. Defaults to None.
            Xy_val_dataset (tf.data.Dataset, optional): TensorFlow dataset for validation. Defaults to None.
            Xy_test_dataset (tf.data.Dataset, optional): TensorFlow dataset for testing. Defaults to None.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test # may not get labels for test set

        if Xy_train_dataset is None:
            Xy_train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        self.Xy_train_dataset = Xy_train_dataset
        self.Xy_val_dataset = Xy_val_dataset
        self.Xy_test_dataset = Xy_test_dataset

    def set_input_data_for_model(self, model_name: str, input_data: tuple[np.ndarray, np.ndarray] | tf.data.Dataset,
                                 validation_data: tuple[np.ndarray, np.ndarray] | tf.data.Dataset,
                                 test_data: tuple[np.ndarray, np.ndarray] | tf.data.Dataset = None) -> None:
        """
        Set the training, validation, and test data for a specific model in the experimentation class.

        Parameters:
            model_name (str): The name of the model.
            input_data (np.ndarray | tf.data.Dataset): Training data (features and labels). This can be either
                a tuple of (X_train, y_train) or a tf.data.Dataset.
            validation_data (np.ndarray | tf.data.Dataset): Validation data (features and labels). This can be either
                a tuple of (X_val, y_val) or a tf.data.Dataset. Validation data is required for making predictions.
            test_data (np.ndarray | tf.data.Dataset, optional): Test data (features and labels). This can be either
                a tuple of (X_test, y_test) or a tf.data.Dataset. Defaults to None.
        """
        if model_name not in self.model_dict:
            raise ValueError(f"Model with name '{model_name}' does not exist in the experimentation class.")
        
        # if data is a tf.data.Dataset, use it directly; otherwise, convert to tf.data.Dataset
        if isinstance(input_data, tf.data.Dataset):
            self.Xy_train_dataset = input_data
        else:
            self.Xy_train_dataset = tf.data.Dataset.from_tensor_slices(input_data)
        
        if isinstance(validation_data, tf.data.Dataset):
            self.Xy_val_dataset = validation_data
        else:
            self.Xy_val_dataset = tf.data.Dataset.from_tensor_slices(validation_data)
        
        if test_data is not None:
            if isinstance(test_data, tf.data.Dataset):
                self.Xy_test_dataset = test_data
            else:
                self.Xy_test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
        else:
            self.Xy_test_dataset = None

        # update model dictionary to indicate that datasets are being used
        self.model_dict[model_name]["input_data"] = self.Xy_train_dataset
        self.model_dict[model_name]["validation_data"] = self.Xy_val_dataset
        self.model_dict[model_name]["test_data"] = self.Xy_test_dataset

    def compile_model(self, model_name: str) -> None:
        """
        Compile the specified model using its stored parameters.

        Parameters:
            model_name (str): The name of the model to compile.
        """
        model_info = self.model_dict.get(model_name)
        if model_info is None:
            raise ValueError(f"Model '{model_name}' not found in the model dictionary.")
        
        model = model_info["model"]
        loss = model_info["loss"]
        optimizer = model_info["optimizer"]
        metrics = model_info["metrics"]

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit_model(self, model_name: str, use_datasets: bool = False) -> None:
        """
        Fit the specified model using its stored parameters.

        Parameters:
            model_name (str): The name of the model to fit.
        """
        model_info = self.model_dict.get(model_name)
        if model_info is None:
            raise ValueError(f"Model '{model_name}' not found in the model dictionary.")
        
        model = model_info["model"]
        batch_size = model_info["batch_size"]
        epochs = model_info["epochs"]
        steps_per_epoch = model_info["steps_per_epoch"]
        validation_steps = model_info["validation_steps"]

        if not use_datasets:
            model.fit(self.X_train, self.y_train,
                      validation_data=(self.X_val, self.y_val),
                      epochs=epochs,
                      batch_size=batch_size,
                      steps_per_epoch=int(steps_per_epoch * len(self.X_train) / batch_size),
                      validation_steps=int(validation_steps * len(self.X_val) / batch_size)
                      )
        else:
            model.fit(self.Xy_train_dataset,
                    validation_data=self.Xy_val_dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    steps_per_epoch=int(steps_per_epoch * len(self.Xy_train)),
                    validation_steps=int(validation_steps * len(self.Xy_val))
                    )
    
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

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] | None = "binary") -> dict:
        """
        Evaluate the model predictions against the true values using various metrics.

        Parameters:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            average (Literal): The averaging method for multi-class classification metrics. Can be
                'micro', 'macro', 'samples', 'weighted', 'binary' or None (if None, the scores for each
                class are returned).

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
    
    def predict_and_evaluate(self, model_name: str) -> dict:
        """
        Make predictions with the specified model and evaluate the results.

        Parameters:
            model_name (str): The name of the model to use for predictions.
            X (np.ndarray): Input data for predictions.
            y_true (np.ndarray): True labels for evaluation.
            average (Literal): The averaging method for multi-class classification metrics.

        Returns:
            dict: A dictionary containing evaluation metrics and prediction times.
        """
        model_info = self.model_dict.get(model_name)
        if model_info is None:
            raise ValueError(f"Model '{model_name}' not found in the model dictionary.")
        
        model = model_info["model"]
        if model is None:
            raise ValueError(f"Model '{model_name}' not found in the model dictionary.")
        
        # get input data
        X = self.X_val
        y_true = self.y_val
        average = model_info["average"]

        total_time, time_per_prediction = self.pred_timer(model, X)
        y_pred_probs = model.predict(X)

        # turn prediction probabilities into predicted classes
        if self.classification_type == 'binary':
            y_pred = tf.squeeze(tf.round(y_pred_probs)).numpy()
        else:
            y_pred = tf.argmax(y_pred_probs, axis=1).numpy()
        
        # get results
        metrics = self.evaluate_model(y_true, y_pred, average)
        results = {
            'Metrics': metrics,
            'Total Prediction Time': total_time,
            'Time per Prediction': time_per_prediction
        }

        self.results.loc[model_name] = results

        # store predictions
        self.model_dict[model_name]["y_pred"] = y_pred
        self.model_dict[model_name]["results"] = results
        self.model_dict[model_name]["y_pred_probs"] = y_pred_probs

        return results
    
    def get_model_results(self, model_name: str) -> pd.Series:
        """
        Retrieve the results for a specific model.

        Parameters:
            model_name (str): The name of the model.
        Returns:
            pd.Series: The results for the specified model.
        """
        return self.results.loc[model_name]
    
    def get_y_pred(self, model_name: str) -> np.ndarray:
        """
        Retrieve the predictions made by a specific model.

        Parameters:
            model_name (str): The name of the model.
        
        Returns:
            np.ndarray: The predictions made by the specified model.
        """
        return self.model_dict[model_name]["y_pred"]
    
    def get_list_of_models(self) -> list[str]:
        """
        Get a list of all model names stored in the experimentation class.

        Returns:
            list[str]: A list of model names.
        """
        return list(self.model_dict.keys())
    
    def run_experiment(self, model_name: str) -> None:
        """Runs the full experiment pipeline for a specific model."""
        if model_name not in self.model_dict:
            logger.critical(f"Running experiment pipeline for model '{model_name}' failed.")
            raise ValueError(f"Model '{model_name}' not found in the model dictionary.")

        
        experiment_pipeline = [
            ("Compile model", self.compile_model),
            ("Fit model", self.fit_model),
            ("Predict and evaluate", self.predict_and_evaluate),
        ]
        try:
            for step_name, step_function in experiment_pipeline:
                logger.info(f"Starting step: {step_name}")
                step_function(model_name)
                logger.success(f"Completed step: {step_name}")
        except Exception as e:
            logger.critical(f"Experiment pipeline for model '{model_name}' failed at step '{step_name}': {e}")
            raise e
        
        logger.info(f"Experiment pipeline for model '{model_name}' completed successfully.")

        model_info = self.model_dict.get(model_name)
        if model_info is not None:
            model_info["results"] = self.get_model_results(model_name)

            logger.info(f"Results for model '{model_name}': {model_info['results']}")

    def get_most_wrong_predictions(self, model_name: str, n: int = 10) -> pd.DataFrame:
        """
        Get the top n most wrong predictions for a specific model.

        Parameters:
            model_name (str): The name of the model.
            n (int): The number of most wrong predictions to retrieve.

        Returns:
            pd.DataFrame: A DataFrame containing the most wrong predictions.
        """
        model_info = self.model_dict.get(model_name)
        if model_info is None:
            raise ValueError(f"Model '{model_name}' not found in the model dictionary.")

        if self.classification_type == 'binary':
            # get necessary data
            y_true = self.y_val
            y_pred = model_info["y_pred"]
            sentences = self.X_val
            y_pred_probs = model_info["y_pred_probs"]

            # create dataframe 
            comparison_df = pd.DataFrame({
                'Sentence': sentences,
                'True Label': y_true,
                'Predicted Label': y_pred,
                'Predicted Probability': tf.squeeze(y_pred_probs).numpy()
            })

            wrong_predictions_df = comparison_df[comparison_df['True Label'] != comparison_df['Predicted Label']]
            wrong_predictions_df['Difference'] = wrong_predictions_df['True Label'] - wrong_predictions_df['Predicted Probability']
            wrong_predictions_df['Absolute Difference'] = wrong_predictions_df['Difference'].abs()

            self.model_dict[model_name]["wrong_predictions_df"] = wrong_predictions_df

            most_wrong_df = wrong_predictions_df.sort_values(by='Absolute Difference', ascending=False).head(n)
            return most_wrong_df
        else:
            # get necessary data
            y_true = self.y_val
            y_pred = model_info["y_pred"]
            sentences = self.X_val
            y_pred_probs = model_info["y_pred_probs"]

            # calculate most incorrect predictions based on probability of the predicted class
            y_true_one_hot = tf.one_hot(y_true, depth=y_pred_probs.shape[1]).numpy()
            prediction_differences = np.abs(y_true_one_hot - y_pred_probs)
            most_wrong_prediction_indices = tf.argmax(prediction_differences, axis=1).numpy()
            most_wrong_prediction_values = tf.reduce_max(prediction_differences, axis=1).numpy()

            # make comparison dataframe with the most incorrect prediction and its probability out of all classes
            comparison_df = pd.DataFrame({
                'Sentence': sentences,
                'True Label': y_true,
                'Predicted Label': y_pred,
                'Most Wrong Predicted Class': most_wrong_prediction_indices,
                'Most Wrong Prediction Value': most_wrong_prediction_values
            })

            wrong_predictions_df = comparison_df[comparison_df['True Label'] != comparison_df['Predicted Label']]

            self.model_dict[model_name]["wrong_predictions_df"] = wrong_predictions_df
            most_wrong_df = wrong_predictions_df.sort_values(by='Most Wrong Prediction Value', ascending=False).head(n)

            return most_wrong_df