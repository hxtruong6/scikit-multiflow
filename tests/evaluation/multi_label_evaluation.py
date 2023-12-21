import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

import scipy.io.arff as arff
import pandas as pd

from skmultiflow.meta.classifier_chains_custom import (
    ProbabilisticClassifierChainCustom,
)

# # Load the ARFF file
# data = arff.loadarff('your_dataset.arff')
# # Convert the ARFF data to a Pandas DataFrame
# df = pd.DataFrame(data[0])

# Now, df contains your data as a Pandas DataFrame
# You can perform various data analysis tasks with df

SEED = 6


# Create static class of EvaluationMetrics
class EvaluationMetrics:
    @staticmethod
    def _check_dimensions(Y_true, Y_pred):
        if Y_true.shape != Y_pred.shape:
            raise Exception("Y_true and Y_pred have different shapes")

    @staticmethod
    def get_loss(Y_true, Y_pred, loss_func):
        EvaluationMetrics._check_dimensions(Y_true, Y_pred)
        return loss_func(Y_true, Y_pred)

    @staticmethod
    def hamming_loss(Y_true, Y_pred):
        """
        Calculate Hamming Loss for multilabel classification.

        Parameters:
        - y_true: NumPy array, true labels (2D array with shape [n_samples, n_labels]).
        - y_pred: NumPy array, predicted labels (2D array with shape [n_samples, n_labels]).

        Returns:
        - float: Hamming Loss.
        """
        EvaluationMetrics._check_dimensions(Y_true, Y_pred)

        # Calculate Hamming Loss
        loss = np.mean(np.not_equal(Y_true, Y_pred))
        return loss

    @staticmethod
    def precision_score(y_true, y_pred):
        """
        Calculate Precision score for binary or multiclass classification.

        Parameters:
        - y_true: NumPy array, true labels.
        - y_pred: NumPy array, predicted labels.

        Returns:
        - float: Precision score.
        """
        # Ensure y_true and y_pred have the same shape
        EvaluationMetrics._check_dimensions(y_true, y_pred)

        # Calculate True Positives and False Positives
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))

        # Calculate Precision
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )

        return precision

    @staticmethod
    def recall_score(y_true, y_pred):
        """
        Calculate Recall score for binary or multiclass classification.

        Parameters:
        - y_true: NumPy array, true labels.
        - y_pred: NumPy array, predicted labels.

        Returns:
        - float: Recall score.
        """
        # Ensure y_true and y_pred have the same shape
        EvaluationMetrics._check_dimensions(y_true, y_pred)

        # Calculate True Positives and False Negatives
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))

        # Calculate Recall
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        return recall

    @staticmethod
    def subset_accuracy(y_true, y_pred):
        """
        Calculate Subset Accuracy for multilabel classification.

        Parameters:
        - y_true: NumPy array, true labels (2D array with shape [n_samples, n_labels]).
        - y_pred: NumPy array, predicted labels (2D array with shape [n_samples, n_labels]).

        Returns:
        - float: Subset Accuracy.
        """
        # Ensure y_true and y_pred have the same shape
        EvaluationMetrics._check_dimensions(y_true, y_pred)

        # Calculate subset accuracy
        correct_samples = np.sum(np.all(y_true == y_pred, axis=1))
        subset_accuracy_value = correct_samples / len(y_true)

        return subset_accuracy_value


class HandleMulanDatasetForMultiLabelArffFile:
    def __init__(self, path, dataset_name):
        self.path = path
        self.data = arff.loadarff(self.path)
        self.df = pd.DataFrame(self.data[0])

        self.dataset_name = dataset_name

        y_split_index = self._get_Y_split_index()

        self.X = self.df.iloc[:, :-y_split_index]
        self.Y = self.df.iloc[:, -y_split_index:].astype(int)

    # Handle custom dataset name to get Y column which is multi-label
    def _get_Y_split_index(self):
        if self.dataset_name == "emotions":
            return 6
        elif self.dataset_name == "corel5k":
            return 374
        elif self.dataset_name == "bitex":
            return 159

        else:
            raise Exception("Dataset name is not supported")


# Define a function to read datasets from JSON files in a folder using a generator function (yield)
def read_datasets_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        # check is folder
        if os.path.isdir(os.path.join(folder_path, filename)):
            # TODO: check if file exist and flexible with testing file

            # Training data
            print(f"Reading {filename} dataset...")
            df_train = HandleMulanDatasetForMultiLabelArffFile(
                os.path.join(folder_path, filename, f"{filename}-train.arff"), filename
            )

            # Testing data
            df_test = HandleMulanDatasetForMultiLabelArffFile(
                os.path.join(folder_path, filename, f"{filename}-test.arff"), filename
            )
            yield df_train, df_test


def calculate_metrics(Y_true, Y_pred, metric_funcs):
    try:
        score_metrics = []

        for metric in metric_funcs:
            # print(f"\n{metric}\n")
            # Check options in metric function
            if "options" in metric:
                metric_name, metric_func, options = (
                    metric["name"],
                    metric["func"],
                    metric["options"],
                )
                score = metric_func(Y_true, Y_pred, **options)
            else:
                metric_name, metric_func = metric["name"], metric["func"]
                score = f"{metric_func(Y_pred,Y_true):.5f}"

            score_metrics.append(
                {
                    "Metric Name": metric_name,
                    "Metric Function": metric_func.__name__,
                    "Score": score,
                }
            )

        # print(f"score_metrics:\t{score_metrics}")
    except Exception as e:
        print("-" * 10)

        print(f"Error: {e}")
        print(f"Y_true:\t{Y_true.shape}\nY_pred:\t{Y_pred.shape}")
        print(f"Metric: \t{metric}")

        print("-" * 10)

    return score_metrics


def training_model(model, X_train, Y_train):
    # ----------------- Fit -----------------
    print(f"Training {model.base_estimator.__class__.__name__} model...")
    model.fit(X_train, Y_train)
    return model


# Define a function to perform model evaluation on each dataset
def evaluate_model(
    model: ProbabilisticClassifierChainCustom,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    predict_funcs: list,
    metric_funcs: list,
) -> list:
    """_summary_

    Args:
        model (ProbabilisticClassifierChainCustom): _description_
        X_test (pd.DataFrame): _description_
        Y_test (pd.DataFrame): _description_
        predict_funcs (list): [{'name': 'Hamming Loss', 'func': predict_HammingLoss}, ...}]
        metric_funcs (list): [{'name': "Predict", 'func': "predict"}, {'name': "Predict Hamming Lost", 'func': 'predict_Hamming'}, ...]

    Returns:
        list: [{'predict_name': "Predict", 'score_metrics': [{'Metric Name': 'Hamming Loss', 'Metric Function': 'hamming_loss', 'Score': 0.0}, ...]}]
    """

    # ----------------- Predict -----------------
    print(f"{'-'*50}\nPredicting {model.__class__.__name__} model...")
    # [{'name': "Predict", 'score_metrics': [
    # {'Metric Name': 'Hamming Loss', 'Metric Function': 'hamming_loss', 'Score': 0.0},
    # {'Metric Name': 'Accuracy Score', 'Metric Function': 'accuracy_score', 'Score': 1.0},
    # {'Metric Name': 'Precision Score', 'Metric Function': 'precision_score', 'Score': 1.0},
    # {'Metric Name': 'Recall Score', 'Metric Function': 'recall_score', 'Score': 1.0},
    # {'Metric Name': 'F1 Score', 'Metric Function': 'f1_score', 'Score': 1.0}]}
    # ]
    loss_score_by_predict_func = []
    # TODO: predict_HammingLoss, predict_Inf, predict_Mar, predict_Neg, predict_Pre, predict_Subset,...
    for predict_func in predict_funcs:
        if predict_func["func"] == "predict":
            Y_pred, _, _ = model.predict(X_test)
        else:
            Y_pred = getattr(model, predict_func["func"])(X_test)

        print(f"Calculating metrics for {predict_func['name']}...")
        score_metrics = calculate_metrics(Y_test, Y_pred, metric_funcs)

        loss_score_by_predict_func.append(
            {"predict_name": predict_func["name"], "score_metrics": score_metrics}
        )
    return loss_score_by_predict_func


def prepare_model_to_evaluate():
    # TODO: add more models
    pcc = [
        # LinearRegression(),
        SGDClassifier(max_iter=100, tol=1e-3, loss="log_loss", random_state=SEED),
        # RandomForestClassifier(random_state=SEED),
        # AdaBoostClassifier(random_state=SEED),
    ]

    # Add more models here if you want to evaluate them
    # Iterate all BOP methods per each classifier such as SGD, etc.
    return [ProbabilisticClassifierChainCustom(model) for model in pcc]


def main():
    # Define the list of models you want to evaluate
    evaluated_models = prepare_model_to_evaluate()

    # Define the folder path containing JSON datasets and the output CSV file name
    folder_path = (
        "/Users/xuantruong/Documents/JAIST/scikit-multiflow/tests/evaluation/dataset"
    )
    output_csv = "/Users/xuantruong/Documents/JAIST/scikit-multiflow/tests/evaluation/result/evaluation_results.csv"

    # -----------------  MAIN -----------------
    # func is same name of the predict function in ProbabilisticClassifierChainCustom
    predict_funcs = [
        # {"name": "Predict", "func": "predict"},
        {"name": "Predict Hamming Loss", "func": "predict_Hamming"},
        {"name": "Predict Subset", "func": "predict_Subset"},
        {"name": "Predict Pre", "func": "predict_Pre"},
        {"name": "Predict Neg", "func": "predict_Neg"},
        {"name": "Predict Mar", "func": "predict_Mar"},
        # {"name": "Predict Inf", "func": "predict_Inf"}, #TODO: WIP
        # {"name": "Predict F Measure", "func": "predict_Fmeasure"}, #TODO: WIP
    ]

    metric_funcs = [
        {"name": "Hamming Loss", "func": EvaluationMetrics.hamming_loss},
        {
            "name": "Precision Score",
            "func": EvaluationMetrics.precision_score,
            # "options": {
            #     "average": "micro"
            # },  # other options: 'micro', 'macro', 'weighted'
        },
        {
            "name": "Recall Score",  #
            "func": EvaluationMetrics.recall_score,
        },
        {
            "name": "Subset Accuracy",
            "func": EvaluationMetrics.subset_accuracy,
        },
        # {
        #     "name": "F Measure",
        #     "func": metrics.f1_score,  # specific case of of F-beta when beta = 1 (harmonic mean of precision and recall)
        # },
        # {
        #     "name": "Markdness",
        #     "func": metrics.markdness,
        # },
        # {
        #     "name": "Informedness",
        #     "func": metrics,
        # },
    ]

    # Create a DataFrame to store the evaluation results
    data = {
        "Dataset": [],
        "Model": [],
        "Predict Function of Model": [],
        "Metric Function": [],
        "Score": [],
    }

    # Iterate over the datasets and models, perform evaluation, and append the results to the DataFrame
    for dataset in read_datasets_from_folder(folder_path):
        print(f"{'-'*50}\nDataset: {dataset[0].dataset_name}")
        df_train, df_test = dataset

        # convert df to numpy array
        X_train, Y_train = df_train.X.to_numpy(), df_train.Y.to_numpy()
        X_test, Y_test = df_test.X.to_numpy(), df_test.Y.to_numpy()

        print(f"X_train:\t{X_train.shape}\nY_train:\t{Y_train.shape}")
        # return

        # For each dataset, iterate over the models and perform evaluation
        for model in evaluated_models:
            trained_model = training_model(model, X_train, Y_train)

            loss_score_by_predict_funcs = evaluate_model(
                trained_model, X_test, Y_test, predict_funcs, metric_funcs
            )
            # Format of loss_score_by_predict_funcs: [
            #   { 'predict_name': "Predict",
            #       'score_metrics': [{'Metric Name': 'Hamming Loss', 'Metric Function': 'hamming_loss', 'Score': 0.0}, ...]
            #   }
            # ]

            # Add the results to the DataFrame.
            print("-" * 10)
            for loss_score_by_predict_func in loss_score_by_predict_funcs:
                print("Metric: ", loss_score_by_predict_func["predict_name"])

                for score_metric in loss_score_by_predict_func["score_metrics"]:
                    data["Dataset"].append(dataset[0].dataset_name)
                    data["Model"].append(model.base_estimator.__class__.__name__)

                    data["Predict Function of Model"].append(
                        loss_score_by_predict_func["predict_name"]
                    )
                    data["Metric Function"].append(score_metric["Metric Name"])
                    data["Score"].append(score_metric["Score"])

    results = pd.DataFrame(data)

    results.to_csv(output_csv, index=False)


if __name__ == "__main__":
    """IDEA:
    - Datasets
        - Models
            - Predict Functions
                - Metric Functions -> Score


    Add all to a DataFrame and save to CSV file
    with format: [
        {
            "Dataset": "emotions",
            "Model": "ProbabilisticClassifierChainCustom",
            "Predict Function of Model": "Predict",
            "Metric Function": "Hamming Loss",
            "Score": 0.0
        },
        {
            "Dataset": "emotions",
            "Model": "ProbabilisticClassifierChainCustom",
            "Predict Function of Model": "Predict Hamming Loss",
            "Metric Function": "Hamming Loss",
            "Score": 0.0
        },
        ...
    ]

    """
    main()

    # TODO:
    # [] 5 datasets
    # [] 3 models: SGDClassifier, RandomForestClassifier, XGBoostClassifier
    # [] F-measure, Informedness
    # [] Implement loss functions
