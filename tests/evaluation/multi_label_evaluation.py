import os
import json
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import LabelEncoder

import scipy.io.arff as arff
import pandas as pd

from skmultiflow.meta.classifier_chains_custom import ProbabilisticClassifierChainCustom

# # Load the ARFF file
# data = arff.loadarff('your_dataset.arff')
# # Convert the ARFF data to a Pandas DataFrame
# df = pd.DataFrame(data[0])

# Now, df contains your data as a Pandas DataFrame
# You can perform various data analysis tasks with df

SEED = 1


class HandleMultiLabelArffFile:
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
            df_train = HandleMultiLabelArffFile(
                os.path.join(folder_path, filename, f"{filename}-train.arff"), filename
            )

            # Testing data
            df_test = HandleMultiLabelArffFile(
                os.path.join(folder_path, filename, f"{filename}-test.arff"), filename
            )
            yield df_train, df_test


# Define a function to perform model evaluation on each dataset
def evaluate_model(datasets, model, metric_funcs):
    df_train, df_test = datasets

    # convert df to numpy array
    X_train, Y_train = df_train.X.to_numpy(), df_train.Y.to_numpy()
    X_test, Y_test = df_test.X.to_numpy(), df_test.Y.to_numpy()

    model.fit(X_train, Y_train)
    # TODO: List of type of prediction
    Y_pred = model.predict(X_test)

    print(f"Y_pred:\t{Y_pred}\nY_test:\t{Y_test}")

    score_metrics = []
    for metric_func in metric_funcs:
        score = metric_func(Y_test, Y_pred)
        print(f"{metric_func.__name__}: {score}")
        score_metrics.append(score)

    return score_metrics


def prepare_model_to_evaluate():
    pcc = [SGDClassifier(max_iter=100, tol=1e-3, loss="log_loss", random_state=SEED)]

    # Add more models here if you want to evaluate them
    return [ProbabilisticClassifierChainCustom(model) for model in pcc]


def main():
    # Define the list of models you want to evaluate
    models_to_evaluate = prepare_model_to_evaluate()

    # Define the folder path containing JSON datasets and the output CSV file name
    folder_path = (
        "/Users/xuantruong/Documents/JAIST/scikit-multiflow/tests/evaluation/dataset"
    )
    output_csv = "/Users/xuantruong/Documents/JAIST/scikit-multiflow/tests/evaluation/result/evaluation_results.csv"

    metric_funcs = [metrics.hamming_loss]

    # Create a DataFrame to store the evaluation results
    results = pd.DataFrame(columns=["Model", "Dataset", "Hamming Loss"])

    # Iterate over the datasets and models, perform evaluation, and append the results to the DataFrame
    for dataset in read_datasets_from_folder(folder_path):
        # df_train, df_test = dataset
        # print(df_train.X)
        # print(df_train.Y)
        # For each dataset, iterate over the models and perform evaluation
        for model in models_to_evaluate:
            loss_score = evaluate_model(dataset, model, metric_funcs)
            results = results.append(
                {
                    "Model": type(model).__name__,
                    "Dataset": dataset["name"],
                    "Hamming Loss": loss_score,
                },
                ignore_index=True,
            )

    # Save the results to a CSV file
    results.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
