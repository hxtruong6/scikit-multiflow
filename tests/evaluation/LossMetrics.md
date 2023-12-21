# Loss metric

## Hamming loss

Hamming Loss is a metric used to measure the accuracy of binary or multilabel classifications. It calculates the fraction of labels that are incorrectly predicted. The formula for Hamming Loss is given by:

$$ \text{Hamming Loss} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{m} \sum_{j=1}^{m} (y_{\text{true},ij} \neq y_{\text{pred},ij}) $$

where:

- $ N $ is the number of samples.
- $ m $ is the number of labels.
- $ y_{\text{true},ij} $ is the true label of the $ j $-th label of the $ i $-th sample.
- $ y_{\text{pred},ij} $ is the predicted label of the $ j $-th label of the $ i $-th sample.

## Precision

Precision is a metric that measures the accuracy of positive predictions made by a classification model. It is defined as the number of true positives divided by the sum of true positives and false positives. The formula for precision is:

$$ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} $$

## Recall

Recall, also known as Sensitivity or True Positive Rate, is a metric that measures the ability of a classification model to capture all positive instances. It is defined as the number of true positives divided by the sum of true positives and false negatives. The formula for recall is:

$$ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $$

## F-measure

## Subset accuracy

Subset accuracy, also known as exact match ratio or accuracy_score, is a classification metric that measures the percentage of samples for which the predicted labels exactly match the true labels. In other words, it calculates the accuracy of predicting all labels for a sample correctly. The formula for subset accuracy is:

$$ \text{Subset Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(\text{predicted labels}_i = \text{true labels}_i) $$

where:

- $N$ is the number of samples.
- $\mathbb{1}(\cdot)$ is the indicator function that returns 1 if the condition inside is true and 0 otherwise.

## Markesness

## Informedness
