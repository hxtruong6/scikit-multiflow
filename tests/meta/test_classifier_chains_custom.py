import pytest

from sklearn.linear_model import SGDClassifier
from sklearn import __version__ as sklearn_version
from sklearn import set_config

from skmultiflow.data import MultilabelGenerator, make_logical
from skmultiflow.meta import (
    ClassifierChain,
    MonteCarloClassifierChain,
    ProbabilisticClassifierChain,
)

import numpy as np

from skmultiflow.meta.classifier_chains_custom import ProbabilisticClassifierChainCustom

# Force sklearn to show only the parameters whose default value have been changed when
# printing an estimator (backwards compatibility with versions prior to sklearn==0.23)
set_config(print_changed_only=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_classifier_chains_all():
    seed = 1
    X, Y = make_logical(random_state=seed)

    # # CC
    # cc = ClassifierChain(SGDClassifier(max_iter=100, tol=1e-3, loss='log', random_state=seed))
    # cc.partial_fit(X, Y)
    # y_predicted = cc.predict(X)
    # y_expected = [[1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0]]
    # assert np.alltrue(y_predicted == y_expected)
    # assert type(cc.predict_proba(X)) == np.ndarray

    # # RCC
    # rcc = ClassifierChain(SGDClassifier(max_iter=100, tol=1e-3, loss='log', random_state=seed),
    #                       order='random', random_state=seed)
    # rcc.partial_fit(X, Y)
    # y_predicted = rcc.predict(X)
    # y_expected = [[1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0]]
    # assert np.alltrue(y_predicted == y_expected)

    # # MCC
    # mcc = MonteCarloClassifierChain(SGDClassifier(max_iter=100, tol=1e-3, loss='log',
    #                                               random_state=seed),
    #                                 M=1000)
    # mcc.partial_fit(X, Y)
    # y_predicted = mcc.predict(X)
    # y_expected = [[1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0]]
    # assert np.alltrue(y_predicted == y_expected)

    # PCC
    pcc = ProbabilisticClassifierChainCustom(
        SGDClassifier(max_iter=100, tol=1e-3, loss="log_loss", random_state=seed)
    )
    pcc.partial_fit(X, Y)
    # y_predicted = pcc.predict(X)
    # y_predicted_proba = pcc.predict_proba(X)

    # print(f"X:\n{X}\n")
    # print(f"y_predicted_proba:\n{y_predicted_proba}\n")
    # print(f"y_predicted:\n{y_predicted}\n")

    y_expected = [[1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 0]]

    # y_hamming = pcc.predict_Hamming(X)
    # print(y_hamming)
    # print(y_hamming == y_expected)
    # assert np.alltrue(y_predicted == y_expected)

    # y_subset = pcc.predict_Subset(X)
    # print(y_subset)
    # print(y_subset == y_expected)

    # y_pre = pcc.predict_Pre(X)
    # y_pre = pcc.predict_Neg(X)
    # y_pre = pcc.predict_Mar(X)
    y_pre = pcc.predict_Inf(X)
    print(f"y_pre:\n{y_pre}\n")
    print(f"y_pre == y_expected:\n{y_pre == y_expected}\n")


if __name__ == "__main__":
    # pytest.main([__file__])
    test_classifier_chains_all()
