import copy

import numpy as np
from sklearn.linear_model import LogisticRegression

from skmultiflow.core import (
    BaseSKMObject,
    ClassifierMixin,
    MetaEstimatorMixin,
    MultiOutputMixin,
)
from skmultiflow.utils import check_random_state


class ClassifierChainCustom(
    BaseSKMObject, ClassifierMixin, MetaEstimatorMixin, MultiOutputMixin
):
    """Classifier Chains for multi-label learning.

    Parameters
    ----------
    base_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator
        (default=LogisticRegression) Each member of the ensemble is
        an instance of the base estimator

    order : str (default=None)
        `None` to use default order, 'random' for random order.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    Examples
    --------
    >>> from skmultiflow.data import make_logical
    >>> from sklearn.linear_model import SGDClassifier
    >>>
    >>> X, Y = make_logical(random_state=1)
    >>>
    >>> print("TRUE: ")
    >>> print(Y)
    >>> print("vs")
    >>>
    >>> print("CC")
    >>> cc = ClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=1))
    >>> cc.fit(X, Y)
    >>> print(cc.predict(X))
    >>>
    >>> print("RCC")
    >>> cc = ClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=1),
    ...                                     order='random', random_state=1)
    >>> cc.fit(X, Y)
    >>> print(cc.predict(X))
    >>>
    TRUE:
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]
    vs
    CC
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]
    RCC
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]



    Notes
    -----
    Classifier Chains [1]_ is a popular method for multi-label learning. It exploits correlation
    between labels by incrementally building binary classifiers for each label.

    scikit-learn also includes 'ClassifierChain'. A difference is probabilistic extensions
    are included here.


    References
    ----------
    .. [1] Read, Jesse, Bernhard Pfahringer, Geoff Holmes, and Eibe Frank. "Classifier chains
        for multi-label classification." In Joint European Conference on Machine Learning and
        Knowledge Discovery in Databases, pp. 254-269. Springer, Berlin, Heidelberg, 2009.

    """

    # TODO: much of this can be shared with Regressor Chains, probably should
    # use a base class to inherit here.

    def __init__(
        self, base_estimator=LogisticRegression(), order=None, random_state=None
    ):
        super().__init__()
        self.base_estimator = base_estimator
        self.order = order
        self.random_state = random_state
        self.chain = None
        self.ensemble = None
        self.L = None
        self._random_state = (
            None  # This is the actual random_state object used internally
        )
        self.__configure()

    def __configure(self):
        self.ensemble = None
        self.L = -1
        self._random_state = check_random_state(self.random_state)

    def fit(self, X, y, classes=None, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples, n_targets)
            An array-like with the labels of all samples in X.

        classes: Not used (default=None)

        sample_weight: Not used (default=None)

        Returns
        -------
        self

        """
        # N is the number of samples, self.L is the number of labels
        N, self.L = y.shape
        L = self.L
        # where D is the number of features
        N, D = X.shape

        # Set the chain order. np.arange(L) is the default order and return [0, 1, ..., L-1]
        self.chain = np.arange(L)
        if self.order == "random":
            self._random_state.shuffle(self.chain)

        # Set the chain order
        y = y[:, self.chain]

        # Train
        self.ensemble = [copy.deepcopy(self.base_estimator) for _ in range(L)]
        XY = np.zeros((N, D + L - 1))
        XY[:, 0:D] = X
        XY[:, D:] = y[:, 0 : L - 1]
        for j in range(self.L):
            self.ensemble[j].fit(XY[:, 0 : D + j], y[:, j])
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the labels of all samples in X.

        classes: Not used (default=None)

        sample_weight: NOT used (default=None)

        Returns
        -------
        self

        """
        if self.ensemble is None:
            # This is the first time that the model is fit
            self.fit(X, y)
            return self

        # N is the number of samples, self.L is the number of labels
        N, self.L = y.shape
        L = self.L
        # D is the number of features
        N, D = X.shape

        # Set the chain order
        y = y[:, self.chain]

        XY = np.zeros((N, D + L - 1))
        XY[:, 0:D] = X
        XY[:, D:] = y[:, 0 : L - 1]
        for j in range(L):
            # partial_fit used the true_label (1/0) of the previous label as input
            # fit used the predicted_label (prob) of the previous label as input
            # using maximum likelihood
            self.ensemble[j].partial_fit(XY[:, 0 : D + j], y[:, j])

        return self

    def predict(self, X):
        """Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the labels for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        """
        N, D = X.shape
        # N is the number of samples, self.L is the number of labels
        # Y is the output matrix with shape (N, self.L)
        Y = np.zeros((N, self.L))
        # For each label j, predict the label of the j-th label
        for j in range(self.L):
            if j > 0:
                X = np.column_stack([X, Y[:, j - 1]])
            Y[:, j] = self.ensemble[j].predict(X)

        # Unset the chain order (back to default)
        return Y[:, np.argsort(self.chain)]

    def predict_proba(self, X):
        """Estimates the probability of each sample in X belonging to each of the class-labels.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The matrix of samples one wants to predict the class probabilities for.

        Returns
        -------
        A numpy.ndarray of shape (n_samples, n_labels), in which each outer entry is associated
        with the X entry of the same index. And where the list in index [i] contains
        len(self.target_values) elements, each of which represents the probability that
        the i-th sample of X belongs to a certain class-label.

        Notes
        -----
        Returns marginals [P(y_1=1|x),...,P(y_L=1|x,y_1,...,y_{L-1})]
        i.e., confidence predictions given inputs, for each instance.

        This function suitable for multi-label (binary) data
        only at the moment (may give index-out-of-bounds error if
        uni- or multi-target (of > 2 values) data is used in training).
        """
        N, D = X.shape
        Y = np.zeros((N, self.L))
        for j in range(self.L):
            if j > 0:
                X = np.column_stack([X, Y[:, j - 1]])
                # Y[:, j - 1] is the predicted label of the previous label
                # e.g., X = [[1, 2, 3], [4, 5, 6]], Y[:, j - 1] = [0, 1, 0], np.column_stack([X, Y[:, j - 1]]) = [[1, 2, 3, 0], [4, 5, 6, 1]]
            Y[:, j] = self.ensemble[j].predict_proba(X)[:, 1]
            # e.g., self.ensemble[j].predict_proba(X) = [[0.9, 0.1], [0.2, 0.8]], Y[:, j] = [0.1, 0.8]
        return Y

    def reset(self):
        self.__configure()
        return self

    def _more_tags(self):
        return {"multioutput": True, "multioutput_only": True}


def P(y, x, cc, payoff=np.prod):
    """Payoff function, P(Y=y|X=x)

    What payoff do we get for predicting y | x, under model cc.

    Parameters
    ----------
    x: input instance
    y: its true labels
    cc: a classifier chain
    payoff: payoff function. Default is np.prod
            example np.prod([0.1, 0.2, 0.3]) = 0.006 (np.prod returns the product of array elements over a given axis.)


    Returns
    -------
    A single number; the payoff of predicting y | x.
    """
    D = len(x)
    # D is the number of features
    L = len(y)
    # L is the number of labels

    p = np.zeros(L)

    # xy is the concatenation of x and y
    # e.g., x = [1, 2, 3], y = [0, 1, 0], xy = [1, 2, 3, 0, 1, 0]
    xy = np.zeros(D + L)

    xy[0:D] = x.copy()

    # For each label j, compute P_j(y_j | x, y_1, ..., y_{j-1})
    for j in range(L):
        # reshape(1,-1) is needed because predict_proba expects a 2D array
        # example: cc.ensemble[j].predict_proba(xy[0:D+j].reshape(1,-1)) = [[0.9, 0.1]]

        P_j = cc.ensemble[j].predict_proba(xy[0 : D + j].reshape(1, -1))[0]
        # e.g., [0.9, 0.1] wrt 0, 1

        xy[D + j] = y[j]  # e.g., 1
        p[j] = P_j[y[j]]
        # e.g., 0.1 or, y[j] = 0 is predicted with probability p[j] = 0.9
    # print(f"p = {p}")

    # The more labels we predict incorrectly, the higher the penalty of the payoff
    # p = [0.99055151 0.00709076 0.99999978]
    # y_ [0 1 0]
    # w_ = 0.007
    return payoff(p)


class ProbabilisticClassifierChainCustom(ClassifierChainCustom):
    """Probabilistic Classifier Chains for multi-label learning.

    Published as 'PCC'

    Parameters
    ----------
    base_estimator: skmultiflow or sklearn model (default=LogisticRegression)
        This is the ensemble classifier type, each ensemble classifier is going
        to be a copy of the base_estimator.

    order : str (default=None)
        `None` to use default order, 'random' for random order.

    random_state: int, RandomState instance or None, optionalseed used by the random number genera (default=None)
        If int, random_state is the tor;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    Examples
    --------
    >>> from skmultiflow.data import make_logical
    >>> from sklearn.linear_model import SGDClassifier
    >>>
    >>> X, Y = make_logical(random_state=1)
    >>>
    >>> print("TRUE: ")
    >>> print(Y)
    >>> print("vs")
    >>> print("PCC")
    >>> pcc = ProbabilisticClassifierChain(SGDClassifier(max_iter=100, loss='log', random_state=1))
    >>> pcc.fit(X, Y)
    >>> print(pcc.predict(X))
    TRUE:
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]
    vs
    PCC
    [[1. 0. 1.]
     [1. 1. 0.]
     [0. 0. 0.]
     [1. 1. 0.]]
    """

    def __init__(
        self, base_estimator=LogisticRegression(), order=None, random_state=None
    ):
        super().__init__(
            base_estimator=base_estimator, order=order, random_state=random_state
        )

    def predict(
        self, X, marginal=False, pairwise=False
    ) -> (np.ndarray, np.ndarray, dict):
        """Predict classes for the passed data.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of data samples to predict the labels for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the samples in X.

        Notes
        -----
        Explores all possible branches of the probability tree
        (i.e., all possible 2^L label combinations).
        """
        N, D = X.shape
        # print(f"X.shape = {X.shape} N = {N} D = {D}")

        Yp = np.zeros((N, self.L))

        # if marginal:
        P_margin_yi_1 = np.zeros((N, self.L))

        # if pairwise:
        P_pair_wise = np.zeros((N, self.L, self.L + 1))
        P_pair_wise0 = np.zeros((N, 1))
        P_pair_wise1 = np.zeros((N, 1))

        # for each instance
        for n in range(N):
            w_max = 0.0

            # s is the number of labels that are 1
            s = 0
            # for each and every possible label combination
            # initialize a list of $L$ elements which encode the $L$ marginal probability masses
            # initialize a $L \times (L+1)$ matrix which encodes the pairwise probability masses
            # (i.e., all possible 2^L label combinations) [0, 1, ..., 2^L-1]
            for b in range(2**self.L):
                # print(f"b = {b}")
                # put together a label vector
                # e.g., b = 3, self.L = 3, y_ = [0, 0, 1] | b = 5, self.L = 3, y_ = [0, 1, 0]
                y_ = np.array(list(map(int, np.binary_repr(b, width=self.L))))

                # print(f"y_ = {y_}")
                # ... and gauge a probability for it (given x)
                w_ = P(y_, X[n], self)
                # print(f"w_ = {round(w_, 3)}")

                # All values of y_ are 0
                if np.sum(y_) == 0:
                    P_pair_wise0[n] = w_

                # All values of y_ are 1
                if np.sum(y_) == self.L:
                    P_pair_wise1[n] = w_

                if pairwise:
                    # is number [0-K]
                    s = np.sum(y_)

                if marginal or pairwise:
                    for label_index in range(self.L):
                        if y_[label_index] == 1:
                            P_margin_yi_1[n, label_index] += w_
                            P_pair_wise[n, label_index, s] += w_

                # Use y_ to check which marginal probability masses and pairwise
                # probability masses should be updated (by adding w_)
                # if it performs well, keep it, and record the max
                if w_ > w_max:
                    Yp[n, :] = y_[:].copy()
                    w_max = w_

                # P(y_1 = 1 | X) = P(y_1 = 1 | X, y_2 = 0) * P(y_2 = 0 | X) + P(y_1 = 1 | X, y_2 = 1) * P(y_2 = 1 | X)

            # # iterate over all labels. self.L is number of labels
            # for label_index in range(self.L):

            #     for index in range(self.L):
            #         P_margin_yi_1[n, label_index] += P([index], X[n], self)

        # print(f"P_margin_yi_1 = {[[round(x, 3) for x in y] for y in P_margin_yi_1]}")

        # print(
        #     f"pair wise {[[[round(x, 3) for x in y] for y in z] for z in P_pair_wise]}"
        # )
        # print(f"Yp = {[[round(x, 3) for x in y] for y in w_max]}")

        return (
            Yp,
            P_margin_yi_1,
            {
                "P_pair_wise": P_pair_wise,
                "P_pair_wise0": P_pair_wise0,
                "P_pair_wise1": P_pair_wise1,
            },
        )
        # return Yp, marginal probability masses and pairwise probability masses
        # for each instance X[n] (we might need to choose some appropriate data structure)

        # We would define other inference algorithms for other loss functions or measures by
        # defining def predict_Hamming(self, X):, def predict_Fmeasure(self, X): and so on

    def predict_Hamming(self, X):
        _, P_margin_yi_1, _ = self.predict(X, marginal=True)
        # print(f"P_margin_yi_1 = {[[round(x, 3) for x in y] for y in P_margin_yi_1]}")

        return np.where(P_margin_yi_1 > 0.5, 1, 0)

    def predict_Subset(self, X):
        predictions, _, _ = self.predict(X)
        return predictions

    def predict_Pre(self, X):
        """Predicts the label combination with the highest precision."""
        N, D = X.shape

        Yp = np.zeros((N, self.L))

        _, P_margin_yi_1, _ = self.predict(X, marginal=True)
        for n in range(N):
            max_index = np.argmax(P_margin_yi_1[n])
            Yp[n, max_index] = 1

        # print(f"Yp = {Yp}")
        # print(f"Y_prob = {Y_prob}")
        # print([[round(x, 4) for x in y] for y in P_margin_yi_1])
        return Yp

    def predict_Neg(self, X):
        """Predicts the label combination with the highest negative correlation."""
        N, _ = X.shape
        _, P_margin_yi_1, _ = self.predict(X, marginal=True)
        # Sort the marginal probability masses in asc order
        # and get the indices of the sorted array
        # print(f"P_margin_yi_1 = {[round(y, 6) for x in P_margin_yi_1 for y in x]}")
        indices = np.argsort(P_margin_yi_1, axis=1)[:]
        # print(f"indices = {indices}")

        # X.shape[0] is the number of instances
        P = np.ones((N, self.L))
        # Set the smallest probability mass to 0 and the rest to 1
        P[np.arange(N)[:, None], indices[:, :1]] = 0

        return P

    def predict_Recall(self, X):
        # The hightest marginal probability.
        # return all array with 1
        return np.ones((X.shape[0], self.L))

    def predict_Mar(self, X):
        """Predicts the label combination with the highest marginal probability."""
        N, _ = X.shape
        _, P_margin_yi_1, _ = self.predict(X, marginal=True)
        # Sort in descending order
        indices = np.argsort(P_margin_yi_1, axis=1)[:][:, ::-1]
        # print(f"indices = {indices}")

        # Expectation of the marginal probability masses
        E = np.zeros((N, self.L))

        # Find the optimal l for each instance of expectation
        for i in range(N):
            # E_0
            E[i][0] = 2 - (1 / self.L) * np.sum(P_margin_yi_1, axis=1)[i]
            E[i][self.L - 1] = 1 + (1 / self.L) * np.sum(P_margin_yi_1, axis=1)[i]

            s1 = np.sum(P_margin_yi_1, axis=1)[i]
            s2 = 0
            # print(f"E = {E}")
            for _l in range(1, self.L - 1):
                s2 = s2 + P_margin_yi_1[i, indices[i, _l]]
                # print(f"l = {l}, s1 = {s1}, s2 = {s2}")
                E[i][_l] = 1 - (1 / (self.L - _l)) * s1 + (1 / (self.L - _l) * _l) * s2

        # TODO: convert to vectorized form
        # print(f"E = {E}")
        l_optimal = np.argsort(E, axis=1)[:, ::-1]
        # print(f"l_optimal = {l_optimal}")
        P = np.zeros((N, self.L))

        for i in range(N):
            # print(f"l_optimal = {l_optimal[i]}")
            # Set l_optimal highest of the descending sorted marginal probability masses to 1
            for _l in range(l_optimal[i][0] + 1):
                P[i][indices[i, _l]] = 1

        # print(f"P_margin_yi_1 = {[[round(x, 5) for x in y] for y in P_margin_yi_1]}")
        return P

    def predict_Fmeasure(self, X, beta=1):
        N, _ = X.shape
        _, _, P_pair_wise_obj = self.predict(X, pairwise=True)

        P_pair_wise, P_pair_wise0, P_pair_wise1 = (
            P_pair_wise_obj["P_pair_wise"],
            P_pair_wise_obj["P_pair_wise0"],
            P_pair_wise_obj["P_pair_wise1"],
        )

        # print(
        #     f"P_pair_wise = {[[[round(x, 3) for x in y] for y in z] for z in P_pair_wise]}"
        # )

        # E[0] , E[L-1], E[L]

        P = np.zeros((N, self.L))

        for i in range(N):  # for each instance
            # q_f_measure[i][top_ranked_label][label]
            q_f_measure = np.zeros((self.L, self.L))
            indices_q_f_measure_desc = []

            expectation_values = np.zeros(self.L + 1)

            # line 9 in the algorithm for F-measure
            expectation_value_0 = P_pair_wise0[i][0]

            # rank label = L -> L + 1 top ranked labels
            for top_ranked_label in range(self.L):
                # l = top ranked labels \bar{y}_{(k)} = 1
                for label in range(self.L):  # for each label
                    for s in range(self.L):
                        # for group of vectors with s relevant labels (label = 1)
                        # + 2 because iterate from 1 to L
                        q_f_measure[top_ranked_label][label] += (1 + beta**2) * (
                            P_pair_wise[i][label][s]
                            / (
                                beta**2 * (s + 1) + top_ranked_label + 1
                            )  # revised indices
                        )
                # sort by descending order indices_q_f_measure_desc[top_ranked_label]
                indices_q_f_measure_desc.append(
                    np.argsort(q_f_measure[top_ranked_label])[::-1].tolist()
                )

                # print(
                #     f"indices_q_f_measure_desc = {indices_q_f_measure_desc} |{indices_q_f_measure_desc[top_ranked_label]} \n {top_ranked_label} \n q_f_measure: {q_f_measure[top_ranked_label]}"
                # )
                # max q_f_measure
                # q_f_measure_max = q_f_measure[i][top_ranked_label][indices_q_f_measure[i][0]]

                # Expectation value at top_ranked_label = sum max q_f_measure from 0 to top_ranked_label
                for i_ in range(top_ranked_label + 1):
                    expectation_values[top_ranked_label] += q_f_measure[
                        top_ranked_label
                    ][int(indices_q_f_measure_desc[top_ranked_label][i_])]

            # print(f"\nexpectation_values = {expectation_values}")
            # print(f"q_f_measure = {q_f_measure}")
            # print(f"indices_q_f_measure_desc = {indices_q_f_measure_desc}")
            # print(f"expectation_value_0 = {expectation_value_0}")

            # Determine ˆy which is ˆyl with the highest E(f (y, ˆyl) where l ∈ [K]0
            # Case 1: Expectation value of 0 > max(expectation_values)
            if expectation_value_0 > np.max(expectation_values):
                P[i] = np.zeros(self.L)
            else:
                # Case 2: Expectation value of 0 <= max(expectation_values)
                # max_expectation_value_index = L_optimal -> optimal top ranked label
                L_optimal_index = np.argmax(expectation_values)
                # print(f"max_expectation_value_index = {L_optimal_index}")
                for _l in range(L_optimal_index + 1):
                    P[i][int(indices_q_f_measure_desc[L_optimal_index][_l])] = 1

        return P

    def predict_Inf(self, X):
        N, _ = X.shape
        _, _, P_pair_wise_obj = self.predict(X, pairwise=True)

        P_pair_wise, P_pair_wise0, P_pair_wise1 = (
            P_pair_wise_obj["P_pair_wise"],
            P_pair_wise_obj["P_pair_wise0"],
            P_pair_wise_obj["P_pair_wise1"],
        )

        print(
            f"P_pair_wise = {[[[round(x, 3) for x in y] for y in z] for z in P_pair_wise]}"
        )

        q_inf = np.zeros((N, self.L - 1))

        # E[0] , E[L-1], E[L]

        P = np.zeros((N, self.L))
        index_L = [0, self.L, self.L - 1]

        for i in range(N):
            for k in range(self.L - 1):
                # s is value
                for s in range(self.L):
                    q_inf[i][k] += P_pair_wise[i][k][s] / (s + 1)

            # sort by descending order
            indices_q = np.argsort(q_inf[i])[::-1]
            print(f"Index: {indices_q}")

            # E[0] = E[0] , E[1] = E[L] ,E[2] = E[L-1]
            E = np.zeros(3)
            E[0] = 1 + P_pair_wise0[i]
            E[1] = 1 + P_pair_wise1[i]
            E[2] = np.sum(q_inf[i][indices_q[0 : self.L - 1]])

            # sort E in descending order
            indices_E = np.argsort(E)[::-1]
            print(f"E: {E} \t indices_E = {indices_E}")
            L_optimal = index_L[indices_E[0]]
            print(f"indices_q: {indices_q}")
            print(f"L_optimal = {L_optimal}")

            for _l in range(L_optimal):
                # revised indices: L_optimal = 0 -> no relevant label, L_optimal = L -> L relevant labels, L_optimal = L-1 -> L-1 relevant labels
                P[i][indices_q[_l]] = 1
                # TODO: check if this is correct

        return P
