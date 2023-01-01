"""Graphical representation of the decision boundary of a Naive Bayes classifier in 2D."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap
from src.generative.naive_bayes.naive_bayes import (
    NaiveBayesGaussian,
    NaivesBayesHyperParams,
)
from sklearn.datasets import make_blobs


def get_meshgrid(x0_range, x1_range, num_points=100):
    x0 = np.linspace(x0_range[0], x0_range[1], num_points)
    x1 = np.linspace(x1_range[0], x1_range[1], num_points)
    return np.meshgrid(x0, x1)


def contour_plot(
    x0_range, x1_range, prob_fn, batch_shape, colours, levels=None, num_points=100
):
    X0, X1 = get_meshgrid(x0_range, x1_range, num_points=num_points)
    Z = prob_fn(np.expand_dims(np.array([X0.ravel(), X1.ravel()]).T, 1))
    Z = np.array(Z).T.reshape(batch_shape, *X0.shape)
    for batch in np.arange(batch_shape):
        if levels:
            plt.contourf(X0, X1, Z[batch], alpha=0.2, colors=colours, levels=levels)
        else:
            plt.contour(X0, X1, Z[batch], colors=colours[batch], alpha=0.3)


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02, ax=None):
    if ax is None:
        ax = plt.gca()
    # setup marker generator and color map
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    print(x1_min, x1_max, x2_min, x2_max)
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    print(xx1, xx2)
    X_input_space = np.array([xx1.ravel(), xx2.ravel()]).T  # N x 2 matrix
    print(X_input_space)
    Z = classifier.predict(X_input_space)
    print(f"Z: {Z}, {np.unique(Z)}")
    Z = Z.reshape(xx1.shape)
    X_11 = X_input_space[0, 0]
    X_12 = X_input_space[0, 1]

    contour = ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    # contour = ax.contour(xx1, xx2, Z, alpha=0.3, cmap=cmap, levels=np.unique(Z))
    # contour = ax.contour(xx1, xx2, Z, alpha=0.3, cmap=cmap, levels=[1])
    print(f"contour levels: {contour.levels}")
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax.scatter(X_11, X_12, c="red", marker="o", s=100, label="test")

    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolor="black",
        )
    # plt.colorbar(contour, ax=ax)
    plt.show()


if __name__ == "__main__":

    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    X_train_2d = X_train[:, :2]

    hparams = {
        "random_state": 1992,
        "num_classes": 3,
    }
    naives_bayes_hyperparams = NaivesBayesHyperParams.from_dict(hparams)

    gnb = NaiveBayesGaussian(naives_bayes_hyperparams)
    gnb.fit(X_train_2d, y_train)

    # graph_boundaries(X_train_2d, gnb, "Gaussian Naive Bayes")
    plot_decision_regions(X_train_2d, y_train, classifier=gnb)

    # sk_gnb = GaussianNB(var_smoothing=0)  # to get exact same results as sklearn
    # sk_gnb.fit(X_train_2d, y_train)

    # plt.figure(figsize=(16, 8))
    # plot_decision_regions(
    #     X_train_2d, y_train, clf=sk_gnb, legend=0, colors="#1f77b4,#ff7f0e,#ffec6e"
    # )
    # print("A")
    # for i in range(3):
    #     plt.scatter(X[np.where(y == i), 0], X[np.where(y == i), 1], s=200)
    # plt.scatter([-2, 0, 6], [5, 0, -0.3], c="k", s=200)
    # plt.xlabel("$x_1$")
    # plt.ylabel("$x_2$")
    # plt.title("Decision Regions")
    # plt.show()
