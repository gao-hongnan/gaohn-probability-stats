"""Graphical representation of the decision boundary of a Naive Bayes classifier in 2D."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from src.generative.naive_bayes.naive_bayes import (
    NaiveBayesGaussian,
    NaivesBayesHyperParams,
)


def graph_boundaries(
    X, model, model_title, n0=100, n1=100, figsize=(7, 5), label_every=4
):

    # Generate X for plotting
    d0_range = np.linspace(X[:, 0].min(), X[:, 0].max(), n0)
    d1_range = np.linspace(X[:, 1].min(), X[:, 1].max(), n1)
    X_plot = np.array(np.meshgrid(d0_range, d1_range)).T.reshape(-1, 2)

    # Get class predictions
    y_plot = model.predict(X_plot).astype(int)
    print(y_plot)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        y_plot.reshape(n0, n1).T,
        cmap=sns.color_palette("Pastel1", 3),
        cbar_kws={"ticks": sorted(np.unique(y_plot))},
    )
    xticks, yticks = ax.get_xticks(), ax.get_yticks()
    # ax.set(
    #     xticks=xticks[::label_every],
    #     xticklabels=d0_range.round(2)[::label_every],
    #     yticks=yticks[::label_every],
    #     yticklabels=d1_range.round(2)[::label_every],
    # )
    ax.set(xlabel="X1", ylabel="X2", title=model_title + " Predictions by X1 and X2")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
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

    graph_boundaries(X_train_2d, gnb, "Gaussian Naive Bayes")

    sk_gnb = GaussianNB(var_smoothing=0)  # to get exact same results as sklearn
    sk_gnb.fit(X_train_2d, y_train)

    plt.figure(figsize=(16, 8))
    plot_decision_regions(
        X_train_2d, y_train, clf=sk_gnb, legend=0, colors="#1f77b4,#ff7f0e,#ffec6e"
    )
    print("A")
    for i in range(3):
        plt.scatter(X[np.where(y == i), 0], X[np.where(y == i), 1], s=200)
    plt.scatter([-2, 0, 6], [5, 0, -0.3], c="k", s=200)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Decision Regions")
    plt.show()
