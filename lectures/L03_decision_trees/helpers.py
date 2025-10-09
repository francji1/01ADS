# Helpers for interactive tree visualisation (auto-generated)
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from matplotlib.axes import Axes
from sklearn.tree import DecisionTreeClassifier
from typing import Any, Optional, Tuple


def visualize_tree(
    estimator: DecisionTreeClassifier,
    X: np.ndarray,
    y: np.ndarray,
    boundaries: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    '''Fit the estimator on `(X, y)` and plot decision boundaries for two features.'''
    ax = ax or plt.gca()
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        s=30,
        cmap='viridis',
        clim=(y.min(), y.max()),
        zorder=3,
    )
    scatter.set_label('training data')
    ax.set_xlabel('feature 1')
    ax.set_ylabel('feature 2')
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()
    estimator.fit(X, y)
    xx, yy = np.meshgrid(
        np.linspace(*xlim, num=200),
        np.linspace(*ylim, num=200),
    )
    mesh_predictions = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    mesh_predictions = mesh_predictions.reshape(xx.shape)
    ax.contourf(
        xx,
        yy,
        mesh_predictions,
        alpha=0.3,
        levels=np.arange(len(np.unique(y)) + 1) - 0.5,
        cmap='viridis',
    )
    if boundaries:
        def plot_boundaries(index: int, x_span: Tuple[float, float], y_span: Tuple[float, float]) -> None:
            if index < 0:
                return
            tree = estimator.tree_
            threshold = tree.threshold[index]
            feature_index = tree.feature[index]
            if feature_index == 0:
                ax.plot([threshold, threshold], y_span, '-k', zorder=2)
                plot_boundaries(tree.children_left[index], (x_span[0], threshold), y_span)
                plot_boundaries(tree.children_right[index], (threshold, x_span[1]), y_span)
            elif feature_index == 1:
                ax.plot(x_span, [threshold, threshold], '-k', zorder=2)
                plot_boundaries(tree.children_left[index], x_span, (y_span[0], threshold))
                plot_boundaries(tree.children_right[index], x_span, (threshold, y_span[1]))
        plot_boundaries(0, xlim, ylim)
    ax.set(xlim=xlim, ylim=ylim)
    return ax


def plot_tree_interactive(X: np.ndarray, y: np.ndarray) -> Any:
    '''Create an interactive widget to explore tree depth on a toy dataset.'''
    def interactive_tree(depth: int) -> Axes:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        return visualize_tree(clf, X, y)
    return interact(interactive_tree, depth=list(range(1, 11)))


def randomized_tree_interactive(X: np.ndarray, y: np.ndarray) -> None:
    '''Illustrate how random sampling changes the learned boundaries.'''
    n_samples = int(0.75 * X.shape[0])
    xlim = (X[:, 0].min(), X[:, 0].max())
    ylim = (X[:, 1].min(), X[:, 1].max())

    def fit_randomized_tree(random_state: int = 0) -> None:
        clf = DecisionTreeClassifier(max_depth=15, random_state=random_state)
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(X.shape[0])[:n_samples]
        visualize_tree(
            clf,
            X[indices],
            y[indices],
            boundaries=False,
            xlim=xlim,
            ylim=ylim,
        )

    interact(fit_randomized_tree, random_state=list(range(0, 501, 50)))
