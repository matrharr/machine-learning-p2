import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
import numpy as np

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, learning_curve

from visualization.plot_learning_curve import plot_learning_curve
from data.load_data import load_bankrupt_data, load_brain_tumor_data
from metrics import get_metrics

algos = [
    'gradient_descent',
    'random_hill_climb',
    'simulated_annealing',
    'genetic_alg'
]

metric_dict = {}

# data = load_bankrupt_data()
# y_col = 'Bankrupt?'
data = load_brain_tumor_data()
y_col = 'Class'

sss = StratifiedShuffleSplit(
    n_splits=2,
    random_state=42,
    test_size=0.2
)

y_all = data[y_col]
x_all = data

chy = SelectKBest(chi2, k=5)
x_all = chy.fit_transform(x_all, y_all)
print('Selected Columns: ', data.columns[chy.get_support()])

for train_index, test_index in sss.split(x_all, y_all):
    train_set = data.loc[train_index]
    test_set = data.loc[test_index]

x_train = train_set.drop(y_col, axis=1)
y_train = train_set[y_col].copy()

x_test = test_set.drop(y_col, axis=1)
y_test = test_set[y_col].copy()

scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

for algo in algos:
    print('--------------New Algo ', algo, '----------------------')

    nn_model = mlrose.NeuralNetwork(
        hidden_nodes=[3],
        activation='relu',
        algorithm=algo,
        max_iters=100,
        bias=True,
        is_classifier=True,
        learning_rate=0.0001,
        early_stopping=False,
        clip_max=1,
        restarts=30,
        schedule=mlrose.ExpDecay(1),
        pop_size=300,
        mutation_prob=0.4,
        max_attempts=10,
        random_state=4,
        curve=False
    )

    print('Fitting model...')
    nn_model.fit(x_train, y_train)

    y_train_pred = nn_model.predict(x_train)

    print('Training Score: ', accuracy_score(y_train, y_train_pred))

    y_test_pred = nn_model.predict(x_test)

    print('Testing Score: ', accuracy_score(y_test, y_test_pred))

    metrics = get_metrics(nn_model, x_train, y_train, x_test, y_test, y_test_pred, x_all, y_all)
    print('loss: ', nn_model.loss)
    metric_dict[algo] = metrics

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    title = f'Learning Curve using {algo}'
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=5)
    axes=axes[:, 0]
    estimator = nn_model
    ylim=(0.1, 1.01)
    train_sizes=np.linspace(.1, 1.0, 5)
    print('here')
    # plot_learning_curve(
    #     estimator,
    #     title,
    #     x_all,
    #     y_all,
    #     axes=axes[:, 0],
    #     ylim=(0.7, 1.01),
    #     cv=cv,
    #     n_jobs=4
    # )
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, x_all, y_all, cv=cv,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    print(train_scores_mean)
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    print(test_scores_mean)
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")
    plt.show()
    fig.savefig(f'figures/nn_{algo}_learning_curve')

