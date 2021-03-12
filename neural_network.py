import mlrose_hiive as mlrose
import numpy as np

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from data.load_data import load_bankrupt_data, load_brain_tumor_data
from metrics import get_metrics

algos = [
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
    test_size=0.3
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
        clip_max=3,
        restarts=0,
        schedule=mlrose.GeomDecay(),
        pop_size=200,
        mutation_prob=0.1,
        max_attempts=10,
        random_state=23,
        curve=False
    )

    print('Fitting model...')
    nn_model.fit(x_train, y_train)

    y_train_pred = nn_model.predict(x_train)

    print('Training Score: ', accuracy_score(y_train, y_train_pred))

    y_test_pred = nn_model.predict(x_test)

    print('Testing Score: ', accuracy_score(y_test, y_test_pred))

    metrics = get_metrics(nn_model, x_train, y_train, x_test, y_test, y_test_pred, x_all, y_all)
    metric_dict[algo] = metrics

