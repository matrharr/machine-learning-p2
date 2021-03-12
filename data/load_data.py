import pandas as pd


def load_brain_tumor_data():
    brain = pd.read_csv('data/brain-tumor.csv')
    # brain = brain.loc[:2000]
    brain.drop(['Image'], axis=1, inplace=True)
    return brain


def load_bankrupt_data():
    bankrupt = pd.read_csv('data/company-bankrupt.csv')
    return bankrupt