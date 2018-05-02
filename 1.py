# Random Forest Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

names = ['game1', 'game2', 'game3', 'game4', 'game5', 'game6', 'game7', 'game8', 'game9']
dataframe = pandas
array = pandas.read_csv('Train.csv', names = names)
X = array[:,0:9]
Y = array[:,9]
seed = 8
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())