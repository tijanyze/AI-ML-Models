from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}



grid = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,            # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1        # use all CPU cores
)


grid.fit(X_train, y_train)


print("Best parameters:", grid.best_params_)


best_svm = grid.best_estimator_

y_pred = best_svm.predict(X_test)

print(classification_report(y_test, y_pred, target_names=iris.target_names))


