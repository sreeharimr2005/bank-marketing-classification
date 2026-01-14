from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class Models:
    def __init__(self, cat_vars = None, numeric_vars = None, pipe = None, param_grid = None, n_folds = 5):

        if pipe is None:
            preprocessor = ColumnTransformer([
                ("transform_categorical", OneHotEncoder(handle_unknown = "ignore"), cat_vars),
                ("imputer", SimpleImputer(strategy="median"), numeric_vars)
            ])

            self.pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", DecisionTreeClassifier())
            ])
        else:
            self.pipe = pipe

        if param_grid is None:
            self.param_grid = [
                {
                    "model": [DecisionTreeClassifier()],
                    "model__criterion": ["gini", "entropy"]
                },
                {
                    "model": [RandomForestClassifier(random_state=42, class_weight="balanced")],
                    "model__n_estimators": [100, 500],
                    "model__criterion": ["gini", "entropy"],
                    "model__bootstrap": [True, False]
                }, {
                    "model": [LogisticRegression(solver="liblinear", class_weight="balanced", l1_ratio=1)],
                    "model__max_iter": [100, 200, 500]
                }
            ]
        else:
            self.param_grid = param_grid

        self.gs = GridSearchCV(self.pipe, self.param_grid, cv = n_folds)

    def fit_models(self, X_train, y_train):
        return self.gs.fit(X_train, y_train)

    def get_best_model(self):
        return self.gs.best_estimator_