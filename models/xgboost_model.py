import xgboost as xgb

class XGBoostModel:
    def __init__(self, task="classification", **kwargs):
        self.task = task
        if task == "classification":
            self.model = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=5,
                eval_metric="mlogloss",
                use_label_encoder=False,
                **kwargs
            )
        elif task == "regression":
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                **kwargs
            )
        else:
            raise ValueError("Unsupported task type. Choose 'classification' or 'regression'.")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        if self.task == "classification":
            return self.model.predict(X)
        else:
            return self.model.predict(X)

    def predict_proba(self, X):
        if self.task == "classification":
            return self.model.predict_proba(X)
        else:
            raise RuntimeError("predict_proba only available for classification.")

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)

