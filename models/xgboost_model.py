import xgboost as xgb

class XGBoostModel:
    def __init__(self, task="binary", **kwargs):
        """
        task: "binary" or "regression"
        kwargs: XGBoost params (tree_method, device, etc.)
        """
        self.task = task
        base_params = {
            "tree_method": "hist",
            "device": "cuda",
            "predictor": "gpu_predictor",
        }
        base_params.update(kwargs)

        if task == "binary":
            self.model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",  # or "aucpr"
                use_label_encoder=False,
                **base_params
            )
        elif task == "regression":
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                **base_params
            )
        else:
            raise ValueError("Unsupported task. Use 'binary' or 'regression'.")

    def fit(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50):
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.task == "binary":
            return self.model.predict_proba(X)
        raise RuntimeError("predict_proba is only available for binary classification.")

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)
