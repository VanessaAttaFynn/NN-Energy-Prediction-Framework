from xgboost import XGBRegressor


def build_model(random_state=42, **kwargs):
    defaults = dict(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=random_state)
    defaults.update(kwargs)
    return XGBRegressor(**defaults)
