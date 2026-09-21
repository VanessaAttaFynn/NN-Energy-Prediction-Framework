from sklearn.ensemble import RandomForestRegressor


def build_model(random_state=42, **kwargs):
    defaults = dict(n_estimators=300, n_jobs=-1, random_state=random_state)
    defaults.update(kwargs)
    return RandomForestRegressor(**defaults)
