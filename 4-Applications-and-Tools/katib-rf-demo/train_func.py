"""
Katib objective function for Random Forest hyperparameter tuning.

Uses scikit-learn's bundled Breast Cancer Wisconsin dataset (569 rows,
30 numeric features, binary target). It ships inside the scikit-learn wheel,
so there is no download step and nothing to bake into the image.

CRITICAL CONSTRAINT
-------------------
Katib's `tune()` API serialises ONLY the source of this single function and
runs it inside the Trial container via `python -c "..."`.

That means:
  * every `import` must live INSIDE the function
  * every helper must be a nested function INSIDE this function
  * module-level globals (loggers, constants) are NOT shipped to the Trial
  * every hyperparameter arrives as a STRING and must be cast explicitly

The file is still importable normally (`from train_func import train`) so you
can unit-test it on your Mac before shipping it to the cluster.
"""


def train(parameters):
    # ------------------------------------------------------------------
    # 1. Imports (must be inside the function)
    # ------------------------------------------------------------------
    import os
    import json

    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    # ------------------------------------------------------------------
    # 2. Nested helpers (must be inside the function)
    # ------------------------------------------------------------------
    def as_bool(value, default=True):
        """Katib passes categoricals as strings: bool("false") is True, so
        never call bool() directly on a Katib parameter."""
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in ("1", "true", "t", "yes", "y")

    # ------------------------------------------------------------------
    # 3. Load data — bundled with scikit-learn, no network access needed
    # ------------------------------------------------------------------
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    print(f"dataset shape: {X.shape[0]} rows, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # 4. Cast hyperparameters (all arrive as strings)
    # ------------------------------------------------------------------
    hp = {
        "max_depth": int(parameters["max_depth"]),
        "min_samples_leaf": float(parameters["min_samples_leaf"]),
        "n_estimators": int(parameters["n_estimators"]),
        "criterion": str(parameters["criterion"]),
        "bootstrap": as_bool(parameters["bootstrap"]),
    }

    # Printed as JSON (colons, not "="), so the Katib stdout metrics collector
    # does not mistake these lines for metrics.
    print("trial hyperparameters: " + json.dumps(hp))

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    clf = RandomForestClassifier(
        max_depth=hp["max_depth"],
        min_samples_leaf=hp["min_samples_leaf"],
        n_estimators=hp["n_estimators"],
        criterion=hp["criterion"],
        bootstrap=hp["bootstrap"],
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 6. Report the objective metric
    #    Katib's stdout collector parses exactly this shape: <name>=<number>
    # ------------------------------------------------------------------
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    accuracy = clf.score(X_test, y_test)

    print(f"auc={auc:.6f}")
    print(f"accuracy={accuracy:.6f}")

    # ------------------------------------------------------------------
    # 7. Optional: save the fitted model
    #    NOTE: the Trial pod filesystem is ephemeral, so this artifact is gone
    #    once the pod is cleaned up. Mount a PVC or push to S3/MinIO to keep it.
    # ------------------------------------------------------------------
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        import joblib

        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "random_forest.pkl")
        joblib.dump(clf, model_path)
        print(f"model written to: {model_path}")
