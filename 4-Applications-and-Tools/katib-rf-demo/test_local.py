"""Smoke test the objective function on your Mac before touching Kubernetes.

    python test_local.py

Everything Katib passes arrives as a string, so this test passes strings too.
"""

from train_func import train

if __name__ == "__main__":
    train(
        {
            "max_depth": "8",
            "min_samples_leaf": "0.05",
            "n_estimators": "100",
            "criterion": "gini",
            "bootstrap": "true",
        }
    )
