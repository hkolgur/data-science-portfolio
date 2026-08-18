"""
Submit the Random Forest hyperparameter tuning Experiment to a local Katib.

Run this on your Mac (NOT inside a container). It talks to the cluster through
your current kubectl context, creates an Experiment CR, waits for it to finish
and prints the winning hyperparameters.

    python submit_experiment.py
"""

import uuid

import kubeflow.katib as katib

from train_func import train

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The katib-standalone install already labels the `kubeflow` namespace for
# metrics-collector injection, so running Trials there needs zero extra setup.
# To use `default` instead, first run:
#   kubectl label namespace default katib.kubeflow.org/metrics-collector-injection=enabled
NAMESPACE = "kubeflow"

# NOTE the tag is v1, NOT latest. Kubernetes defaults imagePullPolicy to
# Always for `:latest`, which makes minikube try to pull from a remote
# registry and fail with ErrImagePull. Any non-latest tag defaults to
# IfNotPresent and uses the locally built image.
BASE_IMAGE = "katib-random-forest:v1"


def main():
    # Unique name per run so repeat demos never collide.
    experiment_name = f"tune-rf-{uuid.uuid4().hex[:8]}"

    # -----------------------------------------------------------------------
    # Search space
    # -----------------------------------------------------------------------
    parameters = {
        # Tree depth. Sensible RF range is single/low-double digits — the
        # original notes searched 100-1000, which just builds identical fully
        # grown trees and wastes every trial.
        "max_depth": katib.search.int(min=2, max=20),
        # Fraction of samples required at a leaf. Pushed toward 0.3 the trees
        # become near-stumps and AUC drops — that visible spread is what makes
        # the parallel-coordinates plot in the Katib UI worth looking at.
        "min_samples_leaf": katib.search.double(min=0.01, max=0.3),
        "n_estimators": katib.search.int(min=20, max=200),
        "criterion": katib.search.categorical(list=["gini", "entropy"]),
        # Categoricals are delivered as strings — train_func casts them safely.
        "bootstrap": katib.search.categorical(list=["true", "false"]),
    }

    client = katib.KatibClient(namespace=NAMESPACE)

    print(f"Submitting Katib Experiment: {experiment_name}")
    client.tune(
        name=experiment_name,
        namespace=NAMESPACE,
        # REQUIRED: without objective=, Katib has no code to run.
        objective=train,
        parameters=parameters,
        base_image=BASE_IMAGE,
        objective_metric_name="auc",
        additional_metric_names=["accuracy"],
        objective_type="maximize",
        algorithm_name="random",
        max_trial_count=12,
        parallel_trial_count=3,
        max_failed_trial_count=3,
        # Keep this modest: 3 parallel trials x 1 CPU must fit inside minikube.
        resources_per_trial={"cpu": "1", "memory": "1Gi"},
        # Keep completed Trial pods around so `kubectl logs` still works.
        # If your SDK version rejects this kwarg, just delete the line.
        retain_trials=True,
    )

    print("Experiment created. Watch it with:")
    print(f"  kubectl get trials -n {NAMESPACE} -l katib.kubeflow.org/experiment={experiment_name} -w")

    client.wait_for_experiment_condition(
        name=experiment_name,
        namespace=NAMESPACE,
        timeout=3600,
    )

    best = client.get_optimal_hyperparameters(
        name=experiment_name,
        namespace=NAMESPACE,
    )
    print("\n=== Best trial ===")
    print(best)


if __name__ == "__main__":
    main()
