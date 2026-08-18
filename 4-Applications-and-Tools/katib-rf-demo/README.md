# Katib on a Local Mac — Random Forest Hyperparameter Tuning Demo

End-to-end notes for running a Kubeflow Katib hyperparameter tuning experiment
on a single MacBook (Intel or Apple Silicon), from zero to a populated Katib UI.

**Time to first result:** ~30–40 min (most of it is image pulls)
**Verified against:** Katib `v0.17.0` standalone, `kubeflow-katib==0.17.0` SDK, minikube + Docker driver, macOS + Python 3.13

---

## Quickstart

If the cluster is already up and Katib is installed, this is the whole loop:

```bash
cd katib-rf-demo
source .venv/bin/activate                 # prompt must show (.venv)

eval $(minikube docker-env)               # this shell only
docker build -t katib-random-forest:v1 .
python submit_experiment.py
```

Then watch it: `kubectl get trials -n kubeflow -w`

Starting from nothing (or from a fresh `git clone`), work through §3 → §12 in
order. Full setup is one-time; only the three commands above repeat.

---

## 0. What was wrong with the original notes

Read this section before copying anything from the old file — the original
script does not run as written. Six blocking bugs, in the order they'd bite you:

| # | Issue | Effect | Fix applied here |
|---|-------|--------|------------------|
| 1 | `tune()` was called without `objective=` | Katib gets no code. The `train()` function was defined and then never referenced. | Pass `objective=train` |
| 2 | Imports + `logger` at module level | `tune()` ships only the *body* of the objective function to the Trial pod. `NameError: pd is not defined` on every trial. | All imports and helpers moved **inside** `train()` |
| 3 | `bool(parameters.get("bootstrap"))` | Katib delivers `"false"` as a **string**, and `bool("false")` is `True`. `bootstrap=False` would never be tested. | Explicit `as_bool()` string parser |
| 4 | `max_depth` searched over `100–1000` | Every tree is fully grown; all 15 trials are effectively identical. Search space wasted. | `max_depth` → `3–20` |
| 5 | Image tagged `:latest` + `localhost:5001` | K8s defaults `imagePullPolicy: Always` for `:latest` → `ErrImagePull`, because minikube's registry addon does not serve `localhost:5001` from the host. | Tag `:v1`, build straight into minikube's Docker daemon, no registry needed |
| 6 | Dataset URL `jbrownlee/Datasets/prostate.csv` | That file does not exist there → the `docker build` step fails. | Dropped entirely — we use `sklearn.datasets.load_breast_cancer()`, which ships inside the scikit-learn wheel |

Two more that cost hours if you hit them:

- **Metrics collector injection label.** Katib's stdout collector is injected by a
  webhook that only fires in labelled namespaces. Trials in an unlabelled
  namespace run fine and report **zero metrics**, so the experiment hangs. See §5.
- **Training Operator is not required.** The `tune()` API creates a plain
  `batch/v1` Job. You only need the Training Operator for distributed
  (PyTorchJob/TFJob) trials. Skipping it saves memory on a laptop.

---

## 1. Feasibility on a laptop

| Goal | Local Mac? | Notes |
|------|-----------|-------|
| Katib control plane + Experiments | ✅ | ~1.5 GB RAM, comfortable |
| Katib UI dashboard | ✅ | via `kubectl port-forward` |
| Parallel trials in containers | ✅ | Keep `parallel_trial_count` ≤ cores − 1 |
| Custom training image | ✅ | Built directly into minikube's Docker daemon |
| **Full Kubeflow Platform** (Pipelines, Notebooks, Central Dashboard, Dex) | ❌ | Needs ~16 vCPU / 32 GB. It will thrash a laptop. |
| Multi-user auth (Dex/OIDC) | ❌ | Not meaningful in a single-user local cluster |

**Apple Silicon (M1–M4):** this stack works natively. Katib publishes multi-arch
images, and `python:3.11-slim` resolves to arm64. Do **not** force
`--platform linux/amd64` unless something actually fails — QEMU emulation makes
scikit-learn trials several times slower. If a control-plane pod crash-loops,
run `kubectl describe pod` and check for `exec format error`, which is the one
symptom that genuinely indicates a missing arm64 image.

---

## 2. Repository layout

Split into separate files — the objective function must be importable on its
own so you can smoke-test it locally before it ever reaches the cluster.

```
katib-rf-demo/
├── train_func.py           # objective function (runs INSIDE each trial pod)
├── submit_experiment.py    # client-side: builds search space, submits, waits
├── test_local.py           # runs train_func on your Mac, no Kubernetes
├── requirements.txt        # deps baked into the trial IMAGE
├── requirements-dev.txt    # deps for your Mac venv (SDK + sklearn)
├── Dockerfile
├── .dockerignore
└── .gitignore
```

`.venv/` is created locally in §6 and is gitignored — never commit it.

No `data/` directory: the demo uses **Breast Cancer Wisconsin**, bundled with
scikit-learn (569 rows, 30 numeric features, binary target). It loads from the
installed wheel, so trial pods need no volume, no download, and no network.

Why this split:

- `train_func.py` is shipped **by value** (its source is serialised into the
  Trial spec), so it must be self-contained. Keeping it alone in a file makes
  that constraint obvious and testable.
- `submit_experiment.py` runs on your Mac against your kubectl context. It
  never runs in a container.
- The **image** and your **venv** need different dependency sets — the image
  doesn't need the Katib SDK, your Mac doesn't need to be reproducible.

---

## 3. Install the toolchain

```bash
# Docker Desktop — launch it once and let it finish starting before continuing
brew install --cask docker

# Cluster tooling
brew install minikube kubernetes-cli

# Sanity check
docker version
minikube version
kubectl version --client
```

In **Docker Desktop → Settings → Resources**, give the VM at least **6 CPUs and
10 GB RAM**. minikube runs *inside* Docker Desktop, so minikube can never have
more than Docker Desktop does — this is the #1 cause of "my cluster won't start".

---

## 4. Start the cluster

```bash
minikube start --cpus=4 --memory=8192 --driver=docker

# Katib's MySQL needs dynamic volume provisioning — enabled by default on
# minikube, but verify you see a StorageClass marked (default):
kubectl get storageclass
```

Expected: `standard (default)  k8s.io/minikube-hostpath`.

---

## 5. Install Katib (standalone)

Full Kubeflow is off the table on a laptop; standalone Katib is the supported
subset and is exactly what the demo needs.

```bash
kubectl apply -k "github.com/kubeflow/katib.git/manifests/v1beta1/installs/katib-standalone?ref=v0.17.0"
```

Wait for all four control-plane pods:

```bash
kubectl get pods -n kubeflow -w
```

```
katib-controller-xxxxxxxxxx-xxxxx   1/1   Running
katib-db-manager-xxxxxxxxx-xxxxx    1/1   Running
katib-mysql-xxxxxxxxx-xxxxx         1/1   Running
katib-ui-xxxxxxxxx-xxxxx            1/1   Running
```

First pull takes several minutes. If `katib-mysql` sits in `Pending`, its PVC
didn't bind — check `kubectl get pvc -n kubeflow`.

### Namespace labelling (the silent failure mode)

Katib injects its metrics-collector sidecar only into namespaces carrying this
label. The standalone install applies it to `kubeflow` for you — **verify it**,
because without it trials run to completion and report no metrics at all:

```bash
kubectl get ns kubeflow --show-labels | grep metrics-collector-injection
```

If it's missing, or if you'd rather run trials in `default`:

```bash
kubectl label namespace default katib.kubeflow.org/metrics-collector-injection=enabled
```

> These notes run Trials in the `kubeflow` namespace (zero extra setup). To use
> `default`, apply the label above and change `NAMESPACE` in `submit_experiment.py`.

---

## 6. Python environment

Use a dedicated venv. The Katib SDK pins an older `kubernetes` client and will
happily wreck a shared environment.

```bash
cd katib-rf-demo

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-dev.txt
```

Your prompt must now start with `(.venv)`. If it doesn't, nothing below will work.

### Three things macOS does that trip people up

1. **There is no bare `python` command outside a venv.** `python --version`
   returns `command not found`; use `python3` until the venv is activated.
   Inside an activated venv, `python` and `pip` both work and point at `.venv/`.
2. **`python3.11` almost certainly doesn't exist either.** Use plain `python3`
   (system Python 3.13 is fine — see below). Install 3.11 only if §14 tells you to.
3. **Activation is per-shell.** A new terminal tab, a VS Code task, or a command
   Claude Code runs in a fresh terminal will *not* be activated. Either re-run
   `source .venv/bin/activate`, or call the interpreter by path — this always
   works regardless of activation state:

   ```bash
   .venv/bin/python test_local.py
   ```

### The client Python version does not need to match the image

The container is `python:3.11-slim`, but your venv can be 3.13. The objective
function is shipped to the Trial as **source text** and executed by the image's
interpreter — your local Python only builds the Experiment spec and talks to the
API server. Python 3.13 is verified working here: `scikit-learn==1.5.2` and
`pandas==2.2.3` both publish cp313 macOS wheels.

### Verify the SDK imports

The ML libraries are fine on 3.13; the Katib SDK is the one component that
predates it, so check it explicitly before going further:

```bash
python -c "import kubeflow.katib as katib; print(katib.KatibClient)"
```

Expected: `<class 'kubeflow.katib.api.katib_client.KatibClient'>`

Keep the SDK version aligned with the control plane (`0.17.0` ↔ `v0.17.0`).

---

## 7. The dataset (nothing to do)

`train_func.py` calls:

```python
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
```

569 rows, 30 numeric features, binary target (malignant / benign). It is packaged
inside the scikit-learn wheel, so it is present the moment `pip install` finishes
— no `curl`, no CSV, no volume mount, no network access from the trial pods.

**One caveat for the demo:** this dataset is easy. Most trials land between
AUC 0.97 and 0.99, and only genuinely bad configurations (shallow trees plus a
large `min_samples_leaf`) drop below that. The search space in
`submit_experiment.py` is deliberately widened to `min_samples_leaf` up to `0.3`
so the Katib UI plot shows real separation rather than a flat line. Other
one-line swaps if you want a different shape:

| Swap in | Why |
|---|---|
| `load_wine(return_X_y=True, as_frame=True)` | 3-class; also change the metric to `roc_auc_score(y_test, clf.predict_proba(X_test), multi_class="ovr")` |
| `load_digits(...)` | 1797 rows, 64 features — slower trials, better for testing parallelism |
| `make_classification(n_samples=5000, n_informative=8, flip_y=0.25, random_state=42)` | Injects label noise, so hyperparameters matter a lot more and the spread is dramatic |

---

## 8. Smoke-test before Kubernetes

Ten seconds here saves a debugging cycle where every failure is buried in a pod log.

```bash
source .venv/bin/activate      # skip if your prompt already shows (.venv)
python test_local.py
```

```
dataset shape: 569 rows, 30 features
trial hyperparameters: {"max_depth": 8, "min_samples_leaf": 0.05, "n_estimators": 100, "criterion": "gini", "bootstrap": true}
auc=0.990410
accuracy=0.938596
```

That `auc=<number>` line is exactly the format Katib's stdout collector parses
(`<metric-name>=<value>`). If it doesn't appear, no experiment will ever
produce a result.

---

## 9. Build the image into minikube

No registry required. `minikube docker-env` points your shell at the Docker
daemon *inside* the minikube node, so the built image is already where the
kubelet looks for it.

```bash
# Applies to THIS shell only — rebuild in the same shell you eval'd in
eval $(minikube docker-env)

docker build -t katib-random-forest:v1 .

# Confirm the image exists inside the node
docker images | grep katib-random-forest
```

**Tag `:v1`, never `:latest`.** Kubernetes defaults `imagePullPolicy` to
`Always` for `:latest`, so it would ignore your local build and try to pull
from Docker Hub. Any other tag defaults to `IfNotPresent`.

Rebuild after editing `train_func.py`? Only if you changed dependencies — the
function source is re-serialised into the Trial spec on every submit, so logic
edits take effect without a rebuild.

To return your shell to the host Docker daemon: `eval $(minikube docker-env -u)`.

---

## 10. Run the experiment

```bash
python submit_experiment.py
```

```
Submitting Katib Experiment: tune-rf-3f9a2c11
Experiment created. Watch it with:
  kubectl get trials -n kubeflow -l katib.kubeflow.org/experiment=tune-rf-3f9a2c11 -w
```

12 trials, 3 at a time, ~1–2 min per trial on this dataset. The script blocks
until the experiment succeeds, then prints the winning hyperparameters:

```
=== Best trial ===
{'best_trial_name': 'tune-rf-3f9a2c11-abc12345',
 'observation': {'metrics': [{'name': 'auc', 'latest': '0.8412', 'max': '0.8412', 'min': '0.8412'}]},
 'parameter_assignments': [{'name': 'max_depth', 'value': '11'}, ...]}
```

---

## 11. Monitoring cheat sheet

```bash
NS=kubeflow

# Experiment status (Running / Succeeded / Failed)
kubectl get experiments -n $NS

# Individual trials and their objective values
kubectl get trials -n $NS

# Underlying jobs and pods
kubectl get jobs,pods -n $NS

# Training output for one trial
kubectl logs <trial-pod-name> -n $NS -c training-container

# Metrics collector sidecar — check here when trials succeed but show no metric
kubectl logs <trial-pod-name> -n $NS -c metrics-logger-and-collector

# Why is a trial stuck?
kubectl describe pod <trial-pod-name> -n $NS

# Full experiment status including conditions
kubectl get experiment <experiment-name> -n $NS -o yaml
```

> The primary container is named `training-container`, not `metrics-logger-and-collector`
> — the original notes had these swapped, which is why the log command returned nothing useful.

---

## 12. Katib UI

```bash
kubectl port-forward svc/katib-ui -n kubeflow 8080:80
```

Open **http://localhost:8080/katib/**

Navigate: sidebar → **Experiments** (older builds: **HP → Monitor**) → select
your `tune-rf-...` run. You get the trial table plus the parallel-coordinates
plot showing which hyperparameter regions produced high AUC.

Leave the port-forward running in its own terminal tab; it dies with the shell.

---

## 13. Cleanup

```bash
# One experiment (also deletes its trials, jobs, pods)
kubectl delete experiment <experiment-name> -n kubeflow

# All of them
kubectl delete experiments --all -n kubeflow

# Free the laptop
minikube stop

# Start over from scratch
minikube delete
```

---

## 14. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ErrImagePull` / `ImagePullBackOff` | Image built on host Docker, not in minikube; or tag is `:latest` | `eval $(minikube docker-env)` then rebuild with tag `:v1` |
| Trials `Succeeded` but experiment never completes, metrics empty | Namespace missing the injection label | `kubectl label namespace <ns> katib.kubeflow.org/metrics-collector-injection=enabled`, then resubmit |
| `NameError: name 'pd' is not defined` in trial logs | An import crept back to module level | Every import must be inside `train()` |
| `bootstrap` always `True` in results | `bool("false")` → `True` | Use the `as_bool()` helper |
| `TypeError: tune() got an unexpected keyword argument` | SDK/control-plane version drift | `python -c "import kubeflow.katib as k; help(k.KatibClient.tune)"` and drop the unsupported kwarg |
| Trials `Pending` forever | Not enough allocatable CPU | Lower `parallel_trial_count`, or restart minikube with more `--cpus` |
| `exec format error` | Image built for the wrong architecture | Rebuild natively; don't pin `--platform` on Apple Silicon |
| `katib-mysql` `Pending` | PVC unbound | `kubectl get pvc -n kubeflow`; confirm a default StorageClass exists |
| Connection refused from the SDK | Wrong kubectl context | `kubectl config current-context` should be `minikube` |
| Trial pods vanish before you can read logs | `retain_trials` not set | Keep `retain_trials=True` in `tune()` |
| `zsh: command not found: python` | No bare `python` on macOS, or the venv isn't active | Check the prompt for `(.venv)`; run `source .venv/bin/activate`, or use `.venv/bin/python` |
| `zsh: command not found: python3.11` | Only system Python is installed | Use `python3`. Install 3.11 only if the SDK import below fails |
| `ls: .venv: No such file or directory` | The venv was never created (the `python3.11` command failed first) | `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt` |
| SDK import fails on Python 3.13 | `kubeflow-katib==0.17.0` predates 3.13 | `brew install python@3.11`, then `rm -rf .venv && /opt/homebrew/bin/python3.11 -m venv .venv` and reinstall |
| `git add .` sweeps in thousands of files | `.gitignore` missing or added after `.venv` was tracked | Ensure `.gitignore` lists `.venv/`; if already tracked, `git rm -r --cached .venv` |

---

## 15. Re-running and day-2 operations

**Same code, new experiment** — each submit generates a fresh `tune-rf-<uuid>`
name, so just run it again. No rebuild needed: the objective function source is
re-serialised into the Trial spec on every submit.

```bash
source .venv/bin/activate
python submit_experiment.py
```

**After editing `requirements.txt`** — rebuild the image:

```bash
eval $(minikube docker-env) && docker build -t katib-random-forest:v1 .
```

Chain those two with `&&` in one command. `minikube docker-env` applies only to
the shell that evaluated it, so running the build in a separate terminal silently
builds to your host Docker daemon and the trials fail with `ErrImagePull`.

**After a reboot** — minikube stops with the machine:

```bash
minikube start          # same flags are remembered
kubectl get pods -n kubeflow
```

Images built into the node survive a `minikube stop`, but **not** a
`minikube delete`. After a delete, redo §5 and §9.

**On a fresh clone of this repo** — everything except `.venv/` is committed, so:
§3 (toolchain) → §4 (cluster) → §5 (Katib) → §6 (venv) → §9 (build) → §10 (run).

---

## 16. Extending the demo

- **Smarter search:** `algorithm_name="bayesianoptimization"` or `"tpe"` — the
  same 12 trials find better optima than `"random"`.
- **Early stopping:** pass `early_stopping_algorithm_name="medianstop"` to kill
  hopeless trials early.
- **Persist models:** the Trial filesystem is ephemeral. Mount a PVC or push to
  MinIO/S3; set `MODEL_DIR` in `env_per_trial` to activate the save path in
  `train_func.py`.
- **Your own data:** swap the `load_breast_cancer()` call in `train_func.py` for
  another sklearn loader (table in §7), or read a CSV. If you switch to a CSV,
  bake it into the image with a `COPY` line — trial pods have no access to your
  Mac's filesystem.
- **Multiple metrics:** anything printed as `name=value` and listed in
  `additional_metric_names` shows up in the UI — `accuracy` is already wired up.

---

## Appendix: the two rules that explain most Katib failures

1. **The objective function travels as source text, not as a Python object.**
   Only the function body reaches the Trial pod. Imports, helpers, and constants
   must all live inside it. Anything at module scope is invisible there.

2. **Every hyperparameter arrives as a string.** `int()`, `float()`, and a real
   boolean parser are mandatory. This is the single most common source of
   "the experiment ran but the results are meaningless".
