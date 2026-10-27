"""Kedro project settings.

All configuration values defined here will override the defaults provided by
the Kedro framework core.
"""

from gym_exercises.hooks import MlflowHook

# Seguimiento de experimentos con MLflow (README §8.8).
HOOKS = (MlflowHook(),)

# conf/<env>/mlflow.yml: configuración del seguimiento.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {"mlflow": ["mlflow*", "mlflow*/**"]},
}
