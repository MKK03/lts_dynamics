from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from pyshred import (
    LSTM,
    GRU,
    DataManager,
    SHRED,
    SHREDEngine,
    SINDy_Forecaster,
)


@dataclass
class ExperimentConfig:
    name: str = ""

    # DataManager
    lags: int = 52
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1

    # Sensors
    n_sensors: int = 3
    sensor_seed: Optional[int] = None

    # Compression (SVD modes)
    compress: int = 20

    # Architecture
    forecaster_type: str = "shred"  # "shred" or "sindy"
    latent_dim: int = 64  # hidden_size of the sequence model
    decoder_model: str = "MLP"

    # Training
    num_epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    patience: int = 20

    # SINDy-specific (only used when forecaster_type == "sindy")
    sindy_regularization: float = 1.0
    sindy_poly_order: int = 1
    sindy_include_sine: bool = False
    sindy_dt: float = 1.0


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    train_mse: float
    val_mse: float
    test_mse: float
    val_errors: np.ndarray
    physical_errors: dict = field(default_factory=dict)
    manager: Optional[DataManager] = None
    shred: Optional[SHRED] = None
    engine: Optional[SHREDEngine] = None


def _build_shred(config: ExperimentConfig) -> SHRED:
    if config.forecaster_type == "sindy":
        sequence_model = GRU(hidden_size=config.latent_dim)
        latent_forecaster = SINDy_Forecaster(
            poly_order=config.sindy_poly_order,
            include_sine=config.sindy_include_sine,
            dt=config.sindy_dt,
        )
    elif config.forecaster_type == "shred":
        sequence_model = LSTM(hidden_size=config.latent_dim)
        latent_forecaster = "LSTM_Forecaster"
    else:
        raise ValueError(
            f"Unknown forecaster_type '{config.forecaster_type}'. "
            "Expected 'shred' or 'sindy'."
        )

    return SHRED(
        sequence_model=sequence_model,
        decoder_model=config.decoder_model,
        latent_forecaster=latent_forecaster,
    )


def _compute_physical_errors(
    data: np.ndarray,
    data_id: str,
    manager: DataManager,
    engine: SHREDEngine,
) -> dict[str, pd.DataFrame]:
    t_train = len(manager.train_sensor_measurements)
    t_val = len(manager.val_sensor_measurements)
    t_test = len(manager.test_sensor_measurements)

    train_Y = {data_id: data[:t_train]}
    val_Y = {data_id: data[t_train : t_train + t_val]}
    test_Y = {data_id: data[-t_test:]}

    train_error = engine.evaluate(manager.train_sensor_measurements, train_Y)
    val_error = engine.evaluate(manager.val_sensor_measurements, val_Y)
    test_error = engine.evaluate(manager.test_sensor_measurements, test_Y)

    return {"train": train_error, "val": val_error, "test": test_error}


def run_experiment(
    data: np.ndarray,
    data_id: str,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Run a single SHRED / SINDy-SHRED experiment.

    Parameters
    ----------
    data : np.ndarray
        Full-state data of shape ``(n_timesteps, n_spatial)``.
    data_id : str
        Identifier string for the dataset (passed to ``DataManager.add_data``).
    config : ExperimentConfig
        Hyperparameter configuration for this experiment.

    Returns
    -------
    ExperimentResult
    """
    manager = DataManager(
        lags=config.lags,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
    )

    add_data_kwargs: dict = dict(
        data=data,
        id=data_id,
        random=config.n_sensors,
        compress=config.compress,
    )
    if config.sensor_seed is not None:
        add_data_kwargs["seed"] = config.sensor_seed

    manager.add_data(**add_data_kwargs)

    train_dataset, val_dataset, test_dataset = manager.prepare()

    shred = _build_shred(config)

    sindy_reg = (
        config.sindy_regularization
        if config.forecaster_type == "sindy"
        else 0
    )

    val_errors = shred.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        patience=config.patience,
        sindy_regularization=sindy_reg,
    )

    train_mse = shred.evaluate(dataset=train_dataset)
    val_mse = shred.evaluate(dataset=val_dataset)
    test_mse = shred.evaluate(dataset=test_dataset)

    engine = SHREDEngine(manager, shred)

    physical_errors = _compute_physical_errors(data, data_id, manager, engine)

    return ExperimentResult(
        config=config,
        train_mse=float(train_mse),
        val_mse=float(val_mse),
        test_mse=float(test_mse),
        val_errors=val_errors,
        physical_errors=physical_errors,
        manager=manager,
        shred=shred,
        engine=engine,
    )


def run_experiments(
    data: np.ndarray,
    data_id: str,
    configs: list[ExperimentConfig],
) -> list[ExperimentResult]:
    """Run a batch of experiments and return all results.

    Parameters
    ----------
    data : np.ndarray
        Full-state data of shape ``(n_timesteps, n_spatial)``.
    data_id : str
        Identifier string for the dataset.
    configs : list[ExperimentConfig]
        List of experiment configurations to run.

    Returns
    -------
    list[ExperimentResult]
    """
    results: list[ExperimentResult] = []
    n = len(configs)

    for i, cfg in enumerate(configs, start=1):
        label = cfg.name or f"experiment-{i}"
        print(f"\n{'=' * 60}")
        print(f"[{i}/{n}] Running: {label}")
        print(f"  forecaster={cfg.forecaster_type}  lags={cfg.lags}  "
              f"latent_dim={cfg.latent_dim}  sensors={cfg.n_sensors}  "
              f"epochs={cfg.num_epochs}")
        print(f"{'=' * 60}")

        result = run_experiment(data, data_id, cfg)
        results.append(result)

        print(f"  -> Train MSE: {result.train_mse:.6f}  "
              f"Val MSE: {result.val_mse:.6f}  "
              f"Test MSE: {result.test_mse:.6f}")

    return results


def summarize_results(results: list[ExperimentResult]) -> pd.DataFrame:
    """Build a summary DataFrame from a list of experiment results.

    Columns include experiment config parameters, latent-space MSE
    (train/val/test), and physical-space metrics for the test set.
    """
    rows = []
    for r in results:
        cfg = r.config
        row: dict = {
            "name": cfg.name,
            "forecaster_type": cfg.forecaster_type,
            "lags": cfg.lags,
            "latent_dim": cfg.latent_dim,
            "n_sensors": cfg.n_sensors,
            "compress": cfg.compress,
            "num_epochs": cfg.num_epochs,
            "decoder_model": cfg.decoder_model,
            "train_mse": r.train_mse,
            "val_mse": r.val_mse,
            "test_mse": r.test_mse,
        }

        test_phys = r.physical_errors.get("test")
        if test_phys is not None and not test_phys.empty:
            first = test_phys.iloc[0]
            row["test_phys_MSE"] = first.get("MSE")
            row["test_phys_RMSE"] = first.get("RMSE")
            row["test_phys_MAE"] = first.get("MAE")
            row["test_phys_R2"] = first.get("R2")

        rows.append(row)

    return pd.DataFrame(rows)
