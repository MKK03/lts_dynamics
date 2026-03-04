# lts_dynamics

Experiment framework for SHRED and SINDy-SHRED reconstruction/forecasting on dynamical systems data (e.g. 1D Kuramoto-Sivashinsky).

## Setup

```bash
pip install -r requirements.txt
```

## Running experiments

The `shred_experiments` module provides a configurable wrapper for running batches of SHRED / SINDy-SHRED experiments with varying hyperparameters.

### Quick start

```python
import numpy as np
from shred_experiments import ExperimentConfig, run_experiments, summarize_results

# data should be shape (n_timesteps, n_spatial)
# e.g. from the KS simulation notebook: data = u.T
data = np.load("my_data.npy")

configs = [
    # SHRED with LSTM forecaster — vary lags and latent dim
    ExperimentConfig(
        name="shred-lag52-ld64",
        lags=52,
        latent_dim=64,
        forecaster_type="shred",
    ),
    ExperimentConfig(
        name="shred-lag100-ld64",
        lags=100,
        latent_dim=64,
        forecaster_type="shred",
    ),
    ExperimentConfig(
        name="shred-lag52-ld32",
        lags=52,
        latent_dim=32,
        forecaster_type="shred",
    ),

    # SINDy-SHRED with GRU + SINDy forecaster
    ExperimentConfig(
        name="sindy-lag52-ld3",
        lags=52,
        latent_dim=3,
        forecaster_type="sindy",
        sindy_include_sine=True,
        sindy_dt=0.05,
    ),
    ExperimentConfig(
        name="sindy-lag100-ld3",
        lags=100,
        latent_dim=3,
        forecaster_type="sindy",
        sindy_include_sine=True,
        sindy_dt=0.05,
    ),
]

results = run_experiments(data, "1dks", configs)
summary = summarize_results(results)
print(summary)
```

### Configuration reference

`ExperimentConfig` fields:

| Field | Default | Description |
|---|---|---|
| `name` | `""` | Label for the experiment |
| `lags` | `52` | Time-lag window size |
| `latent_dim` | `64` | Hidden size of the sequence model (latent space dimensionality) |
| `forecaster_type` | `"shred"` | `"shred"` (LSTM forecaster) or `"sindy"` (SINDy forecaster) |
| `n_sensors` | `3` | Number of random sensors |
| `compress` | `20` | Number of SVD modes for compression |
| `decoder_model` | `"MLP"` | Decoder architecture |
| `num_epochs` | `200` | Maximum training epochs |
| `batch_size` | `64` | Training batch size |
| `lr` | `1e-3` | Learning rate |
| `patience` | `20` | Early stopping patience |
| `train_size` | `0.8` | Training split fraction |
| `val_size` | `0.1` | Validation split fraction |
| `test_size` | `0.1` | Test split fraction |
| `sensor_seed` | `None` | Random seed for sensor placement |
| `sindy_regularization` | `1.0` | SINDy regularization weight (sindy mode only) |
| `sindy_poly_order` | `1` | SINDy polynomial order (sindy mode only) |
| `sindy_include_sine` | `False` | Include sine in SINDy library (sindy mode only) |
| `sindy_dt` | `1.0` | SINDy time step (sindy mode only) |

### Working with results

Each `ExperimentResult` contains:

- `train_mse`, `val_mse`, `test_mse` — latent-space MSE from `shred.evaluate()`
- `val_errors` — per-epoch validation errors (numpy array)
- `physical_errors` — dict with `"train"`, `"val"`, `"test"` DataFrames containing physical-space MSE, RMSE, MAE, and R2
- `manager`, `shred`, `engine` — the fitted pyshred objects for further analysis (decoding, forecasting, etc.)

```python
# Access individual result
result = results[0]

# Physical-space test error
print(result.physical_errors["test"])

# Use the engine for custom decoding/forecasting
engine = result.engine
manager = result.manager
test_latent = engine.sensor_to_latent(manager.test_sensor_measurements)
test_reconstruction = engine.decode(test_latent)
```

`summarize_results()` returns a pandas DataFrame with one row per experiment for side-by-side comparison.
