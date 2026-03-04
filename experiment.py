import numpy as np
from shred_experiments import ExperimentConfig, run_experiments, summarize_results

# data should be shape (n_timesteps, n_spatial)
# e.g. from the KS simulation notebook: data = u.T
data = np.load("1dks.npy")

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