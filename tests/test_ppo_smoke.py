"""PPO smoke test for the modified FinRL PortfolioOptimizationEnv.

Run from Regime_RL_Portfolio root:
    python tests/test_ppo_smoke.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv

SEED = 42
INITIAL = 100_000.0
FEE = 0.001
N_DAYS = 260
TRAIN_STEPS = 2048
TICS = ("SPY", "IEF", "GLD")
OUTPUT_DIR = Path("outputs") / "ppo_smoke"


def make_data():
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2025-01-02", periods=N_DAYS)
    params = {"SPY": (0.0003, 0.010), "IEF": (0.0001, 0.004), "GLD": (0.0002, 0.008)}
    rows = []
    for tic in TICS:
        drift, volatility = params[tic]
        log_returns = rng.normal(drift, volatility, size=N_DAYS - 1)
        closes = np.r_[100.0, 100.0 * np.exp(np.cumsum(log_returns))]
        for date, close in zip(dates, closes):
            rows.append({"date": date, "tic": tic, "close": float(close),
                         "high": float(close * 1.002), "low": float(close * 0.998)})
    return pd.DataFrame(rows)


def make_env(df, output_dir):
    return PortfolioOptimizationEnv(
        df=df.copy(), initial_amount=INITIAL,
        features=["close", "high", "low"], valuation_feature="close",
        time_column="date", tic_column="tic", time_window=3,
        normalize_df="by_previous_time", comission_fee_model="wvm",
        comission_fee_pct=FEE, return_last_action=False,
        new_gym_api=True, cwd=str(output_dir),
    )


def main():
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = make_data()
    assert len(df) == N_DAYS * len(TICS)

    print("\n[1] Gymnasium environment validation")
    check_env(make_env(df, OUTPUT_DIR / "check"), warn=True)
    print("PASS: check_env")

    print("\n[2] PPO training")
    vec_env = DummyVecEnv([lambda: make_env(df, OUTPUT_DIR / "train")])
    model = PPO("MlpPolicy", vec_env, seed=SEED, verbose=0,
                n_steps=256, batch_size=64, device="cpu")
    model.learn(total_timesteps=TRAIN_STEPS)
    print("PASS: PPO learn", TRAIN_STEPS, "timesteps")

    print("\n[3] Independent evaluation episode")
    eval_env = make_env(df, OUTPUT_DIR / "eval")
    obs, info = eval_env.reset(seed=SEED)
    rewards = []
    fees = []
    steps = 0
    terminated = truncated = False
    expected_steps = N_DAYS - eval_env._time_window

    while not (terminated or truncated):
        raw_action, _ = model.predict(obs, deterministic=True)
        raw_action = np.asarray(raw_action, dtype=np.float32)
        assert raw_action.shape == eval_env.action_space.shape
        assert np.all(np.isfinite(raw_action))

        obs, reward, terminated, truncated, info = eval_env.step(raw_action)
        weights = np.asarray(eval_env._final_weights[-1], dtype=float)
        requested = np.asarray(eval_env._actions_memory[-1], dtype=float)
        assert np.all(np.isfinite(weights)) and np.all(weights >= -1e-6)
        assert np.all(np.isfinite(requested)) and np.all(requested >= -1e-6)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
        np.testing.assert_allclose(requested.sum(), 1.0, atol=1e-6)
        assert np.isfinite(reward) and np.isfinite(eval_env._portfolio_value)
        assert eval_env._portfolio_value > 0
        assert "transaction_fee" in info and "fee_mu" in info
        fee = float(info["transaction_fee"])
        mu = float(info["fee_mu"])
        assert np.isfinite(fee) and fee >= -1e-6
        assert 0 < mu <= 1.0 + 1e-8
        rewards.append(float(reward))
        fees.append(fee)
        steps += 1
        assert steps <= expected_steps, "Episode failed to terminate on time"

    assert terminated and not truncated
    assert steps == expected_steps, (steps, expected_steps)
    total_reward = float(np.sum(rewards))
    expected_reward = float(np.log(eval_env._portfolio_value / INITIAL) * eval_env._reward_scaling)
    np.testing.assert_allclose(total_reward, expected_reward, rtol=1e-5, atol=1e-5)
    print("PASS: Evaluation episode")
    print("Asset order:", list(eval_env._tic_list))
    print("Steps:", steps)
    print("Final portfolio value:", round(float(eval_env._portfolio_value), 4))
    print("Total reward:", round(total_reward, 8))
    print("Total fees:", round(sum(fees), 4))
    print("Final weights [cash, assets...]:", np.round(weights, 6))
    print("\nALL PPO SMOKE TESTS PASSED!")
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
