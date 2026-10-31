
"""PPO action-interface validation; run from Regime_RL_Portfolio root.

    python tests/test_ppo_action_interface.py

Keeps the modified FinRL accounting and fee implementation unchanged.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from src.envs.ppo_portfolio_adapter import PPOPortfolioAdapter

SEED = 42
FEE = 0.001
INITIAL = 100_000.0
LOGIT_SCALE = 5.0
OUTPUT = Path("outputs/ppo_action_interface")


def synthetic_prices(n_days=70):
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    rows = []

    for tic, mu, sigma in [
        ("SPY", 0.0003, 0.01),
        ("IEF", 0.0001, 0.004),
        ("GLD", 0.0002, 0.008),
    ]:
        returns = rng.normal(mu, sigma, n_days - 1)
        prices = np.r_[100.0, 100.0 * np.exp(np.cumsum(returns))]

        for dt, price in zip(dates, prices):
            rows.append({
                "date": dt,
                "tic": tic,
                "close": price,
                "high": price * 1.002,
                "low": price * 0.998,
            })

    return pd.DataFrame(rows)


def make_base_env(df, label):
    return PortfolioOptimizationEnv(
        df=df.copy(),
        initial_amount=INITIAL,
        time_window=3,
        features=["close", "high", "low"],
        valuation_feature="close",
        normalize_df="by_previous_time",
        return_last_action=False,
        comission_fee_model="wvm",
        comission_fee_pct=FEE,
        new_gym_api=True,
        cwd=str(OUTPUT / label),
    )



def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    df = synthetic_prices()
    env = PPOPortfolioAdapter(make_base_env(df, "validation"))

    print("[1] Action mapping checks")

    n = env.action_space.shape[0]

    uniform = env.action_to_weights(
        np.zeros(n, dtype=np.float32)
    )

    np.testing.assert_allclose(
        uniform,
        np.full(n, 1 / n),
        atol=1e-7,
    )

    for i in range(n):
        action = -np.ones(n, dtype=np.float32)
        action[i] = 1.0

        weights = env.action_to_weights(action)

        assert weights[i] > 0.99, (i, weights)
        assert np.all(weights > 0)
        assert np.isclose(weights.sum(), 1, atol=1e-6)

    print("PASS: equal weights and >99% single-asset concentration")

    print("[2] Accounting and fee integration")

    env.reset(seed=SEED)

    action = -np.ones(n, dtype=np.float32)
    action[1] = 1.0  # GLD: asset order is cash, GLD, IEF, SPY

    obs, reward, terminated, truncated, info = env.step(action)

    np.testing.assert_allclose(
        env.unwrapped._actions_memory[-1],
        info["target_weights"],
        atol=1e-6,
    )

    assert info["transaction_fee"] > 0
    assert 0 < info["fee_mu"] < 1
    assert np.isfinite(reward)
    assert obs.shape == env.observation_space.shape

    print("PASS: target weights reach unchanged WVM fee engine")

    print("[3] SB3 compatibility")

    check_env(
        PPOPortfolioAdapter(make_base_env(df, "check")),
        warn=True,
    )

    print("PASS: check_env")

    print("[4] Short PPO training and evaluation")

    train_env = PPOPortfolioAdapter(make_base_env(df, "train"))

    model = PPO(
        "MlpPolicy",
        train_env,
        seed=SEED,
        verbose=0,
        n_steps=64,
        batch_size=32,
        device="cpu",
    )

    model.learn(total_timesteps=256)

    eval_env = PPOPortfolioAdapter(make_base_env(df, "eval"))

    obs, _ = eval_env.reset(seed=SEED)

    count = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = eval_env.step(action)

        assert np.isfinite(reward)
        assert np.isclose(info["target_weights"].sum(), 1, atol=1e-6)
        assert info["transaction_fee"] >= 0

        done = terminated or truncated
        count += 1

        assert count <= len(df["date"].unique()) - 3

    assert terminated
    assert count == len(df["date"].unique()) - 3

    print("PASS: PPO train/predict/evaluate, steps:", count)

    print(
        "Final weights:",
        np.round(eval_env.unwrapped._final_weights[-1], 5),
    )

    print("ALL PPO ACTION INTERFACE TESTS PASSED!")

    env.close()
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
