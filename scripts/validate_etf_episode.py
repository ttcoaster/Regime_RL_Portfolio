
"""Validate a full FinRL episode using adjusted ETF prices.

Run:
    python -m scripts.validate_etf_episode
"""

from pathlib import Path

import numpy as np
import pandas as pd

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
    PortfolioOptimizationEnv,
)

from src.envs.ppo_portfolio_adapter import PPOPortfolioAdapter


DATA_PATH = Path("data/processed/etf_adjusted_panel.csv")
OUTPUT_DIR = Path("outputs/etf_episode_validation")

SEED = 42
INITIAL = 100_000.0
FEE = 0.001
TIME_WINDOW = 3


def make_env(df):
    base_env = PortfolioOptimizationEnv(
        df=df.copy(),
        initial_amount=INITIAL,
        time_window=TIME_WINDOW,
        features=["adj_close", "adj_high", "adj_low"],
        valuation_feature="adj_close",
        normalize_df="by_previous_time",
        return_last_action=False,
        comission_fee_model="wvm",
        comission_fee_pct=FEE,
        new_gym_api=True,
        cwd=str(OUTPUT_DIR),
    )

    return PPOPortfolioAdapter(base_env)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    df = df.sort_values(
        ["date", "tic"]
    ).reset_index(drop=True)

    env = make_env(df)

    try:
        obs, info = env.reset(seed=SEED)

        print("[1] Environment initialization")
        print("Observation shape:", obs.shape)
        print("Action shape:", env.action_space.shape)

        assert obs.shape == (36,)
        assert env.action_space.shape == (5,)
        assert np.isfinite(obs).all()

        print("PASS: reset and dimensions")

        print("\n[2] Full episode validation")

        # Equal-weight target across cash and four ETFs.
        action = np.zeros(
            env.action_space.shape,
            dtype=np.float32,
        )

        initial_value = float(
            env.unwrapped._portfolio_value
        )

        assert np.isclose(initial_value, INITIAL)

        rewards = []
        fees = []
        portfolio_values = [initial_value]

        steps = 0
        terminated = False
        truncated = False

        max_steps = df["date"].nunique()

        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            reward = float(reward)

            assert np.isfinite(reward)
            assert np.isfinite(obs).all()
            assert obs.shape == (36,)

            weights = info["target_weights"]

            assert weights.shape == (5,)
            assert np.all(weights >= 0)
            assert np.isclose(weights.sum(), 1, atol=1e-6)

            fee = float(info["transaction_fee"])

            assert np.isfinite(fee)
            assert fee >= 0

            value = float(
                env.unwrapped._portfolio_value
            )

            assert np.isfinite(value)
            assert value > 0

            rewards.append(reward)
            fees.append(fee)
            portfolio_values.append(value)

            steps += 1

            assert steps <= max_steps, (
                "Episode exceeded available trading dates"
            )

        print("PASS: full episode completed")

        print("\n[3] Episode summary")

        values = np.asarray(
            portfolio_values,
            dtype=np.float64,
        )

        running_max = np.maximum.accumulate(values)

        drawdowns = values / running_max - 1.0

        total_return = values[-1] / values[0] - 1.0

        print("Steps:", steps)
        print("Initial value:", round(values[0], 2))
        print("Final value:", round(values[-1], 2))
        print("Total return:", round(total_return, 6))
        print("Total fees:", round(sum(fees), 2))
        print("Maximum drawdown:", round(drawdowns.min(), 6))
        print("Total reward:", round(sum(rewards), 6))
        print("Terminated:", terminated)
        print("Truncated:", truncated)

        assert terminated
        assert not truncated

        print("\nALL REAL ETF EPISODE CHECKS PASSED!")

    finally:
        env.close()


if __name__ == "__main__":
    main()
