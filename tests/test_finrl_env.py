
import pytest
import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3.common.env_checker import check_env

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
    PortfolioOptimizationEnv,
)


# ==================================================
# 1. Synthetic Financial Data
# ==================================================

def make_synthetic_data():
    rng = np.random.default_rng(42)

    dates = pd.bdate_range("2025-01-02", periods=12)
    tickers = ["SPY", "IEF", "GLD"]

    rows = []

    for tic in tickers:
        price = 100.0

        for date in dates:
            daily_return = rng.normal(0.0005, 0.01)
            price *= 1 + daily_return

            rows.append({
                "date": date,
                "tic": tic,
                "close": price,
                "high": price * 1.005,
                "low": price * 0.995,
            })

    return pd.DataFrame(rows)

@pytest.fixture
def df():
    return make_synthetic_data()

# ==================================================
# 2. FinRL Environment Factory
# ==================================================

def make_env(df):
    env=PortfolioOptimizationEnv(
        df=df.copy(),
        initial_amount=100_000,
        features=["close", "high", "low"],
        valuation_feature="close",
        time_column="date",
        tic_column="tic",
        time_window=3,
        normalize_df="by_previous_time",
        comission_fee_model=None,
        return_last_action=False,
        new_gym_api=True,
    )
    env=gym.wrappers.FlattenObservation(env)
    return env




# ==================================================
# 3. Test reset()
# ==================================================

def test_reset(df):
    print("\n[TEST 1] reset()")

    env = make_env(df)

    obs, info = env.reset(seed=42)

    print("Observation shape:", obs.shape)
    print("Observation dtype:", obs.dtype)
    print("Info type:", type(info).__name__)

    assert isinstance(env, gym.Env)
    assert isinstance(info, dict)
    assert obs.shape == (27,)
    assert env.observation_space.contains(obs)

    print("PASS: reset()")


# ==================================================
# 4. Test step()
# ==================================================

def test_step(df):
    print("\n[TEST 2] step()")

    env = make_env(df)
    obs, info = env.reset(seed=42)

    # [Cash, SPY, IEF, GLD]
    action = np.array(
        [0.25, 0.25, 0.25, 0.25],
        dtype=np.float32,
    )

    result = env.step(action)

    assert isinstance(result, tuple)
    assert len(result) == 5

    obs, reward, terminated, truncated, info = result

    print("Observation shape:", obs.shape)
    print("Reward:", reward)
    print("Reward type:", type(reward).__name__)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    print("PASS: step()")


# ==================================================
# 5. Stable-Baselines3 check_env()
# ==================================================

def test_sb3(df):
    print("\n[TEST 3] Stable-Baselines3 check_env()")

    env = make_env(df)

    check_env(env, warn=True)

    print("PASS: SB3 check_env()")



# 기존 import 및 함수 정의들
# make_synthetic_data()
# make_env()
# test_reset()
# test_step()
# test_sb3()


def test_full_episode(df):
    print("\n[TEST 4] Full Episode")

    env = make_env(df)

    obs, info = env.reset(seed=42)

    terminated = False
    truncated = False
    steps = 0
    total_reward = 0.0

    while not (terminated or truncated):
        action = np.array(
            [0.25, 0.25, 0.25, 0.25],
            dtype=np.float32,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        assert np.isfinite(reward)
        assert env.observation_space.contains(obs)

        total_reward += reward
        steps += 1

        if steps > len(df["date"].unique()) + 2:
            raise RuntimeError(
                "Episode did not terminate as expected."
            )

    print("Total steps:", steps)
    print("Total reward:", total_reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    assert terminated or truncated

    print("PASS: Full Episode")

# ==================================================
# Main
# ==================================================

# 반드시 모든 함수 정의보다 아래에 배치
if __name__ == "__main__":
    data = make_synthetic_data()

    print("Synthetic Data Preview:")
    print(data.head())
    print("\nData shape:", data.shape)

    test_reset(data)
    test_step(data)
    test_sb3(data)
    test_full_episode(data)

    print("\nAll four tests passed!")



