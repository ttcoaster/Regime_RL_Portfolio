
import math
import numpy as np
import pandas as pd

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
    PortfolioOptimizationEnv,
)


INITIAL_VALUE = 100_000.0
FEE = 0.001


def make_data():
    """
    Eight dates with a known SPY price path.

    SPY rises 10% on the first tradable transition.
    All subsequent prices remain unchanged.
    """
    prices = [100, 100, 100, 110, 110, 110, 110, 110]
    dates = pd.bdate_range("2025-01-02", periods=len(prices))

    rows = []

    for tic in ["GLD", "IEF", "SPY"]:
        for date, spy_price in zip(dates, prices):
            price = float(spy_price if tic == "SPY" else 100)

            rows.append({
                "date": date,
                "tic": tic,
                "close": price,
                "high": price,
                "low": price,
            })

    return pd.DataFrame(rows)


def make_env(fee=0.0):
    env = PortfolioOptimizationEnv(
        df=make_data(),
        initial_amount=INITIAL_VALUE,
        features=["close", "high", "low"],
        valuation_feature="close",
        time_window=3,
        comission_fee_model="wvm",
        comission_fee_pct=fee,
        reward_scaling=1.0,
        normalize_df="by_previous_time",
        new_gym_api=True,
    )

    env.reset()
    return env


def make_action(env, cash=0.0, SPY=0.0, IEF=0.0, GLD=0.0):
    allocations = {
        "cash": cash,
        "SPY": SPY,
        "IEF": IEF,
        "GLD": GLD,
    }

    names = ["cash"] + list(env._tic_list)

    weights = np.array(
        [allocations[name] for name in names],
        dtype=np.float32,
    )

    assert np.isclose(weights.sum(), 1.0)
    return weights


def assert_close(actual, expected, label, tol=1e-6):
    assert math.isclose(
        float(actual),
        float(expected),
        rel_tol=0,
        abs_tol=tol,
    ), (
        f"{label}: actual={actual}, expected={expected}"
    )


def run_episode(env, target_action):
    rewards = []
    terminated_flags = []
    truncated_flags = []

    while True:
        _, reward, terminated, truncated, _ = env.step(
            target_action
        )

        rewards.append(float(reward))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))

        if terminated or truncated:
            break

        assert len(rewards) <= 20, (
            "Episode failed to terminate within 20 steps."
        )

    return rewards, terminated_flags, truncated_flags


def test_exact_step_count():
    print("\n[TEST 1] Exact episode step count")

    env = make_env(fee=0.0)

    rewards, flags, truncated = run_episode(
        env,
        make_action(env, SPY=1.0),
    )

    expected_steps = (
        len(env._sorted_times) - env._time_window
    )   

    assert len(rewards) == expected_steps
    assert expected_steps == 5
    assert flags[:-1] == [False] * (expected_steps - 1)
    assert flags[-1] is True
    assert not any(truncated)

    print(
        f"PASS: steps={len(rewards)}, "
        f"terminated={flags[-1]}"
    )


def test_final_reward_once():
    print("\n[TEST 2] Final reward counted once")

    env = make_env(fee=0.0)

    rewards, _, _ = run_episode(
        env,
        make_action(env, SPY=1.0),
    )

    expected_final = 110_000.0
    expected_total_reward = math.log(1.10)

    assert_close(
        env._portfolio_value,
        expected_final,
        "Final portfolio value",
        tol=0.05,
    )

    assert_close(
        sum(rewards),
        expected_total_reward,
        "Cumulative reward",
    )

    # The first tradable transition contains the 10% rise.
    assert_close(
        rewards[0],
        math.log(1.10),
        "First step reward",
    )

    # Later transitions have no price change and no fees.
    for index, reward in enumerate(rewards[1:], start=2):
        assert_close(
            reward,
            0.0,
            f"Step {index} reward",
        )

    print(
        f"PASS: final_value={env._portfolio_value:.4f}, "
        f"total_reward={sum(rewards):.8f}"
    )


def test_step_after_termination():
    print("\n[TEST 3] Reject step after termination")

    env = make_env(fee=0.0)
    target = make_action(env, SPY=1.0)

    run_episode(env, target)

    try:
        env.step(target)
    except RuntimeError as error:
        assert "terminated" in str(error).lower()
        print("PASS: RuntimeError raised as expected")
    else:
        raise AssertionError(
            "step() should raise RuntimeError after termination."
        )


def test_reward_with_transaction_fees():
    print("\n[TEST 4] Reward includes transaction fees")

    env = make_env(fee=FEE)

    rewards, flags, _ = run_episode(
        env,
        make_action(env, SPY=1.0),
    )

    # Cash -> SPY 100% at 0.1% fee.
    # Subsequent steps hold SPY without rebalancing.
    expected_final = (
        INITIAL_VALUE / (1.0 + FEE)
    ) * 1.10

    expected_reward = math.log(
        expected_final / INITIAL_VALUE
    )

    assert_close(
        env._portfolio_value,
        expected_final,
        "Fee-adjusted final value",
        tol=0.05,
    )

    assert_close(
        sum(rewards),
        expected_reward,
        "Fee-adjusted cumulative reward",
    )

    assert flags[-1] is True

    print(
        f"PASS: final_value={env._portfolio_value:.4f}, "
        f"total_reward={sum(rewards):.8f}"
    )


def test_reset_after_termination():
    print("\n[TEST 5] Reset after termination")

    env = make_env(fee=0.0)

    run_episode(
        env,
        make_action(env, SPY=1.0),
    )

    observation, info = env.reset()

    assert env._terminal is False
    assert_close(
        env._portfolio_value,
        INITIAL_VALUE,
        "Reset portfolio value",
        tol=0.05,
    )

    assert isinstance(info, dict)
    assert observation is not None

    _, reward, terminated, truncated, _ = env.step(
        make_action(env, cash=1.0)
    )

    assert not terminated
    assert not truncated
    assert_close(reward, 0.0, "Reward after reset")

    print("PASS: reset and new episode work correctly")


if __name__ == "__main__":
    print("\n=== Episode Termination Tests ===")

    test_exact_step_count()
    test_final_reward_once()
    test_step_after_termination()
    test_reward_with_transaction_fees()
    test_reset_after_termination()

    print("\nALL EPISODE TERMINATION TESTS PASSED!")
