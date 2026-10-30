
import math
import numpy as np
import pandas as pd

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
    PortfolioOptimizationEnv,
)


INITIAL_VALUE = 100_000.0
TOL = 0.05


def make_data(spy_prices=None):
    """Create deterministic synthetic ETF data."""
    if spy_prices is None:
        spy_prices = [100.0] * 8

    dates = pd.bdate_range("2025-01-02", periods=len(spy_prices))

    rows = []
    for tic, prices in {
        "GLD": [100.0] * len(dates),
        "IEF": [100.0] * len(dates),
        "SPY": spy_prices,
    }.items():
        for date, price in zip(dates, prices):
            rows.append({
                "date": date,
                "tic": tic,
                "close": float(price),
                "high": float(price),
                "low": float(price),
            })

    return pd.DataFrame(rows)


def make_env(fee=0.001, spy_prices=None):
    env = PortfolioOptimizationEnv(
        df=make_data(spy_prices),
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


def action(env, **allocations):
    """Build action in the environment's actual asset order."""
    weights = {"cash": 0.0, "GLD": 0.0, "IEF": 0.0, "SPY": 0.0}
    weights.update(allocations)

    names = ["cash"] + list(env._tic_list)

    result = np.array(
        [weights[name] for name in names],
        dtype=np.float32,
    )

    assert np.isclose(result.sum(), 1.0)
    return result


def check_close(actual, expected, label, tol=TOL):
    assert math.isclose(
        float(actual), float(expected), abs_tol=tol, rel_tol=0
    ), f"{label}: actual={actual}, expected={expected}"


def check_accounting(env, total_reward):
    final_value = float(env._portfolio_value)

    assert final_value > 0
    assert np.all(np.isfinite(env._final_weights[-1]))
    assert np.all(env._final_weights[-1] >= -1e-8)

    check_close(
        np.sum(env._final_weights[-1]),
        1.0,
        "Weight sum",
        tol=1e-6,
    )

    check_close(
        total_reward,
        math.log(final_value / INITIAL_VALUE),
        "Cumulative reward",
        tol=1e-6,
    )


def run_flat_case(name, fee, allocations, expected_value):
    env = make_env(fee=fee)

    weights = action(env, **allocations)

    _, reward, terminated, truncated, info = env.step(weights)

    actual = float(env._portfolio_value)
    expected_fee = INITIAL_VALUE - expected_value

    check_close(actual, expected_value, name)

    check_close(
        info["transaction_fee"],
        expected_fee,
        name + " fee",
    )

    check_accounting(env, reward)

    assert not terminated
    assert not truncated

    print(
        f"PASS {name}: "
        f"value={actual:.4f}, "
        f"fee={info['transaction_fee']:.4f}, "
        f"reward={reward:.8f}"
    )


def test_zero_fee():
    run_flat_case(
        "Zero fee: Cash -> SPY",
        fee=0.0,
        allocations={"SPY": 1.0},
        expected_value=100_000.0,
    )


def test_full_buy():
    # B = V / (1+c)
    expected = INITIAL_VALUE / 1.001

    run_flat_case(
        "Cash -> SPY 100%",
        fee=0.001,
        allocations={"SPY": 1.0},
        expected_value=expected,
    )


def test_half_buy():
    # mu = 1 / (1 + 0.5c)
    expected = INITIAL_VALUE / (1 + 0.5 * 0.001)

    run_flat_case(
        "Cash -> SPY 50%",
        fee=0.001,
        allocations={"cash": 0.5, "SPY": 0.5},
        expected_value=expected,
    )


def test_full_sell():
    env = make_env(fee=0.001)

    # First buy SPY using the self-financing rule
    _, r1, _, _, _ = env.step(action(env, SPY=1.0))
    value_after_buy = float(env._portfolio_value)

    # Sell all SPY; selling fee = c * current value
    _, r2, _, _, info = env.step(action(env, cash=1.0))

    expected = value_after_buy * (1 - 0.001)

    check_close(env._portfolio_value, expected, "Full sell")
    check_close(
        info["transaction_fee"],
        value_after_buy * 0.001,
        "Full sell fee",
    )
    check_accounting(env, r1 + r2)

    print(
        f"PASS SPY -> Cash: "
        f"value={env._portfolio_value:.4f}"
    )


def test_asset_switch():
    env = make_env(fee=0.001)

    _, r1, _, _, _ = env.step(action(env, SPY=1.0))
    value_after_buy = float(env._portfolio_value)

    _, r2, _, _, info = env.step(action(env, GLD=1.0))

    # Sell SPY and buy GLD, both sides incur fees.
    expected_mu = (1 - 0.001) / (1 + 0.001)
    expected = value_after_buy * expected_mu

    check_close(env._portfolio_value, expected, "SPY -> GLD")
    check_close(
        info["transaction_fee"],
        value_after_buy * (1 - expected_mu),
        "Switch fee",
    )
    check_accounting(env, r1 + r2)

    print(
        f"PASS SPY -> GLD: "
        f"value={env._portfolio_value:.4f}"
    )


def test_drift_rebalance():
    # With time_window=3, the first step uses
    # the price change from index 2 to index 3.
    prices = [100, 100, 100, 110, 110, 110, 110, 110]
    env = make_env(fee=0.001, spy_prices=prices)

    _, r1, _, _, info1 = env.step(
        action(env, cash=0.5, SPY=0.5)
    )

    value_after_rise = float(env._portfolio_value)
    previous_weights = env._final_weights[-1].copy()

    # SPY is now overweight after the 10% rise.
    assert previous_weights[-1] > 0.5

    _, r2, _, _, info2 = env.step(
        action(env, cash=0.5, SPY=0.5)
    )

    # Independently solve the fee equation:
    # mu = 1 - c * abs(mu*0.5 - previous_spy_weight)
    previous_spy = float(previous_weights[-1])
    c = 0.001

    # Selling SPY to return to 50%:
    # mu = (1 - c*previous_spy) / (1 - 0.5*c)
    expected_mu = (1 - c * previous_spy) / (1 - 0.5 * c)
    expected = value_after_rise * expected_mu

    check_close(
        env._portfolio_value,
        expected,
        "Drift rebalance",
    )

    check_close(
        info2["transaction_fee"],
        value_after_rise * (1 - expected_mu),
        "Drift fee",
    )

    check_accounting(env, r1 + r2)

    print(
        f"PASS Drift rebalance: "
        f"previous SPY={previous_spy:.6f}, "
        f"fee={info2['transaction_fee']:.4f}, "
        f"value={env._portfolio_value:.4f}"
    )


if __name__ == "__main__":
    print("\n=== Self-Financing Fee Tests ===\n")

    test_zero_fee()
    test_full_buy()
    test_half_buy()
    test_full_sell()
    test_asset_switch()
    test_drift_rebalance()

    print("\nALL SELF-FINANCING TESTS PASSED!")
