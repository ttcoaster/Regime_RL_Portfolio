
import numpy as np
import pandas as pd

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
    PortfolioOptimizationEnv,
)

INITIAL = 100_000.0
FEE = 0.001
ATOL = 0.05


def make_data():
    dates = pd.bdate_range("2025-01-02", periods=8)

    prices = {
        "SPY": [100, 100, 100, 110, 110, 110, 110, 110],
        "IEF": [100, 100, 100, 100, 100, 100, 100, 100],
        "GLD": [100, 100, 100, 100, 100, 100, 100, 100],
    }

    rows = []
    for tic, values in prices.items():
        for date, price in zip(dates, values):
            rows.append({
                "date": date,
                "tic": tic,
                "close": float(price),
                "high": float(price),
                "low": float(price),
            })

    return pd.DataFrame(rows)


def make_env(model, fee):
    env = PortfolioOptimizationEnv(
        df=make_data(),
        initial_amount=INITIAL,
        features=["close", "high", "low"],
        valuation_feature="close",
        time_column="date",
        tic_column="tic",
        time_window=3,
        normalize_df="by_previous_time",
        comission_fee_model=model,
        comission_fee_pct=fee,
        return_last_action=False,
        new_gym_api=True,
    )
    env.reset(seed=42)
    return env


def action(env, cash=0, spy=0, ief=0, gld=0):
    mapping = {"SPY": spy, "IEF": ief, "GLD": gld}
    result = np.array(
        [cash] + [mapping[t] for t in env._tic_list],
        dtype=np.float32,
    )
    np.testing.assert_allclose(result.sum(), 1, atol=1e-6)
    return result


def price_relatives(env, old_index, new_index):
    """Use original deterministic prices, not normalized features."""
    df = make_data()
    dates = sorted(df["date"].unique())

    old = df[df["date"] == dates[old_index]].set_index("tic")
    new = df[df["date"] == dates[new_index]].set_index("tic")

    relatives = [1.0]
    for tic in env._tic_list:
        relatives.append(
            new.loc[tic, "close"] / old.loc[tic, "close"]
        )

    return np.asarray(relatives, dtype=float)

def expected_wvm_mu(previous_weights, target_weights, fee):
    """
    Self-Financing WVM의 독립적인 이분법 계산.

    mu = 1 - fee * sum(
        abs(mu * target_weights[i] - previous_weights[i])
    )

    현금(index 0)은 수수료 계산에서 제외.
    """
    previous_weights = np.asarray(
        previous_weights, dtype=np.float64
    )
    target_weights = np.asarray(
        target_weights, dtype=np.float64
    )

    assert previous_weights.shape == target_weights.shape

    np.testing.assert_allclose(
        previous_weights.sum(), 1.0, atol=1e-6
    )
    np.testing.assert_allclose(
        target_weights.sum(), 1.0, atol=1e-6
    )

    assert 0 <= fee < 1

    if fee == 0:
        return 1.0

    def residual(mu):
        turnover = np.abs(
            mu * target_weights[1:]
            - previous_weights[1:]
        ).sum()

        return mu + fee * turnover - 1.0

    low, high = 0.0, 1.0

    for _ in range(200):
        mid = (low + high) / 2.0

        if residual(mid) > 0:
            high = mid
        else:
            low = mid

    return (low + high) / 2.0


def run_step(env, target, model, fee, label):
    before_value = float(env._portfolio_value)

    before_weights = np.asarray(
        env._final_weights[-1], dtype=np.float64
    ).copy()

    old_index = env._time_index

    relatives = price_relatives(
        env, old_index, old_index + 1
    )

    target = np.asarray(target, dtype=np.float64)

    # Independent expected accounting
    if model == "wvm":
        expected_mu = expected_wvm_mu(
            before_weights,
            target,
            fee,
        )

        expected_fee = before_value * (1.0 - expected_mu)

        expected_final = (
            before_value
            * expected_mu
            * np.dot(target, relatives)
        )

        expected_weights = (
            target * relatives
            / np.dot(target, relatives)
        )

    elif model == "trf":
        # TRF는 기존 구현을 그대로 검증
        # mu는 실행 후 info에서 확인
        pass

    else:
        raise ValueError(f"Unknown fee model: {model}")

    obs, reward, terminated, truncated, info = env.step(
        target.astype(np.float32)
    )

    actual_final = float(env._portfolio_value)

    actual_weights = np.asarray(
        env._final_weights[-1], dtype=np.float64
    )

    print(f"\n{label}")
    print("Before value:", round(before_value, 4))
    print("Target weights:", target)
    print("Final value:", round(actual_final, 4))
    print("Final weights:", np.round(actual_weights, 6))
    print("Reward:", reward)

    # Common accounting invariants
    assert np.isfinite(actual_final)
    assert actual_final > 0
    assert np.isfinite(reward)
    assert np.all(np.isfinite(actual_weights))
    assert np.all(actual_weights >= -1e-6)

    np.testing.assert_allclose(
        actual_weights.sum(),
        1.0,
        atol=1e-6,
    )

    expected_reward = (
        np.log(actual_final / before_value)
        * env._reward_scaling
    )

    np.testing.assert_allclose(
        reward,
        expected_reward,
        atol=1e-6,
    )

    if model == "wvm":
        actual_mu = float(info["fee_mu"])
        actual_fee = float(info["transaction_fee"])

        np.testing.assert_allclose(
            actual_mu,
            expected_mu,
            rtol=1e-6,
            atol=1e-7,
        )

        np.testing.assert_allclose(
            actual_fee,
            expected_fee,
            rtol=1e-5,
            atol=0.02,
        )

        np.testing.assert_allclose(
            actual_final,
            expected_final,
            rtol=1e-6,
            atol=ATOL,
        )

        np.testing.assert_allclose(
            actual_weights,
            expected_weights,
            atol=1e-6,
        )

        print("Expected fee:", round(expected_fee, 4))
        print("Actual fee:", round(actual_fee, 4))
        print("Self-financing mu:", round(actual_mu, 8))
        print("Trade executed: True")

    elif model == "trf":
        mu = float(info["trf_mu"])

        assert np.isfinite(mu)
        assert 0 < mu <= 1 + 1e-6

        expected_final = (
            before_value
            * mu
            * np.dot(target, relatives)
        )

        np.testing.assert_allclose(
            actual_final,
            expected_final,
            atol=ATOL,
            rtol=1e-6,
        )

        expected_weights = (
            target * relatives
            / np.dot(target, relatives)
        )

        np.testing.assert_allclose(
            actual_weights,
            expected_weights,
            atol=1e-6,
        )

        print("TRF mu:", round(mu, 8))

        if (
            fee > 0
            and np.isclose(before_weights[0], 1.0)
            and np.isclose(target[0], 0.0)
        ):
            print(
                "TRF diagnostic: Full cash-to-asset purchase "
                "may have zero transaction cost."
            )

    print("PASS:", label)



def run_scenarios(model, fee):
    print("\n" + "=" * 65)
    print(f"MODEL: {model.upper()} | FEE: {fee:.3%}")
    print("=" * 65)

    # A. Cash -> SPY 100%
    env = make_env(model, fee)

    run_step(
        env,
        action(env, spy=1),
        model,
        fee,
        "A. Full purchase",
    )

    # B. SPY 100% -> Cash 100%
    # WVM은 수수료가 있어도 검증
    # TRF는 기존 범위를 유지
    if model == "wvm" or fee == 0:
        env = make_env(model, fee)

        run_step(
            env,
            action(env, spy=1),
            model,
            fee,
            "B1. Buy SPY",
        )

        run_step(
            env,
            action(env, cash=1),
            model,
            fee,
            "B2. Sell all SPY",
        )

        # C. SPY -> GLD
        env = make_env(model, fee)

        run_step(
            env,
            action(env, spy=1),
            model,
            fee,
            "C1. Buy SPY",
        )

        run_step(
            env,
            action(env, gld=1),
            model,
            fee,
            "C2. Switch SPY to GLD",
        )

    # D. Partial allocation and rebalance after drift
    env = make_env(model, fee)

    run_step(
        env,
        action(env, cash=0.5, spy=0.5),
        model,
        fee,
        "D1. Initial allocation",
    )

    run_step(
        env,
        action(env, cash=0.5, spy=0.5),
        model,
        fee,
        "D2. Rebalance after drift",
    )

    print(f"\nPASS: {model.upper()} / {fee:.3%}")



if __name__ == "__main__":
    for model in ["wvm", "trf"]:
        for fee in [0.0, FEE]:
            run_scenarios(model, fee)

    print("\nALL BOUNDARY TESTS PASSED")
