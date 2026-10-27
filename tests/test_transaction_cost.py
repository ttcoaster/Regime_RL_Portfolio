
import numpy as np
import pandas as pd

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
    PortfolioOptimizationEnv,
)

INITIAL = 100_000.0
RATES = [0.0, 0.001, 0.01]


def make_data():
    dates = pd.bdate_range("2025-01-02", periods=6)
    rows = []

    prices = {
        "SPY": [100, 100, 100, 110, 110, 110],
        "IEF": [100, 100, 100, 100, 100, 100],
        "GLD": [100, 100, 100, 100, 100, 100],
    }

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
    return PortfolioOptimizationEnv(
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


def action_for(env, cash, spy):
    allocation = {"SPY": spy, "IEF": 0.0, "GLD": 0.0}
    action = np.array(
        [cash] + [allocation[tic] for tic in env._tic_list],
        dtype=np.float32,
    )
    np.testing.assert_allclose(action.sum(), 1.0, atol=1e-6)
    return action


def run_wvm(fee):
    env = make_env("wvm", fee)
    env.reset(seed=42)

    # 초기 현금 100% -> 현금 75%, SPY 25%
    action = action_for(env, cash=0.75, spy=0.25)

    _, reward, _, _, _ = env.step(action)

    # D3 -> D4: SPY +10%
    value = float(env._portfolio_value)

    # 수수료 없이 기대되는 가치
    no_fee_value = INITIAL * (0.75 + 0.25 * 1.1)

    # 원본 WVM 방식의 매수 거래대금
    turnover = INITIAL * 0.25
    expected_fee = fee * turnover

    # 수수료를 현금에서 차감한 뒤 가격변화 적용
    expected_value = (
        INITIAL * 0.75
        - expected_fee
        + INITIAL * 0.25 * 1.1
    )

    return {
        "fee": fee,
        "value": value,
        "no_fee_value": no_fee_value,
        "expected_value": expected_value,
        "implied_cost": no_fee_value - value,
        "expected_cost": expected_fee,
        "reward": reward,
    }


def test_wvm():
    print("\n" + "=" * 55)
    print("[TEST 1] WVM Transaction Cost")
    print("=" * 55)

    results = [run_wvm(fee) for fee in RATES]

    for result in results:
        print(
            f"Fee rate: {result['fee']:.3%} | "
            f"Actual value: {result['value']:.2f} | "
            f"Expected value: {result['expected_value']:.2f} | "
            f"Implied cost: {result['implied_cost']:.2f} | "
            f"Expected cost: {result['expected_cost']:.2f}"
        )

    return results


def trf_fixed_point(last_weights, target_weights, fee):
    """FinRL TRF 식의 독립적인 이분법 해."""
    if fee == 0:
        return 1.0

    def residual(mu):
        numerator = (
            1
            - fee * target_weights[0]
            - (2 * fee - fee**2)
            * np.maximum(
                last_weights[1:] - mu * target_weights[1:],
                0,
            ).sum()
        )
        denominator = 1 - fee * target_weights[0]
        return mu - numerator / denominator

    lo, hi = 0.0, 1.0

    for _ in range(100):
        mid = (lo + hi) / 2
        if residual(mid) > 0:
            hi = mid
        else:
            lo = mid

    return (lo + hi) / 2


def test_trf():
    print("\n" + "=" * 55)
    print("[TEST 2] TRF Transaction Cost")
    print("=" * 55)

    for fee in RATES:
        env = make_env("trf", fee)
        env.reset(seed=42)

        # 첫 번째 행동: 현금 50%, SPY 50%
        first = action_for(env, cash=0.5, spy=0.5)
        env.step(first)

        # 가격변동 후 실제 비중
        previous_weights = np.array(
            env._final_weights[-1], dtype=float
        )
        value_before = float(env._portfolio_value)

        # 두 번째 행동: SPY 일부 매도, 현금 90%
        second = action_for(env, cash=0.9, spy=0.1)

        expected_mu = trf_fixed_point(
            previous_weights, second.astype(float), fee
        )

        _, reward, _, _, info = env.step(second)

        actual_mu = float(info["trf_mu"])

        # D4 -> D5는 가격변동 없음
        expected_value = value_before * expected_mu
        actual_value = float(env._portfolio_value)

        print(
            f"Fee rate: {fee:.3%} | "
            f"mu: {actual_mu:.8f} | "
            f"Value: {actual_value:.2f} | "
            f"Expected: {expected_value:.2f}"
        )

        np.testing.assert_allclose(
            actual_mu, expected_mu, rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            actual_value, expected_value, rtol=1e-6, atol=0.02
        )

        expected_reward = np.log(
            expected_value / value_before
        ) * env._reward_scaling

        np.testing.assert_allclose(
            reward, expected_reward, rtol=1e-5, atol=1e-6
        )

    print("PASS: TRF fixed-point, value and reward")


if __name__ == "__main__":
    wvm_results = test_wvm()
    test_trf()

    print("\n" + "=" * 55)
    print("WVM DIAGNOSTIC")
    print("=" * 55)

    actual_costs = [
        result["implied_cost"] for result in wvm_results
    ]
    expected_costs = [
        result["expected_cost"] for result in wvm_results
    ]

    print("Actual costs:", actual_costs)
    print("Expected costs:", expected_costs)

    if np.allclose(actual_costs, expected_costs, atol=0.02):
        print("PASS: WVM transaction fees")
    else:
        print("FAIL: WVM fees do not match fee-rate calculation")
        print("FinRL WVM implementation needs inspection.")
