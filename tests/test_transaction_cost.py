
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

def expected_wvm_mu(previous_weights, target_weights, fee):
    """
    Self-Financing WVM의 독립적인 이분법 해.

    mu = 1 - fee * sum(
        abs(mu * target_weights[i] - previous_weights[i])
    )

    현금(index 0)은 거래비용 계산에서 제외.
    """
    previous_weights = np.asarray(
        previous_weights, dtype=np.float64
    )
    target_weights = np.asarray(
        target_weights, dtype=np.float64
    )

    assert previous_weights.shape == target_weights.shape
    np.testing.assert_allclose(previous_weights.sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(target_weights.sum(), 1.0, atol=1e-6)
    assert 0 <= fee < 1

    if fee == 0:
        return 1.0

    def residual(mu):
        traded_amount = np.sum(
            np.abs(
                mu * target_weights[1:]
                - previous_weights[1:]
            )
        )
        return mu + fee * traded_amount - 1.0

    low, high = 0.0, 1.0

    for _ in range(200):
        mid = (low + high) / 2.0

        if residual(mid) > 0:
            high = mid
        else:
            low = mid

    return (low + high) / 2.0


def run_wvm(fee):
    env = make_env("wvm", fee)
    env.reset(seed=42)

    # 초기 현금 100% -> 현금 75%, SPY 25%
    action = action_for(env, cash=0.75, spy=0.25)

    previous_weights = np.array(
        env._final_weights[-1], dtype=np.float64
    )

    value_before = float(env._portfolio_value)

    # 독립적인 자기금융 방정식으로 기대 mu 계산
    expected_mu = expected_wvm_mu(
        previous_weights,
        action.astype(np.float64),
        fee,
    )

    # 수수료 차감 직후의 기대 자산 가치
    expected_value_after_fee = value_before * expected_mu

    # 실제 거래 시점에 발생하는 기대 수수료
    expected_fee = value_before * (1.0 - expected_mu)

    # D3 -> D4: SPY +10%
    expected_growth = 0.75 + 0.25 * 1.10

    # 수수료 차감 후 시장 가격변동 적용
    expected_value = expected_value_after_fee * expected_growth

    _, reward, terminated, truncated, info = env.step(action)

    actual_value = float(env._portfolio_value)
    actual_fee = float(info["transaction_fee"])
    actual_mu = float(info["fee_mu"])

    # 수수료 없는 경우의 최종 자산 가치
    no_fee_value = value_before * expected_growth

    # 가격변동 이후 최종 자산 가치 차이
    implied_cost = no_fee_value - actual_value

    # 이 값은 실제 거래 시점의 수수료와 다를 수 있음
    expected_final_value_difference = (
        no_fee_value - expected_value
    )

    # 1. 자기금융 mu 검증
    np.testing.assert_allclose(
        actual_mu,
        expected_mu,
        rtol=1e-6,
        atol=1e-7,
        err_msg="WVM self-financing mu mismatch",
    )

    # 2. 실제 거래 시점의 수수료 검증
    np.testing.assert_allclose(
        actual_fee,
        expected_fee,
        rtol=1e-5,
        atol=0.02,
        err_msg="WVM transaction fee mismatch",
    )

    # 3. 최종 포트폴리오 가치 검증
    np.testing.assert_allclose(
        actual_value,
        expected_value,
        rtol=1e-6,
        atol=0.02,
        err_msg="WVM final portfolio value mismatch",
    )

    # 4. 수수료 차감 이후의 최종 가치 차이 검증
    np.testing.assert_allclose(
        implied_cost,
        expected_final_value_difference,
        rtol=1e-5,
        atol=0.02,
        err_msg="WVM final value difference mismatch",
    )

    # 5. Log-return reward 검증
    expected_reward = np.log(
        expected_value / value_before
    ) * env._reward_scaling

    np.testing.assert_allclose(
        reward,
        expected_reward,
        rtol=1e-5,
        atol=1e-6,
        err_msg="WVM log-return reward mismatch",
    )

    assert terminated is False
    assert truncated is False

    return {
        "fee": fee,
        "value": actual_value,
        "expected_value": expected_value,
        "actual_mu": actual_mu,
        "expected_mu": expected_mu,
        "actual_fee": actual_fee,
        "expected_fee": expected_fee,
        "implied_cost": implied_cost,
        "expected_final_value_difference": expected_final_value_difference,
        "reward": reward,
    }




def test_wvm():
    print("\n" + "=" * 55)
    print("[TEST 1] WVM Self-Financing Transaction Cost")
    print("=" * 55)

    results = [run_wvm(fee) for fee in RATES]

    for result in results:
        print(
            f"Fee rate: {result['fee']:.3%} | "
            f"Actual value: {result['value']:.4f} | "
            f"Expected value: {result['expected_value']:.4f} | "
            f"Actual fee: {result['actual_fee']:.4f} | "
            f"Expected fee: {result['expected_fee']:.4f} | "
            f"mu: {result['actual_mu']:.8f}"
        )

    print("PASS: WVM self-financing mu, fee, value and reward")

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

    actual_fees = [
        result["actual_fee"] for result in wvm_results
    ]
    expected_fees = [
        result["expected_fee"] for result in wvm_results
    ]

    print("Actual transaction fees:", actual_fees)
    print("Expected transaction fees:", expected_fees)

    np.testing.assert_allclose(
        actual_fees,
        expected_fees,
        rtol=1e-5,
        atol=0.02,
    )

    print("PASS: WVM transaction fees match self-financing model")
    print("\nALL TRANSACTION COST TESTS PASSED!")
