
import numpy as np
import pandas as pd

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
    PortfolioOptimizationEnv,
)

INITIAL_VALUE = 100_000.0
TOL = 1e-2


def make_data():
    dates = pd.bdate_range("2025-01-02", periods=6)

    prices = {
        "SPY": [100, 100, 100, 110, 110, 110],
        "IEF": [100, 100, 100, 100, 100, 100],
        "GLD": [100, 100, 100, 100, 100, 100],
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


def make_env(df, fee_model=None, fee_pct=0):
    return PortfolioOptimizationEnv(
        df=df.copy(),
        initial_amount=INITIAL_VALUE,
        features=["close", "high", "low"],
        valuation_feature="close",
        time_column="date",
        tic_column="tic",
        time_window=3,
        normalize_df="by_previous_time",
        comission_fee_model=fee_model,
        comission_fee_pct=fee_pct,
        return_last_action=False,
        new_gym_api=True,
    )


def assert_close(actual, expected, label, atol=TOL):
    np.testing.assert_allclose(
        actual, expected, rtol=1e-6, atol=atol,
        err_msg=label,
    )
    print(f"PASS: {label}")


def test_accounting():
    df = make_data()
    env = make_env(df)
    env.reset(seed=42)

    # FinRL의 실제 자산 순서를 확인
    print("Asset order:", env._tic_list)

    target = {
        "SPY": 0.5,
        "IEF": 0.0,
        "GLD": 0.0,
    }

    # 첫 번째 요소는 현금
    action = np.array(
        [0.5] + [target[tic] for tic in env._tic_list],
        dtype=np.float32,
    )

    assert_close(action.sum(), 1.0, "Action sum", atol=1e-6)

    # D3 -> D4: SPY 10% 상승
    obs, reward, terminated, truncated, info = env.step(action)

    expected_value = INITIAL_VALUE * 1.05
    expected_reward = np.log(1.05)

    print("\n--- Portfolio accounting ---")
    print("Portfolio value:", env._portfolio_value)
    print("Reward:", reward)
    print("Final weights:", env._final_weights[-1])

    assert_close(
        env._portfolio_value,
        expected_value,
        "Portfolio value after price change",
    )

    assert_close(
        reward,
        expected_reward * env._reward_scaling,
        "Log-return reward",
        atol=1e-6,
    )

    actual_weights = dict(
        zip(["CASH"] + list(env._tic_list), env._final_weights[-1])
    )

    assert_close(
        actual_weights["CASH"],
        50_000 / 105_000,
        "Cash weight after drift",
        atol=1e-6,
    )

    assert_close(
        actual_weights["SPY"],
        55_000 / 105_000,
        "SPY weight after drift",
        atol=1e-6,
    )

    assert_close(
        sum(actual_weights.values()),
        1.0,
        "Weights sum after drift",
        atol=1e-6,
    )

    # D4 -> D5: 모든 자산 가격 동일
    # 같은 목표 비중으로 리밸런싱, 거래비용 없음
    obs, reward, terminated, truncated, info = env.step(action)

    assert_close(
        env._portfolio_value,
        expected_value,
        "Value unchanged when prices are flat",
    )

    assert_close(
        reward,
        0.0,
        "Zero reward on flat prices",
        atol=1e-6,
    )

    # D5 -> D6: 마지막 가격 변화
    obs, reward, terminated, truncated, info = env.step(action)

    assert_close(
        env._portfolio_value,
        expected_value,
        "Final price change accounted for",
    )

    value_before_terminal = float(env._portfolio_value)

    # 종료 확인용 추가 step
    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated
    assert_close(
        env._portfolio_value,
        value_before_terminal,
        "No value change on terminal step",
    )

    print("\nALL BASIC ACCOUNTING TESTS PASSED")


if __name__ == "__main__":
    test_accounting()
