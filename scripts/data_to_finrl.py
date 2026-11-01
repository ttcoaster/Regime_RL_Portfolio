import pandas as pd
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
    PortfolioOptimizationEnv,
)

from src.envs.ppo_portfolio_adapter import PPOPortfolioAdapter


df = pd.read_csv(
    "data/processed/etf_daily_panel.csv",
    parse_dates=["date"],
)
df = df.sort_values(["date", "tic"]).reset_index(drop=True)

base_env = PortfolioOptimizationEnv(
    df=df.copy(),
    initial_amount=100_000,
    time_window=3,
    features=["close", "high", "low"],
    valuation_feature="close",
    normalize_df="by_previous_time",
    return_last_action=False,
    comission_fee_model="wvm",
    comission_fee_pct=0.001,
    new_gym_api=True,
    cwd="outputs/etf_env_validation",
)

env = PPOPortfolioAdapter(base_env)
obs, info = env.reset(seed=42)

print("Observation shape:", obs.shape)
print("Action shape:", env.action_space.shape)

env.close()