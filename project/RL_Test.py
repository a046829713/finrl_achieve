

from DQN.lib.Backtest import Strategy, RL_evaluate, Backtest
from utils.AppSetting import AppSetting


# 目前沒有連接 Strategy 和 RL_evaluate
setting = AppSetting.get_DQN_setting()
strategy = Strategy(strategytype="DQN",
                    symbol_name="SOLUSDT",
                    freq_time=30,
                    fee=setting['BACKTEST_DEFAULT_COMMISSION_PERC'],
                    slippage=setting['DEFAULT_SLIPPAGE'],
                    model_count_path=r'DQN\Meta\SOLUSDT.pt')

strategy.load_data(local_data_path=r'DQN\simulation\data\SOLUSDT-F-30-Min.csv')


re_evaluate = RL_evaluate(strategy) 
Backtest(re_evaluate, strategy).order_becktest(
    re_evaluate.record_orders, ifplot=True)
