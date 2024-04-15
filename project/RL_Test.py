

from DQN.lib.Backtest import Strategy, RL_evaluate, Backtest
from utils.AppSetting import AppSetting
import re

# 目前沒有連接 Strategy 和 RL_evaluate
setting = AppSetting.get_DQN_setting()


def creaet_strategy(model_path:str):
    info,feature,data = model_path.split('-')
    feature_len = re.findall('\d+',feature)[0]
    data_len = re.findall('\d+',data)[0]    
    strategytype,_,symbol = info.split('\\')

    
    strategy = Strategy(strategytype=strategytype,
                    symbol_name=symbol,
                    freq_time=int(data_len),
                    model_feature_len = int(feature_len),
                    fee=setting['BACKTEST_DEFAULT_COMMISSION_PERC'],
                    slippage=setting['DEFAULT_SLIPPAGE'],
                    model_count_path=model_path)

    strategy.load_data(local_data_path=f'DQN\simulation\data\{symbol}-F-{data_len}-Min.csv')

    return strategy

strategy = creaet_strategy('DQN\Meta\BTCUSDT-50B-30K.pt')

re_evaluate = RL_evaluate(strategy) 
Backtest(re_evaluate, strategy).order_becktest(
    re_evaluate.record_orders, ifplot=True)
