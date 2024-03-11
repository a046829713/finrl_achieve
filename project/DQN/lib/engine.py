from DQN.lib.Backtest import Strategy, RL_evaluate, Backtest
from utils.AppSetting import AppSetting
import os
import pandas as pd


class EngineBase():
    def __init__(self) -> None:
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        self.setting = AppSetting.get_DQN_setting()
        self.strategy_prepare()

    def get_if_order_map(self, df: pd.DataFrame) -> dict:
        if_order_map = {}
        for each_strategy in self.strategys:
            # 載入所需要的資料
            each_strategy.load_Real_time_data(
                df[df['tic'] == each_strategy.symbol_name])

            re_evaluate = RL_evaluate(each_strategy)
            info = Backtest(re_evaluate, each_strategy).order_becktest(
                re_evaluate.record_orders, ifplot=False)

            if_order_map[each_strategy.symbol_name] = info['marketpostion_array'][-1]
        return if_order_map

    def strategy_prepare(self):
        self.strategys = []

        for each_symbol in self.symbols:
            self.strategys.append(Strategy(strategytype="DQN",
                                           symbol_name=each_symbol,
                                           freq_time=30,
                                           fee=self.setting['BACKTEST_DEFAULT_COMMISSION_PERC'],
                                           slippage=self.setting['DEFAULT_SLIPPAGE'],
                                           model_count_path=os.path.join(
                                               'DQN', 'Meta', f'{each_symbol}.pt'),
                                           formal=True))
