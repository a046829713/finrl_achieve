from DQN.lib.Backtest import Strategy, RL_evaluate, Backtest
from utils.AppSetting import AppSetting
import pandas as pd
import re
import os
import time

class EngineBase():
    def __init__(self) -> None:
        self.symbols = ['BTCUSDT']        
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

            print(info)
            if_order_map[each_strategy.symbol_name] = info['marketpostion_array'][-1]
        return if_order_map
    
    def creaet_strategy(self,model_path:str):
        info,feature,data = model_path.split('-')
        feature_len = re.findall('\d+',feature)[0]
        data_len = re.findall('\d+',data)[0]    
        strategytype,_,symbol = info.split(os.sep)
        
        return Strategy(strategytype=strategytype,
                        symbol_name=symbol,
                        freq_time=int(data_len),
                        model_feature_len = int(feature_len),
                        fee=self.setting['BACKTEST_DEFAULT_COMMISSION_PERC'],
                        slippage=self.setting['DEFAULT_SLIPPAGE'],
                        model_count_path=model_path,
                        formal=True)
        
    
    def strategy_prepare(self):
        # 設定你想查看的資料夾路徑
        folder_path = os.path.join('DQN','Meta')

        # 獲取資料夾中的所有檔案和子資料夾名稱
        files_and_dirs = os.listdir(folder_path)

        # 如果你只想獲取檔案，排除子資料夾，你可以使用以下代碼
        model_files = [os.path.join(folder_path,f) for f in files_and_dirs if os.path.isfile(os.path.join(folder_path, f))]

        self.strategys = []
        for symbol in self.symbols:
            # 模型路徑
            for model_path in model_files:
                if symbol in model_path:            
                    self.strategys.append(self.creaet_strategy(model_path))