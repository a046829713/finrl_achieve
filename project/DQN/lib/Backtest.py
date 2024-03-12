from .Error import InvalidModeError
import pandas as pd
import numpy as np
import torch
from DQN.lib.DataFeature import DataFeature
from DQN.lib import environment
from DQN.lib.environment import State1D
from DQN.lib import environment, models
from Count import nb
import matplotlib.pyplot as plt
import quantstats as qs
from pathlib import Path
import time


class Strategy(object):
    """ 
       神經網絡的模型基底
    """

    def __init__(self,
                 strategytype: str,
                 symbol_name: str,
                 freq_time: int,
                 fee: float,
                 slippage: float,
                 model_count_path: str,
                 init_cash: float = 10000.0,
                 symobl_type: str = "Futures",
                 lookback_date: str = None,
                 formal: bool = False) -> None:
        self.strategytype = strategytype
        self.symbol_name = symbol_name  # 商品名稱
        self.freq_time = freq_time  # 商品週期
        self.fee = fee  # 手續費
        self.slippage = slippage  # 滑價
        self.model_count_path = model_count_path  # 模型路徑
        self.init_cash = init_cash  # 起始資金
        self.symobl_type = symobl_type  # 每個策略會有一個商品別(期貨現貨別)
        self.lookback_date = lookback_date  # 策略回測日期
        self.formal = formal  # 策略是否正式啟動

    def load_data(self, local_data_path: str):
        """
            如果非正式交易的的時候，可以啟用
        """
        if self.formal:
            raise InvalidModeError()

        self.df = pd.read_csv(local_data_path)
        self.df.set_index("Datetime", inplace=True)

    def load_Real_time_data(self, df: pd.DataFrame):
        self.df = df[['date', 'close', 'high', 'low', 'open', 'volume']].copy()
        self.df.rename(columns={"date": 'Datetime',
                                'open': 'Open',
                                "high": "High",
                                "low": "Low",
                                "close": "Close",
                                'volume': 'Volume'
                                }, inplace=True)

        self.df.set_index('Datetime', inplace=True)

    def _strategy_name(self):
        return f"{self.strategytype}-{self.symbol_name}-{self.freq_time}"


class RL_evaluate():
    def __init__(self, strategy: Strategy) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.hyperparameters()

        data = DataFeature().get_train_net_work_data_by_pd(symbol=strategy.symbol_name,
                                                           df=strategy.df)

        # 準備神經網絡的狀態
        state = State1D(bars_count=self.BARS_COUNT,
                        commission_perc=self.MODEL_DEFAULT_COMMISSION_PERC,
                        model_train=False
                        )
        # 製作環境
        self.evaluate_env = environment.Env(
            prices=data, state=state, random_ofs_on_reset=False)

        self.agent = self.load_model(
            model_path=strategy.model_count_path)

        self.test()

    def load_model(self, model_path: str):
        model = models.DQNConv1D(self.evaluate_env.observation_space.shape,
                                 self.evaluate_env.action_space.n).to(self.device)
        checkpoint = torch.load(
            model_path, map_location=self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # 將模型設置為評估模式
        return model

    def test(self):
        done = False
        rewards = []
        record_orders = []
        obs = self.evaluate_env.reset()
        state = torch.from_numpy(obs).to(self.device)
        state = state.unsqueeze(0)

        while not done:
            action = self.agent(state)
            action_idx = action.max(dim=1)[1].item()
            record_orders.append(self._parser_order(action_idx))
            _state, reward, done, info = self.evaluate_env.step(action_idx)
            state = torch.from_numpy(_state).to(self.device)
            state = state.unsqueeze(0)
            rewards.append(reward)

        self.record_orders = record_orders

    def hyperparameters(self):
        self.BARS_COUNT = 50  # 用來準備要取樣的特徵長度,例如:開高低收成交量各取10根K棒
        self.MODEL_DEFAULT_COMMISSION_PERC = 0.002  # 後來決定不要乘上100
        self.REWARD_ON_CLOSE = False  # 結束之後才給獎勵

    def _parser_order(self, action_value: int):
        if action_value == 2:
            return -1
        return action_value


class Backtest(object):
    def __init__(self, re_evaluate: RL_evaluate, strategy: Strategy) -> None:
        self.strategy = strategy
        self.bars_count = re_evaluate.BARS_COUNT
        self.Symbol_data = self.strategy.df

        self._cwd = Path("./")
        # results file
        self._results_file = self._cwd / "results" / "rl"
        self._results_file.mkdir(parents=True, exist_ok=True)

    def order_becktest(self, order: list, ifplot: bool):
        """

        order (list):
            類神經網絡所產生的訂單

        params = {'shiftorder': array([0, 0, 0, ..., 0, 0, 0], dtype=int64),
                'open_array': array([ 146.  ,  146.  ,  146.  , ..., 1631.48, 1627.78, 1628.54]),
                'Length': 135450,
                'init_cash': 10000.0,
                'slippage': 0.0025,
                'size': 1.0,
                'fee': 0.002}
        """
        # 從類神經網絡拿order的一個狀態
        self.shiftorder = np.array(order)
        self.shiftorder = np.roll(self.shiftorder, 1)
        self.shiftorder[0] = 0  # 一率將其歸零即可
        datetime_list = self.Symbol_data.index.to_list()

        # # 前面10個當樣本
        datetime_list = datetime_list[self.bars_count:]

        # # 最後一個不計算
        datetime_list = datetime_list[:-1]

        # open 平倉版本
        self.Open = self.Symbol_data['Open'].to_numpy()

        # # 前面10個當樣本
        self.Open = self.Open[self.bars_count:]

        # # 最後一個不計算
        self.Open = self.Open[:-1]

        params = {'shiftorder': self.shiftorder,
                  'open_array': self.Open,
                  'Length': len(self.Open),
                  'init_cash': self.strategy.init_cash,
                  'slippage': self.strategy.slippage,
                  'size': 1.0,
                  'fee': self.strategy.fee}

        orders, marketpostion_array, entryprice_array, buy_Fees_array, sell_Fees_array, OpenPostionprofit_array, ClosedPostionprofit_array, profit_array, Gross_profit_array, Gross_loss_array, all_Fees_array, netprofit_array = nb.logic_order(
            **params
        )

        if ifplot:
            self._plot(ClosedPostionprofit_array, "Portfolio Value", "Time",
                       "Portfolio value", self._results_file, "portfolio_value.png", specified_color='r')

            self.plot_max_drawdown(ClosedPostionprofit_array)

        return {"marketpostion_array": marketpostion_array}

    def plot_max_drawdown(self, data):
        index = pd.to_datetime(
            self.Symbol_data.index[self.bars_count:-1])  # 转换为DatetimeIndex
        data_series = pd.Series(data, index=index)  # 将数据转换为Series，并设置索引

        # 计算最大回撤
        max_dd = qs.stats.max_drawdown(data_series)
        print("Maximum DrawDown: {}".format(max_dd))
        # 绘制最大回撤图
        plt.figure(figsize=(10, 6))
        qs.plots.drawdown(data_series, show=False,
                          savefig=self._results_file / 'max_drawdown.png')
        plt.close()

    def _plot(self, data, title: str, xlabel: str, ylabel, save_path, file_name, specified_color: str = None, special=False):
        """
            绘制并保存图表。

            参数:
            - data: 要绘制的数据。
            - title: 图表标题。
            - xlabel: x轴标签。
            - ylabel: y轴标签。
            - save_path: 保存图表的路径。
            - file_name: 保存的文件名。
            - special:是否需要特殊處理
        """
        plt.figure()
        if special:
            # 转置data，以便每次循环处理一个资产的数据
            for i, asset_data in enumerate(np.array(data).T):
                local_tic = ['Cash_asset']
                local_tic.extend(self._tic_list)
                plt.plot(asset_data, label=local_tic[i])
        else:
            if specified_color is not None:
                plt.plot(data, specified_color)
            else:
                plt.plot(data)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()  # 显示图例
        plt.savefig(Path(save_path) / file_name)
        plt.close()
