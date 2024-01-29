## Hide matplotlib warnings
# import warnings
# warnings.filterwarnings('ignore')
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

import gym
from common import PG
import random
import functools
import torch.nn.functional as F
import pandas as pd
# from tqdm.notebook import tqdm
from torch import nn

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from environment import PortfolioOptimizationEnv
from AppSettings import AppSettings
import torch


TOP_BRL = AppSettings.get_Train_config()['TOP_BRL']

# print(TOP_BRL)

# portfolio_raw_df = YahooDownloader(start_date = '2011-01-01',
#                                 end_date = '2019-12-31',
#                                 ticker_list = TOP_BRL).fetch_data()


# print(portfolio_raw_df)
# portfolio_raw_df.to_csv('train_data.csv')

portfolio_raw_df = pd.read_csv('train_data.csv')
portfolio_raw_df.drop(columns=['Unnamed: 0'],inplace=True)




df_portfolio = portfolio_raw_df[["date", "tic", "close", "high", "low"]]

environment = PortfolioOptimizationEnv(
        df_portfolio,
        initial_amount=100000,
        comission_fee_pct=0.0025,
        time_window=50,
        features=["close", "high", "low"]
    )

algo = PG(environment, lr=0.0001)
algo.train(episodes=500)

torch.save(algo.target_policy.state_dict(), "policy_EIIE.pt")