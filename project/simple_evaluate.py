from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from AppSettings import AppSettings
from environment import PortfolioOptimizationEnv
from common import PVM
import numpy as np
from model import GradientPolicy
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import pandas as pd




TOP_BRL = AppSettings.get_evaluate_config()['TOP_BRL']
PORTFOLIO_SIZE = AppSettings.get_evaluate_config()['PORTFOLIO_SIZE']


# portfolio_raw_df = YahooDownloader(start_date = '2011-01-01',
#                                 end_date = '2019-12-31',
#                                 ticker_list = TOP_BRL).fetch_data()
portfolio_raw_df = pd.read_csv('train_data.csv')
portfolio_raw_df.drop(columns=['Unnamed: 0'],inplace=True)

# portfolio_test_raw_df = YahooDownloader(start_date = '2020-01-01',
#                                 end_date = '2023-12-31',
#                                 ticker_list = TOP_BRL).fetch_data()

df_portfolio = portfolio_raw_df[["date", "tic", "close", "high", "low"]]
# df_portfolio_test = portfolio_test_raw_df[["date", "tic", "close", "high", "low"]]

environment = PortfolioOptimizationEnv(
    df_portfolio,
    initial_amount=100000,
    comission_fee_pct=0.0025,
    time_window=50,
    features=["close", "high", "low"]
)


# environment_test = PortfolioOptimizationEnv(
#     df_portfolio_test,
#     initial_amount=100000,
#     comission_fee_pct=0.0025,
#     time_window=50,
#     features=["close", "high", "low"]
# )


EIIE_results = {
    "train": {},
    "test": {},
}

policy = GradientPolicy()
policy.load_state_dict(torch.load("policy_EIIE.pt"))
policy = policy.to(policy.device)


merge_list = [
    ('train',environment),
    # ('test',environment_test),

]


def update_test_model_performance(environment:PortfolioOptimizationEnv,policy,Tag,results:dict):
    """_summary_

    Args:
        environment (PortfolioOptimizationEnv): _description_
        policy (_type_): _description_
        Tag (_type_): _description_
        results (dict): 
            EIIE_results = {
                "train": {},
                "2020": {},
                "2021": {},
                "2022": {}
            }
    """
    done = False
    obs = environment.reset()
    pvm = PVM(environment.episode_length)
    while not done:
        last_action = pvm.retrieve()
        print(last_action)
        obs_batch = np.expand_dims(obs, axis=0)
        last_action_batch = np.expand_dims(last_action, axis=0)
        
        # return numpy.ndarray
        action = policy(obs_batch, last_action_batch)

        pvm.add(action)
        obs, _, done, _ = environment.step(action)
    
    results[Tag]["value"] = environment._asset_memory["final"]

for each_value in merge_list:
    update_test_model_performance(each_value[1],policy,each_value[0],EIIE_results)