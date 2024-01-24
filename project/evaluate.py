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

TOP_BRL = AppSettings.get_evaluate_config()['TOP_BRL']
PORTFOLIO_SIZE = AppSettings.get_evaluate_config()['PORTFOLIO_SIZE']

portfolio_raw_df = YahooDownloader(start_date = '2011-01-01',
                                end_date = '2019-12-31',
                                ticker_list = TOP_BRL).fetch_data()

portfolio_2020_raw_df = YahooDownloader(start_date = '2020-01-01',
                                end_date = '2020-12-31',
                                ticker_list = TOP_BRL).fetch_data()
portfolio_2021_raw_df = YahooDownloader(start_date = '2021-01-01',
                                end_date = '2021-12-31',
                                ticker_list = TOP_BRL).fetch_data()
portfolio_2022_raw_df = YahooDownloader(start_date = '2022-01-01',
                                end_date = '2022-12-31',
                                ticker_list = TOP_BRL).fetch_data()

df_portfolio = portfolio_raw_df[["date", "tic", "close", "high", "low"]]
df_portfolio_2020 = portfolio_2020_raw_df[["date", "tic", "close", "high", "low"]]
df_portfolio_2021 = portfolio_2021_raw_df[["date", "tic", "close", "high", "low"]]
df_portfolio_2022 = portfolio_2022_raw_df[["date", "tic", "close", "high", "low"]]

environment = PortfolioOptimizationEnv(
    df_portfolio,
    initial_amount=100000,
    comission_fee_pct=0.0025,
    time_window=50,
    features=["close", "high", "low"]
)



environment_2020 = PortfolioOptimizationEnv(
    df_portfolio_2020,
    initial_amount=100000,
    comission_fee_pct=0.0025,
    time_window=50,
    features=["close", "high", "low"]
)

environment_2021 = PortfolioOptimizationEnv(
    df_portfolio_2021,
    initial_amount=100000,
    comission_fee_pct=0.0025,
    time_window=50,
    features=["close", "high", "low"]
)

environment_2022 = PortfolioOptimizationEnv(
    df_portfolio_2022,
    initial_amount=100000,
    comission_fee_pct=0.0025,
    time_window=50,
    features=["close", "high", "low"]
)


EIIE_results = {
    "train": {},
    "2020": {},
    "2021": {},
    "2022": {}
}

policy = GradientPolicy()
policy.load_state_dict(torch.load("policy_EIIE.pt"))
policy = policy.to(policy.device)


merge_list = [
    ('train',environment),
    ('2020',environment_2020),
    ('2021',environment_2021),
    ('2022',environment_2022),
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
        obs_batch = np.expand_dims(obs, axis=0)
        last_action_batch = np.expand_dims(last_action, axis=0)
        action = policy(obs_batch, last_action_batch)
        pvm.add(action)
        obs, _, done, _ = environment.step(action)
    results[Tag]["value"] = environment._asset_memory["final"]

for each_value in merge_list:
    update_test_model_performance(each_value[1],policy,each_value[0],EIIE_results)






UBAH_results = {
    "train": {},
    "2020": {},
    "2021": {},
    "2022": {}
}


def BuyAndHold(environment:PortfolioOptimizationEnv,Tag,results:dict):
    """_summary_

    Args:
        environment (_type_): _description_
        Tag (_type_): _description_
        results (dict): 
            UBAH_results = {
                "train": {},
                "2020": {},
                "2021": {},
                "2022": {}
            }

    """
    # train period
    terminated = False
    environment.reset()
    while not terminated:
        action = [0] + [1/PORTFOLIO_SIZE] * PORTFOLIO_SIZE
        _, _, terminated, _ = environment.step(action)
    results[Tag]["value"] = environment._asset_memory["final"]


for each_value in merge_list:
    BuyAndHold(each_value[1],each_value[0],UBAH_results)



def plot_performance(UBAH_results, EIIE_results, period, title):
    """
    绘制指定时间段的投资组合价值表现。

    :param UBAH_results: 买入并持有策略的结果字典。
    :param EIIE_results: EIIE 策略的结果字典。
    :param period: 要绘制的时间段（例如 "train", "2020", "2021", "2022"）。
    :param title: 图表的标题。
    """
    plt.plot(UBAH_results[period]["value"], label="Buy and Hold")
    plt.plot(EIIE_results[period]["value"], label="EIIE")

    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.title(title)
    plt.legend()
    plt.show()

# 使用示例
plot_performance(UBAH_results, EIIE_results, "train", "Performance in train period")
plot_performance(UBAH_results, EIIE_results, "2020", "Performance in 2020")
plot_performance(UBAH_results, EIIE_results, "2021", "Performance in 2021")
plot_performance(UBAH_results, EIIE_results, "2022", "Performance in 2022")