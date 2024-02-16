from Major.DataProvider import DataProvider
from Major.Datatransformer import Datatransformer
from EIIE.lib import Train_neural_networks
import matplotlib.pyplot as plt
from pathlib import Path
from  EIIE.lib.simple_evaluate import evaluate_train_test_performance

def example_get_symboldata():
    """
        introduction:
            this function is for download history data to experiment.
            
    """
    symbols =['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'TRBUSDT', 'BNBUSDT', 'ETCUSDT', 'INJUSDT', 'LTCUSDT', 'BCHUSDT', 'MKRUSDT']
    for _each_symbol_name in symbols:
        DataProvider().Downloader(symbol_name=_each_symbol_name,save=True, freq=30)


def example_get_target_symbol():
    """
        introduction:
            this function is for filter 
    """
    all_symbols = DataProvider().get_symbols_history_data(
            symbol_type='FUTURES', time_type='1d') 
    
    example = Datatransformer().get_volume_top_filter_symobl(all_symbols,max_symbols = 11)



def example_reload_all_data(time_type:str):
    """
    Args:
        time_type (str): '1m','1d'
    """
    DataProvider().reload_all_data(time_type='1d',symbol_type ='FUTURES')
        


def example_Train_neural_networks():
    Train_neural_networks.train(Train_data_path='EIIE/simulation/train_data.csv',
                                Meta_path="EIIE\Train\policy_EIIE.pt", #EIIE\Meta\policy_EIIE.pt
                                episodes=200,
                                save=True,
                                pre_train=True) # True


def example_simple_evaluate(): 
    evaluate_train_test_performance(Train_data_path=r'EIIE\simulation\train_data.csv',
                                    Test_data_path=r'EIIE\simulation\test_data.csv',
                                    Meta_path=r'EIIE\Meta\policy_EIIE.pt')

example_reload_all_data(time_type = '1d')