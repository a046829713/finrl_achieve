from Major.DataProvider import DataProvider


def example_get_symboldata():
    """
        introduction:
            this function is for download history data to experiment.
            
    """
    symbols =['BTCUSDT','ETHUSDT']
    
    for _each_symbol_name in symbols:
        DataProvider().Downloader(symbol_name=_each_symbol_name,save=True, freq=30)





# DataProvider().reload_all_data(time_type='1m',symbol_type ='FUTURES')
        
example_get_symboldata()