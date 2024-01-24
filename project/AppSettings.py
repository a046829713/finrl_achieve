class AppSettings:
    def __init__(self, config_file):
        pass
    
    @staticmethod
    def get_Train_config():
        Train_symbol = ["2330.TW"]
        
        
        Train_config = {
            'TOP_BRL':Train_symbol,            
            'PORTFOLIO_SIZE' :len(Train_symbol)# Define constants
        }
        return Train_config

    def get_evaluate_config():
        Test_symobl = ['2330.TW']
        evaluate_config = {
            'TOP_BRL':Test_symobl,
            'PORTFOLIO_SIZE' :len(Test_symobl)# Define constants
        }
        return evaluate_config