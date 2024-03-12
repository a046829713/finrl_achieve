class AppSetting():
    def __init__(self) -> None:
        pass

    @staticmethod    
    def systeam_setting():
        permission_data ={
            'execute_orders': True,
            'emergency_times':50
        }   
        return permission_data
         
    @staticmethod
    def engine_setting():
        data = {
            'FREQ_TIME':30,
            'LEVERAGE':3
        }
        return data
    
    @staticmethod
    def get_DQN_setting() -> str:
        setting_data = {
            # 手續費用(傭金)(乘上100 類神經網絡會更有反應)(影響reward)
            "BACKTEST_DEFAULT_COMMISSION_PERC": 0.002,  # 回測時要來計算金額,所以要修正
            "DEFAULT_SLIPPAGE": 0.0025,  # 注意滑價是不能乘上100的,因為reward 本來就會乘上100 
        }

        return setting_data        