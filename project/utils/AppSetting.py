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
            'LEVERAGE':4
        }
        return data       