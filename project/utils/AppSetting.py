class AppSetting():
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_setting():
        data = {"Quantify_systeam": {

            "online": {
                "Attributes": {"AVAXUSDT-15K-OB": {'symbol': 'AVAXUSDT', 'freq_time': 15, "size": 1.0, "fee": 0.002, "slippage": 0.0025, "lookback_date": '2020-01-01'},
                               "COMPUSDT-15K-OB": {'symbol': 'COMPUSDT', 'freq_time': 15, "size": 1.0, "fee": 0.002, "slippage": 0.0025, "lookback_date": '2020-01-01'},
                               "SOLUSDT-15K-OB": {'symbol': 'SOLUSDT', 'freq_time': 15, "size": 1.0, "fee": 0.002, "slippage": 0.0025, "lookback_date": '2020-01-01'},
                               "AAVEUSDT-15K-OB": {'symbol': 'AAVEUSDT', 'freq_time': 15, "size": 1.0, "fee": 0.002, "slippage": 0.0025, "lookback_date": '2020-01-01'},
                               "DEFIUSDT-15K-OB": {'symbol': 'DEFIUSDT', 'freq_time': 15, "size": 1.0, "fee": 0.002, "slippage": 0.0025, "lookback_date": '2020-01-01'},
                               },

                "parameter":  {
                    "AVAXUSDT-15K-OB": {'highest_n1': 570, 'lowest_n2': 370, 'ATR_short1': 100.0, 'ATR_long2': 190.0},
                    "COMPUSDT-15K-OB": {'highest_n1': 670, 'lowest_n2': 130, 'ATR_short1': 160.0, 'ATR_long2': 60.0},
                    "SOLUSDT-15K-OB": {'highest_n1': 490, 'lowest_n2': 290, 'ATR_short1': 100.0, 'ATR_long2': 60.0},
                    "AAVEUSDT-15K-OB": {'highest_n1': 730, 'lowest_n2': 490, 'ATR_short1': 170.0, 'ATR_long2': 150.0},
                    "DEFIUSDT-15K-OB": {'highest_n1': 90, 'lowest_n2': 490, 'ATR_short1': 70.0, 'ATR_long2': 180.0},
                }
            },
            "history": {
                "Attributes": {"AVAXUSDT-15K-OB": {'symbol': 'AVAXUSDT', 'freq_time': 15, "size": 1.0, "fee": 0.002, "slippage": 0.0025, "lookback_date": '2020-01-01'},
                               "COMPUSDT-15K-OB": {'symbol': 'COMPUSDT', 'freq_time': 15, "size": 1.0, "fee": 0.002, "slippage": 0.0025, "lookback_date": '2020-01-01'},
                               "SOLUSDT-15K-OB": {'symbol': 'SOLUSDT', 'freq_time': 15, "size": 1.0, "fee": 0.002, "slippage": 0.0025, "lookback_date": '2020-01-01'},
                               "AAVEUSDT-15K-OB": {'symbol': 'AAVEUSDT', 'freq_time': 15, "size": 1.0, "fee": 0.002, "slippage": 0.0025, "lookback_date": '2020-01-01'},
                               "DEFIUSDT-15K-OB": {'symbol': 'DEFIUSDT', 'freq_time': 15, "size": 1.0, "fee": 0.002, "slippage": 0.0025, "lookback_date": '2020-01-01'},
                               },

                "parameter": {
                    "AVAXUSDT-15K-OB": {'highest_n1': 570, 'lowest_n2': 370, 'ATR_short1': 100.0, 'ATR_long2': 190.0},
                    "COMPUSDT-15K-OB": {'highest_n1': 670, 'lowest_n2': 130, 'ATR_short1': 160.0, 'ATR_long2': 60.0},
                    "SOLUSDT-15K-OB": {'highest_n1': 490, 'lowest_n2': 290, 'ATR_short1': 100.0, 'ATR_long2': 60.0},
                    "AAVEUSDT-15K-OB": {'highest_n1': 730, 'lowest_n2': 490, 'ATR_short1': 170.0, 'ATR_long2': 150.0},
                    "DEFIUSDT-15K-OB": {'highest_n1': 90, 'lowest_n2': 490, 'ATR_short1': 70.0, 'ATR_long2': 180.0},
                }

            },
            "sharemode": {
                "Attributes": {"SHARE-15K-OB": {'freq_time': 15, "size": 1.0, "fee": 0.002, "slippage": 0.0025},

                               },

            }

        },
        }

        return data

    @staticmethod
    def get_UserDeadline():
        data = {
            "48d326d82ea14efc6710e4043722c204ee230b001f0524d1f7b3f37091542136": "2025-12-31",
            "094cb2eaec7a7eb0eb8f7dce3a5e1d082af20e9424ac70413ff79fc47d9dcecb": "2023-12-10"  # UTTER
        }

        return data

    @staticmethod
    def get_version() -> str:
        return '2023-07-02'

    @staticmethod
    def get_emergency_times() -> str:
        return 20

    @staticmethod
    def get_DQN_setting() -> str:
        setting_data = {
            "SAVES_PATH": "saves",  # 儲存的路徑
            "LEARNING_RATE": 0.0001,  # optim 的學習率,
            "BARS_COUNT": 50,  # 用來準備要取樣的特徵長度,例如:開高低收成交量各取10根K棒
            "VOLUMES_TURNON": True,  # 特徵是否要採用成交量
            'BATCH_SIZE': 32,  # 每次要從buffer提取的資料筆數,用來給神經網絡更新權重
            # 手續費用(傭金)(乘上100 類神經網絡會更有反應)(影響reward)
            "BACKTEST_DEFAULT_COMMISSION_PERC": 0.002,  # 回測時要來計算金額,所以要修正
            "DEFAULT_SLIPPAGE": 0.0025,  # 注意滑價是不能乘上100的,因為reward 本來就會乘上100
            "MODEL_DEFAULT_COMMISSION_PERC": 0.002,  # 後來決定不要乘上100
            "REWARD_ON_CLOSE": False,  # 結束之後才給獎勵
            # "MODEL_COUNT_PATH": r'DQN\20231019-154809-50k-False\mean_val-2.278.data',  # 正式模型
            "MODEL_COUNT_PATH": r'saves\20231218-172648-50k-False\checkpoint-59.pt',  # 測試模型
            'STATE_1D': True,  # 使否要採用捲積神經網絡
            'EPSILON_START': 1.0,  # 起始機率(一開始都隨機運行)
            'EVAL_EVERY_STEP': 10000, # 每一萬步驗證一次
            'NUM_EVAL_EPISODES': 10,  # 每次评估的样本数
            'STATES_TO_EVALUATE':10000, # 每次驗證一萬筆資料
            'RESET_ON_CLOSE':True # 結束之後是否重置
        }
        return setting_data
        
    @staticmethod
    def get_ooas_setting() -> str:
        setting_data = {
            "SAVES_PATH": "saves",  # 儲存的路徑
            "BARS_COUNT": 50,  # 用來準備要取樣的特徵長度,例如:開高低收成交量各取10根K棒
            "REWARD_ON_CLOSE": False,  # 結束之後才給獎勵
            'RESET_ON_CLOSE':True, # 結束之後是否隨機選擇標的和步數
            "DEFAULT_SLIPPAGE": 0.0025,  # 注意滑價是不能乘上100的,因為reward 本來就會乘上100
            "MODEL_DEFAULT_COMMISSION_PERC": 0.002,  # 後來決定不要乘上100
            'BATCH_SIZE': 32,  # 每次要從buffer提取的資料筆數,用來給神經網絡更新權重
            
        }
        return setting_data
    
    @staticmethod
    def get_trading_permission() -> bool:
        """用來控制整個交易的最高權限

        Returns:
            bool: _description_
        """
        permission_data = {
            'Data_transmission': True,
            'execute_orders': False
        }
        return permission_data
    
    @staticmethod
    def get_trading_per() -> bool:
        """用來控制交易環境

        Returns:
            bool: _description_
        """
        per_data = {
            'RSIKPERCENT': 0.0025,
            
        }
        return per_data
    
    
    @staticmethod
    def get_LSTM_DQN_setting() -> str:
        setting_data = {
            "BARS_COUNT": 50,  # 用來準備要取樣的特徵長度,例如:開高低收成交量各取10根K棒
            'RESET_ON_CLOSE':True, # 結束之後是否重置
            'EPSILON_START': 1.0,  # 起始機率(一開始都隨機運行)
            "LEARNING_RATE": 0.0001,  # optim 的學習率
            "SAVES_PATH": "saves",  # 儲存的路徑
            "MODEL_DEFAULT_COMMISSION_PERC": 0.002,  # 後來決定不要乘上100
            "DEFAULT_SLIPPAGE": 0.0025,  # 注意滑價是不能乘上100的,因為reward 本來就會乘上100
            'BATCH_SIZE': 32,  # 每次要從buffer提取的資料筆數,用來給神經網絡更新權重
            'EVAL_EVERY_STEP': 10000, # 每一萬步驗證一次
            'NUM_EVAL_EPISODES': 10,  # 每次评估的样本数
            'STATES_TO_EVALUATE':10000, # 每次驗證一萬筆資料
            "MODEL_COUNT_PATH": r'saves\20231228-115417-50k-True\checkpoint-72.pt',  # 測試模型
            "BACKTEST_DEFAULT_COMMISSION_PERC": 0.002,  # 回測時要來計算金額,所以要修正
        }
        return setting_data
    
    @staticmethod
    def get_evo_setting() -> str:
        setting_data = {
            "BACKTEST_DEFAULT_COMMISSION_PERC": 0.002,
            "MODEL_DEFAULT_COMMISSION_PERC": 0.002,  
            "DEFAULT_SLIPPAGE": 0.0025,
            "SAVES_PATH": "saves",  # 儲存的路徑 
            "BARS_COUNT": 50,  # 用來準備要取樣的特徵長度,例如:開高低收成交量各取10根K棒
            "MODEL_COUNT_PATH": r'saves\20240102-093220-50k\best_agent_params_3_2.2969335795074386.pt',  # 測試模型
            "GAME_MAX_COUNT":1000, # 每場遊戲所需要採集的樣本數量 
        }
        
        return setting_data
    
    @staticmethod
    def get_es_setting() -> str:
        setting_data = {
            "BACKTEST_DEFAULT_COMMISSION_PERC": 0.002,
            "MODEL_DEFAULT_COMMISSION_PERC": 0.002,
            "DEFAULT_SLIPPAGE": 0.0025,
            "SAVES_PATH": "saves",  # 儲存的路徑
            "BARS_COUNT": 50,  # 用來準備要取樣的特徵長度,例如:開高低收成交量各取10根K棒 
            "INITIAL_INVESTMENT":10000
        }
        
        return setting_data