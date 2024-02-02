from Database.BackUp import DatabasePreparator
from Major.DataProvider import DataProvider, AsyncDataProvider
from utils.AppSetting import AppSetting
from datetime import timedelta
import datetime

class Trading_system():
    def __init__(self) -> None:
        DatabasePreparator()  # 系統初始化等檢查
        self.dataprovider = DataProvider()  # 創建資料庫的連線
        # self.engine = self.buildEngine() # 建立引擎
        self.systeam_setting = AppSetting.systeam_setting()
        
    
    

    def checkDailydata(self):
        """
            檢查資料庫中的日資料是否已經回補
            if already update then contiune
        """
        data = self.dataprovider.SQL.get_db_data(
            """select *  from `btcusdt-f-d` order by Datetime desc limit 1""")
        sql_date = str(data[0][0]).split(' ')[0]
        _todaydate = str((datetime.datetime.today() -
                         timedelta(days=1))).split(' ')[0]

        if sql_date != _todaydate:
            # 更新日資料 並且在回補完成後才繼續進行 即時行情的回補
            self.dataprovider.reload_all_data(
                time_type='1d', symbol_type='FUTURES')
        else:
            # 判斷幣安裡面所有可交易的標的
            allsymobl = self.dataprovider.Binanceapp.get_targetsymobls()
            all_tables = self.dataprovider.SQL.read_Dateframe(
                """show tables""")
            all_tables = [
                i for i in all_tables['Tables_in_crypto_data'].to_list() if '-f-d' in i]

            # 當有新的商品出現之後,會導致有錯誤,錯誤修正
            if list(filter(lambda x: False if x.lower() + "-f-d" in all_tables else True, allsymobl)):
                self.dataprovider.reload_all_data(
                    time_type='1d', symbol_type='FUTURES')    
    
    def get_target_symbol(self):
        """ 
            取得交易標的
        """
        all_symbols = self.dataprovider.get_symbols_history_data(
            symbol_type='FUTURES', time_type='1d')
        
        return self.dataprovider.filter_useful_symbol(all_symbols,tag="VOLUME_TYPE")


    def buildEngine(self):
        """ 用來創建回測系統

        Returns:
            _type_: _description_
        """
        return FastCreateOrder(formal=AppSetting.get_trading_permission()['Data_transmission'])

class AsyncTrading_system(Trading_system):
    def __init__(self) -> None:
        """
            這邊才是正式開始交易的地方，父類別大多擔任準備工作和函數提供
        """
        super().__init__()
        self.checkDailydata()
        self.process_target_symbol()

    def update_interval_record(self):

            self.dataprovider.SQL.change_db_data(f"""UPDATE interval_record SET lastportfolioadjustmenttime = '{datetime.datetime.now()}'WHERE id = '1';""")


    def process_target_symbol(self):
        """
            Retrieve the latest targets from the market every 5 days.
        """
        old_symbol = self.dataprovider.Binanceapp.getfutures_account_name() # 第一次運行會是空的
        diff_time = datetime.datetime.today() - self.dataprovider.last_profolio_adjust_time()
        if diff_time.days > 5 :
            # 取得新的標的
            self.targetsymbols =self.get_target_symbol()
        else:
            # 維持原本的標的   
             
            self.targetsymbols = self.dataprovider.target_symobl_merge(
                , old_symbol)

if __name__ == '__main__':
    trading_system = AsyncTrading_system()
