from DQN.lib import environment, models, common
from DQN.lib.environment import State1D, State_time_step
import os
import numpy as np
import torch
import torch.optim as optim
from DQN import ptan
from DQN.lib.DataFeature import DataFeature
from datetime import datetime
from tensorboardX import SummaryWriter
import time
from DQN.lib import offical_transformer


class RL_Train():
    def prepare_data(self):
        return DataFeature().get_train_net_work_data_by_path(self.symbols)

    def __init__(self, symbols: list) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.symbols = list(set(symbols))  # 避免重複

        # 設定檔
        # setting = AppSetting.get_DQN_setting()

        # 超參數設定
        self.hyperparameters()
        data = self.prepare_data()

        self.writer = SummaryWriter(
            log_dir=os.path.join(
                'C:\\', 'runs', datetime.strftime(
                    datetime.now(), "%Y%m%d-%H%M%S") + '-conv-'))

        # 準備神經網絡的狀態
        state = State_time_step(bars_count=self.BARS_COUNT,
                                commission_perc=self.MODEL_DEFAULT_COMMISSION_PERC,
                                model_train=True
                                )
        # 製作環境
        train_env = environment.Env(
            prices=data, state=state, random_ofs_on_reset=True)

        engine_info = train_env.engine_info()

        self.net = offical_transformer.TransformerDuelingModel(
            d_model=engine_info['input_size'],
            nhead=2,
            d_hid=2048,
            nlayers=1,
            num_actions=train_env.action_space.n,  # 假设有5种可能的动作
            hidden_size=64,  # 使用隐藏层
            seq_dim = self.BARS_COUNT,
            dropout=0.1  # 适度的dropout以防过拟合
        ).to(self.device)

        self.tgt_net = ptan.agent.TargetNet(self.net)
        # 貪婪的選擇器
        self.selector = ptan.actions.EpsilonGreedyActionSelector(
            self.EPSILON_START)

        agent = ptan.agent.DQNAgent(
            self.net, self.selector, device=self.device)

        self.exp_source = ptan.experience.ExperienceSourceFirstLast(
            train_env, agent, self.GAMMA, steps_count=self.REWARD_STEPS)

        self.buffer = ptan.experience.ExperienceReplayBuffer(
            self.exp_source, self.REPLAY_SIZE)

        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.LEARNING_RATE)

        self.load_pre_train_model_state()
        self.train()

    def load_pre_train_model_state(self):
        # 加載檢查點如果存在的話
        checkpoint_path = r''
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print("資料繼續運算模式")
            saves_path = checkpoint_path.split('\\')
            self.saves_path = os.path.join(saves_path[0], saves_path[1])
            checkpoint = torch.load(checkpoint_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.step_idx = self.EPSILON_STEPS * 0.
            
        else:
            print("建立新的儲存點")
            # 用來儲存的位置
            self.saves_path = os.path.join(self.SAVES_PATH, datetime.strftime(
                datetime.now(), "%Y%m%d-%H%M%S") + '-' + str(self.BARS_COUNT) + 'k-')

            os.makedirs(self.saves_path, exist_ok=True)
            self.step_idx = 0
            

    def train(self):
        with common.RewardTracker(self.writer, np.inf, group_rewards=2) as reward_tracker:
            while True:
                self.step_idx += 1
                self.buffer.populate(1)
                # 探索率
                self.selector.epsilon = max(
                    self.EPSILON_STOP, self.EPSILON_START - self.step_idx / self.EPSILON_STEPS)

                # [(-2.5305491551459296, 10)]
                # 跑了一輪之後,清空原本的數據,並且取得獎勵
                new_rewards = self.exp_source.pop_rewards_steps()

                if new_rewards:
                    reward_tracker.reward(
                        new_rewards[0], self.step_idx, self.selector.epsilon)

                if len(self.buffer) < self.REPLAY_INITIAL:
                    continue

                self.optimizer.zero_grad()
                batch = self.buffer.sample(self.BATCH_SIZE)
                
                loss_v = common.calc_loss(
                    batch, self.net, self.tgt_net.target_model, self.GAMMA ** self.REWARD_STEPS, device=self.device)
                if self.step_idx % self.WRITER_EVERY_STEP == 0:
                    self.writer.add_scalar(
                        "Loss_Value", loss_v.item(), self.step_idx)
                loss_v.backward()
                self.optimizer.step()

                if self.step_idx % self.TARGET_NET_SYNC == 0:
                    self.tgt_net.sync()

                # 在主訓練循環中的合適位置插入保存檢查點的代碼
                if self.step_idx % self.CHECKPOINT_EVERY_STEP == 0:
                    idx = self.step_idx // self.CHECKPOINT_EVERY_STEP
                    checkpoint = {
                        'step_idx': self.step_idx,
                        'model_state_dict': self.net.state_dict(),                
                        'selector_state': self.selector.epsilon,
                        
                    }
                    self.save_checkpoint(checkpoint, os.path.join(
                        self.saves_path, f"checkpoint-{idx}.pt"))

                # if self.step_idx > self.terminate_times:
                #     break

    def hyperparameters(self):
        self.BARS_COUNT = 300  # 用來準備要取樣的特徵長度,例如:開高低收成交量各取10根K棒
        self.GAMMA = 0.99
        self.MODEL_DEFAULT_COMMISSION_PERC = 0.002  # 後來決定不要乘上100
        self.REWARD_STEPS = 2
        self.REPLAY_SIZE = 100000
        self.REPLAY_INITIAL = 10000
        self.LEARNING_RATE = 0.0001  # optim 的學習率
        self.EPSILON_START = 1.0  # 起始機率(一開始都隨機運行)
        self.SAVES_PATH = "saves"  # 儲存的路徑
        self.EPSILON_STOP = 0.1
        self.TARGET_NET_SYNC = 1000
        self.CHECKPOINT_EVERY_STEP = 20000
        self.VALIDATION_EVERY_STEP = 100000
        self.WRITER_EVERY_STEP = 100
        self.EPSILON_STEPS = 1000000 * len(self.symbols)
        self.EVAL_EVERY_STEP = 10000  # 每一萬步驗證一次
        self.NUM_EVAL_EPISODES = 10  # 每次评估的样本数
        self.BATCH_SIZE = 32  # 每次要從buffer提取的資料筆數,用來給神經網絡更新權重
        self.STATES_TO_EVALUATE = 10000  # 每次驗證一萬筆資料
        self.terminate_times = 8000000

    def save_checkpoint(self, state, filename):
        # 保存檢查點的函數
        torch.save(state, filename)


# 我認為可以訓練出通用的模型了
# 多數據供應
# RL_Train(symbols=['ENSUSDT','LPTUSDT','GMXUSDT','TRBUSDT','ARUSDT','XMRUSDT','ETHUSDT', 'AAVEUSDT',  'ZECUSDT', 'SOLUSDT', 'DEFIUSDT',  'ETCUSDT', 'LTCUSDT', 'BCHUSDT'])
RL_Train(symbols=['ETHUSDT'])
