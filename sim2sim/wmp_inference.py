
import torch, os
import numpy as np
from timeit import default_timer as timer
from contextlib import contextmanager

# 定义 time-block（毫秒单位）
@contextmanager
def time_block(name="block"):
    start = timer()
    yield
    end = timer()
    elapsed_ms = (end - start) * 1000
    print(f"{name} 执行时间: {elapsed_ms:.3f} ms")


class Model_test():
    def __init__(self, jit_load_path):
        self.jit_load_path = jit_load_path
        self.init_context()
        self.load_policy()
        # self.policy_inference()

    def load_policy(self):
        # World Model and Policy 加载
        self.world_model = torch.jit.load(os.path.join(self.jit_load_path, "world_model.pt"),map_location=self.device).to(self.device)
        self.policy = torch.jit.load(os.path.join(self.jit_load_path, "policy.pt"),map_location=self.device).to(self.device)
        self.world_model.eval()
        self.policy.eval()

    def init_context(self):
        self.device = "cpu"
        self.num_hist = 5
        self.counter = 0
        self.wm_update_interval = 5
        self.policy_step_counter = 0  
        self.num_actions = 12
        self.num_obs = 33

        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.policy_prop = np.zeros(self.num_obs, dtype=np.float32)
        self.wm_prop = np.zeros(33, dtype=np.float32)
        self.obs_history = np.zeros((self.num_hist, 42), dtype=np.float32)
        self.wm_image = torch.zeros((1, 64, 64, 1), device=self.device)
        self.cmd = torch.zeros((1,3), device=self.device)
        
        self.wm_is_first = torch.ones(1, device=self.device)
        self.wm_logit = torch.zeros(1, 32, 32, device=self.device)
        self.wm_stoch = torch.zeros(1, 32, 32, device=self.device)
        self.wm_deter = torch.zeros(1, 512, device=self.device)
        self.wm_feature = torch.zeros(1, 512, device=self.device)

        self.wm_action_history = torch.zeros(1, self.wm_update_interval, self.num_actions,device=self.device)  
        self.action_tensor = torch.zeros(1, 1, self.num_actions,device=self.device)  
        self.wm_action = torch.zeros(1, 60, device=self.device)

    def policy_inference(self):
        input_wm_prop = torch.tensor(self.wm_prop, dtype=torch.float32, device=self.device).unsqueeze(0)
        history_tensor = torch.tensor(self.obs_history, dtype=torch.float32, device=self.device).unsqueeze(0)
        cmd_tensor = torch.tensor(self.cmd[:3], dtype=torch.float32, device=self.device)

        with time_block("wm推理"):
                self.wm_logit, self.wm_stoch, self.wm_deter, self.wm_feature = self.world_model(
                    input_wm_prop,
                    self.wm_image,
                    self.wm_logit,
                    self.wm_stoch,
                    self.wm_deter,
                    self.wm_action,
                    self.wm_is_first,
                )
        with time_block("policy推理"):
            action = self.policy( cmd_tensor.detach(),history_tensor.flatten(1).detach(),self.wm_feature.detach())

# ------------------------- 主入口 -------------------------
if __name__ == "__main__":
    jit_load_path = '/data/train/wmp/WMP-panda3/logs/panda3_amp_example/Feb24_11-31-32_WMP copy/exported/policies'
    demo = Model_test(jit_load_path)
    while 1:
        demo.policy_inference()
