from abc import *
import numpy as np
import os
import sys
import gym
import d4rl
import pickle as pkl
from jaxbc.utils.jaxbc_utils import yielding
from jaxbc.buffers.d4rlbuffer import d4rlBuffer
from jaxbc.buffers.rlbenchbuffer import RlbenchStateBuffer
from rlbench.gym.skillgrounding_env import SkillGroundingEnv
from rlbench.tasks.pick_and_lift_block2 import PickAndLiftBlock2
#from RLBench.rlbench.action_modes.action_mode import MoveArmThenGripper
#from RLBench.rlbench.action_modes.arm_action_modes import JointVelocity
#from RLBench.rlbench.action_modes.gripper_action_modes import Discrete
#from RLBench.rlbench.environment import Environment
#from RLBench.rlbench.tasks import ReachTarget


class set_env(ABC):
    def __init__(self,cfg) -> None:
        self.configs = cfg
        self.env_name = cfg.env.env_name
        self.task_name = cfg.env.task_name
        print(f"train info -> task: {self.task_name} | policy: {self.configs.policy} ")

    def make_env(slef) -> None:
        pass

    def make_buffer(slef) -> None:
        pass


class set_d4rl(set_env):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def make_env(self):
        env = gym.make(self.task_name)
        replay_buffer = self.make_buffer(env)
        # observation -> type: numpy.ndarray, shape: (timestep,state)
        return replay_buffer, env
    
    def make_buffer(self,env) -> None:
        episodes = d4rl.sequence_dataset(env)
        episodes = list(yielding(episodes))
        replay_buffer = d4rlBuffer(self.configs,env=env)
        replay_buffer.add_episodes_from_d4rl(episodes)
        return replay_buffer
    
    
class set_rlbench(set_env):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        if self.configs.model.data_type == "sensor":
            self.observation_mode = "state"
        else:
            self.observation_mode = "vision"
        if self.configs.model.image_dim:
            self.image_dim = self.configs.model.image_dim
        else:
            self.image_dim = None

    def make_env(self):
        task = getattr(sys.modules[__name__], self.task_name)
        env = SkillGroundingEnv(task_class=task,observation_mode=self.observation_mode,render_mode='rgb_array',expert_mode=False)
                # data loading
        replay_buffer = self.make_buffer(env)
        return replay_buffer, env
    
    def make_buffer(self,env) -> None:
        replay_buffer = RlbenchStateBuffer(self.configs,env=env)
        if self.observation_mode == "vision":
            replay_buffer.observation_dim = env.observation_space["state"].shape
        try:
            data_path = self.configs.model.data_path
        except:
            print("No Directory or wrong path plz check again")
        episodes = []
        for filename in os.listdir(data_path):
            episode_path = os.path.join(data_path, filename)
            episode = {}
            with open(episode_path,'rb') as f:
                data = pkl.load(f)
            episode['observations'] = data["observations"]["sensor"]
            episode['images'] = data["observations"]["image"]
            if self.image_dim == None:
                self.image_dim = np.zeros_like(episode['images'][0])
            episode['actions'] =  data["actions"]
            episodes.append(episode)
        replay_buffer.image_dim = self.image_dim
        replay_buffer.add_episodes_from_rlbench(episodes)
        return replay_buffer