from abc import *
import numpy as np
from abc import *
import gym
import d4rl
import numpy as np
from jaxbc.utils.jaxbc_utils import yielding
from jaxbc.modules.trainer import BCTrainer,OnlineBCTrainer
from jaxbc.buffers.d4rlbuffer import d4rlBuffer
from jaxbc.buffers.rlbenchbuffer import RlbenchStateBuffer
from jaxbc.utils.jaxbc_utils import yielding
#import RLBench.rlbench.gym
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


class set_d4rl(set_env):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)


    def make_env(self):
        env = gym.make(self.task_name)
        episodes = d4rl.sequence_dataset(env)
        episodes = list(yielding(episodes))
        replay_buffer = d4rlBuffer(self.configs,env=env)
        replay_buffer.add_episodes_from_d4rl(episodes)
        # observation -> type: numpy.ndarray, shape: (timestep,state)
        return replay_buffer, env
    
    
class set_rlbench(set_env):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def make_env(self):
        obs_type = "vision"
        env = gym.make(self.task_name + '-' + obs_type + '-v0')
        return env
        # action_mode = MoveArmThenGripper(
        # arm_action_mode=JointVelocity(),
        # gripper_action_mode=Discrete()
        # )
        # env = Environment(action_mode)
        # env.launch()
        # task = env.get_task(ReachTarget)
        # print(env.action_shape)
        # print(env.ob)

        # return task, env
            