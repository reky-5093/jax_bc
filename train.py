import os
import json
import pprint
import hydra
import hydra.utils
import omegaconf
import d4rl
import numpy as np
import pickle as pkl
from jaxbc.utils.jaxbc_utils import yielding
from jaxbc.modules.trainer import BCTrainer,OnlineBCTrainer
import envs.env_set 



@hydra.main(config_path="configs", config_name="defaults")
def run_experiment(cfg: omegaconf.DictConfig) -> None:
    pprint.pprint(cfg)
    ### env ###
    functions = getattr(envs.env_set , f'set_{cfg.env.env_name}')(cfg)
    replay_buffer, env = functions.make_env()
    cfg.args.observation_dim = env.observation_space.shape
    cfg.args.action_dim = int(np.prod(env.action_space.shape))
    '''
    if cfg['env_name'] == "rlbench":

        # cfg['observation_dim'] = 512 + int(np.prod(env.observation_space['state'].shape)) # visualk feature size + state
        # cfg['observation_dim'] = int(np.prod(env.observation_space['state'].shape)) 
        # cfg['action_dim'] = int(np.prod(env.action_space.shape))


        # data loading
        print("loading data..")
        data_path = cfg['info']['data_path'] + '/variation0'
        # load task_name 
        with open(data_path+'/variation_descriptions.pkl','rb') as f:
            data = pkl.load(f)
            task_name = data[0]
        print(task_name)
        
        # load data

  
        episodes = []
        for filename in os.listdir(data_path+"/episodes"):
            episode_path = os.path.join(data_path+"/episodes", filename)
            episode = {}
            with open(episode_path+'/low_dim_obs.pkl','rb') as f:
                data = pkl.load(f)._observations
            # observations = np.concatenate([obs.task_low_dim_state[np.newaxis,:] for obs in data],axis=0)
            observations = np.concatenate([obs.get_low_dim_data()[np.newaxis,:] for obs in data],axis=0)
            # min max normalization for joint_force
            # max_value = 31.173511505126953
            # min_value = -21.414112091064453   
            max_value = 82.78092956542969
            min_value = -77.0235595703125

            data_to_minmax = observations[:,15:22]
            output = (data_to_minmax - min_value) / (max_value - min_value)
            observations = np.concatenate([observations[:,:15],output,observations[:,22:]],axis=1)

            actions = np.concatenate([np.append(obs.joint_velocities,[obs.gripper_open])[np.newaxis,:] for obs in data],axis=0)
            # actions = np.concatenate([np.append(obs.joint_positions,[obs.gripper_open])[np.newaxis,:] for obs in data],axis=0)
    
            episode['observations'] = observations
            episode['actions'] = actions

            episodes.append(episode)            

        replay_buffer = RlbenchStateBuffer(cfg,env=env)
        replay_buffer.add_episodes_from_rlbench(episodes)
    '''
    trainer = BCTrainer(cfg=cfg)
    
    # train
    trainer.run(replay_buffer,env)
    

if __name__ == '__main__':
    run_experiment()