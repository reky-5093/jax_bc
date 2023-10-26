import pprint
import hydra
import omegaconf
import numpy as np
from jaxbc.utils.jaxbc_utils import yielding
from jaxbc.modules.trainer import BCTrainer
import envs.env_set 



@hydra.main(version_base=None,config_path="configs", config_name="defaults")
def run_experiment(cfg: omegaconf.DictConfig) -> None:
    pprint.pprint(cfg)
    ### env ###
    functions = getattr(envs.env_set , f'set_{cfg.env.env_name}')(cfg)
    replay_buffer, env = functions.make_env()
    if cfg.model.observation_dim == None:
        cfg.model.observation_dim = env.observation_space.shape
    if cfg.model.observation_dim == None:
        cfg.model.action_dim = int(np.prod(env.action_space.shape))
    trainer = BCTrainer(cfg=cfg)
    # train
    trainer.run(replay_buffer,env)
    

if __name__ == '__main__':
    run_experiment()