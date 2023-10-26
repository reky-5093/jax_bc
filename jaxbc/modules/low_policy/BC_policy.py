import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Dict
import sys
from flax.core import FrozenDict,frozen_dict
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxbc.modules.common import Model,Img_Model
from jaxbc.modules.updates import bc_mlp_updt,image_bc_mlp_updt
from jaxbc.modules.mlp.mlp  import MLP, Img_MLP_model,ResNetMLP ,ResMLP
from jaxbc.modules.forwards import mlp_forward, image_mlp_forward
from jax_resnet import pretrained_resnet
from jax_resnet.common import  slice_variables
from flax import traverse_util



class BCpolicy():
    
    # initial variables
    def __init__(
            self,
            cfg: Dict,
            init_build_model: bool = True,
    ):
        self.seed = cfg.seed
        self.cfg = cfg
        self.observation_dim = self.cfg.model.observation_dim
        self.action_dim = self.cfg.model.action_dim

        self.rng = jax.random.PRNGKey(self.seed)

        self.__model = None

        if init_build_model:
            self.build_model()

    @property
    def model(self) -> Model:
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value      

    def build_model(self):
        act_scale = False
        # net_arch = [256]*4
        net_arch = self.cfg.model.architecture
        activation_fn = nn.relu
        dropout = 0.0
        squash_output = self.cfg.model.tanh_action
        layer_norm = False

        mlp = MLP(
            act_scale=act_scale,
            output_dim=self.action_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            dropout=dropout,
            squash_output=squash_output,
            layer_norm=layer_norm
        )

        init_obs = np.expand_dims(np.zeros((self.observation_dim)),axis=0)

        rng, param_key, dropout_key, batch_key = jax.random.split(self.rng, 4)

        self.rng = rng
        rngs = {"params": param_key, "dropout": dropout_key, "batch_stats": batch_key}
        tx = optax.adam(self.cfg.parameter.lr)


        self.model = Model.create(model_def=mlp, inputs=[rngs, init_obs], tx=tx)

    def update(self, replay_data):  
        obs = []
        actions = []
        maskings = []

        for data in replay_data:
            obs.append(data['obs'])
            actions.append(data['actions'])
        
        obs = np.concatenate(obs)
        actions = np.concatenate(actions)

        # TODO
        maskings = None

        new_model, info = bc_mlp_updt(
			rng=self.rng,
			mlp=self.model,
			observations=obs,
			actions=actions,
			maskings=maskings
		)  
 
        self.model = new_model
        self.rng, _ = jax.random.split(self.rng)

        return info

    def predict(
        self,
        observations: np.ndarray,
        to_np: bool = True,
        squeeze: bool = False,
        *args, **kwargs  # Do not remove these dummy parameters.
    ) -> np.ndarray:

        self.rng, prediction = mlp_forward(
            rng=self.rng,
            model=self.model,
            observations=observations,
        )

        if squeeze:
            prediction = np.squeeze(prediction, axis=0)

        if to_np:
            return np.array(prediction)
        else:
            return prediction

    def evaluate(
        self,
        replay_data
    ) -> Dict:
        observations = replay_data.observations
        actions = replay_data.actions[:, -1, ...]
        if self.cfg.use_optimal_lang:
            raise NotImplementedError("Obsolete")
        maskings = replay_data.maskings[:, -1]

        if maskings is None:
            raise NotImplementedError("No mask")
        maskings = maskings.reshape(-1, 1)
        
        pred_actions = self.predict(observations=observations)

        pred_actions = pred_actions.reshape(-1, self.action_dim) * maskings
        target_actions = actions.reshape(-1, self.action_dim) * maskings
        mse_error = np.sum(np.mean((pred_actions - target_actions) ** 2, axis=-1)) / np.sum(maskings)
        eval_info = {
            "decoder/mse_error": mse_error,
            "decoder/mse_error_scaled(x100)": mse_error * 100
        }
        return eval_info

    def save(self, path: str) -> None:
        self.model.save_dict_from_path(path)

    def load(self, path: str) -> None:
        self.model = self.model.load_dict_from_path(path)


class Image_BCpolicy(BCpolicy):
    def __init__(
        self,
        cfg: Dict,
        init_build_model: bool = True,
    ):
        self.image_dim = cfg.model.image_dim
        super().__init__(cfg,
        init_build_model)
        
    def build_model(self):
        act_scale = False
        net_arch = self.cfg.model.architecture
        activation_fn = nn.relu
        dropout = 0.0
        squash_output = self.cfg.model.tanh_action
        layer_norm = False
        resnet_tmpl, params = pretrained_resnet(18)
        resnet = resnet_tmpl()
        resnet = nn.Sequential(resnet.layers[0:-1])
        backbone_params = slice_variables(params,0, -1)
        mlp = ResMLP(
            act_scale=act_scale,
            output_dim=self.action_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            dropout=dropout,
            squash_output=squash_output,
            layer_norm=layer_norm,
            resnet= resnet
        )
        rng, param_key, dropout_key, batch_key = jax.random.split(self.rng, 4)
        self.rng = rng
        x = np.expand_dims(np.zeros((self.image_dim)),axis=0)
        y = np.expand_dims(np.zeros((self.observation_dim)),axis=0)
        rngs = {"params": param_key, "dropout": dropout_key, "batch_stats": batch_key}
        variables = mlp.init(rngs= rngs, observations=x, states=y)
        variables = variables.unfreeze()
        variables['params']["resnet"] = backbone_params['params'].unfreeze()
        variables['batch_stats']["resnet"] = backbone_params['batch_stats'].unfreeze()
        partition_optimizers = {'trainable': optax.adam(self.cfg.parameter.lr), 'frozen': optax.set_to_zero()}
        param_partitions = traverse_util.path_aware_map(
            lambda path, v: 'frozen' if 'resnet' in path else 'trainable', variables["params"])
        tx = optax.multi_transform(partition_optimizers, param_partitions)
        self.model = Img_Model.create(model_def=mlp, variables = variables  ,tx= tx)
        

    def update(self, replay_data):  
        states = []
        actions = []
        observations = []
        maskings = []

        for data in replay_data:
            states.append(data['obs'])
            observations.append(data['images'])
            actions.append(data['actions'])
        
        observations = np.concatenate(observations) 
        states = np.concatenate(states)
        actions = np.concatenate(actions)
        # TODO
        maskings = None
    
        new_model, info = image_bc_mlp_updt(
			rng=self.rng,
			mlp=self.model,
			observations=observations,
			actions=actions,
            states=states,
			maskings=maskings
		)  

        
        #print(info['__decoder/gripper_actions'])
        #print(info['__decoder/target_gripper_actions'])
        #print(info['decoder/ce_loss'])
        #sys.exit()
 
        self.model = new_model
        self.rng, _ = jax.random.split(self.rng)

        return info

    def predict(
        self,
        observations: np.ndarray,
        states: np.ndarray,
        to_np: bool = True,
        squeeze: bool = False,
        *args, **kwargs  # Do not remove these dummy parameters.
    ) -> np.ndarray:

        self.rng, prediction = image_mlp_forward(
            rng=self.rng,
            model=self.model,
            observations=observations,
            states = states,
        )

        if squeeze:
            prediction = np.squeeze(prediction, axis=0)

        if to_np:
            return np.array(prediction)
        else:
            return prediction

    def evaluate(
        self,
        replay_data
    ) -> Dict:
        observations = replay_data.images
        states = replay_data.observations
        actions = replay_data.actions[:, -1, ...]
        if self.cfg.use_optimal_lang:
            raise NotImplementedError("Obsolete")
        maskings = replay_data.maskings[:, -1]

        if maskings is None:
            raise NotImplementedError("No mask")
        maskings = maskings.reshape(-1, 1)
        
        pred_actions = self.predict(observations=observations,states=states)

        pred_actions = pred_actions.reshape(-1, self.action_dim) * maskings
        target_actions = actions.reshape(-1, self.action_dim) * maskings
        mse_error = np.sum(np.mean((pred_actions - target_actions) ** 2, axis=-1)) / np.sum(maskings)
        eval_info = {
            "decoder/mse_error": mse_error,
            "decoder/mse_error_scaled(x100)": mse_error * 100
        }
        return eval_info