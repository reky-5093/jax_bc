from typing import List, Callable
import sys
import flax.linen as nn
import jax.numpy as jnp
from jax_resnet import pretrained_resnet,ResNet18
from jax_resnet.common import  slice_variables

from jaxbc.modules.type_aliases import (
	PRNGKey,
	Shape,
	Dtype,
	Array
)

from jaxbc.modules.mlp.mlp_layer import create_mlp , create_res_mlp



class MLP(nn.Module):
	act_scale: float
	output_dim: int
	net_arch: List
	activation_fn: nn.Module = nn.relu
	dropout: float = 0.0
	squash_output: bool = False
	layer_norm: bool = False
	batch_norm: bool = False
	use_bias: bool = True
	kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal()
	bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

	mlp = None

	def setup(self) -> None:
		
		mlp = create_mlp(
			output_dim=self.output_dim,
			net_arch=self.net_arch,
			activation_fn=self.activation_fn,
			dropout=self.dropout,
			squash_output=self.squash_output,
			layer_norm=self.layer_norm,
			batch_norm=self.batch_norm,
			use_bias=self.use_bias,
			kernel_init=self.kernel_init,
			bias_init=self.bias_init
		)

		self.mlp = mlp

		# self.mlp = Scaler(base_model=mlp, scale=jnp.array(self.act_scale))

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def forward(
		self,
		observations: jnp.ndarray,
		# skills: jnp.ndarray,  # [b, l, d]
		deterministic: bool = False,
		training: bool = True,
		*args, **kwargs		# Do not remove this
	):
		mlp_input = observations
		y = self.mlp(mlp_input, deterministic=deterministic, training=training)
		return y



class ResNetMLP(nn.Module):
	act_scale: float
	output_dim: int
	net_arch: List
	activation_fn: nn.Module = nn.relu
	dropout: float = 0.0
	squash_output: bool = False
	layer_norm: bool = False
	batch_norm: bool = False
	use_bias: bool = True
	kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal()
	bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
	mlp = None

	def setup(self) -> None:
		#resnet18, params = pretrained_resnet(18)
		fe_model = ResNet18(n_classes=10, norm_cls = False)
		self.resnet = nn.Sequential(fe_model.layers[0:-1])
		mlp = create_mlp(
			output_dim=self.output_dim,
			net_arch=self.net_arch,
			activation_fn=self.activation_fn,
			dropout=self.dropout,
			squash_output=self.squash_output,
			layer_norm=self.layer_norm,
			batch_norm=self.batch_norm,
			use_bias=self.use_bias,
			kernel_init=self.kernel_init,
			bias_init=self.bias_init
		)
		self.mlp = mlp

		# self.mlp = Scaler(base_model=mlp, scale=jnp.array(self.act_scale))

	def __call__(self,observations,states, *args, **kwargs):
		return self.forward(observations,states,*args,**kwargs)

	def forward(
		self,
		observations: jnp.ndarray,
		states: jnp.ndarray,
		# skills: jnp.ndarray,  # [b, l, d]
		deterministic: bool = False,
		training: bool = True,
		*args, **kwargs		# Do not remove this
	):
        # fe()
		x = self.resnet(observations)
		mlp_input = jnp.concatenate((x,states),axis=1)
		y = self.mlp(mlp_input, deterministic=deterministic, training=training)
		#y = self.mlp(observations,states, deterministic=deterministic, training=training)
		return y
	
class ResMLP(nn.Module):
	act_scale: float
	output_dim: int
	net_arch: List
	activation_fn: nn.Module = nn.relu
	dropout: float = 0.0
	squash_output: bool = False
	layer_norm: bool = False
	batch_norm: bool = False
	use_bias: bool = True
	kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal()
	bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
	resnet : nn.Module = nn.Sequential
	mlp = None

	def setup(self) -> None:
		mlp = create_res_mlp(
			output_dim=self.output_dim,
			net_arch=self.net_arch,
			activation_fn=self.activation_fn,
			dropout=self.dropout,
			squash_output=self.squash_output,
			layer_norm=self.layer_norm,
			batch_norm=self.batch_norm,
			use_bias=self.use_bias,
			kernel_init=self.kernel_init,
			bias_init=self.bias_init,
			resnet = self.resnet
		)
		self.mlp = mlp

		# self.mlp = Scaler(base_model=mlp, scale=jnp.array(self.act_scale))

	def __call__(self,observations,states, *args, **kwargs):
		return self.forward(observations,states,*args,**kwargs)

	def forward(
		self,
		observations: jnp.ndarray,
		states: jnp.ndarray,
		# skills: jnp.ndarray,  # [b, l, d]
		deterministic: bool = False,
		training: bool = True,
		*args, **kwargs		# Do not remove this
	):
		
		y = self.mlp(observations,states, deterministic=deterministic, training=training)
		return y




class Img_MLP_model(nn.Module):
	resnet: nn.Sequential
	mlp: MLP

	def __call__(self,observations,states,*args, **kwargs):
		return self.forward(observations,states,*args, **kwargs)

	def forward(self,
		observations : jnp.ndarray,
		states : jnp.ndarray,
		deterministic: bool = False,
		training: bool = True,
		*args, **kwargs
	):	
		print(kwargs)
		x = self.resnet(observations)
		x = jnp.concatenate((x,states),axis=1)
		y = self.mlp(x, deterministic = False, training= True)
		return y




# PrimGoalMLP = PrimBCMLP