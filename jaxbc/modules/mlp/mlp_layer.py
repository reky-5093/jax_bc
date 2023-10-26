from typing import  Callable, List ,Any
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


from jaxbc.modules.type_aliases import (
	PRNGKey,
	Shape,
	Dtype,
	Array,
	Params
)

def create_mlp(
	output_dim: int,
    net_arch: List[int],
    activation_fn: Callable = nn.relu,
    dropout: float = 0.0,
    squash_output: bool = False,
    layer_norm: bool = False,
    batch_norm: bool = False,
    use_bias: bool = True,
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal(),
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
):
	if output_dim > 0:
		net_arch = list(net_arch)
		net_arch.append(output_dim)

	return MLP_model(
		net_arch=net_arch,
		activation_fn=activation_fn,
		dropout=dropout,
		squash_output=squash_output,
		layer_norm=layer_norm,
		batch_norm=batch_norm,
		use_bias=use_bias,
		kernel_init=kernel_init,
		bias_init=bias_init
	)

def create_res_mlp(
	output_dim: int,
	resnet: nn.Module,
    net_arch: List[int],
    activation_fn: Callable = nn.relu,
    dropout: float = 0.0,
    squash_output: bool = False,
    layer_norm: bool = False,
    batch_norm: bool = False,
    use_bias: bool = True,
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal(),
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
	
):
	if output_dim > 0:
		net_arch = list(net_arch)
		net_arch.append(output_dim)

	return Res_MLP_model(
		net_arch=net_arch,
		activation_fn=activation_fn,
		dropout=dropout,
		squash_output=squash_output,
		layer_norm=layer_norm,
		batch_norm=batch_norm,
		use_bias=use_bias,
		kernel_init=kernel_init,
		bias_init=bias_init,
		resnet = resnet
	)




class MLP_model(nn.Module):
	net_arch: List
	activation_fn: nn.Module
	dropout: float = 0.0
	squash_output: bool = False
	layer_norm: bool = False
	batch_norm: bool = False
	use_bias: bool = True
	kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal()
	bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

	@nn.compact
	def __call__(self, x: jnp.ndarray, deterministic: bool = False, training: bool = True):
		for features in self.net_arch[: -1]:
			x = nn.Dense(features=features, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
			if self.batch_norm:
				x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
			if self.layer_norm:
				x = nn.LayerNorm()(x)

			x = self.activation_fn(x)
			x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)

		if len(self.net_arch) > 0:
			x = nn.Dense(features=self.net_arch[-1], kernel_init=self.kernel_init, bias_init=self.bias_init)(x)

		if self.squash_output:
			return nn.tanh(x)
		else:
			return x
		


class Res_MLP_model(nn.Module):
	net_arch: List
	activation_fn: nn.Module
	dropout: float = 0.0
	squash_output: bool = False
	layer_norm: bool = False
	batch_norm: bool = False
	use_bias: bool = True
	kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_normal()
	bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
	resnet: nn.Sequential = None


	@nn.compact
	def __call__(self, x: jnp.ndarray, y: jnp.ndarray, deterministic: bool = False, training: bool = True):
		x = self.resnet(x)
		x = nn.Dense(features=32)(x)
		x = jnp.concatenate((x,y),axis=1)
		for features in self.net_arch[: -1]:
			x = nn.Dense(features=features, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
			if self.batch_norm:
				x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
			if self.layer_norm:
				x = nn.LayerNorm()(x)

			x = self.activation_fn(x)
			x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)

		if len(self.net_arch) > 0:
			x = nn.Dense(features=self.net_arch[-1], kernel_init=self.kernel_init, bias_init=self.bias_init)(x)

		if self.squash_output:
			return nn.tanh(x)
		else:
			return x
		