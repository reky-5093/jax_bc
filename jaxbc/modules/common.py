import os
import sys
from typing import Any, Optional, Tuple, Union, Callable, Sequence, TypeVar, List, Dict

from flax import traverse_util
T = TypeVar('T')

from flax.core import frozen_dict,FrozenDict
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import traverse_util
import optax
from jax_resnet import pretrained_resnet
from jax_resnet.common import  slice_variables

from jaxbc.modules.type_aliases import (
	PRNGKey,
	Shape,
	Dtype,
	Array,
	Params
)

def print_tree(d, depth=0, print_value=False):
    for k in d.keys():
        if isinstance(d[k], Dict):
            print('  ' * depth, k)
            print_tree(d[k], depth + 1, print_value)
        else:
            if print_value:
                print('  ' * depth, k, d[k])
            else:
                print('  ' * depth, k)



def create_mask(params, label_fn):
    def _map(params, mask, label_fn):
        for k in params:
            if label_fn(k):
                mask[k] = 'zero'
            else:
                if isinstance(params[k], Dict):
                    mask[k] = {}
                    _map(params[k], mask[k], label_fn)
                else:
                    mask[k] = 'adam'
    mask = {}
    _map(params, mask, label_fn)
    return frozen_dict.freeze(mask)

def zero_grads():
    # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)


@flax.struct.dataclass
class Model:
	step: int
	apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
	params: Params
	batch_stats: Union[Params]
	tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
	opt_state: Optional[optax.OptState] = None
	# model_cls: Type = None


	@classmethod
	def create(
		cls,
		model_def: [nn.Module],
		inputs: Sequence[jnp.ndarray],
		tx: Optional[optax.GradientTransformation] = None,
		**kwargs
	) -> 'Model':

		variables = model_def.init(*inputs)

		_, params = variables.pop('params')
		"""
		NOTE:
			Here we unfreeze the parameter. 
			This is because some optimizer classes in optax must receive a dict, not a frozendict, which is annoying.
			https://github.com/deepmind/optax/issues/160
			And ... if we can access to the params, then why it should be freezed ? 
		"""
		params = params.unfreeze()


		if tx is not None:
			opt_state = tx.init(params)
		else:
			opt_state = None

		
		# NOTE : Unfreeze the parameters !!!!!


		# Frozendict's 'pop' method does not support default value. So we use get method instead.
		
		batch_stats = variables.get("batch_stats", None)


		return cls(
			step=1,
			apply_fn=model_def.apply,
			params=params,
			batch_stats=batch_stats,
			tx=tx,
			opt_state=opt_state,
			**kwargs
		)

	def __call__(self, *args, **kwargs):
		return self.apply_fn({"params": self.params, }, *args, **kwargs)

	def apply_gradient(
		self,
		loss_fn: Optional[Callable[[Params], Any]] = None,
		grads: Optional[Any] = None,
		has_aux: bool = True
	) -> Union[Tuple['Model', Any], 'Model']:

		assert ((loss_fn is not None) or (grads is not None), 'Either a loss function or grads must be specified.')

		if grads is None:
			grad_fn = jax.grad(loss_fn, has_aux=has_aux)
			if has_aux:
				grads, aux = grad_fn(self.params)
			else:
				grads = grad_fn(self.params)
		else:
			assert (has_aux, 'When grads are provided, expects no aux outputs.')

		updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
		new_params = optax.apply_updates(self.params, updates)
		new_model = self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state)

		if has_aux:
			return new_model, aux
		else:
			return new_model

	def save_dict_from_path(self, save_path: str) -> Params:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		with open(save_path, 'wb') as f:
			f.write(flax.serialization.to_bytes(self.params))
		return self.params

	def load_dict_from_path(self, load_path: str) -> "Model":
		with open(load_path, 'rb') as f:
			params = flax.serialization.from_bytes(self.params, f.read())
		return self.replace(params=params)

	def save_batch_stats_from_path(self, save_path: str) -> Params:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		with open(save_path, 'wb') as f:
			f.write(flax.serialization.to_bytes(self.batch_stats))
		return self.batch_stats

	def load_batch_stats_from_path(self, load_path: str) -> "Model":
		with open(load_path, 'rb') as f:
			batch_stats = flax.serialization.from_bytes(self.batch_stats, f.read())
		return self.replace(batch_stats=batch_stats)

	def load_dict(self, params: bytes) -> 'Model':
		params = flax.serialization.from_bytes(self.params, params)
		return self.replace(params=params)


@flax.struct.dataclass
class Img_Model:
	step: int
	apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
	params: Params
	batch_stats: Union[Params]
	tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
	opt_state: Optional[optax.OptState] = None
	# model_cls: Type = None


	@classmethod
	def create(
		cls,
		model_def: [nn.Module],
		variables,
		tx: Optional[optax.GradientTransformation] = None,
		**kwargs
	) -> 'Model':

		params = variables["params"]
		if tx is not None:
			opt_state = tx.init(params)
		else:
			opt_state = None

		# NOTE : Unfreeze the parameters !!!!!
		# Frozendict's 'pop' method does not support default value. So we use get method instead.
		batch_stats = variables.get("batch_stats", None)

		return cls(
			step=1,
			apply_fn=model_def.apply,
			params=params,
			batch_stats=batch_stats,
			tx=tx,
			opt_state=opt_state,
			**kwargs
		)

	def __call__(self, *args, **kwargs):
		return self.apply_fn({"params": self.params , 'batch_stats': self.batch_stats}, *args, **kwargs)

	def apply_gradient(
		self,
		loss_fn: Optional[Callable[[Params], Any]] = None,
		grads: Optional[Any] = None,
		has_aux: bool = True
	) -> Union[Tuple['Model', Any], 'Model']:

		assert ((loss_fn is not None) or (grads is not None), 'Either a loss function or grads must be specified.')

		if grads is None:
			grad_fn = jax.grad(loss_fn, has_aux=has_aux)
			if has_aux:
				#print(jax.tree_map(jnp.shape, self.params))
				grads, aux = grad_fn(self.params, self.batch_stats)
			else:
				grads = grad_fn(self.params , self.batch_stats)
		else:
			assert (has_aux, 'When grads are provided, expects no aux outputs.')

			
		updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
		new_params = optax.apply_updates(self.params, updates)
		new_model = self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state ,batch_stats = aux["batch_stats"])

		if has_aux:
			return new_model, aux
		else:
			return new_model

	def save_dict_from_path(self, save_path: str) -> Params:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		with open(save_path, 'wb') as f:
			f.write(flax.serialization.to_bytes(self.params))
		return self.params

	def load_dict_from_path(self, load_path: str) -> "Model":
		with open(load_path, 'rb') as f:
			params = flax.serialization.from_bytes(self.params, f.read())
		return self.replace(params=params)

	def save_batch_stats_from_path(self, save_path: str) -> Params:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		with open(save_path, 'wb') as f:
			f.write(flax.serialization.to_bytes(self.batch_stats))
		return self.batch_stats

	def load_batch_stats_from_path(self, load_path: str) -> "Model":
		with open(load_path, 'rb') as f:
			batch_stats = flax.serialization.from_bytes(self.batch_stats, f.read())
		return self.replace(batch_stats=batch_stats)

	def load_dict(self, params: bytes) -> 'Model':
		params = flax.serialization.from_bytes(self.params, params)
		return self.replace(params=params)
