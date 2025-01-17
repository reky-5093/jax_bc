from typing import Tuple

import jax
from jax import numpy as jnp

from jaxbc.modules.common import Model
from jaxbc.modules.type_aliases import PRNGKey

@jax.jit
def mlp_forward(
	rng: jnp.ndarray,
	model: Model,
	observations: jnp.ndarray,
) -> Tuple[PRNGKey, jnp.ndarray]:

	rng, dropout_key = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params},
		observations=observations,
		rngs={"dropout": dropout_key},
		deterministic=True,
		training=False
	)
	return rng, prediction


@jax.jit
def image_mlp_forward(
	rng: jnp.ndarray,
	model: Model,
	observations: jnp.ndarray,
	states :jnp.ndarray,
) -> Tuple[PRNGKey, jnp.ndarray]:

	rng, dropout_key = jax.random.split(rng)
	prediction = model.apply_fn(
		{"params": model.params,
   		'batch_stats': model.batch_stats},
		observations=observations,
		states =states,
		rngs={"dropout": dropout_key},
		deterministic=True,
		training=False,
	)
	return rng, prediction

