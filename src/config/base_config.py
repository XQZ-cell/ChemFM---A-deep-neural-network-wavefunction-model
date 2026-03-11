# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default base configuration for molecular VMC calculations."""

import enum

import ml_collections
from ml_collections import config_dict


class SystemType(enum.IntEnum):
  """Enum for system types.

  WARNING: enum members cannot be serialised readily so use
  SystemType.member.value in such cases.
  """
  MOLECULE = enum.auto()

  @classmethod
  def has_value(cls, value):
    return any(value is item or value == item.value for item in cls)


def default() -> ml_collections.ConfigDict:
  """Create set of default parameters for running qmc.py.

  Note: placeholders (cfg.system.molecule and cfg.system.electrons) must be
  replaced with appropriate values.

  Returns:
    ml_collections.ConfigDict containing default settings.
  """
  # wavefunction output.
  cfg = ml_collections.ConfigDict({
      'batch_size': 4096,  # batch size
      # Config module used. Should be set in get_config function as either the
      # absolute module or relative to the configs subdirectory. Relative
      # imports must start with a '.' (e.g. .atom). Do *not* override on
      # command-line. Do *not* set using __name__ from inside a get_config
      # function, as config_flags overrides this when importing the module using
      # importlib.import_module.
      'mode': 'training',  # or: 'inference', 'analysis'
      'config_module': __name__,
      'optim': {
          'iterations': 100000,  # number of iterations
          'optimizer': 'kfac',  # one of adam, kfac, lamb, none
          'laplacian': 'folx',  # of of default or folx (for forward lapl)
          # If 0, use standard vmap. If >0, the max batch size for batched_vmap
          'lr': {
              'rate': 0.05,  # learning rate
              'decay': 1.0,  # exponent of learning rate decay
              'delay': 10000.0,  # term that sets the scale of the rate decay
          },
          # If greater than zero, scale (at which to clip local energy) in units
          # of the mean deviation from the mean.
          'clip_local_energy': 5.0,
          # If true, center the clipping window around the median rather than
          # the mean. More "correct" for removing outliers, but also potentially
          # slow, especially with multihost training.
          'clip_median': True,
          # If true, center the local energy differences in the gradient at the
          # average clipped energy rather than average energy, guaranteeing that
          # the average energy difference will be zero in each batch.
          'center_at_clip': True,
          # KFAC hyperparameters. See KFAC documentation for details.
          'kfac': {
              'invert_every': 1,
              'cov_update_every': 1,
              'damping': 0.001,
              'cov_ema_decay': 0.95,
              'momentum': 0.0,
              'momentum_type': 'regular',
              # Warning: adaptive damping is not currently available.
              'min_damping': 1.0e-4,
              'norm_constraint': 0.001,
              'mean_center': True,
              'l2_reg': 0.0,
              'register_only_generic': False,
          },
          # ADAM hyperparameters. See optax documentation for details.
          'adam': {
              'b1': 0.9,
              'b2': 0.999,
              'eps': 1.0e-8,
              'eps_root': 0.0,
          },
          'reg_weight': 0.0,  # Weight of entropy regularization term.
      },
      'log': {
          'stats_frequency': 1,  # iterations between logging of stats
          'max_to_keep': 20, 
          'save_interval_steps': 5000,
          'save_path': '',
          # Path containing checkpoint to restore network from.
          # Ignored if falsy or save_path contains a checkpoint.
          'restore_path': '',
          # Remaining log options are currently not functional.  Whether or not
          # to log the values of all walkers every iteration Use with caution!!!
          # Produces a lot of data very quickly.
          'show_det_weights': False,
          'show_params': False, 
          'show_jastrow_factor_params': False,
      },
      'system': {
          'type': SystemType.MOLECULE.value,
          # Specify the system.
          # 1. Specify the system by setting variables below.
          # list of system.Atom objects with element type and position.
          'molecule': config_dict.placeholder(list),
          # number of spin up, spin-down electrons
          'electrons': tuple(),
          # Dimensionality. Change with care. FermiNet implementation currently
          # assumes 3D systems.
          'ndim': 3,
          # Units of *input* coords of atoms. Either 'bohr' or
          # 'angstrom'. Internally work in a.u.; positions in
          # Angstroms are converged to Bohr.
          'units': 'bohr',
      },
      'mcmc': {
          # Note: HMC options are not currently used.
          # Number of burn in steps after pretraining.  If zero do not burn in
          # or reinitialize walkers.
          'burn_in': 100,
          'steps': 30,  # Number of MCMC steps to make between network updates.
          # Width of (atom-centred) Gaussian used to generate initial electron
          # configurations.
          'init_width': 1.0,
          # Width of Gaussian used for random moves for RMW or step size for
          # HMC.
          'move_width': 0.02,
          # Number of steps after which to update the adaptive MCMC step size
          'adapt_frequency': 100,
          'proposal': 'random_walk', 
          'max_norm': 5.0, # Max norm of gradients of network output to electronic positions for MALA
          'sampler_params': {},
      },
      'network': {
          'network_type': 'psiformer',
          # Only used if network_type is 'psiformer'.
          'psiformer': {
              # PsiFormer architecture: von Glehn, Spencer, Pfau, ICLR 2023.
              'num_layers': 4,
              'num_heads': 4,
              'heads_dim': 64,
              'mlp_hidden_dims': 256,
              'use_layer_norm': True,
              'separate_spin_channels': True,
              'use_res': True,
              'use_gate': False,
              'use_edge_bias': False,
          },
          # Config common to all architectures.
          'determinants': 16,  # Number of determinants.
          # If true, determinants are dense rather than block-sparse
          'full_det': True,
          # If specified, include a pre-determinant Jastrow factor.
          # One of 'default' (use network_type default), 'none', or 'simple_ee'.
          'jastrow': 'simple',
          # If true, rescale the inputs so they grow as log(|r|)
          'rescale_inputs': True,
          'envelope': 'simple',
          'activation_fun': 'tanh',
          'bias_orbitals': True,
      },
      'debug': {
          'deterministic': False,  # Use a deterministic seed.
      },
      'pretrain': {
          'method': 'hf',  # Currently only 'hf' is supported.
          'iterations': 1000,  # Only used if method is 'hf'.
          'basis': 'ccpvdz',  # Larger than STO-6G, but good for excited states
          # Fraction of SCF to use in pretraining MCMC. This enables pretraining
          # similar to the original FermiNet paper.
          'scf_fraction': 0.5,
      },
  })

  return cfg


def resolve(cfg):
  """Resolve any ml_collections.config_dict.FieldReference values in a ConfigDict for qmc.

  Any FieldReferences in the coords array for each element in
  cfg.system.molecule are treated specially as nested references are not
  resolved by ConfigDict.copy_and_resolve_references. Similar cases should be
  added here as needed.

  Args:
    cfg: ml_collections.ConfigDict containing settings.

  Returns:
    ml_collections.ConfigDict with ml_collections.FieldReference values resolved
    (as far as possible).

  Raises:
    RuntimeError: If an atomic position is non-numeric.
  """
  if 'set_molecule' in cfg.system and callable(cfg.system.set_molecule):
    cfg = cfg.system.set_molecule(cfg)
    with cfg.ignore_type():
      # Replace the function with its name so we know how the molecule was set
      # This makes the ConfigDict object serialisable.
      if callable(cfg.system.set_molecule):
        cfg.system.set_molecule = cfg.system.set_molecule.__name__
  cfg = cfg.copy_and_resolve_references()
  return cfg
