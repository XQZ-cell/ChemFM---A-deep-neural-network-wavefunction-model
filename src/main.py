# Local module imports
from src.config import base_config
from src.trainer.train import train
from src.utils import system


############################################################ 
#                                                          #
#                   Configuration Setting                  #
#                                                          #
############################################################
cfg = base_config.default()
# Set up molecule spins, atomic types and positions
cfg.system.electrons = (1, 1)
atoms_angstrom = [
    ('H', [- 0.5, 0.0, 0.0]),
    ('H', [0.5, 0.0, 0.0])
    ]  # unit: Ang

ang_to_bohr = 1.0 / 0.529177

cfg.system.molecule = []
for symbol, coords_ang in atoms_angstrom:
    coords_bohr = [coord * ang_to_bohr for coord in coords_ang]
    cfg.system.molecule.append(system.Atom(symbol, coords_bohr))

# Debug mode settngs
cfg.debug.deterministic = False

# Mode
cfg.mode = 'training'

# Network settings
cfg.batch_size = 256
cfg.optim.laplacian = 'folx'
cfg.network.psiformer.separate_spin_channels = True
cfg.network.determinants = 16
cfg.network.psiformer.num_layers = 4
cfg.network.psiformer.mlp_hidden_dims = 256

# MCMC settings
#cfg.mcmc.init_width = 1.0
cfg.mcmc.proposal = 'random_walk'
cfg.mcmc.move_width = 0.02
cfg.mcmc.burn_in = 100
cfg.mcmc.adapt_frequency = 100
#cfg.mcmc.max_norm = 5.0

# Training settings
cfg.optim.optimizer = 'kfac'
cfg.pretrain.iterations = 100
cfg.optim.iterations = 100
#cfg.optim.reg_weight = 1e-3  # use entropy regularization

# Checkpoint settings
cfg.log.max_to_keep = 10
cfg.log.save_interval_steps = 100
cfg.log.restore_path = ""
cfg.log.save_path = "/home/yinchang/ChemFM/checkpoints"


def main():
    train(cfg)