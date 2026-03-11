# Standard library imports
import time
from datetime import datetime


# Third-party library imports
import jax
import kfac_jax
import numpy as np
import ml_collections
from jax import numpy as jnp
from absl import logging


# Local module imports
from src.utils import utils
from src import checkpoint, pretrain, constants
from src.trainer import train, inference
from src.initialization import initializer


def main(cfg: ml_collections.ConfigDict):
    logging.set_verbosity(logging.INFO)  # Logging level
    logging.info(cfg)
    ############################################################ 
    #                                                          #
    #                      Initialization                      #
    #                                                          #
    ############################################################
    # Random seed and key
    if cfg.debug.deterministic:
        seed = 23
        logging.info(f"DEBUG mode enabled, using a fixed random number seed={seed}. "
                    f"This will be overridden when checkpoint loaded.")
    else:
        seed = int(time.time() * 1e6)
    key = jax.random.PRNGKey(seed)

    # Create QMCInitializer
    init = initializer.QMCInitializer(cfg, key)
    # Input data
    data = init.init_data()
    # Model and parameters
    network, params = init.init_model(data)
    num_params = init.get_param_size(params)
    logging.info(f"Number of total network parameters: {num_params}")
    # Network functions
    batch_network, batch_orbitals, batch_det_weights, get_inter_value, mcmc_step, loss_fn = init.make_functions(network)
    # Optimizer
    train_step, opt_state, sharded_key = init.init_optimizer(
        params, data, mcmc_step, loss_fn)  # if cfg.mode == 'inferce', train_step is in fact infer_step

    # Other arguments
    iteration = 0  # Number of completed training steps
    adapt_frequency = cfg.mcmc.adapt_frequency
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(
        cfg.mcmc.move_width)
    pmoves = np.zeros(adapt_frequency)
    beta = 0.9  # EWMA weight

    ############################################################ 
    #                                                          #
    #                    Checkpoint Settings                   #
    #                                                          #
    ############################################################
    # Make checkpoint example
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    ckp_example = checkpoint.CheckpointData(
        iteration = iteration,
        params = params,
        data = data,
        opt_state = opt_state,
        key = subkeys,
        mcmc_width = mcmc_width,
        loss_ewma = 0.0,
        pmoves = pmoves
        )
    del subkeys

    # Restore datas from checkpoint file
    ckp_data = checkpoint.restore_from_ckp(
        restore_path = cfg.log.restore_path,
        ckp_example = ckp_example
        )
    params = ckp_data.params
    iteration = ckp_data.iteration
    data = ckp_data.data
    opt_state = ckp_data.opt_state
    key = ckp_data.key
    mcmc_width = ckp_data.mcmc_width
    loss_ewma = ckp_data.loss_ewma
    pmoves = ckp_data.pmoves
            
    ############################################################ 
    #                                                          #
    #                      Main iterations                     #
    #                                                          #
    ############################################################
    if iteration == 0:  # A new training/infernce run
        if cfg.mode == 'training':  # A new training run
            logging.info(f"Training mode: will start {cfg.optim.iterations} training "
                        f"steps with optimizer: {cfg.optim.optimizer}")
            if cfg.pretrain.iterations > 0:
                ############################################################ 
                #                                                          #
                #                  Hartree-Fock Pretraining                #
                #                                                          #
                ############################################################
                logging.info(f"Start {cfg.pretrain.iterations} Hartree-Fock pretraining steps:")
                hartree_fock = pretrain.get_hf(
                    molecule = cfg.system.molecule,
                    nspins = init.nspins,
                    basis = cfg.pretrain.basis,
                    pyscf_mol = cfg.system.get('pyscf_mol'),
                    restricted = False
                    )  # Get Hartree-Fock solutions
                sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
                params, data = pretrain.pretrain_hartree_fock(
                    params = params,
                    data = data,
                    batch_network = batch_network,
                    batch_orbitals = batch_orbitals,
                    sharded_key = subkeys,
                    nspins = init.nspins,
                    scf_approx = hartree_fock,
                    iterations = cfg.pretrain.iterations,
                    batch_size = init.device_batch_size,
                    scf_fraction = cfg.pretrain.scf_fraction,
                    mcmc_width = 0.02
                    )
                del subkeys
                logging.info("Hartree-Fock pretraining completed.")
        
            ############################################################ 
            #                                                          #
            #                       MCMC Burn-in                       #         
            #                                                          #
            ############################################################
            if cfg.mcmc.burn_in > 0:
                logging.info(f"Start MCMC burn-in ({cfg.mcmc.burn_in} steps):")
                p_mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
                for i in range(1, cfg.mcmc.burn_in + 1):
                    sharded_key, mcmc_keys = kfac_jax.utils.p_split(sharded_key)
                    data, pmove = p_mcmc_step(params, data, mcmc_keys, mcmc_width)
                    del mcmc_keys
                    if i % 10 == 0:
                        logging.info(f'{i}: pmove = {pmove[0].item():.4f}')
                logging.info("MCMC burn-in completed.")

    if cfg.mode == 'training':  # Training mode
        # Create save manager
        save_interval_steps = cfg.log.save_interval_steps  # Only save checkpoints for training
        save_mngr, save_path = checkpoint.create_save_mngr(
            save_path = cfg.log.save_path,
            save_interval_steps = save_interval_steps,
            max_to_keep = cfg.log.max_to_keep
            )
        ############################################################ 
        #                                                          #
        #                      Training Loop                       #
        #                                                          #
        ############################################################
        # Training start time
        start_timestamp = time.time()
        start_datetime = datetime.now()
        start_time_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"==================== Training Start [{start_time_str}] ====================")
        # Training iteration
        sharded_key = train.train(
            train_step = train_step,
            iter_start = iteration,
            iter_final = cfg.optim.iterations,
            params = params,
            data = data,
            opt_state = opt_state,
            key = sharded_key,
            mcmc_width = mcmc_width,
            pmoves = pmoves,
            adapt_frequency = adapt_frequency,
            loss_ewma = loss_ewma,
            beta = beta,
            save_interval_steps = save_interval_steps,
            save_mngr = save_mngr,
            show_params = cfg.log.show_params
            )
        # Training end time
        end_timestamp = time.time()
        end_datetime = datetime.now()
        end_time_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"==================== Training End [{end_time_str}] ====================")

        logging.info(f"{cfg.optim.iterations} training steps completed. All checkpoints saved in {save_path}.")

        # Report total training time (including compiling) and average speed
        total_duration = end_timestamp - start_timestamp  # in second
        total_steps = cfg.optim.iterations - iteration
        avg_time_per_step = total_duration / total_steps
            
        logging.info(f"Total training time: {total_duration:.2f} seconds (including compiling)")
        logging.info(f"Average speed: {avg_time_per_step:.3f} seconds / step")
        

    elif cfg.mode == 'inference':  # Inference mode
        ############################################################ 
        #                                                          #
        #                      Inference Loop                      #
        #                                                          #
        ############################################################
        logging.info("Inference mode: no checkpoint will be saved.")
        sharded_key = inference.inference(
            infer_step = train_step,
            iter_final = cfg.optim.iterations,
            params = params,
            data = data,
            key = sharded_key,
            mcmc_width = mcmc_width,
            pmoves = pmoves,
            adapt_frequency = adapt_frequency,
            beta = beta
            )

        logging.info(f"{cfg.optim.iterations} inference steps completed.")
    
    elif cfg.mode == 'analysis':  # Analysis mode
        ############################################################ 
        #                                                          #
        #                         Analysis                         #
        #                                                          #
        ############################################################
        logging.info("Analysis mode: no checkpoint will be saved.")
        # Show parameters
        params_local = jax.device_get(params)
        logging.info(f"parameters: {params_local}")
        # Show attention map
        inter_dict = jax.device_get(get_inter_value(
            params, data.positions, data.spins, data.atoms, data.charges)
            )
        # 1st electron sample, 1st attention layer
        pos = data.positions[0][0].reshape((-1, cfg.system.ndim))
        _, r_ee = utils.make_real_space_vector_diff(
            pos, pos, cfg.system.ndim)
        logging.info(f"Electron distances : \n{r_ee.squeeze(-1)}")

        # Show det weights
        det_weights = constants.pmap(batch_det_weights)(
            params, data.positions, data.spins, data.atoms, data.charges)
        logging.info(f"Det weights (Ave): {jnp.mean(constants.pmean(det_weights)[0], axis=0)}")  # 1st device

        '''
        attn_weights = inter_dict['intermediates']['PsiformerOrbitals']['psiformer_layers_0']['attention']['attn_weights'][0][0][0].transpose(2, 0, 1)
        num_heads = attn_weights.shape[0]

        # 添加最简洁的热力图绘制功能
        import matplotlib.pyplot as plt
        import os
        
        # 1. 为每个注意力头绘制热力图
        for i in range(num_heads):
            plt.figure(figsize=(6, 5))
            plt.imshow(attn_weights[i], cmap='viridis')
            plt.colorbar()
            plt.title(f'Attention Head {i}')
            plt.xlabel('Key Electron')
            plt.ylabel('Query Electron')
            
            # 保存图片
            os.makedirs("attention_maps", exist_ok=True)
            plt.savefig(f"attention_maps/head_{i}.png", dpi=200, bbox_inches='tight')
            plt.close()

        for i in range(num_heads):
            logging.info(f"head {i}: \n{attn_weights[i]}")
        '''

    else:
        raise NotImplementedError(f"Unknown mode: {cfg.mode}.")
    