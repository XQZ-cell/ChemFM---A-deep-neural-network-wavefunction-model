
# ChemFM

A demonstration platform for training a deep neural network wavefunction model using Variational Monte Carlo (VMC). Given a molecular structure, the model automatically generates electronic configurations via Monte Carlo sampling and estimates the ground-state energy of the molecule. Our long-term goal is to build a foundation wavefunction model pretrained on diverse molecules, capable of fine-tuning for accurate predictions on unseen molecular systems.

Current features include:
 - Multi-device deployment and training of wavefunction models
 - Standard Markov Chain Monte Carlo (MCMC) sampling algorithms
 - Hartree–Fock pretraining
 - Inference for predicting molecular ground-state energy