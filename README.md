# Orbital-Koopman

This repository contains all the source code from the paper *Deep Learning Based Dynamics Identification and Linerization of Orbital Problems using Koopman Theory*.

The code is all written in Jupyter Notebook and the pretrained models are provided as .pt files which can be loaded in without the need for training (run all cells except the training cell). The code is organised as follows:

- CR3BP
    - *CR3BP_Data_Generator.ipynb* : Generates all the training data required to train the CR3BP model.
    - *CR3BP_RK4.ipynb* : Trains the Koopman model on the CR3BP dataset and provides all the plots and metric analysis from the paper.
      
- Two Body Problem
    - *Data_Generator.ipynb* : This creates the datasets for all three simulation types, purely circular, ellpitical or perturbed 2BP. In order to create circular 2BP datasets, ensure the 
                               *perturb* variable is set to 0 and the eccentricity is also set to 0. To create elliptical datasets ensure that the *perturb* variable is set to 0 and the 
                               eccentricity is also set to whatever value you wish to use. To create perturbed datasets ensure that the *perturb* variable is set to 1 and the eccentricity is also 
                               set to 0.
      
    - *Circular_RK4.ipynb* : Trains the Koopman model on the purely circular or elliptical dataset and provides all the plots and metric analysis from the paper. It also contains the simulations   
                             for the Moon and Jupiter centered 2BP. In order to simulate the purely circular trajectories the model must be trained on the circular datasets (X_Data.pt etc.) and 
                             the eccentricities set to zero for the prediction post training. If you would like to see the elliptical model, you must load in the elliptical datasets (X_DataE.pt 
                             etc.) and set the corresponding eccentricity in the post training prediction loop. We have pretrained models for various eccentricities as seen in the paper (the 
                             suffix 01, 02, 05, 08, designate the eccentricity used to generate that model.)
      
    - *Circular_RK4_perturb.ipynb* : Trains the Koopman model on the 2BP with J2 and SRP perturbations and provides all the associated plots and metric analysis from the paper. This model does not 
                                     contain simulations around the Moon and Jupiter as the J2 and SRP perturbations are only accounted for the Earth model. In order to ensure that the nonlinear 
                                     dynamics use the perturbations, in the post training simulation ensure that the *perturb* variable is set to 1. 
