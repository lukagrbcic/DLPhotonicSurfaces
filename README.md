Implementation of DL models (Tandem Neural Networks and Conditional Generative Adversarial Networks) for photonic surface inverse design used for comparison with the Multi-fidelity (MF) ensemble framework detailed in https://github.com/lukagrbcic/MFEnsemblePhotonicSurfaces and https://arxiv.org/abs/2406.01471.

_______
**Dataset Details**
_______

The dataset used to train the models can be downloaded at:  https://osf.io/dwgtf/

The dataset should be put in the main directory in a folder named **inconel_data**.

Further dataset details are given in the MF ensemble github repository: https://github.com/lukagrbcic/MFEnsemblePhotonicSurfaces

_______
**Tandem Neural Network Details**
_______

Tandem Neural Network (TNN) implementation can be found in the TNN folder. 

To reproduce the training, testing and the postprocessing of the TNN model, it is neccessary to run **train_test_tnn.py** script.

The forward Deep Neural Network (DNN) training is given in the script **train_forward.py**.

The DNN architecutre details for both the inverse and forward DNN are given in the **inverse_forward.py** file.

_______
**Conditional Generative Adversarial Networks Details**
_______

Conditional Generative Adversarial Networks  (cGAN) implementation can be found in the CGAN folder. 
