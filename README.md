# endomondo_fun
Final Project for CSC2515 at University of Toronto, using the Endomondo dataset from Ni, et al [WWW 2019]


## Models
We train and evaluate two sequence-based models using the endomondo dataset:
 * An augmented the FitRec-Attn model from Ni, et al (found in [fitrec_attn](https://github.com/twkillian/endomondo_fun/tree/main/fitrec_attn) folder)
 * An implementation of a Temporal Convolutional Neural Network (found in [conv_experiments](https://github.com/twkillian/endomondo_fun/tree/main/conv_experiments) folder)

Each of these folders have examples of how we trained and evaluated these models as well as some generating some figures.

## Data
The endomondo dataset can be accessed from the Ni, et al repo: https://github.com/nijianmo/fit-rec

Our project focused on the short-term prediction task, meaning that we used the `endomondoHR_proper_temporal_dataset.pkl` file of sampled and filtered user workout data.

## Requirements
All models were built using Pytorch v. 1.7.0
