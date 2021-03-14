# Explainable AI (XAI)
This repository is an attempt to explain the predictions made by Deep Learning models.
Deep Learning models are considered black box models as we often do not know why a prediction was made. With the advancements 
in Deep Learning it is important that we develop methods that can be used to explain the predictions
made by complex models. 
For the purpose of explaining the predictions the [shap](https://shap.readthedocs.io/en/latest/) package is used. This package is based
on calculating the shapley values for the features.

## XAI in NLP 
The currently available techniques for explanations in NLP computes relevance of single words only.
The goal of this project is to devise techniques for NLP tasks that can be used to compute the
relevance of a group of words, phrases etc. 
The project combines the concepts of Constituency parsing with Deep neural networks to build 
an interpretable model. Shap is then run to generate the explanations for the predictions.

## Results
The results are quite interesting. Instead of having one word as explanations, we can see the word along with its context as explanations.
![Alt text](images/grammar1.png?raw=true)

![Alt text](images/grammar2.png?raw=true)

> Note: This project was done as a part of XAI lab course at TUM.