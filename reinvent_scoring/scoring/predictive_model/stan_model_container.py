import os
import json

import pandas as pd
import numpy as np

from typing import List, Dict
from reinvent_chemistry.descriptors import Descriptors
from reinvent_scoring.scoring.predictive_model.base_model_container import BaseModelContainer

from scipy.stats import bernoulli

class StanModelContainer(BaseModelContainer):
    def __init__(self, activity_model, specific_parameters):
        """
        :type activity_model: stan type of model object
        :type model_type: can be "classification" or "regression"
        """
        self._activity_model = activity_model
        #self._model_type = model_type
        self._molecules_to_descriptors = self._load_descriptor(specific_parameters) #ecfp_counts
        self._selected_feat_idx = specific_parameters["selected_feat_idx"]
        self._json_dir = specific_parameters["json_dir"]
        self._hitl_step = specific_parameters["hitl_step"]

    def predict(self, molecules: List, parameters: Dict) -> np.array:
        """
        Takes as input RDKit molecules and uses a stan model to predict activities.
        :param molecules: This is a list of rdkit.Chem.Mol objects
        :param parameters: Those are descriptor-specific parameters.
        :return: numpy.array with activity predictions
        """
        return self.predict_from_mols(molecules, parameters)

    def predict_from_mols(self, molecules: List, parameters: dict):
        if len(molecules) == 0:
            return np.empty([])
        fps = self._molecules_to_descriptors(molecules, parameters)
        activity = self.predict_from_fingerprints(fps)
        return activity

    def predict_from_fingerprints(self, fps):
        activity = self.predict_proba(fps)

        return activity

    def predict_proba(self, fps):
        activity = self.calculate_predictions(fps)

        return activity

    def calculate_predictions(self, fps):
        np.random.seed(125235)

        fps = np.array(fps)

        if self._selected_feat_idx != "none":
            fps = fps[:, self._selected_feat_idx]

        N = fps.shape[0]
        D = fps.shape[1]

        fit = self._activity_model["fit"]

        la = fit.extract(permuted=True)

        df_trace_beta = pd.DataFrame(la["beta"])
        df_trace_alpha = pd.DataFrame(la["alpha"])

        # draw parameter values from posterior distributions (TODO: get thetas that minimize entropy)
        beta_samples = dict()
        for i in range(len(df_trace_beta.columns)):
            beta_samples[i] = df_trace_beta.iloc[:,i].sample(1).item()
        alpha_sample = df_trace_alpha.iloc[:,0].sample(1).item()

        predicted_prob = []

        for n in range(len(fps)):
            sum = 0
            # iterate over coefficients
            for i in range(D):
                sum += beta_samples[i] * fps[n,i]
            sum += alpha_sample
            predicted_prob.append(self._inv_logit(sum))

        compute_dists = True
        # compute predictive posterior distributions
        if compute_dists:
            prob_dists = []
            # for each molecule to be predicted, we get a distribution of predicted labels from 
            # the different samples of the model parameters
            for n in range(len(fps)):
                preds_per_mol = []
                for i in range(len(df_trace_beta)):
                    beta_vec_i = df_trace_beta.iloc[i]
                    if len(df_trace_alpha) == len(df_trace_beta):
                        alpha_i = df_trace_alpha.iloc[i].values[0]
                    z_i = fps[n,:].dot(beta_vec_i) + alpha_i
                    p_i = self._inv_logit(z_i)
                    label_i = bernoulli.rvs(p_i, size = 1).item()
                    preds_per_mol.append(label_i)
                if len(preds_per_mol) == len(df_trace_beta) == len(df_trace_alpha):
                    if len(np.unique(preds_per_mol)) > 1: # if the two classes are predicted
                        pos_labels = np.unique(preds_per_mol, return_counts = True)[1][1] # for label == 1
                        neg_labels = np.unique(preds_per_mol, return_counts = True)[1][0] # for label == 0
                        pos_percent = pos_labels / len(preds_per_mol)
                        neg_percent = neg_labels / len(preds_per_mol)
                    else:
                        if np.unique(preds_per_mol, return_counts = True)[0].item() == 1:
                            pos_labels = np.unique(preds_per_mol, return_counts = True)[1].item()
                            pos_percent = pos_labels / len(preds_per_mol)
                            neg_percent = 0.0
                        if np.unique(preds_per_mol, return_counts = True)[0].item() == 0:
                            neg_labels = np.unique(preds_per_mol, return_counts = True)[1].item()
                            neg_percent = neg_labels / len(preds_per_mol)
                            pos_percent = 0.0

                prob_dists.append((neg_percent, pos_percent))
            
            #with open(f"{self._json_dir}/{self._hitl_step}.json", "w") as write_json_file:
            #    json.dump(prob_dists, write_json_file, indent = 4)

            prob_dists = np.array(prob_dists)

        return predicted_prob, prob_dists

    def _load_descriptor(self, parameters: {}):
        descriptors = Descriptors()
        descriptor = descriptors.load_descriptor(parameters)
        return descriptor

    # logistic regression
    def _inv_logit(self, x):
        return np.exp(x) / (1+np.exp(x))