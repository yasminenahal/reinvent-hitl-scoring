import os
import json
import math
import torch

import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict
from reinvent_chemistry.descriptors import Descriptors
from reinvent_scoring.scoring.predictive_model.base_model_container import BaseModelContainer


class TorchModelContainer(BaseModelContainer):
    def __init__(self, activity_model, specific_parameters):
        """
        :type activity_model: stan type of model object
        :type model_type: can be "classification" or "regression"
        """
        #self._model_type = model_type
        self._molecules_to_descriptors = self._load_descriptor(specific_parameters) #ecfp_counts
        self._selected_feat_idx = specific_parameters["selected_feat_idx"]
        self._activity_model = activity_model

    def predict(self, molecules: List, parameters: Dict, classifier_only) -> np.array:
        """
        Takes as input RDKit molecules and uses a stan model to predict activities.
        :param molecules: This is a list of rdkit.Chem.Mol objects
        :param parameters: Those are descriptor-specific parameters.
        :return: numpy.array with activity predictions
        """
        return self.predict_from_mols(molecules, parameters, classifier_only)

    def predict_from_mols(self, molecules: List, parameters: dict, classifier_only):
        if len(molecules) == 0:
            return np.empty([])
        fps = self._molecules_to_descriptors(molecules, parameters)
        activity = self.predict_from_fingerprints(fps, classifier_only)
        return activity

    def predict_from_fingerprints(self, fps, classifier_only):
        if self._selected_feat_idx != "none":
            fps = fps[:, self._selected_feat_idx]

        if classifier_only == True:

            final_preds = self._activity_model(torch.tensor(fps, dtype=torch.float32))
            final_preds = final_preds.cpu().detach().numpy()

        if classifier_only == False:

            preds, decision_outs = self._activity_model(torch.tensor(fps, dtype=torch.float32))

            boolean = (
            (decision_outs[:, 0] > preds[:, 0])
            * (
                (preds[:, 0] > 0.5)
                * (preds[:, 1] > preds[:, 0])
                + (preds[:, 0] < 0.5)
                * (preds[:, 1] < preds[:, 0])
            )
            ).float()
            final_preds = (boolean * preds[:, 1]) + (1 - boolean) * preds[:, 0]

            print(f"Percentage of Deferred : {boolean.mean()*100:.2f}")

            final_preds = final_preds.cpu().detach().numpy()

        return final_preds

    def _load_descriptor(self, parameters: {}):
        descriptors = Descriptors()
        descriptor = descriptors.load_descriptor(parameters)
        return descriptor


class NonLinearNetDeferSoftmax(nn.Module):
    def __init__(self, num_features, dropout):
        super(NonLinearNetDeferSoftmax, self).__init__()

        # define architecture for classifier 1
        self.classifier1 = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(
                512, 1
            ),  # output layer with one neuron for binary classification
            nn.Sigmoid(),  # sigmoid activation for probability output (g_y)
        )

        # define architecture for classifier 2 or human model (similar to classifier 1)
        self.classifier2 = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
            nn.Sigmoid(),  # sigmoid activation for probability output (g_h)
        )

        # define the architecture for the rejector
        self.decision_classifier = nn.Sequential(
            nn.Linear(
                2, 3
            ),  # input size is 2: takes in as input the outputs of classifier 1 and classifier 2
            nn.Softmax()
            #nn.Softmax(dim=1)  # softmax activation for probability distribution
            # in this way we have 3 output heads [g_y, g_h, g_perp]
        )

    def forward(self, x):
        # forward pass for classifier 1
        out1 = self.classifier1(x)

        # forward pass for classifier 2
        out2 = self.classifier2(x)

        # combine the outputs of classifier 1 and classifier 2
        combined_output = torch.cat((out1, out2), dim=1)

        # forward pass for the rejector to decide which classifier predicts
        final_output = self.decision_classifier(combined_output)

        # return combined_output from classifier1 and classifier 2 trained separately
        # and final output which is the probability distribution p(y=1,h=1,r=1)
        return combined_output, final_output


class NonLinearNetDefer(nn.Module):
    def __init__(self, num_features, dropout):
        super(NonLinearNetDefer, self).__init__()

        # define architecture for classifier 1
        self.classifier1 = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(
                512, 1
            ),  # output layer with one neuron for binary classification
            nn.Sigmoid(),  # sigmoid activation for probability output (g_y)
        )

        # define architecture for classifier 2 or human model (similar to classifier 1)
        self.classifier2 = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
            nn.Sigmoid(),  # sigmoid activation for probability output (g_h)
        )

        # define the architecture for the rejector
        self.decision_classifier = nn.Sequential(
            nn.Linear(
                2, 1
            ),  # input size is 2: takes in as input the outputs of classifier 1 and classifier 2
            nn.Sigmoid()
            #nn.Softmax(dim=1)  # softmax activation for probability distribution
            # in this way we have 3 output heads [g_y, g_h, g_perp]
        )

    def forward(self, x):
        # forward pass for classifier 1
        out1 = self.classifier1(x)

        # forward pass for classifier 2
        out2 = self.classifier2(x)

        # combine the outputs of classifier 1 and classifier 2
        combined_output = torch.cat((out1, out2), dim=1)

        # forward pass for the rejector to decide which classifier predicts
        final_output = self.decision_classifier(combined_output)

        # return combined_output from classifier1 and classifier 2 trained separately
        # and final output which is the probability distribution p(y=1,h=1,r=1)
        return combined_output, final_output
    

class ClassifierSimple(nn.Module):
    def __init__(self, num_features, dropout):
        super(ClassifierSimple, self).__init__()

        # define architecture for classifier 1
        self.classifier1 = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),  # output layer with one neuron for binary classification
            nn.Sigmoid(),  # sigmoid activation for probability output (g_y)
        )

    def forward(self, x):
        # forward pass for classifier 1
        out = self.classifier1(x)

        return out