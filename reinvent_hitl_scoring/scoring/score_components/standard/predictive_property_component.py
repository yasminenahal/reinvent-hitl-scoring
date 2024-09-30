import torch
import pickle

from typing import List

from reinvent_hitl_scoring.scoring.predictive_model.model_container import ModelContainer
from reinvent_hitl_scoring.scoring.predictive_model.torch_model_container import NonLinearNetDefer, ClassifierSimple
from reinvent_hitl_scoring.scoring.component_parameters import ComponentParameters
from reinvent_hitl_scoring.scoring.score_components import BaseScoreComponent
from reinvent_hitl_scoring.scoring.score_summary import ComponentSummary
from reinvent_hitl_scoring.scoring.score_transformations import TransformationFactory
from reinvent_hitl_scoring.scoring.enums import TransformationTypeEnum, TransformationParametersEnum

scikit = True
stan = False
torch_model = False

class PredictivePropertyComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.activity_model = self._load_model(parameters)
        self._transformation_function = self._assign_transformation(parameters.specific_parameters)
        self.model_path = self.parameters.specific_parameters.get(self.component_specific_parameters.MODEL_PATH, "")

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary:
        if scikit:
            score, raw_score = self._predict_and_transform(molecules)
            score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        if torch_model:
            score, raw_score = self._predict_and_transform(molecules)
            score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        if stan:
            score, raw_score, prob_dists = self._predict_and_transform(molecules)
            score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score, prob_dists=prob_dists)
        return score_summary

    def _predict_and_transform(self, molecules: List, scikit = scikit):
        if scikit:
            score = self.activity_model.predict(molecules, self.parameters.specific_parameters)
            transformed_score = self._apply_transformation(score, self.parameters.specific_parameters)
            return transformed_score, score
        if torch_model:
            if self.model_path == "models/simple_classifier.pt":
                score = self.activity_model.predict(molecules, self.parameters.specific_parameters, classifier_only = True)
            else:
                score = self.activity_model.predict(molecules, self.parameters.specific_parameters, classifier_only = False)
            transformed_score = self._apply_transformation(score, self.parameters.specific_parameters)
            return transformed_score, score
        if stan:
            score, prob_dists = self.activity_model.predict(molecules, self.parameters.specific_parameters)
            transformed_score = self._apply_transformation(score, self.parameters.specific_parameters)
            return transformed_score, score, prob_dists

    def _load_model(self, parameters: ComponentParameters):
        #try:
        activity_model = self._load_container(parameters)
        #except:
        #    model_path = self.parameters.specific_parameters.get(self.component_specific_parameters.MODEL_PATH, "")
        #    raise Exception(f"The loaded file `{model_path}` isn't a valid model")
        return activity_model

    def _load_container(self, parameters: ComponentParameters):
        model_path = self.parameters.specific_parameters.get(self.component_specific_parameters.MODEL_PATH, "")
        
        if not scikit:
            #initialize
            out_dim = self.parameters.specific_parameters.get(self.component_specific_parameters.OUT_DIM, "")
            dropout = self.parameters.specific_parameters.get(self.component_specific_parameters.DROPOUT, "")
            input_dim = self.parameters.specific_parameters.get(self.component_specific_parameters.SIZE, "")
            layers = self.parameters.specific_parameters.get(self.component_specific_parameters.LAYERS, "")
            
            model = NonLinearNetDefer(
                input_dim,
                #out_dim,
                #layers,
                dropout
            )

            if "models/simple_classifier.pt" in model_path:
                model = ClassifierSimple(input_dim, dropout)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
        if scikit:
            model = pickle.load(open(model_path, "rb"))
        packaged_model = ModelContainer(model, parameters.specific_parameters)
        return packaged_model

    def _apply_transformation(self, predicted_activity, parameters: dict):
        transform_params = parameters.get(self.component_specific_parameters.TRANSFORMATION)
        if transform_params:
            activity = self._transformation_function(predicted_activity, transform_params)
        else:
            activity = predicted_activity
        return activity

    def _assign_transformation(self, specific_parameters: dict):
        transformation_type = TransformationTypeEnum()
        transform_params = specific_parameters.get(self.component_specific_parameters.TRANSFORMATION)
        if not transform_params:
            specific_parameters[self.component_specific_parameters.TRANSFORMATION] = {
                    TransformationParametersEnum.TRANSFORMATION_TYPE: transformation_type.NO_TRANSFORMATION
                }
        factory = TransformationFactory()
        transform_function = factory.get_transformation_function(transform_params)
        return transform_function
