from typing import Any, Tuple

from src.model.ctb.training import CTBTrainer
from src.model.ctb.initialize import CTBInit
from src.model.ctb.inference import CTBInference
from src.model.ctb.explainer import CTBExplainer

from src.base.model.pipeline import ModelPipeline

class CTBPipeline(ModelPipeline, CTBTrainer, CTBExplainer, CTBInference):
    def __init__(self, 
            experiment_name:str, 
            params_ctb: dict[str, Any],
            metric_eval: str,
            config_dict: dict[str, Any], data_columns: Tuple[str],
            exclude_feature_list: list[str] = [],
            log_evaluation:int =1, fold_name: str = 'fold_info', 
            evaluate_stability: bool=False, evaluate_shap: bool=False
        ):
        CTBInit.__init__(
            self, experiment_name=experiment_name, params_ctb=params_ctb,
            metric_eval=metric_eval, config_dict=config_dict,
            data_columns=data_columns, exclude_feature_list=exclude_feature_list,
            log_evaluation=log_evaluation, fold_name=fold_name
        )
        self.evaluate_stability: bool = evaluate_stability
        self.evaluate_shap: bool = evaluate_shap
        
    def activate_inference(self) -> None:
        self.load_model()
        self.inference = True
        
    def run_train(self) -> None:
        self.train()
        self.save_model()
        
    def explain_model(self) -> None:
        self.evaluate_score()
        self.get_feature_importance()
        self.get_oof_prediction()
        self.get_oof_insight()
        
    def train_explain(self) -> None:
        self.create_experiment_structure()
        self.run_train()
        self.explain_model()