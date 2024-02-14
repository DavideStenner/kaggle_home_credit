from typing import Any

from src.model.lgbm.training import LgbmTrainer
from src.model.lgbm.explainer import LgbmExplainer
from src.model.lgbm.initialize import LgbmInit
from src.model.lgbm.inference import LgbmInference

from src.base.model.pipeline import ModelPipeline

class LgbmPipeline(ModelPipeline, LgbmTrainer, LgbmExplainer, LgbmInference):
    def __init__(self, 
            experiment_name:str, 
            params_lgb: dict[str, Any],
            metric_eval: str,
            config_dict: dict[str, Any], inference_setup: str=None,
            log_evaluation:int =1, fold_name: str = 'fold_info', 
        ):
        LgbmInit.__init__(
            self, experiment_name=experiment_name, params_lgb=params_lgb,
            metric_eval=metric_eval, config_dict=config_dict, inference_setup=inference_setup,
            log_evaluation=log_evaluation, fold_name=fold_name
        )

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
        
    def train_explain(self) -> None:
        self.create_experiment_structure()
        self.run_train()
        self.explain_model()