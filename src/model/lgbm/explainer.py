import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Union
from src.model.lgbm.initialize import LgbmInit

class LgbmExplainer(LgbmInit):       
    def plot_train_curve(self, 
            progress_df: pd.DataFrame, 
            variable_to_plot: Union[str, list], 
            name_plot: str, 
            best_epoch_lgb:int
        ) -> None:
        
        if isinstance(variable_to_plot, str):
            variable_to_plot = [variable_to_plot]
                        
        fig = plt.figure(figsize=(12,8))
        sns.lineplot(
            data=progress_df[['time'] + variable_to_plot].melt(
                id_vars='time',
                value_vars=variable_to_plot,
                var_name='metric_fold', value_name=self.metric_eval
            ), 
            x="time", y=self.metric_eval, hue='metric_fold'
        )
        plt.axvline(x=best_epoch_lgb, color='blue', linestyle='--')

        plt.title(f"Training plot curve of {self.metric_eval}")

        fig.savefig(
            os.path.join(self.experiment_path, f'{name_plot}.png')
        )
        plt.close(fig)

    def evaluate_score(self) -> None:    
        #load feature list
        self.load_used_feature()
        
        # Find best epoch
        self.load_progress_list()

        progress_dict = {
            'time': range(self.params_lgb['n_round']),
        }

        progress_dict.update(
                {
                    f"{self.metric_eval}_fold_{i}": self.progress_list[i]['valid'][self.metric_eval]
                    for i in range(self.n_fold)
                }
            )

        progress_df = pd.DataFrame(progress_dict)
        progress_df[f"average_{self.metric_eval}"] = progress_df.loc[
            :, [self.metric_eval in x for x in progress_df.columns]
        ].mean(axis =1)
        
        progress_df[f"std_{self.metric_eval}"] = progress_df.loc[
            :, [self.metric_eval in x for x in progress_df.columns]
        ].std(axis =1)

        best_epoch_lgb = int(progress_df[f"average_{self.metric_eval}"].argmin())
        best_score_lgb = progress_df.loc[
            best_epoch_lgb,
            f"average_{self.metric_eval}"
        ]
        lgb_std = progress_df.loc[
            best_epoch_lgb, f"std_{self.metric_eval}"
        ]

        print(f'Best epoch: {best_epoch_lgb}, CV-L1: {best_score_lgb:.5f} ± {lgb_std:.5f}')

        self.best_result = {
            'best_epoch': best_epoch_lgb+1,
            'best_score': best_score_lgb
        }
        #plot cv score
        self.plot_train_curve(
            progress_df=progress_df, 
            variable_to_plot=f'average_{self.metric_eval}', name_plot='average_training_curve', 
            best_epoch_lgb=best_epoch_lgb
        )
        #plot std score
        self.plot_train_curve(
            progress_df=progress_df, 
            variable_to_plot=f'std_{self.metric_eval}', name_plot='std_training_curve', 
            best_epoch_lgb=best_epoch_lgb
        )
        #plot every fold score
        self.plot_train_curve(
            progress_df=progress_df, 
            variable_to_plot=[f'{self.metric_eval}_fold_{x}' for x in range(self.n_fold)], 
            name_plot='training_curve_by_fold', 
            best_epoch_lgb=best_epoch_lgb
        )
        
        self.save_best_result()
        
    def get_feature_importance(self) -> None:
   
        self.load_pickle_model_list()

        feature_importances = pd.DataFrame()
        feature_importances['feature'] = self.feature_list

        for fold_, model in enumerate(self.model_list):
            feature_importances[f'fold_{fold_}'] = model.feature_importance(
                importance_type='gain', iteration=self.best_result['best_epoch']
            )

        feature_importances['average'] = feature_importances[
            [f'fold_{fold_}' for fold_ in range(self.n_fold)]
        ].mean(axis=1)
        feature_importances = (
            feature_importances[['feature', 'average']]
            .sort_values(by='average', ascending=False)
        )
        
        fig = plt.figure(figsize=(12,8))
        sns.barplot(data=feature_importances.head(50), x='average', y='feature')
        plt.title(f"50 TOP feature importance over {self.n_fold} average")

        fig.savefig(
            os.path.join(self.experiment_path, 'importance_plot.png')
        )
        plt.close(fig)
        
        #feature importance excel
        feature_importances.to_excel(
            os.path.join(self.experiment_path, 'feature_importances.xlsx'),
            index=False
        )
        
    def get_oof_prediction(self) -> None:
        print('Warning: method not implemented')