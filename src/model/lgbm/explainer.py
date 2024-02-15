import os
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Union
from sklearn.metrics import roc_auc_score
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
            os.path.join(
                self.experiment_insight_path, f'{name_plot}.png'
            )
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

        best_epoch_lgb = int(progress_df[f"average_{self.metric_eval}"].argmax())
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
        
        #plain feature
        fig = plt.figure(figsize=(12,8))
        sns.barplot(data=feature_importances.head(50), x='average', y='feature')
        plt.title(f"50 TOP feature importance over {self.n_fold} average")

        fig.savefig(
            os.path.join(self.experiment_insight_path, 'importance_plot.png')
        )
        plt.close(fig)
        
        #feature importance excel
        feature_importances.to_excel(
            os.path.join(self.experiment_path, 'feature_importances.xlsx'),
            index=False
        )
        #add dataset
        feature_importances_dataset = feature_importances.merge(
            self.feature_dataset, how='left',
            on='feature'
        )

        feature_importances_dataset['rank_average'] = feature_importances_dataset['average'].rank(ascending=False)

        #plain feature top dataset
        fig = plt.figure(figsize=(12,8))
        plot_ = sns.barplot(
            data=feature_importances_dataset, 
            x='rank_average', y='average', hue='dataset', 
        )
        plot_.set(xticklabels=[])
        plt.title(f"Rank Top feature by dataset")
        
        fig.savefig(
            os.path.join(self.experiment_insight_path, 'top_feature_by_dataset.png')
        )
        plt.close(fig)

        #for each dataset print top feature
        for dataset_name in feature_importances_dataset['dataset'].unique():
            fig = plt.figure(figsize=(12,8))
            temp_dataset_feature = feature_importances_dataset.loc[
                feature_importances_dataset['dataset'] == dataset_name
            ]
            sns.barplot(data=temp_dataset_feature.head(50), x='average', y='feature')
            plt.title(f"50 TOP feature importance over {self.n_fold} average for {dataset_name}")

            fig.savefig(
                os.path.join(
                    self.experiment_insight_path, 
                    f'importance_plot_{dataset_name}.png'
                )
            )
            plt.close(fig)

        #get information about top dataset on mean gai and rank gain
        feature_importances_dataset = feature_importances_dataset.groupby(
            'dataset'
        )[['average', 'rank_average']].mean().reset_index()
        
        #top mean gain for each dataset
        fig = plt.figure(figsize=(12,8))
        sns.barplot(data=feature_importances_dataset, x='average', y='dataset')
        plt.title(f"Top dataset importance mean gain")

        fig.savefig(
            os.path.join(self.experiment_insight_path, 'dataset_importance_plot.png')
        )
        plt.close(fig)

        #top rank gain for each dataset
        fig = plt.figure(figsize=(12,8))
        sns.barplot(data=feature_importances_dataset, x='rank_average', y='dataset')
        plt.title(f"Top dataset importance mean rank gain")

        fig.savefig(
            os.path.join(self.experiment_insight_path, 'dataset_importance_rank_plot.png')
        )
        plt.close(fig)
        
    def get_oof_insight(self) -> None:
        #read data
        oof_prediction = pl.read_parquet(
            os.path.join(self.experiment_path, 'oof_prediction.parquet')
        )
        
        #score plot
        fig = plt.figure(figsize=(12,8))
        sns.lineplot(
            data=oof_prediction, 
            x="date_decision", y="score", hue='fold'
        )
        plt.title(f"Score prediction over date_decision")
        
        fig.savefig(
            os.path.join(self.experiment_insight_path, 'score_over_date.png')
        )
        plt.close(fig)

        #auc over time
        gini_in_time = (
            oof_prediction.to_pandas()
            .sort_values("WEEK_NUM")
            .groupby(["WEEK_NUM", "fold"])[["target", "score"]]
            .apply(
                lambda x: 2*roc_auc_score(x["target"], x["score"])-1
            )
        ).reset_index().rename(columns={0: 'auc'})

        fig = plt.figure(figsize=(12,8))
        sns.lineplot(
            data=gini_in_time, 
            x="WEEK_NUM", y="auc", hue='fold'
        )
        plt.title(f"AUC over WEEK_NUM")
        
        fig.savefig(
            os.path.join(self.experiment_insight_path, 'auc_over_week.png')
        )
        plt.close(fig)

    def get_oof_prediction(self) -> None:
        self.load_pickle_model_list()
        self.load_used_feature()
        
        prediction_list: list[pd.DataFrame] = []
        
        for fold_ in range(self.n_fold):

            fold_data = pl.scan_parquet(
                os.path.join(
                    self.config_dict['PATH_PARQUET_DATA'],
                    'data.parquet'
                )
            ).with_columns(
                (
                    pl.col('fold_info').str.split(', ')
                    .list.get(fold_).alias('current_fold')
                )
            ).filter(
                (pl.col('current_fold') == 'v')
            )
            
            test_feature = fold_data.select(self.feature_list).collect().to_pandas().to_numpy('float32')
            
            oof_prediction = self.model_list[fold_].predict(
                test_feature, 
                num_iteration=self.best_result['best_epoch']
            )
            prediction_df = fold_data.select(
                [
                    'case_id', 'date_decision',
                    'MONTH', 'WEEK_NUM', 'target'
                ]
            ).collect().to_pandas()
            
            prediction_df['fold'] = fold_
            prediction_df['score'] = oof_prediction
            prediction_list.append(prediction_df)
        
        (
            pl.from_dataframe(
                pd.concat(
                    prediction_list, axis=0
                )
            )
            .with_columns(
                pl.col('case_id').cast(pl.UInt32),
                pl.col('date_decision').cast(pl.Date),
                pl.col('MONTH').cast(pl.UInt32),
                pl.col('WEEK_NUM').cast(pl.UInt8),
                pl.col('fold').cast(pl.UInt8),
            )
            .sort(
                ['case_id', 'date_decision']
            )
            .write_parquet(
                os.path.join(self.experiment_path, 'oof_prediction.parquet')
            )
        )