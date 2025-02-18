import os
import shap
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Union, Tuple
from itertools import chain
from sklearn.metrics import roc_auc_score, log_loss
from src.model.lgbm.initialize import LgbmInit

class LgbmExplainer(LgbmInit):       
    def plot_train_curve(self, 
            progress_df: pd.DataFrame, 
            variable_to_plot: Union[str, list],  metric_to_eval: str,
            name_plot: str, 
            best_epoch_lgb:int
        ) -> None:
        
        if isinstance(variable_to_plot, str):
            variable_to_plot = [variable_to_plot]
                        
        fig = plt.figure(figsize=(18,8))
        sns.lineplot(
            data=progress_df[['time'] + variable_to_plot].melt(
                id_vars='time',
                value_vars=variable_to_plot,
                var_name='metric_fold', value_name=metric_to_eval
            ), 
            x="time", y=metric_to_eval, hue='metric_fold'
        )
        plt.axvline(x=best_epoch_lgb, color='blue', linestyle='--')

        plt.title(f"Training plot curve of {metric_to_eval}")

        fig.savefig(
            os.path.join(
                self.experiment_insight_train_path, f'{name_plot}.png'
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

        list_metric = self.progress_list[0]['valid'].keys()
        
        for metric_ in list_metric:
            progress_dict.update(
                {
                    f"{metric_}_fold_{i}": self.progress_list[i]['valid'][metric_]
                    for i in range(self.n_fold)
                }
            )

        progress_df = pd.DataFrame(progress_dict)
        
        for metric_ in list_metric:
            
            progress_df[f"average_{metric_}"] = progress_df.loc[
                :, [metric_ in x for x in progress_df.columns]
            ].mean(axis =1)
        
            progress_df[f"std_{metric_}"] = progress_df.loc[
                :, [metric_ in x for x in progress_df.columns]
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
        
        for metric_ in list_metric:
            #plot cv score
            self.plot_train_curve(
                progress_df=progress_df, 
                variable_to_plot=f'average_{metric_}', metric_to_eval=metric_,
                name_plot=f'average_{metric_}_training_curve', 
                best_epoch_lgb=best_epoch_lgb
            )
            #plot every fold score
            self.plot_train_curve(
                progress_df=progress_df, 
                variable_to_plot=[f'{metric_}_fold_{x}' for x in range(self.n_fold)],
                metric_to_eval=metric_,
                name_plot=f'training_{metric_}_curve_by_fold', 
                best_epoch_lgb=best_epoch_lgb
            )

        #plot std score
        self.plot_train_curve(
            progress_df=progress_df, 
            variable_to_plot=f'std_{self.metric_eval}', metric_to_eval=self.metric_eval,
            name_plot='std_training_curve', 
            best_epoch_lgb=best_epoch_lgb
        )
        
        self.save_best_result()
        
    def get_feature_importance(self) -> None:
        self.get_dataset_columns()
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
        feature_importances['rank_average'] = feature_importances['average'].rank(ascending=False)
        feature_importances['type_feature'] = feature_importances['feature'].apply(
            lambda x: 
                x[-1] if x[-1] in self.config_dict['TYPE_FEATURE'] else 'other'
        )

        #plain feature
        fig = plt.figure(figsize=(18,8))
        sns.barplot(data=feature_importances.head(50), x='average', y='feature')
        plt.title(f"50 TOP feature importance over {self.n_fold} average")

        fig.savefig(
            os.path.join(self.experiment_insight_feat_imp_path, 'importance_plot.png')
        )
        plt.close(fig)
        
        #feature importance excel
        feature_importances.to_excel(
            os.path.join(self.experiment_insight_feat_imp_path, 'feature_importances.xlsx'),
            index=False
        )
        #add dataset
        feature_importances_dataset = feature_importances.merge(
            self.feature_dataset, how='inner',
            on='feature'
        )

        #feature type
        fig = plt.figure(figsize=(18,8))
        sns.barplot(
            data=feature_importances, 
            x='average', y='type_feature'
        )
        plt.title(f"Top type feature")
        
        fig.savefig(
            os.path.join(self.experiment_insight_feat_imp_path, 'top_type_feature.png')
        )
        plt.close(fig)
        
        #feature type by dataset
        fig = plt.figure(figsize=(18,8))
        sns.barplot(
            data=feature_importances_dataset, 
            x='average', y='type_feature', hue='dataset'
        )
        plt.title(f"Top type feature by dataset")
        
        fig.savefig(
            os.path.join(self.experiment_insight_feat_imp_path, 'top_type_feature_by_dataset.png')
        )
        plt.close(fig)

        #plain feature top dataset
        fig = plt.figure(figsize=(18,8))
        plot_ = sns.barplot(
            data=feature_importances_dataset.head(50), 
            x='rank_average', y='average', hue='dataset', 
        )
        plot_.set(xticklabels=[])
        plt.title(f"Rank Top feature by dataset")
        
        fig.savefig(
            os.path.join(self.experiment_insight_feat_imp_path, 'top_feature_by_dataset.png')
        )
        plt.close(fig)

        #for each dataset print top feature
        for dataset_name in feature_importances_dataset['dataset'].unique():
            fig = plt.figure(figsize=(18,8))
            temp_dataset_feature = feature_importances_dataset.loc[
                feature_importances_dataset['dataset'] == dataset_name
            ]
            sns.barplot(data=temp_dataset_feature.head(50), x='average', y='feature')
            plt.title(f"50 TOP feature importance over {self.n_fold} average for {dataset_name}")

            fig.savefig(
                os.path.join(
                    self.experiment_insight_feat_imp_path, 
                    f'importance_plot_{dataset_name}.png'
                )
            )
            plt.close(fig)

        #add information about basic feature to see the best one with mean and average aggregation
        #showing only top 50 over each dataset
        result_to_excel: dict[str, dict[str, pd.DataFrame]] = {
            'mean': {},
            'sum': {}
        }
        
        for dataset_name in feature_importances_dataset['dataset'].unique():
            list_base_mean_feature: list[Tuple[str, float]] = []
            list_base_sum_feature: list[Tuple[str, float]] = []

            feature_of_dataset_list = [
                row['feature']
                for _, row in self.original_feature_dataset.iterrows()
                if row['dataset'] == dataset_name
            ]
            #over each feature find list of transformation and calculate average and sum importance
            for feature in feature_of_dataset_list:
                transformation_of_feature_list = [
                    col for col in self.feature_list
                    if 
                        (dataset_name in col) &
                        (feature in col)
                ]


                temp_feature = feature_importances_dataset.loc[
                    feature_importances_dataset['feature'].isin(transformation_of_feature_list)
                ]
            
                if temp_feature.shape[0] > 0:
                    list_base_mean_feature.append([dataset_name, feature, temp_feature['average'].mean()])
                    list_base_sum_feature.append([dataset_name, feature, temp_feature['average'].sum()])
            
            for list_importance, name_plot in [
                [list_base_mean_feature, 'mean'],
                [list_base_sum_feature, 'sum']
            ]:
                if len(list_importance) > 0:
                    temp_dataset_feature = pd.DataFrame(
                        list_importance, columns=['dataset', 'feature', 'importance']
                    ).sort_values(by='importance', ascending=False)

                    #save result in excel also
                    result_to_excel[name_plot][dataset_name] = temp_dataset_feature.copy()
                    
                    fig = plt.figure(figsize=(18,8))
                    sns.barplot(data=temp_dataset_feature.head(50), x='importance', y='feature')
                    plt.title(f"50 TOP base feature importance over {self.n_fold} average for {dataset_name}")

                    fig.savefig(
                        os.path.join(
                            self.experiment_insight_feat_imp_base_path, 
                            f'importance_plot_{name_plot}_{dataset_name}.png'
                        )
                    )
                    plt.close(fig)
        
        for operation in result_to_excel.keys():
            with pd.ExcelWriter(
                os.path.join(
                    self.experiment_insight_feat_imp_base_path, 
                    f'importance_{operation}.xlsx'
                ), engine='xlsxwriter'
            ) as writer:
                for dataset_name, dataset_df in result_to_excel[operation].items():
                    dataset_df.to_excel(writer, sheet_name=dataset_name)

        #add information about basic aggregation to see which is the best
        aggregation_list: list[str] = [
            'minnozero', 'max', 'mean', 'std', 'sum', 'numerical_range', 
            'filtered_mean', 'filtered_min', 'filtered_max',
            'not_hashed_missing_mode', 'mode', 'first', 'last', 'n_unique',
            'count_not_missing_not_hashednull', 'count_not_missing',
            'date_range', 'date_mean', 'date_min', 'date_max'
        ]
        pattern_columns_to_retrieve = '{dataset}_{operation}_{feature}'

        result_aggregation_imp_count: list[Tuple[str, float]] = []
        result_aggregation_imp_mean: list[Tuple[str, float]] = []
        result_aggregation_imp_sum: list[Tuple[str, float]] = []

        for operation in aggregation_list:

            aggregation_list_columns: list[str] = [ 
                pattern_columns_to_retrieve.format(
                    dataset=row['dataset'], 
                    operation=operation,
                    feature=row['feature']
                )
                for _, row in self.original_feature_dataset.iterrows()
                if pattern_columns_to_retrieve.format(
                    dataset=row['dataset'], 
                    operation=operation,
                    feature=row['feature']
                ) in self.feature_list
            ]

            temp_feature = feature_importances_dataset.loc[
                feature_importances_dataset['feature'].isin(aggregation_list_columns)
            ]
            
            if temp_feature.shape[0] > 0:
                result_aggregation_imp_sum.append([operation, temp_feature['average'].sum()])
                result_aggregation_imp_mean.append([operation, temp_feature['average'].mean()])
                result_aggregation_imp_count.append([operation, temp_feature['average'].count()])
                
        for list_importance, name_plot in [
            [result_aggregation_imp_sum, 'sum'],
            [result_aggregation_imp_mean, 'mean'],
            [result_aggregation_imp_count, 'count']
        ]:
            result_aggregation_importance = pd.DataFrame(
                list_importance, columns=['operation', 'importance']
            )
            
            fig = plt.figure(figsize=(18,8))
            sns.barplot(data=result_aggregation_importance, x='importance', y='operation')
            plt.title(f"Top operation ({name_plot}) over {self.n_fold}")

            fig.savefig(
                os.path.join(
                    self.experiment_insight_feat_imp_path, 
                    f'importance_operation_{name_plot}_plot.png'
                )
            )
            plt.close(fig)
        
        #mean over categorical
        result_aggregation_imp_mean: list[Tuple[str, float]] = []

        for operation in aggregation_list:

            aggregation_list_columns: list[str] = [ 
                pattern_columns_to_retrieve.format(
                    dataset=row['dataset'], 
                    operation=operation,
                    feature=row['feature']
                )
                for _, row in self.original_feature_dataset.iterrows()
                if pattern_columns_to_retrieve.format(
                    dataset=row['dataset'], 
                    operation=operation,
                    feature=row['feature']
                ) in self.categorical_col_list
            ]

            temp_feature = feature_importances_dataset.loc[
                feature_importances_dataset['feature'].isin(aggregation_list_columns)
            ]
            
            if temp_feature.shape[0] > 0:
                result_aggregation_imp_mean.append([operation, temp_feature['average'].mean()])
        
        result_aggregation_importance = pd.DataFrame(
                result_aggregation_imp_mean, columns=['operation', 'importance']
            )
            
        fig = plt.figure(figsize=(18,8))
        sns.barplot(data=result_aggregation_importance, x='importance', y='operation')
        plt.title(f"Top operation mean on categorical")

        fig.savefig(
            os.path.join(
                self.experiment_insight_feat_imp_path, 
                f'importance_operation_mean_categorical_plot.png'
            )
        )
        plt.close(fig)       
                
        #get information about top dataset on mean gai and rank gain
        feature_importances_dataset_mean = feature_importances_dataset.groupby(
            'dataset'
        )[['average', 'rank_average']].mean().reset_index()
        
        #top mean gain for each dataset
        fig = plt.figure(figsize=(18,8))
        sns.barplot(data=feature_importances_dataset_mean, x='average', y='dataset')
        plt.title(f"Top dataset importance mean gain")

        fig.savefig(
            os.path.join(self.experiment_insight_feat_imp_path, 'dataset_importance_plot.png')
        )
        plt.close(fig)

        #top rank gain for each dataset
        fig = plt.figure(figsize=(18,8))
        sns.barplot(data=feature_importances_dataset_mean, x='rank_average', y='dataset')
        plt.title(f"Top dataset importance mean rank gain")

        fig.savefig(
            os.path.join(self.experiment_insight_feat_imp_path, 'dataset_importance_rank_plot.png')
        )
        plt.close(fig)

        #get information about top dataset on sum fe
        feature_importances_dataset_sum = feature_importances_dataset.groupby(
            'dataset'
        )[['average']].sum().reset_index()
        #top mean gain for each dataset
        
        fig = plt.figure(figsize=(18,8))
        sns.barplot(data=feature_importances_dataset_sum, x='average', y='dataset')
        plt.title(f"Top dataset total contribution")

        fig.savefig(
            os.path.join(self.experiment_insight_feat_imp_path, 'dataset_total_importance_plot.png')
        )
        plt.close(fig)

        #get information about top dataset on sum fe
        feature_importances_dataset_count = feature_importances_dataset.groupby(
            'dataset'
        ).size().reset_index().rename(columns={0: 'count'})
        #top mean gain for each dataset

        fig = plt.figure(figsize=(18,8))
        sns.barplot(data=feature_importances_dataset_count, x='count', y='dataset')
        plt.title(f"Number of feature for dataset")

        fig.savefig(
            os.path.join(self.experiment_insight_feat_imp_path, 'dataset_number_feature.png')
        )
        plt.close(fig)
        
        #get information about 'total importance of dataset'/'number of feature'
        feature_importances_dataset_mean_sum= feature_importances_dataset.groupby(
            'dataset'
        ).agg(
            {'average': 'sum', 'feature': 'size'}
        ).reset_index().rename(columns={'average': 'total_imp', 'feature': 'num_features'})
        feature_importances_dataset_mean_sum['average_total_imp'] = (
            feature_importances_dataset_mean_sum['total_imp']/
            feature_importances_dataset_mean_sum['num_features']
        )
        #top mean gain for each dataset

        fig = plt.figure(figsize=(18,8))
        sns.barplot(data=feature_importances_dataset_mean_sum, x='average_total_imp', y='dataset')
        plt.title(f"Total Importance vs # feature for dataset")

        fig.savefig(
            os.path.join(self.experiment_insight_feat_imp_path, 'dataset_total_imp_vs_num_feature.png')
        )
        plt.close(fig)

    def get_stability_feature_importance(self) -> None:
        self.load_best_result()
        self.get_dataset_columns()
        self.model_list_stability = self.load_custom_pickle_model_list(file_name='model_list_stability.pkl')
        
        feature_importances = pd.DataFrame()
        feature_importances['feature'] = self.feature_list
                
        data = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_PARQUET_DATA'],
                'data.parquet'
            )
        )
                                    
        for fold_, model in enumerate(self.model_list_stability):
            print(f'Stability fold {fold_}')
            feature_importances[f'fold_{fold_}'] = model.feature_importance(
                importance_type='gain'
            )
            feature_importances[f'fold_{fold_}_rank'] = feature_importances[f'fold_{fold_}'].rank(ascending=False)

            test_feature = (
                data.with_columns(
                    (
                        pl.col('fold_info').str.split(', ')
                        .list.get(fold_).alias('current_fold')
                    )
                )
                .filter(
                    (pl.col('current_fold') == 'v') &
                    (pl.col('target') == 1)
                )
                .select(self.feature_list)
                .collect().to_pandas()
            )
            if test_feature.shape[0] <= 25_000:
                print(f'Target 1 rows: {test_feature.shape[0]}')
            else:
                test_feature = test_feature.sample(25_000)
                
            test_feature = test_feature.to_numpy('float32')
            
            #calculate shap on sampled feature. exclude last which is bias
            shap_oof: np.ndarray = self.model_list_stability[fold_].predict(
                data=test_feature,
                num_iteration=self.best_result['best_epoch'],
                pred_contrib=True
            )[:, :-1]
            
            feature_importances[f'mean_shap_fold_{fold_}'] = shap_oof.mean(axis=0)
            feature_importances[f'imp_shap_fold_{fold_}'] = np.abs(shap_oof).mean(axis=0)

        feature_importances['type_feature'] = feature_importances['feature'].apply(
            lambda x: 
                x[-1] if x[-1] in self.config_dict['TYPE_FEATURE'] else 'other'
        )
        
        feature_importances['mean_importance_shap'] = feature_importances[
            [f'imp_shap_fold_{fold_}' for fold_ in range(self.n_fold)]
        ].mean(axis=1)
        
        feature_importances['mean_stability_shap'] = feature_importances[
            [f'mean_shap_fold_{fold_}' for fold_ in range(self.n_fold)]
        ].mean(axis=1)
        feature_importances['std_stability_shap'] = feature_importances[
            [f'mean_shap_fold_{fold_}' for fold_ in range(self.n_fold)]
        ].std(axis=1)
        
        feature_importances['stability_ratio'] = (
            feature_importances['std_stability_shap']/
            feature_importances['mean_stability_shap']
        )

        feature_importances['average'] = feature_importances[
            [f'fold_{fold_}' for fold_ in range(self.n_fold)]
        ].mean(axis=1)
        feature_importances['std'] = feature_importances[
            [f'fold_{fold_}' for fold_ in range(self.n_fold)]
        ].std(axis=1)
        
        feature_importances['rank_average'] = feature_importances[
            [f'fold_{fold_}_rank' for fold_ in range(self.n_fold)]
        ].mean(axis=1)

        feature_importances['rank_std'] = feature_importances[
            [f'fold_{fold_}_rank' for fold_ in range(self.n_fold)]
        ].std(axis=1)

        feature_importances['count_useless_on_fold'] = (
            feature_importances[
                [f'fold_{fold_}' for fold_ in range(self.n_fold)]
            ]==0
        ).sum(axis=1)
        
        feature_importances = (
            feature_importances
            .sort_values(by='rank_average', ascending=True)
        )
        feature_importances['rank_norm'] = (
            feature_importances['rank_average'] + feature_importances['rank_std'] * 1.96
        )
        feature_importances['rank_ratio'] = (
            feature_importances['rank_std']/feature_importances['rank_average']
        )

        feature_importances['average_norm'] = (
            feature_importances['average'] + feature_importances['std'] * 1.96
        )
        feature_importances['average_ratio'] = (
            feature_importances['std']/feature_importances['average']
        )


        feature_importances.to_excel(
            self.feature_stability_path,
            index=False
        )

        feature_importances_dataset = feature_importances.merge(
            self.feature_dataset, how='inner',
            on='feature'
        )
        #stability over class feature and dataset
        fig = plt.figure(figsize=(18,8))
        sns.barplot(
            data=feature_importances_dataset, 
            x='rank_average', y='type_feature', hue='dataset'
        )
        plt.title(f"Top rank average by type and feature")
        
        fig.savefig(
            os.path.join(self.experiment_insight_path, 'top_rank_feature_by_type_dataset.png')
        )
        plt.close(fig)

    def get_oof_insight(self) -> None:
        #read data
        oof_prediction = pl.read_parquet(
            os.path.join(self.experiment_path, 'oof_prediction.parquet')
        )
        
        #score plot
        fig = plt.figure(figsize=(18,8))
        sns.lineplot(
            data=oof_prediction.filter(
                pl.len().over('date_decision')>30
            ), 
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

        fig = plt.figure(figsize=(18,8))
        sns.lineplot(
            data=gini_in_time, 
            x="WEEK_NUM", y="auc", hue='fold'
        )
        plt.title(f"AUC over WEEK_NUM")
        
        fig.savefig(
            os.path.join(self.experiment_insight_path, 'auc_over_week.png')
        )
        plt.close(fig)

        #binary cross entropy over week
        #auc over time
        logloss_in_time = (
            oof_prediction.to_pandas()
            .sort_values("WEEK_NUM")
            .groupby(["WEEK_NUM", "fold"])[["target", "score"]]
            .apply(
                lambda x: log_loss(x["target"], x["score"])
            )
        ).reset_index().rename(columns={0: 'log_loss'})

        fig = plt.figure(figsize=(18,8))
        sns.lineplot(
            data=logloss_in_time, 
            x="WEEK_NUM", y="log_loss", hue='fold'
        )
        plt.title(f"Log Loss over WEEK_NUM")
        
        fig.savefig(
            os.path.join(self.experiment_insight_path, 'logloss_over_week.png')
        )
        plt.close(fig)

        #TARGET OVER TIME
        target_in_time = (
            oof_prediction.to_pandas()
            .sort_values("date_decision")
            .groupby(["date_decision", "fold"])[["target"]]
            .mean()
        ).reset_index().rename(columns={0: 'target'})

        fig = plt.figure(figsize=(18,8))
        sns.lineplot(
            data=target_in_time, 
            x="date_decision", y="target", hue='fold'
        )
        plt.title(f"Target mean over date decision")
        
        fig.savefig(
            os.path.join(self.experiment_insight_path, 'target_over_date.png')
        )
        plt.close(fig)

        #STD over time
        target_in_time = (
            oof_prediction.to_pandas()
            .sort_values("date_decision")
            .groupby(["date_decision", "fold"])[["target"]]
            .std()
        ).reset_index().rename(columns={0: 'target'})

        fig = plt.figure(figsize=(18,8))
        sns.lineplot(
            data=target_in_time, 
            x="date_decision", y="target", hue='fold'
        )
        plt.title(f"Target std over date decision")
        
        fig.savefig(
            os.path.join(self.experiment_insight_path, 'target_std_over_date.png')
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
        
    def get_shap_insight(
        self, 
        sample_shap_: int = 10_000,
        top_interaction: int=5,
        select_fold: int = None
    ) -> None:
        #define private function
        print('Starting to calculate shap')
        def get_corrected_corr_matrix(
                shap_array: np.ndarray, 
                noise_: float=0.01
            ) -> np.ndarray:

            #add noise on constant columns
            constant_col = np.where(
                np.std(shap_array, axis=0) == 0
            )[0]
            
            shap_array[:, constant_col] += (
                np.random.random(
                    (shap_array.shape[0], len(constant_col))
                )*noise_
            )
            corr_matrix = np.corrcoef(shap_array.T)
            return corr_matrix

        #best interaction as best correlation feature
        def get_best_interaction(
                idx: int, feature_list: list[str],
                corr_matrix: np.ndarray, top_n: int
            ) -> Tuple[Tuple[str, float]]:

            assert corr_matrix.shape[1] == len(feature_list)    
            
            best_position_ = np.argsort(
                np.abs(corr_matrix), axis=1
            )[idx, -(top_n+1):-1]
            return [
                [feature_list[position], corr_matrix[idx, position]]
                for position in best_position_
            ]

        self.load_best_result()
        self.load_model_list()
        self.load_used_feature()
        
        shap_list: list[np.ndarray] = []
        
        for fold_ in range(self.n_fold):
            if select_fold is not None:
                if fold_!=select_fold:
                    continue
                
            print(f'Shap folder {fold_}')
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
            
            test_feature = fold_data.select(self.feature_list).collect().to_pandas()
            
            #calculate shap on sampled feature
            shap_ = self.model_list[fold_].predict(
                data=test_feature.sample(sample_shap_).to_numpy('float32'),
                num_iteration=self.best_result['best_epoch'],
                pred_contrib=True
            )
                
            shap_list.append(shap_[:-1])
            
        shap_array = np.concatenate(
            shap_list, axis=0
        )[:, :-1]

        corr_matrix = get_corrected_corr_matrix(
            shap_array=shap_array
        )

        #get ordered best feature
        top_feature_list = pd.read_excel(
            os.path.join(self.experiment_path, 'feature_importances.xlsx'),
            usecols=['feature'],
        )['feature'].tolist()


        top_interaction_list = list(
            chain(
                *[
                    [
                        [
                            rank_base_feature, 
                            feature, feature_interaction, corr_coef, 
                            rank
                        ]
                        for rank, (feature_interaction, corr_coef) in enumerate(
                            get_best_interaction(
                                idx=rank_base_feature, 
                                feature_list=self.feature_list,
                                corr_matrix=corr_matrix,
                                top_n=top_interaction
                            )
                        )
                    ] for rank_base_feature, feature in enumerate(top_feature_list)
                ]
            )
        )
        
        top_interactive_df = pd.DataFrame(
            top_interaction_list,
            columns=[
                'rank_base_feature', 
                'top_feature', 'top_interaction', 'corr_coef',
                'rank_interaction'
            ]   
        )

        shap_df = pd.DataFrame(
            shap_array,
            columns=self.feature_list
        )
        
        #save info
        top_interactive_df.to_csv(
            os.path.join(self.experiment_shap_path, 'top_feature_interaction.csv'),
            index=False
        )
        shap_df.to_csv(
            os.path.join(self.experiment_shap_path, 'array_shap_interaction.csv'),
            index=False
        )
        
    def get_shap_beeswark(
        self, 
        select_fold: int,
        sample_shap_: int = 10_000,
    ) -> None:
        #define private function
        print('Starting to calculate shap')

        self.load_best_result()
        self.load_model_list()
        self.load_used_feature()
        
        feature_data = (
            pl.scan_parquet(
                os.path.join(
                    self.config_dict['PATH_PARQUET_DATA'],
                    'data.parquet'
                )
            )
            .with_columns(
                (
                    pl.col('fold_info').str.split(', ')
                    .list.get(select_fold).alias('current_fold')
                )
            )
            .filter(
                (pl.col('current_fold') == 'v')
            )
            .select(self.feature_list)
            .collect().to_pandas()
            .sample(sample_shap_)
            .to_numpy('float32')
        )
        model = self.model_list[select_fold]
        shap_values = shap.TreeExplainer(model).shap_values(feature_data)
        
        shap.summary_plot(shap_values, features=feature_data, feature_names=self.feature_list)
