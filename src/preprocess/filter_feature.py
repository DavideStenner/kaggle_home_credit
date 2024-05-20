import polars as pl

from typing import Dict

from src.base.preprocess.cv_fold import BaseCVFold
from src.preprocess.initialize import PreprocessInit

class PreprocessFilterFeature(BaseCVFold, PreprocessInit):
    def filter_feature_by_correlation(self) -> None:
        numerical_feature_list = [
            col for col in self.data.columns
            if col not in (
                self.config_dict['SPECIAL_COLUMNS'] + 
                [
                    'fold_info', 'current_fold', 'date_order_kfold'
                ]
            )
        ]
        group_list = self.get_correlated_group_features(col_list=numerical_feature_list, data=self.data)
        self.exclude_feature_list = self.reduce_group(group_list=group_list, data=self.data)
    
    def reduce_group(self, group_list: list[list[str]], data:pl.DataFrame) -> list[str]:
        drop_list = []

        for current_group in group_list:
            
            max_unique= 0
            select_col = current_group[0]
            
            for current_col in current_group:
                col_n_unique = data.select(pl.n_unique(current_col)).collect().item()
                
                if col_n_unique > max_unique:
                    max_unique = col_n_unique
                    select_col = current_col

            drop_list += [col for col in current_group if col != select_col]
            
        return drop_list

    def get_correlated_group_features(
            self,
            col_list: list[str], data: pl.DataFrame,
            null_pct_treshold: float=0.95, corr_treshold: float=0.8
        ) -> list[list[str]]:
        
        statistic_utils_dict: Dict[str, int] = {
            current_col: data.select(
                pl.col(current_col).is_not_null().sum()
            ).item()
            for current_col in col_list
        }
        group_list = []
        remaining_cols = col_list
        print(f'Starting col: {len(remaining_cols)}')
        
        while remaining_cols:
            current_col = remaining_cols.pop(0)
            print(f'remaining {len(remaining_cols)} col',end='\r')

            
            current_group = [current_col]
            correlated_cols = [current_col]
            
            for other_col in remaining_cols:
                common_not_null_pct: float = (
                    data.select(
                        (
                            (pl.col(current_col).is_not_null())&
                            (pl.col(other_col).is_not_null())
                        ).sum()
                    ).item()/
                    max(
                        statistic_utils_dict[current_col],
                        statistic_utils_dict[other_col]
                    )
                )
            
                if common_not_null_pct >= null_pct_treshold:
                    corr_between_: float = data.select(
                        pl.corr(current_col, other_col)
                    ).item()
                    if corr_between_ >= corr_treshold:

                        current_group.append(other_col)
                        correlated_cols.append(other_col)

            group_list.append(current_group)
            
            remaining_cols = [
                col 
                for col in remaining_cols 
                if col not in correlated_cols
            ]
            
        return group_list
