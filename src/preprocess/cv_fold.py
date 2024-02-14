import pandas as pd
import numpy as np

import polars as pl

from typing import Iterator, Tuple

from src.base.preprocess.cv_fold import BaseCVFold
from src.preprocess.initialize import PreprocessInit

def get_time_series_cross_val_splits(
    data: pd.DataFrame, time_col: str, num_fold: int, 
    embargo: int, min_time_to_use: int,
    percent_split: bool
) -> Iterator[Tuple[list[int], list[int]]]:
    #https://github.com/numerai/example-scripts/blob/495d3c13153c2068a87cf8c33196a787b2a0871f/utils.py#L79
    
    all_train_eras = data[time_col].unique()    
    
    if percent_split:
        min_eras = all_train_eras.min()
        max_eras = all_train_eras.max()

        if min_time_to_use > 0:
            print('Min time to use not implemented for percent split')
            
        cumulative_size = data[[time_col]].sort_values(time_col).groupby(time_col).size()

        cum_size_time = cumulative_size.cumsum()/cumulative_size.sum()

        percent_split = 1/num_fold
        time_split = [cum_size_time[cum_size_time >= percent_split*i].index[0] for i in range(num_fold)]
        max_eras = data[time_col].max()

        test_splits = [
            all_train_eras[
                (all_train_eras >= (time_split[i] if i > 0 else min_eras)) &
                (all_train_eras <= (time_split[i+1] if i < (num_fold - 1) else max_eras))
            ] for i in range(num_fold)
        ]
    
    else:
        number_eras = len(all_train_eras)
       
        #each test split has this length
        len_split = (
            number_eras - min_time_to_use
        ) // num_fold

        #create kfold split by selecting also a min time to use --> first min_time_to_use won't be use for test split
        #fix the last test split to have all the last eras, in case the number of eras wasn't divisible by cv
        test_splits = [
            all_train_eras[
                (min_time_to_use + (i * len_split)):
            (min_time_to_use + (i + 1) * len_split) if i < (num_fold - 1) else number_eras
            ] for i in range(num_fold)
        ]

    train_splits = []
    for test_split in test_splits:
        
        #get boundaries
        test_split_min, test_split_max = int(np.min(test_split)), int(np.max(test_split))

        # get all of the eras that aren't in the test split
        train_split_not_embargoed = [e for e in all_train_eras if not (test_split_min <= int(e) <= test_split_max)]
        
        # embargo the train split so we have no leakage.
        train_split = [
            e for e in train_split_not_embargoed if
            abs(int(e) - test_split_max) > embargo and abs(int(e) - test_split_min) > embargo
        ]
        train_splits.append(train_split)

    # convenient way to iterate over train and test splits
    train_test_zip = zip(train_splits, test_splits)
    return train_test_zip

def get_fold(
        data: pd.DataFrame, time_col: str,
        embargo: int, 
        num_fold: int, 
        min_time_to_use: int = 0, 
        percent_split: bool=True, 
        return_index: bool=True
    ) -> list[list[np.ndarray[int]]]:
    
    fold_embargo_zip = get_time_series_cross_val_splits(
        data=data, time_col=time_col, num_fold=num_fold, embargo=embargo, min_time_to_use=min_time_to_use,
        percent_split=percent_split
    )
    if return_index:
        fold_split = [
            [
                np.where(data[time_col].isin(train_index))[0], 
                np.where(data[time_col].isin(test_index))[0]
            ]
            for train_index, test_index in fold_embargo_zip
        ]
    else:
        fold_split = [
            [train_index, test_index]
            for train_index, test_index in fold_embargo_zip
        ]
    return fold_split

class PreprocessFoldCreator(BaseCVFold, PreprocessInit):
    def create_fold(self):        
        data_for_split = self.data.select([self.fold_time_col]).to_pandas() 
        fold_split = get_fold(
            data_for_split, time_col=self.fold_time_col, 
            embargo=self.embarko_skip,
            num_fold=self.n_folds, return_index=False
        )
        self.data = self.data.with_columns(
            (
                (
                    pl.when(
                        pl.col(self.fold_time_col)
                        .is_in(fold_split[fold_][0])
                    )
                    .then(pl.lit('t'))
                    .when(
                        pl.col(self.fold_time_col)
                        .is_in(fold_split[fold_][1])
                    ).then(pl.lit('v')).otherwise(pl.lit('n'))
                    .alias(f'fold_{fold_}')
                )
                for fold_ in range(self.n_folds)
            )
        ).with_columns(
            pl.concat_str(
                [f'fold_{x}' for x in range(self.n_folds)],
                separator=', '
            ).alias('fold_info')
        ).drop([f'fold_{x}' for x in range(self.n_folds)])