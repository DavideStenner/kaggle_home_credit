import warnings
import numpy as np
import polars as pl

from typing import Union, Dict, Optional
from itertools import product, chain
from functools import partial

from src.utils.other import change_name_with_type
from src.base.preprocess.add_feature import BaseFeature
from src.preprocess.initialize import PreprocessInit
from src.utils.dtype import TYPE_MAPPING

class PreprocessAddFeature(BaseFeature, PreprocessInit):
    def add_generic_feature(
            self, 
            data: Union[pl.DataFrame, pl.LazyFrame], 
            dataset_name: str
        ) -> list[pl.Expr]:
        """
        Generical expresion which can be combined in an agg(case_id) adding more expression
        Used when no domain knowledge can be applied

        Args:
            data (Union[pl.DataFrame, pl.LazyFrame]): initial dataset

        Returns:
            pl.Expr: list of expression
        """
        mapper_column_cast = {
            col: TYPE_MAPPING[dtype_str]
            for col, dtype_str in self.mapper_dtype[dataset_name].items()
        }
        categorical_columns_list :list[str] = [
            col
            for col in data.columns 
            if col in self.mapper_mask[dataset_name].keys()
        ]
        numerical_columns_list :list[str] = [
            col 
            for col in data.columns 
            if 
                (col[-1] in ['A', 'L', 'P', 'T']) &
                (col not in categorical_columns_list)
        ]
        date_columns_list :list[str] = [
            col
            for col in data.columns 
            if col[-1] == 'D'
        ]
        categorical_columns_with_hashed_null: list[str] = [
            col for col in categorical_columns_list
            if self.hashed_missing_label in self.mapper_mask[dataset_name][col].keys()
        ]
        #numerical expression
        base_numerical_expr_list: list[pl.Expr] = [
            (
                pl_operator(col_name)
                .alias(f'{pl_operator.__name__}_{col_name}')
            )
            for pl_operator, col_name in product(
                self.numerical_aggregator,
                numerical_columns_list
            )
        ]
        numerical_expr_list: list[pl.Expr] = []
        
        for pl_expr in base_numerical_expr_list:
            name_expression: str = pl_expr.meta.output_name()
            operator_applied: str = name_expression.split('_', 1)[0]
            original_col_name: str = name_expression.split('_', 1)[-1]
            
            #mean, std -> float32
            if (operator_applied in ['mean', 'std']):
                numerical_expr_list.append(
                    pl_expr.cast(pl.Float32)
                )
            #min, max -> inherit original cast
            elif (operator_applied in ['min', 'max']):
                numerical_expr_list.append(
                    pl_expr.cast(mapper_column_cast[original_col_name])
                )
            #cast is uint16
            elif (operator_applied == 'count'):
                numerical_expr_list.append(
                    pl_expr.cast(pl.UInt16)
                )
            else:
                numerical_expr_list.append(
                    pl_expr
                )

        categorical_expr_list: list[pl.Expr] = (
            [
                (
                    pl.col(col)
                    .filter(
                        pl.col(col)!=
                        self.mapper_mask[dataset_name][col][self.hashed_missing_label]
                    )
                    .drop_nulls().mode().sort().first()
                    .alias(f'not_hashed_missing_mode_{col}')
                    .cast(mapper_column_cast[col])
                )
                for col in categorical_columns_with_hashed_null
            ] + 
            [
                (
                    pl.col(col)
                    .drop_nulls().mode().sort().first()
                    .alias(f'mode_{col}')
                    .cast(mapper_column_cast[col])
                )
                for col in categorical_columns_list
            ]
        )

        date_expr_list: list[pl.Expr] = [
            (
                pl_operator(col_name)
                .alias(f'{pl_operator.__name__}_{col_name}')
                .cast(pl.Date)
            )
            for pl_operator, col_name in product(
                self.date_aggregator,
                date_columns_list
            )
        ]
        date_expr_list += [
            (
                (
                    pl.max(col_name) - pl.min(col_name)
                ).dt.total_days()
                .cast(pl.UInt32)
                .alias(f'date_range_{col_name}')
            )
            for col_name in date_columns_list
        ]
        count_expr_list: list[pl.Expr] = (
            [
                (
                    pl.col(col).n_unique()
                    .alias(f'max_{col}')
                    .cast(pl.UInt16)
                )
                for col in ['num_group1', 'num_group2']
                if col in data.columns
            ]
        )
        
        if 'num_group2' not in data.columns:
            first_expression_list: list[pl.Expr] = [
                (
                    pl.col(col)
                    .filter(pl.col('num_group1')==0)
                    .first()
                    .alias(f'first_{col}')
                    .cast(
                        (
                            pl.Date if col[-1] == 'D'
                            else mapper_column_cast[col]
                        )
                    )
                )
                for col in categorical_columns_list + numerical_columns_list
            ]

            last_expression_list: list[pl.Expr] = [
                (
                    pl.col(col)
                    .filter(pl.col('num_group1')==pl.col('num_group1').filter(pl.col(col).is_not_null()).max())
                    .last()
                    .alias(f'last_{col}')
                    .cast(
                        (
                            pl.Date if col[-1] == 'D'
                            else mapper_column_cast[col]
                        )
                    )
                )
                for col in categorical_columns_list + numerical_columns_list
            ]
        else:
            first_expression_list: list[pl.Expr] = [
                (
                    pl.col(col)
                    .filter(pl.col('num_group2')==0)
                    .first()
                    .alias(f'first_{col}')
                    .cast(
                        (
                            pl.Date if col[-1] == 'D'
                            else mapper_column_cast[col]
                        )
                    )
                )
                for col in categorical_columns_list + numerical_columns_list
            ]

            last_expression_list: list[pl.Expr] = [
                (
                    pl.col(col)
                    .filter(pl.col('num_group2')==pl.col('num_group2').filter(pl.col(col).is_not_null()).max())
                    .last()
                    .alias(f'last_{col}')
                    .cast(
                        (
                            pl.Date if col[-1] == 'D'
                            else mapper_column_cast[col]
                        )
                    )
                )
                for col in categorical_columns_list + numerical_columns_list
            ]
            
        result_expr_list: list[pl.Expr] = (
            numerical_expr_list +
            categorical_expr_list +
            date_expr_list +
            count_expr_list +
            first_expression_list +
            last_expression_list
        )

        return result_expr_list

    def filter_and_select_first(
        self, 
        data: Union[pl.LazyFrame, pl.DataFrame],
        filter_col: pl.Expr, col_list: Optional[list[str]], group_by: str = 'case_id'
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
        
        if col_list is None:
            col_list = data.columns
            
        data = (
            data
            .filter(filter_col)
            .select(col_list)
            .group_by(group_by)
            .agg(pl.all().first())
        )
        return data

    
    def filter_and_select_first_non_blank(
        self, 
        data: Union[pl.LazyFrame, pl.DataFrame],
        filter_col: pl.Expr, col_list: Optional[list[str]], group_by: str = 'case_id'
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
        
        if col_list is None:
            col_list = data.columns
            
        data = (
            data.filter(
                filter_col
            ).select(col_list).group_by(group_by).agg(
                (
                    pl.col(col).filter(pl.col(col).is_not_null()).first()
                    for col in col_list if col not in self.special_column_list
                )
            )
        )
        return data
    
    def add_fold_column(self) -> None:
        self.base_data = self.base_data.with_columns(
            pl.col('WEEK_NUM')
            .alias(self.fold_time_col)
            .cast(pl.Int16)
        )
    
    def create_credit_bureau_a_1_feature(self) -> None:
        self.credit_bureau_a_1 = self.credit_bureau_a_1.with_columns(
            pl.date(
                pl.col('dpdmaxdateyear_896T'),
                pl.col('dpdmaxdatemonth_442T'),
                1
            ).cast(pl.Date).alias('dpdmaxdate_closed_D'),
            pl.date(
                pl.col('dpdmaxdateyear_596T'),
                pl.col('dpdmaxdatemonth_89T'),
                1
            ).cast(pl.Date).alias('dpdmaxdate_active_D'),
            pl.date(
                pl.col('overdueamountmaxdateyear_994T'),
                pl.col('overdueamountmaxdatemonth_284T'),
                1
            ).cast(pl.Date).alias('overdueamountmaxdate_closed_D'),
            pl.date(
                pl.col('overdueamountmaxdateyear_2T'),
                pl.col('overdueamountmaxdatemonth_365T'),
                1
            ).cast(pl.Date).alias('overdueamountmaxdate_active_D')
        ).drop(
            'refreshdate_3813885D',
            'dpdmaxdatemonth_442T', 'dpdmaxdatemonth_89T',
            'dpdmaxdateyear_596T', 'dpdmaxdateyear_896T',
            'overdueamountmaxdatemonth_284T', 'overdueamountmaxdatemonth_365T',
            'overdueamountmaxdateyear_2T', 'overdueamountmaxdateyear_994T'
        )
        
        list_generic_feature: list[pl.Expr] = self.add_generic_feature(
            self.credit_bureau_a_1, 'credit_bureau_a_1'
        )
        self.credit_bureau_a_1 = (
            self.credit_bureau_a_1
            .group_by('case_id')
            .agg(list_generic_feature)
        )

    def create_credit_bureau_a_2_feature(self) -> None:
        numerical_features: list[str] = [
            'collater_valueofguarantee_1124L', 'collater_valueofguarantee_876L',
            'pmts_dpd_1073P', 'pmts_dpd_303P',
            'pmts_overdue_1140A', 'pmts_overdue_1152A'
        ]

        self.credit_bureau_a_2 = self.credit_bureau_a_2.select(
            ['case_id', 'num_group1'] +
            [
                pl.col(col).cast(pl.Float32).alias(col)
                for col in numerical_features
            ]
        )
        
        self.credit_bureau_a_2 = self.credit_bureau_a_2.group_by(
            'case_id'
        ).agg(
            [
                pl.col('num_group1').max()
                .cast(pl.UInt32)
                .alias('max_num_group1'),
                pl.len()
                .cast(pl.UInt32)
                .alias('num_rowsX')
            ] +
            [
                (
                    pl.col(num_col)
                    .max()
                    .cast(pl.Float32)
                    .alias(f'max_{num_col}')
                )
                for num_col in numerical_features
            ] +
            [
                (
                    pl.col(num_col)
                    .min()
                    .cast(pl.Float32)
                    .alias(f'min_{num_col}')
                )
                for num_col in numerical_features
            ]
        )

        
    def create_credit_bureau_b_1_feature(self) -> None:
        self.credit_bureau_b_1 = (
            self.credit_bureau_b_1
            .with_columns(
                pl.date(
                    pl.col('dpdmaxdateyear_742T'),
                    pl.col('dpdmaxdatemonth_804T'),
                    1
                ).cast(pl.Date).alias('dpdmaxdate_due_D'),
                pl.date(
                    pl.col('overdueamountmaxdateyear_432T'),
                    pl.col('overdueamountmaxdatemonth_494T'),
                    1
                ).cast(pl.Date).alias('overdueamountmaxdate_D')
            ).drop(
                'dpdmaxdatemonth_804T', 'dpdmaxdateyear_742T',
                'overdueamountmaxdatemonth_494T', 'overdueamountmaxdateyear_432T',
                'credlmt_1052A', 'residualamount_1093A', 'residualamount_127A'
            )
        )
        
        list_generic_feature: list[pl.Expr] = self.add_generic_feature(
            self.credit_bureau_b_1, 'credit_bureau_b_1'
        )

        self.credit_bureau_b_1 = (
            self.credit_bureau_b_1
            .group_by('case_id')
            .agg(
                [
                    (
                        pl.col('contractmaturitydate_151D') - pl.col('contractdate_551D')
                    ).dt.total_days().mean()
                    .cast(pl.Float32)
                    .alias('mean_duration_contractmaturitydate_151D_contractdate_551D'),
                    (
                        pl.col('lastupdate_260D') - pl.col('contractdate_551D')
                    ).dt.total_days().mean()
                    .cast(pl.Float32)
                    .alias('mean_duration_lastupdate_260D_contractdate_551D'),
                    (
                        pl.col('contractmaturitydate_151D') - pl.col('lastupdate_260D')
                    ).dt.total_days().mean()
                    .cast(pl.Float32)
                    .alias('mean_duration_contractmaturitydate_151D_lastupdate_260D')
                ] +
                list_generic_feature
            )
        )

    def create_credit_bureau_b_2_feature(self) -> None:
        numeric_columns: list[str] = ['pmts_dpdvalue_108P', 'pmts_pmtsoverdue_635A']
        product_operator_numerical_col = product(
            self.numerical_filter_aggregator, 
            numeric_columns
        )
        aggregation_level_group_list: list[pl.Expr] = (
            list(
                chain(
                    *[
                        [
                            #num pmt
                            pl.col('num_group2').filter(filter_pl).max().alias(f'{name_pl}_num_pmt_X').cast(pl.UInt32),
                            #range date
                            (
                                (
                                    pl.col('pmts_date_1107D').filter(filter_pl).max() - 
                                    pl.col('pmts_date_1107D').filter(filter_pl).min()
                                ).dt.total_days()
                                .cast(pl.UInt32)
                                .alias(f'{name_pl}_range_pmts_date_1107D')
                            )
                        ] +
                        #date operator
                        [
                            (
                                pl.col('pmts_date_1107D').filter(filter_pl).max()
                                .alias(f'{name_pl}_max_pmts_date_1107D')
                            ),
                            (
                                pl.col('pmts_date_1107D').filter(filter_pl).min()
                                .alias(f'{name_pl}_min_pmts_date_1107D')
                            )
                        ] +
                        # stat on all numerical
                        [
                            (
                                pl_operator(col_name, filter_pl)
                                .alias(f'{name_pl}_{pl_operator.__name__}_{col_name}')
                                .cast(pl.Float32)
                            )
                            for pl_operator, col_name in product_operator_numerical_col
                        ] +
                        # #stats on all numerical != 0
                        [
                            (
                                pl_operator(col_name, filter_pl).filter(pl.col(col_name)!=0)
                                .alias(f'{name_pl}_{pl_operator.__name__}_no_0_{col_name}')
                                .cast(pl.Float32)
                            )
                            for pl_operator, col_name in product_operator_numerical_col    
                        ] +
                        [
                            #count on 0 and not
                            pl.col('pmts_dpdvalue_108P')
                            .filter(
                                (pl.col('pmts_dpdvalue_108P')==0)&
                                filter_pl
                            )
                            .count()
                            .alias(f'{name_pl}_equal_0_pmts_dpdvalue_108P').cast(pl.UInt32),
                            
                            pl.col('pmts_dpdvalue_108P')
                            .filter(
                                (pl.col('pmts_dpdvalue_108P')!=0)&
                                filter_pl
                            )
                            .count()
                            .alias(f'{name_pl}_unequal_0_pmts_dpdvalue_108P').cast(pl.UInt32),
                            
                            pl.col('pmts_pmtsoverdue_635A')
                            .filter(
                                (pl.col('pmts_pmtsoverdue_635A')==0)&
                                filter_pl
                            )
                            .count()
                            .alias(f'{name_pl}_equal_0_pmts_pmtsoverdue_635A').cast(pl.UInt32),
                            
                            pl.col('pmts_pmtsoverdue_635A')
                            .filter(
                                (pl.col('pmts_pmtsoverdue_635A')!=0) &
                                filter_pl
                            )
                            .count()
                            .alias(f'{name_pl}_unequal_0_pmts_pmtsoverdue_635A').cast(pl.UInt32),
                        ] +
                        #from 0 to not 0 count and statistic on these breaking point
                        [
                            (
                                pl.col(col)
                                .filter(
                                    filter_pl &
                                    (pl.col(col)!=0) &
                                    (pl.col(col).shift()==0)
                                )
                                .count()
                                .cast(pl.UInt32)
                                .alias(f'{name_pl}_count_change_0_not_0_{col}')
                            )
                            for col in ['pmts_dpdvalue_108P', 'pmts_pmtsoverdue_635A']
                        ] +
                        [
                            (
                                pl_operator(col_name, filter_pl)
                                .filter(
                                    (pl.col(col_name)!=0) &
                                    (pl.col(col_name).shift()==0)
                                )
                                .cast(pl.Float32)
                                .alias(f'{name_pl}_{pl_operator.__name__}_change_0_not_0_{col_name}')
                            )
                            for pl_operator, col_name in product_operator_numerical_col 
                        ] +
                        #from not 0 to 0 count on these breaking point
                        [
                            (
                                pl.col(col)
                                .filter(
                                    filter_pl &
                                    (pl.col(col)==0) &
                                    (pl.col(col).shift()!=0)
                                )
                                .count()
                                .cast(pl.UInt32)
                                .alias(f'{name_pl}_count_change_not_0_to_0_{col}')
                            )
                            for col in ['pmts_dpdvalue_108P', 'pmts_pmtsoverdue_635A']
                        ]
                        for name_pl, filter_pl in [
                            ['group1', pl.col('num_group1')==0],
                            ['all', pl.lit(True)]
                        ]
                    ]
                )
            )
        )
        aggregation_level_group_list += (
            [
                pl.col(col_name).first()
                .alias(f'first_{col_name}')
                for col_name in numeric_columns
            ] +
            [
                pl.col(col_name)
                .filter(
                    (pl.col(col_name).is_not_null()) &
                    (pl.col(col_name) != 0)
                )
                .last()
                .alias(f'last_{col_name}')
                for col_name in numeric_columns
            ] 
        )
        #get feature for each contract
        self.credit_bureau_b_2 = (
            self.credit_bureau_b_2.sort(
                [
                    'case_id', 'num_group1', 'num_group2'
                ]
            ).group_by(by=['case_id', 'num_group1'], maintain_order=True).agg(
                aggregation_level_group_list
            )
        )
        #aggregate contract for each case id
        self.credit_bureau_b_2 = (
            self.credit_bureau_b_2
            .group_by('case_id')
            .agg(
                [
                    pl.col('num_group1').max().alias('num_contract_X').cast(pl.UInt32),
                ] +
                [
                    pl.col(col_name).filter(pl.col('num_group1')==0).first()
                    for col_name in [f'first_{col}' for col in numeric_columns]
                ] +
                [
                    pl.col(col_name)
                    .filter(
                        pl.col('num_group1')==
                        pl.col('num_group1').filter(
                            (pl.col(col_name).is_not_null()) &
                            (pl.col(col_name) != 0)
                        ).max()
                    )
                    .last()
                    for col_name in [f'last_{col}' for col in numeric_columns]
                ] +
                [
                    pl_operation(col_name)
                    .alias(f'{pl_operation.__name__}_over_group2_{col_name}')
                    for pl_operation, col_name in product(
                        self.numerical_aggregator,
                        [
                            col 
                            for col in self.credit_bureau_b_2.columns 
                            if (col not in self.special_column_list) & (col[-1]!='D')
                        ]
                    )
                ] +
                [
                    pl_operation(col_name)
                    .alias(f'{pl_operation.__name__}_over_group2_{col_name}')
                    .cast(pl.Date)
                    for pl_operation, col_name in product(
                        self.date_aggregator + [pl.mean],
                        [
                            col 
                            for col in self.credit_bureau_b_2.columns 
                            if (col not in self.special_column_list) & (col[-1]=='D')
                        ]
                    )
                ]
            )
        )
        
        #downcast to float32
        col_list = self.credit_bureau_b_2.columns
        dtype_list = self.credit_bureau_b_2.dtypes
        col_to_float32 = [
            col_list[i]
            for i in range(len(col_list))
            if dtype_list[i] == pl.Float64
        ]

        self.credit_bureau_b_2 = self.credit_bureau_b_2.with_columns(
            [
                pl.col(col).cast(pl.Float32)
                for col in col_to_float32
            ]
        )
        
        
    def create_debitcard_1_feature(self) -> None:
        
        col_to_retrieve_list: list[str] = [
            'last180dayaveragebalance_704A', 
            'last180dayturnover_1134A', 'last30dayturnover_651A'
        ]

        list_generic_feature: list[pl.Expr] = self.add_generic_feature(
            self.debitcard_1, 'debitcard_1'
        )
        self.debitcard_1 = (
            self.debitcard_1
            .sort(['case_id', 'num_group1'])
            .group_by('case_id', maintain_order=True)
            .agg(
                #first info
                [
                    (
                        pl.col(col_)
                        .filter(pl.col(col_).is_not_null())
                        .first()
                        .cast(pl.Float32)
                        .alias(f'first_not_null_{col_}')
                    )
                    for col_ in col_to_retrieve_list
                ] +
                #last info
                [
                    (
                        pl.col(col_)
                        .filter(pl.col(col_).is_not_null())
                        .last()
                        .cast(pl.Float32)
                        .alias(f'last_not_null_{col_}')
                    )
                    for col_ in col_to_retrieve_list
                ] +
                list_generic_feature
            )
        )

    def create_deposit_1_feature(self) -> None:
        closed_contract_expr: pl.Expr = (
            pl.col('amount_416A')
            .filter(pl.col('contractenddate_991D').is_not_null())
            .cast(pl.Float32)
        )
        open_contract_expr: pl.Expr = (
            pl.col('amount_416A')
            .filter(pl.col('contractenddate_991D').is_null())
            .cast(pl.Float32)
        )
        list_generic_feature: list[pl.Expr] = self.add_generic_feature(
            self.deposit_1, 'deposit_1'
        )

        self.deposit_1 = (
            self.deposit_1
            .group_by('case_id')
            .agg(
                #average range diff for closed contract
                [
                    (
                        (
                            pl.col('contractenddate_991D').cast(pl.Date) - 
                            pl.col('openingdate_313D').cast(pl.Date)
                        )
                        .filter(pl.col('contractenddate_991D').is_not_null())
                        .dt.total_days()
                        .mean()
                        .alias('mean_duration_closed_contract_amount_416A')
                        .cast(pl.Float32)
                    ),
                    (
                        (
                            pl.col('contractenddate_991D').cast(pl.Date) - 
                            pl.col('openingdate_313D').cast(pl.Date)
                        )
                        .filter(
                            (pl.col('contractenddate_991D').is_not_null())&
                            (pl.col('amount_416A') == 0.)
                        )
                        .dt.total_days()
                        .mean()
                        .alias('mean_duration_empty_closed_contract_amount_416A')
                        .cast(pl.Float32)
                    ),
                ] +
                #total amount closed contract
                [
                    (
                        closed_contract_expr
                        .mean()
                        .alias(f'mean_closed_contract_amount_416A')
                    ),
                    (
                        closed_contract_expr
                        .sum()
                        .alias(f'sum_closed_contract_amount_416A')
                    ),
                    (
                        closed_contract_expr
                        .std()
                        .alias(f'std_closed_contract_amount_416A')
                    ),
                    (
                        closed_contract_expr
                        .min()
                        .alias(f'min_closed_contract_amount_416A')
                    ),
                    (
                        closed_contract_expr
                        .max()
                        .alias(f'max_closed_contract_amount_416A')
                    )

                ] + 
                #total amount open contract
                [
                    (
                        open_contract_expr
                        .sum()
                        .alias(f'sum_open_contract_amount_416A')
                    ),
                                (
                        open_contract_expr
                        .mean()
                        .alias(f'mean_open_contract_amount_416A')
                    ),
                    (
                        open_contract_expr
                        .std()
                        .alias(f'std_open_contract_amount_416A')
                    ),
                    (
                        open_contract_expr
                        .min()
                        .alias(f'min_open_contract_amount_416A')
                    ),
                    (
                        open_contract_expr
                        .max()
                        .alias(f'max_open_contract_amount_416A')
                    )

                ] +
                #number close empty contract
                [
                    (
                        
                        pl.col('amount_416A')
                        .filter(
                            (pl.col('contractenddate_991D').is_not_null()) &
                            (pl.col('amount_416A') == 0.)
                        )
                        .count()
                        .alias(f'number_empty_closed_contract_amount_416A')
                        .cast(pl.UInt16)
                    ),
                    (
                        
                        pl.col('amount_416A')
                        .filter(
                            (pl.col('contractenddate_991D').is_null()) &
                            (pl.col('amount_416A') == 0.)
                        )
                        .count()
                        .alias(f'number_empty_open_contract_amount_416A')
                        .cast(pl.UInt16)
                    )
                ] +
                # #number non empty contract
                [
                    (
                        
                        pl.col('amount_416A')
                        .filter(
                            (pl.col('contractenddate_991D').is_not_null()) &
                            (pl.col('amount_416A') > 0.)
                        )
                        .count()
                        .alias(f'number_not_empty_closed_contract_amount_416A')
                        .cast(pl.UInt16)
                    ),
                    (
                        
                        pl.col('amount_416A')
                        .filter(
                            (pl.col('contractenddate_991D').is_null()) &
                            (pl.col('amount_416A') > 0.)
                        )
                        .count()
                        .alias(f'number_not_empty_open_contract_amount_416A')
                        .cast(pl.UInt16)
                    ),
                ] +
                [
                    (
                        pl.col('num_group1')
                        .filter(pl.col('contractenddate_991D').is_not_null())
                        .max()
                        .alias('number_not_closed_contractX')
                        .cast(pl.UInt16)
                    ),
                    (
                        pl.col('num_group1')
                        .filter(pl.col('contractenddate_991D').is_null())
                        .max()
                        .alias('number_closed_contractX')
                        .cast(pl.UInt16)
                    )
                ] +
                #generic feature
                list_generic_feature
            )
        )

    def create_static_0_feature(self) -> None:
        self.static_0 = self.filter_and_select_first(
            data=self.static_0, filter_col=pl.lit(True),
            col_list=self.static_0.columns
        )
        
        self.static_0 = self.static_0.drop(
            'previouscontdistrict_112M', 'deferredmnthsnum_166L',
            'applicationcnt_361L'
        )
        self.static_0 = self.static_0.with_columns(            
            #NUMERIC FEATURE
            (pl.col('maxdpdlast24m_143P')-pl.col('avgdbddpdlast24m_3658932P')).alias('delinquency_interactionX').cast(pl.Int32),
            (pl.col('numinstpaidearly_338L')-pl.col('pctinstlsallpaidlate1d_3546856L')).alias('pmt_efficiencyX').cast(pl.Float32),
            (pl.col('totaldebt_9A')-pl.col('avgpmtlast12m_4525200A')).alias('dept_burden_pmtX').cast(pl.Float32),
            (pl.col('clientscnt_304L')-pl.col('maxdpdlast12m_727P')).alias('network_delinquencyX').cast(pl.Int32),
            (pl.col('credamount_770A')-pl.col('numinstlswithdpd10_728L')).alias('loan_size_riskX').cast(pl.Float32),
            (pl.col('maininc_215A')-pl.col('totaldebt_9A')).alias('income_debtX').cast(pl.Float32),
            (pl.col('clientscnt_1022L')-pl.col('avgdbddpdlast3m_4187120P')).alias('mobile_rmptX').cast(pl.Int32),
            (pl.col('numpmtchanneldd_318L')-pl.col('maxdpdlast6m_474P')).alias('pmt_channel_delinquencyX').cast(pl.Int32),
            
            (
                (
                    pl.col('numinstpaidearly_338L')/
                    (
                        pl.lit(1) +
                        pl.col('numinstpaidearly_338L') + 
                        pl.col('pctinstlsallpaidlate1d_3546856L')
                    )
                ).alias('dept_burden_rpmt_interactionX').cast(pl.Float32)
            ),
            (
                (
                    pl.col('maxdpdlast12m_727P')/
                    (
                        pl.lit(1) +
                        pl.col('clientscnt_304L')
                    )
                ).alias('client_past_due_intX').cast(pl.Float32)
            ),
            (
                pl.col('credamount_770A')/
                (
                    pl.lit(1) +
                    pl.col('numinstlswithdpd10_728L')
                )
            ).alias('trend_highloan_overdue_instX').cast(pl.Float32),
        )
    
    def create_static_cb_0_feature(self) -> None:
        def consecutive_pairs(list_col: list[str]) -> list[list[str]]:
            return [
                [list_col[i], list_col[j]] 
                for i in range(len(list_col)) 
                for j in range(i + 1, len(list_col))
            ]
        list_utils_col = {
            'number_cba_queries': [
                'days30_165L', 'days90_310L', 
                'days120_123L', 'days180_256L', 'days360_512L'
            ],
            'credit_history': [
                'fortoday_1092L', 'forweek_528L', 'formonth_535L', 'forquarter_634L', 
                'foryear_850L', 'for3years_504L'
            ],
            'number_results': [
                'firstquarter_103L', 'secondquarter_766L', 'thirdquarter_1082L',
                'fourthquarter_440L'
            ],
            'number_cancelations': [
                'foryear_818L', 'for3years_584L'
            ]
        }
        list_operator = []
        for name_feature, list_col in list_utils_col.items():
            list_operator.append(
                pl.sum_horizontal(
                    pl.col(list_col)
                ).alias(f'{name_feature}_sum_sumX').cast(pl.Int32)
            )
            #difference
            list_operator += [
                (
                    (pl.col(col_2) - pl.col(col_1))
                ).alias(f'{col_1}_{col_2}_diffX').cast(pl.Int32)
                for col_1, col_2 in consecutive_pairs(list_col)
            ]

        list_operator += [
            (
                pl.col(col_1) -
                pl.col(col_2)
            ).dt.total_days().cast(pl.Int32).alias(f'{col_1}_diff_{col_2}')
            for col_1, col_2 in [
                ['responsedate_1012D', 'assignmentdate_238D'],
                #always 0
                ['responsedate_4527233D', 'assignmentdate_4527235D'],
                ['responsedate_4917613D', 'assignmentdate_4955616D']
            ]
        ]

        #ensure no duplicates
        self.static_cb_0 = self.filter_and_select_first(
            data=self.static_cb_0, filter_col=pl.lit(True),
            col_list=self.static_cb_0.columns
        )
        
        self.static_cb_0 = self.static_cb_0.with_columns(
            list_operator
        ).with_columns(
            #coalesce multiple inffo
            pl.coalesce(
                pl.col(
                    [
                        'responsedate_1012D_diff_assignmentdate_238D',
                        'responsedate_4917613D_diff_assignmentdate_4955616D',
                        'responsedate_4527233D_diff_assignmentdate_4527235D',
                    ]
                ).alias('coalesce_responsedate_assignmentdateX').cast(pl.Int32)
            ),
            pl.coalesce(
                pl.col(
                    [
                        'responsedate_1012D', 'responsedate_4917613D', 'responsedate_4527233D'
                    ]
                ).alias('coalesce_responsedate_D').cast(pl.Int32)
            ),
            pl.coalesce(
                pl.col(
                    [
                        'assignmentdate_238D', 'assignmentdate_4955616D', 'assignmentdate_4527235D'
                    ]
                ).alias('coalesce_assignmentdate_D').cast(pl.Int32)
            ),
            (
                pl.sum_horizontal(
                    [
                        'pmtaverage_3A', 'pmtaverage_4527227A', 
                        'pmtaverage_4955615A'
                    ]
                )/3
            ).alias('mean_pmtaverageX').cast(pl.Float32),
            (
                pl.sum_horizontal(
                    [
                        'pmtcount_4527229L', 'pmtcount_4955617L', 
                        'pmtcount_693L', 'pmtscount_423L',
                    ]
                )/4
            ).alias('mean_pmtcountX').cast(pl.UInt16),
            (
                pl.sum_horizontal(
                    ['contractssum_5085716L', 'pmtssum_45A']
                )/2
            ).alias('mean_pmtssum').cast(pl.Float32)
        ).drop(
            [
                'birthdate_574D', 'dateofbirth_337D', 'dateofbirth_342D',
                'for3years_128L',
                'description_5085714M', 
                'forweek_601L', 'forquarter_462L', 'foryear_618L', 
                'formonth_118L',
                'forweek_1077L', 'formonth_206L', 'forquarter_1017L', 
                'responsedate_1012D', 'responsedate_4527233D', 'responsedate_4917613D',
                'assignmentdate_238D', 'assignmentdate_4527235D', 'assignmentdate_4955616D',
                'pmtaverage_3A', 'pmtaverage_4527227A', 'pmtaverage_4955615A',
                'pmtcount_4527229L', 'pmtcount_4955617L', 'pmtcount_693L', 'pmtscount_423L',
                'contractssum_5085716L', 'pmtssum_45A', 'requesttype_4525192L',
            ]
        )

    def create_tax_registry_a_1_feature(self) -> None:
        self.tax_registry_a_1 = (
            self.tax_registry_a_1
            .group_by('case_id')
            .agg(
                (
                    pl.col('amount_4527230A').filter(pl.col('num_group1')==0)
                    .first()
                    .cast(pl.Float32).alias('first_amount_4527230A')                    
                ),
                (
                    pl.col('amount_4527230A').filter(pl.col('num_group1')==pl.col('num_group1').max())
                    .first()
                    .cast(pl.Float32).alias('last_amount_4527230A')                    
                ),
                (
                    pl.col('amount_4527230A').min()
                    .cast(pl.Float32).alias('min_amount_4527230A')
                ),
                (
                    pl.col('amount_4527230A').max()
                    .cast(pl.Float32).alias('max_amount_4527230A')
                ),
                (
                    pl.col('amount_4527230A').sum()
                    .cast(pl.Float32).alias('sum_amount_4527230A')
                ),
                (
                    pl.col('amount_4527230A').std()
                    .cast(pl.Float32).alias('std_amount_4527230A')
                ),
                (
                    pl.col('amount_4527230A').mean()
                    .cast(pl.Float32).alias('mean_amount_4527230A')
                ),
                (
                    pl.col('num_group1').max()
                    .cast(pl.UInt16).alias('num_deductionX')
                ),
                (
                    pl.col('name_4527232M').n_unique()
                    .cast(pl.UInt16).alias('number_workerX')
                ),
                (
                    pl.col('recorddate_4527225D').first()
                    .cast(pl.Date)
                )
            )
        )
        
    def create_tax_registry_b_1_feature(self) -> None:
        self.tax_registry_b_1 = (
            self.tax_registry_b_1
            .group_by('case_id')
            .agg(
                (
                    pl.col('amount_4917619A').filter(pl.col('num_group1')==0)
                    .first()
                    .cast(pl.Float32).alias('first_amount_4917619A')                    
                ),
                (
                    pl.col('amount_4917619A').filter(pl.col('num_group1')==pl.col('num_group1').max())
                    .first()
                    .cast(pl.Float32).alias('last_amount_4917619A')                    
                ),
                (
                    pl.col('amount_4917619A').min()
                    .cast(pl.Float32).alias('min_amount_4917619A')
                ),
                (
                    pl.col('amount_4917619A').max()
                    .cast(pl.Float32).alias('max_amount_4917619A')
                ),
                (
                    pl.col('amount_4917619A').sum()
                    .cast(pl.Float32).alias('sum_amount_4917619A')
                ),
                (
                    pl.col('amount_4917619A').std()
                    .cast(pl.Float32).alias('std_amount_4917619A')
                ),
                (
                    pl.col('amount_4917619A').mean()
                    .cast(pl.Float32).alias('mean_amount_4917619A')
                ),
                (
                    pl.col('num_group1').max()
                    .cast(pl.UInt16).alias('num_deductionX')
                ),
                (
                    pl.col('name_4917606M').n_unique()
                    .cast(pl.UInt16).alias('number_workerX')
                ),
                (
                    pl.col('deductiondate_4917603D').min()
                    .cast(pl.Date)
                    .alias('min_deductiondate_4917603D')
                ),
                (
                    pl.col('deductiondate_4917603D').max()
                    .cast(pl.Date)
                    .alias('max_deductiondate_4917603D')
                ),
                (
                    (
                        pl.max('deductiondate_4917603D') - pl.min('deductiondate_4917603D')
                    ).dt.total_days()
                    .cast(pl.UInt16)
                    .alias(f'range_deductiondate_4917603D')
                )
            )
        )
        
    def create_tax_registry_c_1_feature(self) -> None:
        self.tax_registry_c_1 = (
            self.tax_registry_c_1
            .group_by('case_id')
            .agg(
                (
                    pl.col('pmtamount_36A').filter(pl.col('num_group1')==0)
                    .first()
                    .cast(pl.Float32).alias('first_pmtamount_36A')                    
                ),
                (
                    pl.col('pmtamount_36A').filter(pl.col('num_group1')==pl.col('num_group1').max())
                    .first()
                    .cast(pl.Float32).alias('last_pmtamount_36A')                    
                ),
                (
                    pl.col('pmtamount_36A').min()
                    .cast(pl.Float32).alias('min_pmtamount_36A')
                ),
                (
                    pl.col('pmtamount_36A').max()
                    .cast(pl.Float32).alias('max_pmtamount_36A')
                ),
                (
                    pl.col('pmtamount_36A').sum()
                    .cast(pl.Float32).alias('sum_pmtamount_36A')
                ),
                (
                    pl.col('pmtamount_36A').std()
                    .cast(pl.Float32).alias('std_pmtamount_36A')
                ),
                (
                    pl.col('pmtamount_36A').mean()
                    .cast(pl.Float32).alias('mean_pmtamount_36A')
                ),
                (
                    pl.col('num_group1').max()
                    .cast(pl.UInt16).alias('num_deductionX')
                ),
                (
                    pl.col('employername_160M').n_unique()
                    .cast(pl.UInt16).alias('number_workerX')
                ),
                (
                    pl.col('processingdate_168D').min()
                    .cast(pl.Date)
                    .alias('min_processingdate_168D')
                ),
                (
                    pl.col('processingdate_168D').max()
                    .cast(pl.Date)
                    .alias('max_processingdate_168D')
                ),
                (
                    (
                        pl.max('processingdate_168D') - pl.min('processingdate_168D')
                    ).dt.total_days()
                    .cast(pl.UInt16)
                    .alias(f'range_processingdate_168D')
                )
            )
        )                
        
    def create_person_1_feature(self) -> None:
        select_col_group_1 = [
            'case_id',
            'birth_259D', 
            'contaddr_matchlist_1032L', 'contaddr_smempladdr_334L', 
            'education_927M',
            'empl_employedfrom_271D',
            'empl_industry_691L', 'familystate_447L',
            'housetype_905L', 
            'incometype_1044T', 'isreference_387L', 'language1_981M',
            'mainoccupationinc_384A',
            'role_1084L', 'role_993L',
            'safeguarantyflag_411L', 'type_25L', 'sex_738L'
        ]
        person_1 = self.filter_and_select_first(
            data=self.person_1,
            filter_col=pl.col('num_group1')==0,
            col_list=select_col_group_1
        ).with_columns(
            (
                pl.col('empl_employedfrom_271D') - 
                pl.col('birth_259D')
            ).dt.total_days()
            .alias('empl_employedfrom_271D_diff_birth_259D')
            .cast(pl.Int32)
        )

        dict_agg_info: Dict[str, list[str]] = {
            'education_927M': [
                "a55475b1", "P33_146_175",
                "P97_36_170",
            ],
            'gender_992L': ['M', 'F'],
            'housingtype_772L': [
                'OWNED', 'PARENTAL'
            ],
            'maritalst_703L': [
                'MARRIED', 
                'SINGLE'
            ],
            'persontype_1072L': [
                1, 4, 5
            ],
            'persontype_792L': [4, 5],
            'relationshiptoclient_415T': [
                'SPOUSE', 'OTHER_RELATIVE', 
                'COLLEAGUE', 'GRAND_PARENT',
                'NEIGHBOR', 'OTHER', 'PARENT',
                'SIBLING', 'CHILD', 'FRIEND'
            ],
            'relationshiptoclient_642T': [
                'SIBLING', 'SPOUSE', 'OTHER', 'COLLEAGUE', 
                'PARENT', 'FRIEND', 'NEIGHBOR', 'GRAND_PARENT', 
                'CHILD', 'OTHER_RELATIVE'
            ],
            'role_1084L': ['CL', 'EM', 'PE'],
            'role_993L': ['FULL'],
            'type_25L': [
                'HOME_PHONE', 'PRIMARY_MOBILE', 
                'SECONDARY_MOBILE', 'ALTERNATIVE_PHONE', 
                'PHONE'
            ]
        }

        person_1_related_info: Union[pl.LazyFrame, pl.DataFrame] = (
            self.person_1.filter(
                pl.col('num_group1')!=0
            ).group_by('case_id').agg(
                [
                    pl.len().alias('number_rowsX').cast(pl.UInt16),
                    pl.col('childnum_185L').max().cast(pl.Int16),
                ] +
                [
                    (
                        pl.col(column_name)
                        .filter(
                            (
                                pl.col(column_name)==
                                (
                                    #not string
                                    col_value 
                                    if column_name not in self.mapper_mask['person_1'].keys()
                                    #use mapper
                                    else self.mapper_mask['person_1'][column_name][col_value]
                                )
                            )
                        )
                        .count()
                        .alias(f'{column_name}_{col_value}_count_X')
                        .cast(pl.UInt16)
                    )
                    for column_name, col_value in [
                        (column_name, col_value) 
                        for column_name, col_value_list in dict_agg_info.items() 
                        for col_value in col_value_list
                    ]
                ] +
                [
                    (
                        pl.col(column_name)
                        .filter(
                            (pl.col(column_name).is_null())
                        )
                        .len()
                        .alias(f'{column_name}_countnull_X')
                        .cast(pl.UInt16)
                    )
                    for column_name in [
                        'gender_992L', 'housingtype_772L', 
                        'maritalst_703L', 'persontype_1072L', 
                        'persontype_792L', 'relationshiptoclient_415T', 
                        'relationshiptoclient_642T', 'role_1084L', 'role_993L', 
                        'type_25L'
                    ]
                ]
            )
        ).with_columns(
            [
                (
                    (
                        pl.col('number_rowsX') - 
                        pl.col(f'{col}_countnull_X')
                    )
                    .alias(f'{col}_number_info_x')
                    .cast(pl.UInt16)
                )
                for col in [
                    'gender_992L', 'housingtype_772L', 
                    'maritalst_703L', 'persontype_1072L', 
                    'persontype_792L', 'relationshiptoclient_415T', 
                    'relationshiptoclient_642T', 'role_1084L', 'role_993L', 
                    'type_25L'
                ]
            ]
        )
        
        self.person_1 = (
            person_1.join(
                person_1_related_info, 
                on='case_id', how='left'
            )
        )
        
        
    def create_person_2_feature(self) -> None:
        zero_n_list = {
            'addres_role_871L': ['CONTACT', 'PERMANENT'],
            'conts_role_79M': ['P38_92_157', 'a55475b1']
        }
        n_0_list = {
            'addres_role_871L': ['TEMPORARY', 'REGISTERED', 'CONTACT', 'PERMANENT'],
            'conts_role_79M': [
                'P125_14_176',
                'P177_137_98', 'P115_147_77',
                'a55475b1', 'P125_105_50', 
                'P38_92_157', 'P7_147_157',
            ],
            'relatedpersons_role_762T': [
                    'OTHER', 'CHILD', 'SIBLING',
                    'PARENT', 'OTHER_RELATIVE',
                    'COLLEAGUE', 'SPOUSE',
                    'FRIEND'
            ]
        }
        #for num group1 == 0
        zero_n_list = list(
            chain(
                *[
                    list(product([key], value))
                    for key, value in zero_n_list.items()
                ]
            )
        )
        #for num group1 != 0
        n_0_list = list(
            chain(
                *[
                    list(product([key], value))
                    for key, value in n_0_list.items()
                ]
            )
        )

        date_expression: list[pl.Expr] = [
            #first not blank
            (
                pl.col('empls_employedfrom_796D')
                .drop_nulls()
                .max()
                .alias('date_empls_employedfrom_796D')
                .cast(pl.Date)
            )
        ]

        pl_group_2_expression: list[pl.Expr] = (
            date_expression +
            [
                #max num group1
                (
                    pl.col('num_group2').max()
                    .alias('max_num_group2')
                    .cast(pl.UInt16)
                )
            ] +
            [
                (
                    pl.col(col).filter(
                        (pl.col(col)==self.mapper_mask['person_2'][col][single_value])&
                        (pl.col('num_group1')==0)
                    )
                    .count()
                    .alias(f'{col[:-1]}_{single_value}_0_n_' + col[-1])
                    .cast(pl.UInt16)
                )
                for col, single_value in zero_n_list
            ] +
            [
                (
                    pl.col(col).filter(
                        (pl.col(col)==self.mapper_mask['person_2'][col][single_value])&
                        (pl.col('num_group1')!=0)
                    )
                    .count()
                    .alias(f'{col[:-1]}_{single_value}_n_0_' + col[-1])
                    .cast(pl.UInt16)
                )
                for col, single_value in n_0_list
            ] +
            #total unique
            [
                (
                    pl.col(col_name)
                    .n_unique()
                    .alias(f'n_unique_{col_name}')
                    .cast(pl.UInt16)
                )
                for col_name in [
                    'addres_district_368M', 'addres_zip_823M',
                    'addres_role_871L', 'conts_role_79M',
                    'relatedpersons_role_762T',
                    'empls_economicalst_849M', 
                ]
            ] +
            #how many not hashed unique
            [
                (
                    pl.col(col_name)
                    .filter(
                        (
                            pl.col(col_name) != 
                            (self.mapper_mask['person_2'][col_name][self.hashed_missing_label])
                        )
                    )
                    .n_unique()
                    .alias(f'n_unique_not_hashed_{col_name}')
                    .cast(pl.UInt16)
                )
                for col_name in [
                    'addres_district_368M', 'addres_zip_823M',
                    'conts_role_79M',
                    'empls_economicalst_849M', 'empls_employer_name_740M'
                ]
            ] +
            #how many hashed
            [
                (
                    pl.col(col_name)
                    .filter(
                        (
                            pl.col(col_name) == 
                            (self.mapper_mask['person_2'][col_name][self.hashed_missing_label])
                        )
                    )
                    .count()
                    .alias(f'n_unique_hashed_{col_name}')
                    .cast(pl.UInt16)
                )
                for col_name in [
                    'addres_district_368M', 'addres_zip_823M',
                    'conts_role_79M', 'empls_economicalst_849M',
                    'empls_employer_name_740M'
                ]
            ]
        )        
        
        self.person_2 = self.person_2.group_by(
            'case_id', 'num_group1'
        ).agg(pl_group_2_expression)
        
        date_col_list = ['date_empls_employedfrom_796D']
        
        column_to_split = [
            col for col in 
            self.person_2.columns
            if col not in ['case_id', 'num_group1']
        ]
        
        col_group_zero = [
            col 
            for col in column_to_split 
            if 
                ('_n_0_' not in col) &
                (col not in date_col_list)
        ]
        
        col_group_not_zero = [
            col 
            for col in column_to_split 
            if 
                ('_0_n_' not in col) &
                (col not in date_col_list)
        ]

        self.person_2 = self.person_2.group_by('case_id').agg(
            [
                #max num group1
                (
                    pl.col('num_group1').max()
                    .alias('max_num_group1')
                    .cast(pl.UInt16)
                )
            ] +
            [
                pl.max('date_empls_employedfrom_796D').alias(f'date_empls_employedfrom_796D'),
            ] +
            [
                (
                    getattr(
                        pl.col(col_name)
                        .filter(pl.col('num_group1')==0),
                        pl_expression
                    )()
                    .alias(f'group_0_{pl_expression}_{col_name}')
                )
                for pl_expression, col_name in product(
                    ['min', 'max', 'mean'],
                    col_group_zero
                )
            ] +
            [
                getattr(
                    pl.col(col_name)
                    .filter(pl.col('num_group1')!=0),
                    pl_expression
                )()
                .alias(f'group_not_0_{pl_expression}_{col_name}')
                for pl_expression, col_name in product(
                    ['min', 'max', 'mean'],
                    col_group_not_zero
                )
            ]
        ).with_columns(
            [
                pl.col(col)
                .cast(pl.Float32)
                for col in 
                    [
                        f'group_0_mean_{col_name}'
                        for col_name in col_group_zero
                    ]+ 
                    [
                        f'group_not_0_mean_{col_name}'
                        for col_name in col_group_not_zero
                    ]
            ]
        )
    
    def create_applprev_1_feature(self) -> None:
        self.applprev_1 = self.applprev_1.drop('district_544M', 'profession_152M', 'postype_4733339M')

        #filter past
        self.applprev_1 = self.applprev_1.join(
            self.base_data.select('case_id', 'date_decision'),
            on='case_id', how='left'
        ).filter(
            (pl.col('date_decision') > pl.col('creationdate_885D'))
        ).drop('date_decision')

        list_generic_feature: list[pl.Expr] = self.add_generic_feature(
            self.applprev_1, 'applprev_1'
        )
        date_col_diff: list[str] = [
            'dateactivated_425D', 'dtlastpmt_581D', 
            'dtlastpmtallstes_3545839D', 
            'employedfrom_700D', 'firstnonzeroinstldate_307D'
        ]
        dtype_select = {
            'min': pl.Int32,
            'max': pl.Int32,
            'mean': pl.Float32
        }
        self.applprev_1 = (
            self.applprev_1
            #utils for aggregation
            .with_columns(
                [
                    (pl.col(col) - pl.col('creationdate_885D')).dt.total_days()
                    .alias(f'diff_{col}_creationdate_885D')
                    for col in date_col_diff
                ] +
                [
                    (pl.col(col) - pl.col('approvaldate_319D')).dt.total_days()
                    .alias(f'diff_{col}_approvaldate_319D')
                    for col in date_col_diff
                ]
            )
            .group_by('case_id')
            .agg(
                #diff aggregation to creationdate_885D
                [
                    pl_operator(f'diff_{col}_creationdate_885D')
                    .alias(f'{pl_operator.__name__}_duration_{col}_creationdate_885D')
                    .cast(dtype_select[pl_operator.__name__])
                    for pl_operator, col in product(
                        [pl.mean, pl.min, pl.max],
                        date_col_diff
                    )
                ] +
                #diff to approvaldate_319D
                [
                    pl_operator(f'diff_{col}_approvaldate_319D')
                    .alias(f'{pl_operator.__name__}_duration_{col}_approvaldate_319D')
                    .cast(dtype_select[pl_operator.__name__])
                    for pl_operator, col in product(
                        [pl.mean, pl.min, pl.max],
                        date_col_diff
                    )
                ] +
                list_generic_feature
            )
        )
            
    def create_applprev_2_feature(self) -> None:

        categorical_columns_list = [
            'cacccardblochreas_147M', 'conts_type_509L',
            'credacc_cards_status_52L'
        ]
        features_list = self.add_generic_feature(self.applprev_2, 'applprev_2')
        
        self.applprev_2 = self.applprev_2.group_by(
            ['case_id', 'num_group1']
        ).agg(features_list)
        
        operation_aggregation_list: list[pl.Expr] = (
            [
                pl.col(col_name).filter(pl.col('num_group1')==0)
                .first()
                for col_name in [f'first_{col}' for col in categorical_columns_list]
            ] +
            [
                pl.col(col_name)
                .filter(pl.col('num_group1')==pl.col('num_group1').filter(pl.col(col_name).is_not_null()).max())
                .last()
                for col_name in [f'last_{col}' for col in categorical_columns_list]
            ] +
            [
                pl.col('num_group1')
                .max()
                .alias('max_num_group1')
            ]
        )
        
        operation_aggregation_list += (
            list(
                chain(
                    *[
                        (
                            [
                                pl.col(f'not_hashed_missing_mode_cacccardblochreas_147M')
                                .filter(filter_pl)
                                .drop_nulls()
                                .mode()
                                .sort().first()
                                .alias(f'{name_pl}_mode_not_hashed_missing_mode_cacccardblochreas_147M'),
                            ] +
                            [
                                pl_operator(col_name, filter_pl)
                                .alias(f'{pl_operator.__name__}_{name_pl}_{col_name}')
                                for pl_operator, col_name in product(
                                    self.numerical_filter_aggregator,
                                    ['max_num_group2']
                                )
                            ] +
                            [
                                pl.col(f'mode_{col}')
                                .filter(filter_pl)
                                .mode().sort().first()
                                .alias(f'num_{name_pl}_mode_mode_{col}')
                                for col in categorical_columns_list
                            ]
                        )
                        for name_pl, filter_pl in [
                            ['group1', pl.col('num_group1')==0],
                            ['all', pl.lit(True)]
                        ]

                    ]
                )
            )
        )
        self.applprev_2 = self.applprev_2.group_by(
            'case_id'
        ).agg(
            operation_aggregation_list     
        )

    def create_other_1_feature(self) -> None:
        self.other_1 = self.filter_and_select_first(
            data=self.other_1,
            filter_col=pl.lit(True),
            col_list=[
                'case_id',
                'amtdebitincoming_4809443A', 'amtdebitoutgoing_4809440A', 
                'amtdepositbalance_4809441A', 'amtdepositincoming_4809444A',	
                'amtdepositoutgoing_4809442A'
            ]
        ).with_columns(
            (
                pl.col('amtdebitincoming_4809443A')-
                pl.col('amtdebitoutgoing_4809440A')
            ).alias('amtdebitnetX').cast(pl.Float32),
            (
                pl.col('amtdepositincoming_4809444A')-
                pl.col('amtdepositoutgoing_4809442A')
            ).alias('amtdepositnetX').cast(pl.Float32),
        )

    def create_null_feature(self, dataset_name: str) -> None:
        current_dataset: Union[pl.LazyFrame, pl.DataFrame] = getattr(
            self, dataset_name
        )
        current_dataset = current_dataset.with_columns(
            [
                pl.sum_horizontal(
                    pl.col(
                        [col for col in current_dataset.columns if col[-1] == type_col]
                    ).is_null()
                ).alias(f'count_null_{type_col}').cast(pl.UInt16)
                for type_col in ['A', 'L', 'P', 'T', 'D', 'M']
                if any([col for col in current_dataset.columns if col[-1] == type_col])
            ] +
            [
                pl.sum_horizontal(
                    pl.col(
                        [col for col in current_dataset.columns if col[-1] in ['A', 'L', 'P', 'T', 'D', 'M']]
                    ).is_null()
                ).alias('all_count_null_X').cast(pl.UInt16)
            ]
        )
        setattr(
            self,
            dataset_name,
            current_dataset
        )

    def create_feature(self) -> None:        
        for dataset in self.used_dataset:
            current_dataset_fe_pipeline: callable = getattr(
                self, f'create_{dataset}_feature'
            )
            current_dataset_fe_pipeline()
            if dataset != 'base_data':
                self.create_null_feature(dataset_name=dataset)
            
        if not self.inference:
            self.add_fold_column()

    def add_dataset_name_to_feature(self) -> None: 
        for dataset in self.used_dataset:
            current_dataset: Union[pl.LazyFrame, pl.DataFrame] = getattr(
                self, dataset
            )
            assert not any(
                [
                    col in self.special_column_list
                    for col in current_dataset.columns
                    if col != 'case_id'
                ]
            )
            setattr(
                self, 
                dataset,  
                current_dataset.rename(
                    {
                        col: dataset + '_' + col
                        for col in current_dataset.columns
                        if col != 'case_id'
                    }
                )
            )
    def drop_tax_reg_feature(self) -> None:
        """Drop tax registry a, b, c feature"""
        self.data = self.data.drop(
            [
                col
                for col in self.data.columns
                if
                    ('tax_registry_a_1' in col) |
                    ('tax_registry_b_1' in col) |
                    ('tax_registry_c_1' in col)
            ]
        )
    def drop_useless_feature_manual(self) -> None:
        """Drop manually useless feature"""
        #drop useless null features
        useless_feature = [
            col for col in [
                'other_1_count_null_A', 'other_1_all_count_null_X',
                'tax_registry_a_1_count_null_D', 'tax_registry_b_1_count_null_D', 'tax_registry_c_1_count_null_D',
                'applprev_1_min_isbidproduct_390L', 'static_cb_0_count_null_M',
                'static_0_count_null_M', 'person_1_count_null_A',
                'person_1_count_null_T', 'person_1_count_null_M',
                'credit_bureau_a_1_std_debtoverdue_47A', 'credit_bureau_a_1_std_debtoutstand_525A',
                'credit_bureau_b_2_count_null_A', 'credit_bureau_b_2_count_null_P',
                'credit_bureau_b_2_count_null_D', 'credit_bureau_b_2_all_count_null_X',
                'applprev_2_filtered_std_group1_max_num_group2'
            ]
            if col in self.data.columns
        ]
        self.data = self.data.drop(useless_feature)

    def add_null_feature(self) -> None:
        """Add null count feature"""
        self.data = self.data.with_columns(
            pl.sum_horizontal(
                pl.col(
                    [
                        col for col in self.data.columns
                        if 'all_count_null_X' in col
                    ]
                )
            ).cast(pl.UInt16).alias('base_all_count_null_X')
        )
        
        
    def add_tax_registration_merge(self) -> None:
        """
        Tax registration a1, b2, c2 are different dataset merge them on a same feature
        doing average over main feature
        """
        amount_list: list[str] = [
            'tax_registry_a_1_{operation}_amount_4527230A',
            'tax_registry_b_1_{operation}_amount_4917619A',
            'tax_registry_c_1_{operation}_pmtamount_36A',
        ]
        self.data = self.data.with_columns(
            [
                (
                    pl.sum_horizontal(
                        pl.col(
                            [
                                amount.format(operation=operation) 
                                for amount in amount_list
                            ]
                        )/3
                    )
                    .alias(f'tax_registry_1_{operation}_amountX')
                    .cast(pl.Float32)
                )
                for operation in ['first', 'last', 'min', 'max', 'sum', 'mean', 'std']
            ] +
            [
                (
                    pl.sum_horizontal(
                        pl.col(
                            [
                                'tax_registry_a_1_num_deductionX',
                                'tax_registry_b_1_num_deductionX',
                                'tax_registry_c_1_num_deductionX'
                            ]
                        )/3
                    )
                    .alias(f'tax_registry_1_num_deductionX')
                    .cast(pl.Float32)
                )
            ] +
            [
                (
                    pl.sum_horizontal(
                        pl.col(
                            [
                                'tax_registry_a_1_number_workerX',
                                'tax_registry_b_1_number_workerX',
                                'tax_registry_c_1_number_workerX'
                            ]
                        )/3
                    )
                    .alias(f'tax_registry_1_number_workerX')
                    .cast(pl.Float32)
                )
            ] +
            [
                (
                    pl.sum_horizontal(
                        pl.col(
                            [
                                'tax_registry_b_1_range_deductiondate_4917603D',
                                'tax_registry_c_1_range_processingdate_168D'
                            ]
                        )/2
                    )
                    .alias(f'tax_registry_1_range_date')
                    .cast(pl.UInt16)
                )
            ]
        )
        
    def add_difference_to_date_decision(self) -> None:
        """
        Everything which isn't touched until now and it's a date
        will be confrontend with date decision
        """
        type_list = self.data.dtypes
        
        dates_to_transform = [
            col for i, col in enumerate(self.data.columns)
            if (col[-1]=='D') & (type_list[i] == pl.Date)
        ]
        not_allowed_negative_dates = [
            col
            for col in dates_to_transform
            if col not in self.negative_allowed_dates_date_decision
        ]
        
        #calculate day diff respect to date_decision
        self.data = self.data.with_columns(
            [
                (
                    (
                        pl.col('date_decision') - pl.col(col)
                    )
                    .dt.total_days()
                    .cast(pl.Int32).alias(col)
                )
                for col in dates_to_transform
            ]
        ).with_columns(
            #put blank wrong dates
            [
                (
                    pl.when((pl.col(col) <0))
                    .then(None)
                    .otherwise(pl.col(col))
                    .cast(pl.UInt32).alias(col)
                )
                for col in not_allowed_negative_dates
            ]
        )

    def filter_useless_columns(self, dataset: str) -> None:
        self.filter_empty_columns(dataset=dataset)
        self.filter_sparse_categorical(dataset=dataset)
        if dataset != 'person_1':
            self.filter_only_hashed_categorical(dataset=dataset)
        
    def filter_only_hashed_categorical(self, dataset: str) -> None:
        data: pl.LazyFrame = getattr(self, dataset)
        categorical_list_col: list[str] = [
            col for col in self.mapper_mask[dataset].keys()
            if col in data.columns
        ]
        for col in categorical_list_col:
            if col not in self.special_column_list:
                if 'hashed_pct' in self.mapper_statistic[dataset][col].keys():
                    hashed_pct = self.mapper_statistic[dataset][col]['hashed_pct']
                    pct_null = self.mapper_statistic[dataset][col]['pct_null']
                    if ((hashed_pct + pct_null) > 0.5):
                        data = data.drop(col)
        
        setattr(
            self, dataset, data
        )

    def filter_sparse_categorical(self, dataset: str) -> None:
        data: pl.LazyFrame = getattr(self, dataset)
        categorical_list_col: list[str] = [
            col for col in self.mapper_mask[dataset].keys()
            if col in data.columns
        ]
        for col in categorical_list_col:
            if col not in self.special_column_list:
                n_unique = self.mapper_statistic[dataset][col]['n_unique']
                if (n_unique == 1) | (n_unique > 200):
                    data = data.drop(col)
        
        setattr(
            self, dataset, data
        )

    def filter_empty_columns(self, dataset: str) -> None:
        data: pl.LazyFrame = getattr(self, dataset)
        for col in data.columns:
            if col not in self.special_column_list:
                pct_null = self.mapper_statistic[dataset][col]['pct_null']
                if pct_null > 0.7:
                    data = data.drop(col)
        
        setattr(
            self, dataset, data
        )
        
    def add_additional_feature(self) -> None:
        self.add_difference_to_date_decision()
        self.add_tax_registration_merge()
        self.drop_useless_feature_manual()
        self.drop_tax_reg_feature()
        self.add_null_feature()
                
    def merge_all(self) -> None:
        self.add_dataset_name_to_feature()
        
        self.data = self.base_data
        
        n_rows_begin = self._collect_item_utils(
            self.data.select(pl.len())
        )

        for dataset in self.used_dataset:
            current_dataset: Union[pl.LazyFrame, pl.DataFrame] = getattr(
                self, dataset
            )
            self.data = self.data.join(
                current_dataset, how='left', 
                on=['case_id']
            )
        
        n_rows_end = self._collect_item_utils(
            self.data.select(pl.len())
        )
        assert n_rows_begin == n_rows_end

