import polars as pl

from typing import Union
from itertools import product, chain

from src.utils.other import change_name_with_type
from src.base.preprocess.add_feature import BaseFeature
from src.preprocess.initialize import PreprocessInit

class PreprocessAddFeature(BaseFeature, PreprocessInit):
    
    def add_fold_column(self) -> None:
        self.base_data = self.base_data.with_columns(
            pl.col('WEEK_NUM')
            .alias(self.fold_time_col)
            .cast(pl.UInt16)
        )
    
    def create_credit_bureau_a_1_feature(self) -> None:
        print('Only considering credit_bureau_a_1 info not related person for now...')
        self.credit_bureau_a_1 = self.credit_bureau_a_1.filter(
            pl.col('num_group1')==0
        ).drop('num_group1')
        
        list_date_non_negative_operation = [
            #range active contract to end
            (pl.col('dateofcredend_289D') - pl.col('dateofcredstart_181D')).alias('range_active_D'),
            #range closed contract to end
            (pl.col('dateofcredend_353D') - pl.col('dateofcredstart_739D')).alias('range_close_D'),
            #range of update between active and closed
            (pl.col('lastupdate_1112D')- pl.col('lastupdate_388D')).alias('range_update_D'),
            #end - last update active
            (pl.col('dateofcredend_289D') - pl.col('lastupdate_1112D')).alias('end_active_last_update_D'),
            #update - start active
            (pl.col('lastupdate_1112D')- pl.col('dateofcredstart_181D')).alias('update_start_active_D'),
            #end - last update closed
            (pl.col('dateofcredend_353D') - pl.col('lastupdate_388D')).alias('end_closed_last_update_D'),
            #update - start closed
            (pl.col('lastupdate_388D')- pl.col('dateofcredstart_739D')).alias('update_start_closed_D'),
            #% difference worst date with maximum number of past due active vs closed
            (pl.col('numberofoverdueinstlmaxdat_641D') - pl.col('numberofoverdueinstlmaxdat_148D')).alias('numberofoverdueinstlmaxdat_active_closed_D'),
            #difference worst date with maximum number of past due active vs start of closed
            (pl.col('numberofoverdueinstlmaxdat_148D') - pl.col('dateofcredstart_739D')).alias('numberofoverdueinstlmaxdat_closed_start_D'),
            #difference worst date with maximum number of past due active vs start of active
            (pl.col('numberofoverdueinstlmaxdat_641D') - pl.col('dateofcredstart_181D')).alias('numberofoverdueinstlmaxdat_active_start_D'),
            #difference worst date with maximum number of past due active vs end of closed
            (pl.col('dateofcredend_289D') - pl.col('numberofoverdueinstlmaxdat_148D')).alias('numberofoverdueinstlmaxdat_closed_end_D'),
            #difference worst date with maximum number of past due active vs end of active
            (pl.col('dateofcredend_353D') - pl.col('numberofoverdueinstlmaxdat_641D')).alias('numberofoverdueinstlmaxdat_active_end_D'),
            #% difference worst date with highest outstanding of past due active vs closed
            (pl.col('overdueamountmax2date_1142D') - pl.col('overdueamountmax2date_1002D')).alias('overdueamountmax2date_active_closed_D'),
            #difference worst date with highest outstanding of past due active vs start of closed
            (pl.col('overdueamountmax2date_1002D') - pl.col('dateofcredstart_739D')).alias('overdueamountmax2date_active_closed_start_D'),
            #difference worst date with highest outstanding of past due active vs start of active
            (pl.col('overdueamountmax2date_1142D') - pl.col('dateofcredstart_181D')).alias('overdueamountmax2date_active_active_start_D'),
            #difference worst date with highest outstanding of past due active vs end of closed
            (pl.col('dateofcredend_289D') - pl.col('overdueamountmax2date_1002D')).alias('overdueamountmax2date_active_closed_end_D'),
            #difference worst date with highest outstanding of past due active vs end of active
            (pl.col('dateofcredend_353D') - pl.col('overdueamountmax2date_1142D')).alias('overdueamountmax2date_active_active_end_D'),
        ]

        list_generic_operation = [
            #difference of end active and closed
            (pl.col('dateofcredend_289D') - pl.col('dateofcredend_353D')).alias('range_end_active_closed_D'),
            #difference of start active and closed
            (pl.col('dateofcredstart_181D') - pl.col('dateofcredstart_739D')).alias('range_start_active_closed_D'),
            #difference end activate and refresh date
            (pl.col('dateofcredend_289D') - pl.col('refreshdate_3813885D')).alias('range_end_active_refresh_D'),
            #difference start activate and refresh date
            (pl.col('dateofcredstart_181D') - pl.col('refreshdate_3813885D')).alias('range_start_active_refresh_D'),
            #difference end closed and refresh date
            (pl.col('dateofcredend_353D') - pl.col('refreshdate_3813885D')).alias('range_end_closed_refresh_D'),
            #difference start closed and refresh date
            (pl.col('dateofcredstart_739D') - pl.col('refreshdate_3813885D')).alias('range_start_closed_refresh_D'),
            #difference between end of repayment and closure -> renotiation
            (pl.col('dateofcredend_353D') - pl.col('dateofrealrepmt_138D')).alias('range_end_repmt_closed_D'),
        ]

        self.credit_bureau_a_1 = self.credit_bureau_a_1.with_columns(
            [
                (
                    pl.when(
                        operation.dt.total_days()<0
                    ).then(None).otherwise(operation.dt.total_days())
                ).alias(operation.meta.output_name()).cast(pl.Int64)
                for operation in list_date_non_negative_operation
            ]
        ).with_columns(
            [
                (
                    operation.dt.total_days()
                    .alias(operation.meta.output_name())
                    .cast(pl.Int64)
                )
                for operation in list_generic_operation
            ]
        )
        
    def create_debitcard_1_feature(self) -> None:
        print('Only considering debitcard_1 info not related person for now...')
        self.debitcard_1 = self.debitcard_1.filter(
            pl.col('num_group1')==0
        ).select(
            [
                'case_id', 'last180dayaveragebalance_704A',
                'last180dayturnover_1134A', 'last30dayturnover_651A',
                'openingdate_857D'
            ]
        )
    def create_deposit_1_feature(self) -> None:
        print('Only considering deposit_1 info not related person for now...')
        self.deposit_1 = self.deposit_1.filter(
            pl.col('num_group1')==0
        ).select(
            [
                'case_id', 'amount_416A',
                'contractenddate_991D', 'openingdate_313D'
            ]
        ).with_columns(
            (
                (
                    pl.col('contractenddate_991D') - 
                    pl.col('openingdate_313D')
                ).dt.total_days()
                .alias('duration_contract_date_D')
                .cast(pl.Int32)
            ),
            (
                (
                    (
                        pl.col('contractenddate_991D') - 
                        pl.col('openingdate_313D')
                    ).dt.total_days()//365
                )
                .alias('duration_contract_date_year_diff_D')
                .cast(pl.Int32)
            )
        )
    def create_static_0_feature(self) -> None:
        pass
    
    def create_static_cb_0_feature(self) -> None:
        self.static_cb_0 = self.static_cb_0.drop(
            [
                'birthdate_574D', 'dateofbirth_337D', 'dateofbirth_342D'
            ]
        )

    def create_tax_registry_1_feature(self) -> None:
        print('Only considering tax_registry_a_1 info not related person for now...')
        self.tax_registry_a_1 = self.tax_registry_a_1.filter(
            pl.col('num_group1')==0
        ).select(
            [
                'case_id', 'amount_4527230A',
                'recorddate_4527225D'
            ]
        )
        self.tax_registry_b_1 = self.tax_registry_b_1.filter(
            pl.col('num_group1')==0
        ).select(
            [
                'case_id', 'amount_4917619A',
                'deductiondate_4917603D'
            ]
        )
        
        self.tax_registry_c_1 = self.tax_registry_c_1.filter(
            pl.col('num_group1')==0
        ).select(
            [
                'case_id', 'pmtamount_36A',
                'processingdate_168D'
            ]
        )
                
        
    def create_person_1_feature(self) -> None:
        print('Only considering person_1 info not related person for now...')
        self.person_1 = self.person_1.filter(
            pl.col('num_group1')==0
        ).select(
            [
                'case_id',
                'birth_259D', 'education_927M',
                'empl_employedfrom_271D',
                'empl_industry_691L', 'familystate_447L',
                'housetype_905L', 
                'incometype_1044T', 'isreference_387L', 'language1_981M',
                'mainoccupationinc_384A', 'maritalst_703L',
                'role_1084L', 'role_993L',
                'safeguarantyflag_411L', 'type_25L', 'sex_738L'
            ]
        )

    def create_person_2_feature(self) -> None:
        print('Only considering person_2 info relate to 0...')
        zero_n_list = {
            'addres_role_871L': ['PERMANENT'],
            'conts_role_79M': ['P38_92_157', 'P177_137_98', 'a55475b1']
        }
        n_0_list = {
            'addres_role_871L': ['TEMPORARY', 'REGISTERED', 'CONTACT', 'PERMANENT'],
            'conts_role_79M': [
                'P206_38_166', 'P58_79_51',
                'P7_147_157', 'P115_147_77',
                'P125_105_50', 'P177_137_98', 
                'P125_14_176', 'P124_137_181',
                'a55475b1', 'P38_92_157'
            ],
            'relatedpersons_role_762T': [
                    'OTHER', 'CHILD', 'SIBLING',
                    'PARENT', 'OTHER_RELATIVE',
                    'COLLEAGUE', 'SPOUSE', 'NEIGHBOR',
                    'GRAND_PARENT', 'FRIEND'
            ]
        }

        zero_n_list = list(
            chain(
                *[
                    list(product([key], value))
                    for key, value in zero_n_list.items()
                ]
            )
        )
        n_0_list = list(
            chain(
                *[
                    list(product([key], value))
                    for key, value in n_0_list.items()
                ]
            )
        )

        self.person_2 = self.person_2.group_by('case_id').agg(
            (
                [
                    pl.col('case_id').filter(
                        (pl.col('num_group1')!=0) &
                        (pl.col('num_group2')==0)                
                    ).count().alias('related_n_0_X').cast(pl.UInt16),
                    pl.col('case_id').filter(
                        (pl.col('num_group1')==0) &
                        (pl.col('num_group2')!=0)                
                    ).count().alias('related_0_n_X').cast(pl.UInt16),   
                ] +
                [
                    (
                        pl.col(col).filter(
                            (pl.col(col)==self.mapper_mask['person_2'][col][single_value])&
                            (pl.col('num_group1')==0) &
                            (pl.col('num_group2')!=0)
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
                            (pl.col('num_group1')!=0) &
                            (pl.col('num_group2')==0)
                        )
                        .count()
                        .alias(f'{col[:-1]}_{single_value}_n_0_' + col[-1])
                        .cast(pl.UInt16)
                    )
                    for col, single_value in n_0_list
                ]
            )
        )

    def create_applprev_1_feature(self) -> None:
        print('Only considering applprev_1 info not related person for now...')
        self.applprev_1 = self.applprev_1.filter(
            pl.col('num_group1')==0
        ).select(
            [
                'case_id',
                'actualdpd_943P', 'annuity_853A', 'approvaldate_319D',
                'byoccupationinc_3656910L', 'cancelreason_3545846M',
                'childnum_21L', 'creationdate_885D', 'credacc_actualbalance_314A',
                'credacc_credlmt_575A', 'credacc_maxhisbal_375A',
                'credacc_minhisbal_90A', 'credacc_status_367L',
                'credacc_transactions_402L', 'credamount_590A',
                'credtype_587L', 'currdebt_94A', 'dateactivated_425D',
                'downpmt_134A', 'dtlastpmt_581D', 'dtlastpmtallstes_3545839D',
                'education_1138M', 'employedfrom_700D', 'familystate_726L',
                'firstnonzeroinstldate_307D', 'inittransactioncode_279L', 
                'isbidproduct_390L', 'isdebitcard_527L', 'mainoccupationinc_437A',
                'maxdpdtolerance_577P', 'pmtnum_8L', 'postype_4733339M',
                'profession_152M', 'rejectreason_755M', 'rejectreasonclient_4145042M',
                'revolvingaccount_394A', 'status_219L', 'tenor_203L'
            ]
        )
        self.applprev_1 = self.applprev_1.with_columns(
            #add day diff
            [
                ( 
                    pl.col(col) -
                    pl.col('creationdate_885D')
                ).dt.total_days()
                .cast(pl.Int32).alias(col)
                for col in self.applprev_1.columns
                if (col[-1] == "D") & (col != 'creationdate_885D')
            ]
        ).with_columns(
            #add also year diff
            [
                (
                    (pl.col(col)//365)
                    .cast(pl.Int32).alias(
                        change_name_with_type(
                            col, '_year_diff_'
                        )
                    )
                )
                for col in self.applprev_1.columns
                if (col[-1] == "D") & (col != 'creationdate_885D')
            ]
        )
    
    def create_applprev_2_feature(self) -> None:
        print('Only considering applprev_2 info relate to 0...')

        category_list = {
            "cacccardblochreas_147M": [
                'P127_74_114', 'a55475b1',
                'P133_119_56', 'P23_105_103',
                'P19_60_110', 'P33_145_161',
                'P201_63_60', 'P17_56_144',
                'P41_107_150'
            ],
            "conts_type_509L": ['PHONE',
                'SECONDARY_MOBILE', 'EMPLOYMENT_PHONE',
                'PRIMARY_EMAIL', 'ALTERNATIVE_PHONE',
                'PRIMARY_MOBILE', 'WHATSAPP',
                'HOME_PHONE',
            ],
            "credacc_cards_status_52L": ['RENEWED', 'CANCELLED', 'UNCONFIRMED', 'ACTIVE', 'BLOCKED', 'INACTIVE']
        }
        category_list = list(
            chain(
                *[
                    list(product([key], value))
                    for key, value in category_list.items()
                ]
            )
        )
        self.applprev_2 = self.applprev_2.group_by('case_id').agg(
            (
                [
                    pl.col('case_id').filter(
                        (pl.col('num_group1')!=0) &
                        (pl.col('num_group2')==0)                
                    ).count().alias('related_n_0_X').cast(pl.UInt16),
                    pl.col('case_id').filter(
                        (pl.col('num_group1')==0) &
                        (pl.col('num_group2')!=0)                
                    ).count().alias('related_0_n_X').cast(pl.UInt16),   
                ] +
                [
                    (
                        pl.col(col).filter(
                            (pl.col(col)==self.mapper_mask['applprev_2'][col][single_value])&
                            (pl.col('num_group1')==0) &
                            (pl.col('num_group2')!=0)
                        )
                        .count()
                        .alias(f'{col[:-1]}_{single_value}_0_n_' + col[-1])
                        .cast(pl.UInt16)
                    )
                    for col, single_value in category_list
                ] + 
                [
                    pl.col(col).filter(
                        (pl.col(col)==self.mapper_mask['applprev_2'][col][single_value])&
                        (pl.col('num_group1')!=0) &
                        (pl.col('num_group2')==0)
                    ).count().alias(f'{col[:-1]}_{single_value}_n_0_' + col[-1]).cast(pl.UInt16)
                    for col, single_value in category_list
                ]
            )
        )

    def create_other_1(self) -> None:
        self.other_1 = self.other_1.select(
            [
                'case_id',
                'amtdebitincoming_4809443A', 'amtdebitoutgoing_4809440A', 
                'amtdepositbalance_4809441A', 'amtdepositincoming_4809444A',	
                'amtdepositoutgoing_4809442A'
            ]
        )
    def create_feature(self) -> None:
        self.create_static_0_feature()
        self.create_static_cb_0_feature()
        self.create_person_1_feature()
        self.create_applprev_1_feature()
        self.create_other_1()
        self.create_tax_registry_1_feature()
        self.create_deposit_1_feature()
        self.create_debitcard_1_feature()
        self.create_credit_bureau_a_1_feature()
        
        self.create_person_2_feature()
        self.create_applprev_2_feature()
        
        if not self.inference:
            self.add_fold_column()

    def add_dataset_name_to_feature(self) -> None: 
        for dataset in self.used_dataset:
            current_dataset: Union[pl.LazyFrame, pl.DataFrame] = getattr(
                self, dataset
            )
            setattr(
                self, 
                dataset,  
                current_dataset.rename(
                    {
                        col: dataset + '_' + col
                        for col in current_dataset.columns
                        if col not in self.special_column_list
                    }
                )
            )

    def add_difference_to_date_decision(self) -> None:
        """
        Everything which isn't touched until now and it's a date
        will be confrontend with date decision
        """
        temp_row_data = self.data.first().collect()
        
        dates_to_transform = [
            col for col in self.data.columns 
            if (col[-1]=='D') & (temp_row_data[col].dtype == pl.Date) &
            (col not in ['applprev_1_creationdate_885D'])
        ]
        not_allowed_negative_dates = [
            col
            for col in dates_to_transform
            if col not in self.negative_allowed_dates_date_decision
        ]
        assert all([col in dates_to_transform for col in self.calc_also_year_dates_date_decision])
        
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
            ] + [
                (
                    (
                        (
                            pl.col('date_decision') - pl.col(col)
                        )
                        .dt.total_days()//365
                    )
                    .cast(pl.Int32).alias(
                        change_name_with_type(
                            col, '_year_diff_'
                        )
                    )
                )
                for col in self.calc_also_year_dates_date_decision
            ]
        ).with_columns(
            #put blank wrong dates
            [
                (
                    pl.when((pl.col(col) <0))
                    .then(None)
                    .otherwise(pl.col(col))
                    .cast(pl.Int32).alias(col)
                )
                for col in (
                    not_allowed_negative_dates + 
                    #add also year calculation
                    [
                        change_name_with_type(
                            col, '_year_diff_'
                        )
                        for col in not_allowed_negative_dates
                        if col in self.calc_also_year_dates_date_decision
                    ]
                )
            ]
        )

        #correct applprev_1 columns depending on date decision -> no future decision
        self.data = self.data.with_columns(
            
            [
                (
                    pl.when(
                        (pl.col('date_decision') - pl.col('applprev_1_creationdate_885D'))
                        .dt.total_days() <0
                    )
                    .then(None)
                    .otherwise(pl.col(col))
                    .cast(pl.Int32).alias(col)
                )
                for col in self.data.columns
                if (col[-1]=='D') & ('applprev_1_' in col) & (col != 'applprev_1_creationdate_885D')
            ]
        ).drop('applprev_1_creationdate_885D')
        
        temp_row_data = self.data.first().collect()
        assert all(
            [
                temp_row_data[col].dtype != pl.Date 
                for col in self.data.columns 
                if (col[-1]=='D')
            ]
        )

    def add_additional_feature(self) -> None:
        self.add_difference_to_date_decision()
        
    def merge_all(self) -> None:
        self.add_dataset_name_to_feature()
        
        self.data = self.base_data
        
        n_rows_begin = self._collect_item_utils(
            self.data.select(pl.count())
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
            self.data.select(pl.count())
        )
        assert n_rows_begin == n_rows_end

