import warnings
import polars as pl

from typing import Union, Dict
from itertools import product, chain

from src.utils.other import change_name_with_type
from src.base.preprocess.add_feature import BaseFeature
from src.preprocess.initialize import PreprocessInit

class PreprocessAddFeature(BaseFeature, PreprocessInit):
    
    def add_fold_column(self) -> None:
        self.base_data = self.base_data.with_columns(
            pl.col('WEEK_NUM')
            .alias(self.fold_time_col)
            .cast(pl.Int16)
        )
    
    def create_credit_bureau_a_1_feature(self) -> None:
        warnings.warn('Only considering credit_bureau_a_1 info not related person for now...', UserWarning)
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
    
    def create_credit_bureau_a_2_feature(self) -> None:
        warnings.warn('Only considering credit_bureau_a_2 num1==0', UserWarning)
        #aggregate and take first element
        credit_bureau_a_2_first_categorical = self.credit_bureau_a_2.with_columns(
            #number of num_group1
            (1+ pl.col('num_group1').max()).over('case_id').alias('num_group1_X').cast(pl.Int32),
            #num group1 different group2
            (1 + pl.col('num_group2').max()).over('case_id', 'num_group1').alias('num_group2_X').cast(pl.Int32),
        ).filter(
            #current case
            (pl.col('num_group1')==0) &
            (pl.col('num_group2')==0)
        ).select(
            'case_id', 
            'num_group1_X', 'num_group2_X',
            'collater_typofvalofguarant_298M', 'collater_typofvalofguarant_407M',
            'collater_valueofguarantee_1124L', 'collater_valueofguarantee_876L',
            'collaterals_typeofguarante_359M', 'collaterals_typeofguarante_669M',
        )
        #optimization required
        operation_list = [
            {
                'filter': (
                    (pl.col('collater_valueofguarantee_1124L')!=0)&
                    (pl.col('num_group1')==0)
                ),
                'agg':
                    #has another collateral
                    (
                        pl.col('collater_valueofguarantee_1124L')
                        .count().alias('non_0_collater_valueofguarantee_1124L')
                        .cast(pl.Int32)
                    )
            },
            {
                'filter': (
                    (
                        (pl.col('pmts_dpd_303P')==0) |
                        (pl.col('pmts_dpd_1073P')==0)
                    ) &
                    (pl.col('num_group1')==0)

                ),
                'agg': (
                    pl.col('pmts_dpd_1073P').count().alias('equal_0_pmts_dpd_1073P').cast(pl.Int32),
                    pl.col('pmts_dpd_303P').count().alias('equal_0_pmts_dpd_303P').cast(pl.Int32),
                )
            }
        ]
        operation_list += [
            {
                'filter': (
                    (pl.col(col)!=0) &
                    (pl.col('num_group1')==0)
                ),
                'agg': (
                    pl.col(col).sum().alias(f'sum_no_0_{col}').cast(pl.Float32),
                    pl.col(col).mean().alias(f'mean_no_0_{col}').cast(pl.Float32),
                    pl.col(col).std().alias(f'std_no_0_{col}').cast(pl.Float32),
                )
            }
            for col in ['pmts_dpd_303P', 'pmts_dpd_1073P']

        ]

        credit_bureau_a_2 = self.credit_bureau_a_2.select('case_id').unique().join(
            credit_bureau_a_2_first_categorical,
            on='case_id', how='left'
        )

        for operator_dict in operation_list:
            credit_bureau_a_2 = credit_bureau_a_2.join(
                (
                    self.credit_bureau_a_2.filter(
                        operator_dict['filter']
                    ).group_by('case_id').agg(
                        operator_dict['agg']
                    )
                ),
                on='case_id', how='left'
            )
        
        self.credit_bureau_a_2 = credit_bureau_a_2
        
    def create_credit_bureau_b_1_feature(self) -> None:
        warnings.warn('Only considering credit_bureau_b_1 info not related person for now...', UserWarning)
        self.credit_bureau_b_1 = self.credit_bureau_b_1.filter(
            pl.col('num_group1')==0
        ).drop('num_group1')
        
        list_operation = [
            #end - start active
            (pl.col('contractmaturitydate_151D') - pl.col('contractdate_551D')).alias('range_start_end_active_D'),
            #update - start active
            (pl.col('lastupdate_260D') - pl.col('contractdate_551D')).alias('range_start_update_active_D'),
            #end - update active
            (pl.col('contractmaturitydate_151D') - pl.col('lastupdate_260D')).alias('range_end_update_active_D'),
        ]
        self.credit_bureau_b_1 = self.credit_bureau_b_1.with_columns(
            [
                (
                    operation.dt.total_days()
                    .alias(operation.meta.output_name())
                )
                for operation in list_operation
            ]
        )

    def create_credit_bureau_b_2_feature(self) -> None:
        warnings.warn('Only considering credit_bureau_b_2 num1==0', UserWarning)
        self.credit_bureau_b_2 = (
            self.credit_bureau_b_2.with_columns(
                #number of num_group1
                pl.col('num_group1').n_unique().over('case_id').alias('num_group1_X').cast(pl.Int32),
            ).filter(
                #current case
                (pl.col('num_group1')==0)
            ).sort(
                [
                    'num_group1', 'num_group2'
                ]
            ).group_by('case_id', maintain_order=True).agg(
                pl.col('num_group1_X').first(),
                #num group1 different group2
                pl.col('num_group2').n_unique().alias('num_group2_X').cast(pl.Int32),

                #stat on all
                pl.col('pmts_dpdvalue_108P').sum().alias('sum_pmts_dpdvalue_108P').cast(pl.Float32),
                pl.col('pmts_dpdvalue_108P').mean().alias('mean_pmts_dpdvalue_108P').cast(pl.Float32),
                pl.col('pmts_dpdvalue_108P').std().alias('std_pmts_dpdvalue_108P').cast(pl.Float32),
                
                pl.col('pmts_pmtsoverdue_635A').sum().alias('sum_pmts_pmtsoverdue_635A').cast(pl.Float32),
                pl.col('pmts_pmtsoverdue_635A').mean().alias('mean_pmts_pmtsoverdue_635A').cast(pl.Float32),
                pl.col('pmts_pmtsoverdue_635A').std().alias('std_pmts_pmtsoverdue_635A').cast(pl.Float32),

                #stat on not '
                pl.col('pmts_dpdvalue_108P').filter(pl.col('pmts_dpdvalue_108P')!=0).sum().alias('sum_no_0_pmts_dpdvalue_108P').cast(pl.Float32),
                pl.col('pmts_dpdvalue_108P').filter(pl.col('pmts_dpdvalue_108P')!=0).mean().alias('mean_no_0_pmts_dpdvalue_108P').cast(pl.Float32),
                pl.col('pmts_dpdvalue_108P').filter(pl.col('pmts_dpdvalue_108P')!=0).std().alias('std_no_0_pmts_dpdvalue_108P').cast(pl.Float32),

                pl.col('pmts_pmtsoverdue_635A').filter(pl.col('pmts_pmtsoverdue_635A')!=0).sum().alias('sum_no_0_pmts_pmtsoverdue_635A').cast(pl.Float32),
                pl.col('pmts_pmtsoverdue_635A').filter(pl.col('pmts_pmtsoverdue_635A')!=0).mean().alias('mean_no_0_pmts_pmtsoverdue_635A').cast(pl.Float32),
                pl.col('pmts_pmtsoverdue_635A').filter(pl.col('pmts_pmtsoverdue_635A')!=0).std().alias('std_no_0_pmts_pmtsoverdue_635A').cast(pl.Float32),

                #count on 0 and not '
                pl.col('pmts_dpdvalue_108P').filter(pl.col('pmts_dpdvalue_108P')==0).count().alias('equal_0_pmts_dpdvalue_108P').cast(pl.Int32),
                pl.col('pmts_dpdvalue_108P').filter(pl.col('pmts_dpdvalue_108P')!=0).count().alias('unequal_0_pmts_dpdvalue_108P').cast(pl.Int32),

                #how many change num->0 and 0->num
                (
                    pl.when(
                        (pl.col('pmts_dpdvalue_108P')!=0) &
                        (pl.col('pmts_dpdvalue_108P').shift()==0)
                    )
                    .then(1).otherwise(0)
                    .mean().alias('mean_change_0_not_0_pmts_dpdvalue_108P').cast(pl.Float32)
                ),
                (
                    pl.when(
                        (pl.col('pmts_dpdvalue_108P')==0) &
                        (pl.col('pmts_dpdvalue_108P').shift()!=0)
                    )
                    .then(1).otherwise(0)
                    .mean().alias('mean_change_not_0_0_pmts_dpdvalue_108P').cast(pl.Float32)
                )
            )
        )
        
    def create_debitcard_1_feature(self) -> None:
        warnings.warn('Only considering debitcard_1 info not related person for now...', UserWarning)
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
        warnings.warn('Only considering deposit_1 info not related person for now...', UserWarning)
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
        self.static_0 = self.static_0.with_columns(
            #INTRA DATASET DATE FEATURE
            (pl.col('dtlastpmtallstes_4499206D') - pl.col('firstdatedue_489D')).dt.total_days().alias('pmt_activity_gap_D').cast(pl.Int32),
            (pl.col('lastrepayingdate_696D') - pl.col('firstdatedue_489D')).dt.total_days().alias('pmt_activity_gap_2_D').cast(pl.Int32),
            (pl.col('datelastinstal40dpd_247D') - pl.col('maxdpdinstldate_3546855D')).dt.total_days().alias('delniquency_severity_D').cast(pl.Int32),
            (pl.col('validfrom_1069D') - pl.col('firstclxcampaign_1125D')).dt.total_days().alias('gap_campaign_D').cast(pl.Int32),
            (pl.col('lastapplicationdate_877D') - pl.col('lastactivateddate_801D')).dt.total_days().alias('last_active_range_D').cast(pl.Int32),
            (pl.col('lastapprdate_640D') - pl.col('lastapplicationdate_877D')).dt.total_days().alias('approval_application_D').cast(pl.Int32),
            (pl.col('lastrejectdate_50D') - pl.col('lastapplicationdate_877D')).dt.total_days().alias('reject_application_D').cast(pl.Int32),
            (pl.col('datelastunpaid_3546854D') - pl.col('lastrepayingdate_696D')).dt.total_days().alias('repayment_consistency_D').cast(pl.Int32),
            (pl.col('payvacationpostpone_4187118D') - pl.col('datelastunpaid_3546854D')).dt.total_days().alias('payment_holiday_D').cast(pl.Int32),
            (pl.col('lastdelinqdate_224D') - pl.col('datelastinstal40dpd_247D')).dt.total_days().alias('delinquency_frequency_D').cast(pl.Int32),
            (pl.col('datelastinstal40dpd_247D') - pl.col('dtlastpmtallstes_4499206D')).dt.total_days().alias('delinquency_pmt_speed_D').cast(pl.Int32),
            (pl.col('datelastinstal40dpd_247D') - pl.col('lastrepayingdate_696D')).dt.total_days().alias('delinquency_pmt_speed_2_D').cast(pl.Int32),
            (pl.col('validfrom_1069D') - pl.col('lastactivateddate_801D')).dt.total_days().alias('client_campaign_overlap_D').cast(pl.Int32),
            (pl.col('dtlastpmtallstes_4499206D') - pl.col('lastrepayingdate_696D')).dt.total_days().alias('payment_method_analysis_D').cast(pl.Int32),
            
            #NUMERIC FEATURE
            (pl.col('maxdpdlast24m_143P')-pl.col('avgdbddpdlast24m_3658932P')).alias('delinquency_interactionX').cast(pl.Int32),
            (pl.col('numinstpaidearly_338L')-pl.col('pctinstlsallpaidlate1d_3546856L')).alias('pmt_efficiencyX').cast(pl.Float32),
            (pl.col('totaldebt_9A')-pl.col('avgpmtlast12m_4525200A')).alias('dept_burden_pmtX').cast(pl.Float64),
            (pl.col('clientscnt_304L')-pl.col('maxdpdlast12m_727P')).alias('network_delinquencyX').cast(pl.Int32),
            (pl.col('credamount_770A')-pl.col('numinstlswithdpd10_728L')).alias('loan_size_riskX').cast(pl.Float64),
            (pl.col('maininc_215A')-pl.col('totaldebt_9A')).alias('income_debtX').cast(pl.Float64),
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
            ).alias('trend_highloan_overdue_instX').cast(pl.Float64),
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
            'number_rejected': [
                'forweek_601L', 'formonth_118L', 
                'forquarter_462L', 'foryear_618L', 
                'for3years_128L'
            ],
            'number_cancelations': [
                'forweek_1077L', 'formonth_206L', 'forquarter_1017L', 'foryear_818L',
                'for3years_584L'
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

        self.static_cb_0 = self.static_cb_0.with_columns(
            list_operator
        ).drop(
            [
                'birthdate_574D', 'dateofbirth_337D', 'dateofbirth_342D',
                'numberofqueries_373L'
            ]
        )

    def create_tax_registry_1_feature(self) -> None:
        warnings.warn('Only considering tax_registry_a_1 info not related person for now...', UserWarning)
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
        person_1 = self.person_1.filter(
            pl.col('num_group1')==0
        ).select(
            [
                'case_id',
                'birth_259D', 
                'contaddr_matchlist_1032L', 'contaddr_smempladdr_334L', 
                'education_927M',
                'empl_employedfrom_271D',
                'empl_industry_691L', 'familystate_447L',
                'housetype_905L', 
                'incometype_1044T', 'isreference_387L', 'language1_981M',
                'mainoccupationinc_384A', 'maritalst_703L',
                'role_1084L', 'role_993L',
                'safeguarantyflag_411L', 'type_25L', 'sex_738L'
            ]
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
                pl.len().alias('number_rowsX').cast(pl.Int16),
                pl.col('childnum_185L').max().cast(pl.Int16),
                *[
                    (
                        pl.col(column_name)
                        .filter(
                            (pl.col(column_name)==self.mapper_mask['person_1'][column_name][self.hashed_missing_label])
                        ).count().alias(f'{column_name}_a55475b1countX').cast(pl.Int16)
                    )
                    for column_name in [
                        'contaddr_district_15M', 'contaddr_zipcode_807M', 
                        'empladdr_district_926M', 'empladdr_zipcode_114M',
                        'registaddr_zipcode_184M'
                    ]
                ],
                *[
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
                        .alias(f'{column_name}_{col_value}X')
                        .cast(pl.Int16)
                    )
                    for column_name, col_value in [
                        (column_name, col_value) 
                        for column_name, col_value_list in dict_agg_info.items() 
                        for col_value in col_value_list
                    ]
                ],
                *[
                    (
                        pl.col(column_name)
                        .filter(
                            (pl.col(column_name).is_null())
                        )
                        .len()
                        .alias(f'{column_name}_nullX')
                        .cast(pl.Int16)
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
        )
        
        self.person_1 = (
            person_1.join(
                person_1_related_info, 
                on='case_id', how='left'
            )
        )
        
        
    def create_person_2_feature(self) -> None:
        warnings.warn('Only considering person_2 info relate to 0...', UserWarning)
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
                    ).count().alias('related_n_0_X').cast(pl.Int16),
                    pl.col('case_id').filter(
                        (pl.col('num_group1')==0) &
                        (pl.col('num_group2')!=0)                
                    ).count().alias('related_0_n_X').cast(pl.Int16),   
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
                        .cast(pl.Int16)
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
                        .cast(pl.Int16)
                    )
                    for col, single_value in n_0_list
                ]
            )
        )

    def create_applprev_1_feature(self) -> None:
        warnings.warn('Only considering applprev_1 info not related person for now...', UserWarning)
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
        warnings.warn('Only considering applprev_2 info relate to 0...', UserWarning)

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
                    ).count().alias('related_n_0_X').cast(pl.Int16),
                    pl.col('case_id').filter(
                        (pl.col('num_group1')==0) &
                        (pl.col('num_group2')!=0)                
                    ).count().alias('related_0_n_X').cast(pl.Int16),   
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
                        .cast(pl.Int16)
                    )
                    for col, single_value in category_list
                ] + 
                [
                    pl.col(col).filter(
                        (pl.col(col)==self.mapper_mask['applprev_2'][col][single_value])&
                        (pl.col('num_group1')!=0) &
                        (pl.col('num_group2')==0)
                    ).count().alias(f'{col[:-1]}_{single_value}_n_0_' + col[-1]).cast(pl.Int16)
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
        self.create_credit_bureau_b_1_feature()
        
        self.create_person_2_feature()
        self.create_applprev_2_feature()
        self.create_credit_bureau_a_2_feature()
        self.create_credit_bureau_b_2_feature()
        
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

    def add_difference_to_date_decision(self) -> None:
        """
        Everything which isn't touched until now and it's a date
        will be confrontend with date decision
        """
        type_list = self.data.dtypes
        
        dates_to_transform = [
            col for i, col in enumerate(self.data.columns)
            if (col[-1]=='D') & (type_list[i] == pl.Date) &
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

    def add_difference_to_date_birth(self) -> None:
        """
        Everything which isn't touched until now and it's a date
        will be confrontend with date decision
        """
        type_list = self.data.dtypes
        
        dates_to_transform = [
            col for i, col in enumerate(self.data.columns)
            if (col[-1]=='D') & (type_list[i] == pl.Date)
        ]
        
        #calculate day diff respect to date_decision
        self.data = self.data.with_columns(
            [
                (
                    (
                        pl.col('person_1_birth_259D') - pl.col(col)
                    )
                    .dt.total_days()
                    .alias(
                        change_name_with_type(
                            col, '_birth_diff_'
                        )
                    )
                    .cast(pl.Int32)
                )
                for col in dates_to_transform
            ] + [
                (
                    (
                        (
                            pl.col('person_1_birth_259D') - pl.col(col)
                        )
                        .dt.total_days()//365
                    )
                    .alias(
                        change_name_with_type(
                            col, '_birth_year_diff_'
                        )
                    )
                    .cast(pl.Int32)
                )
                for col in self.calc_also_year_dates_date_decision
            ]
        )

    def add_additional_feature(self) -> None:
        self.add_difference_to_date_birth()
        self.add_difference_to_date_decision()
        
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

