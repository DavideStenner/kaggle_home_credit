import polars as pl

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
    
    def create_static_0_feature(self) -> None:
        pass
    
    def create_static_cb_0_feature(self) -> None:
        self.static_cb_0 = self.static_cb_0.drop(
            [
                'birthdate_574D', 'dateofbirth_337D', 'dateofbirth_342D'
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
        
    def create_feature(self) -> None:
        self.create_static_0_feature()
        self.create_static_cb_0_feature()
        self.create_person_1_feature()
        self.create_applprev_1_feature()
        
        if not self.inference:
            self.add_fold_column()

    def add_dataset_name_to_feature(self) -> None: 
        self.static_0 = self.static_0.rename(
            {
                col: 'static_0_' + col
                for col in self.static_0.columns
                if col not in self.special_column_list
            }
        )
        self.static_cb_0 = self.static_cb_0.rename(
            {
                col: 'static_cb_0_' + col
                for col in self.static_cb_0.columns
                if col not in self.special_column_list
            }
        )
        self.person_1 = self.person_1.rename(
            {
                col: 'person_1_' + col
                for col in self.person_1.columns
                if col not in self.special_column_list
            }
        )
        self.applprev_1 = self.applprev_1.rename(
            {
                col: 'applprev_1_' + col
                for col in self.applprev_1.columns
                if col not in self.special_column_list
            }
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
                if (col[-1]=='D') & ('applprev_1_' in col)
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

        self.data = self.data.join(
            self.static_0, how='left', 
            on=['case_id']
        )
        self.data = self.data.join(
            self.static_cb_0, how='left', 
            on=['case_id']
        )

        self.data = self.data.join(
            self.person_1, how='left', 
            on=['case_id']
        )

        self.data = self.data.join(
            self.applprev_1, how='left', 
            on=['case_id']
        )

        n_rows_end = self._collect_item_utils(
            self.data.select(pl.count())
        )
        assert n_rows_begin == n_rows_end

