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

    def create_person_1_feature(self) -> None:
        print('Only considering client info not related person for now...')
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

    def create_feature(self) -> None:
        self.create_person_1_feature()
        
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

    def add_additional_feature(self) -> None:
        not_allowed_negative_dates = [
            col
            for col in self.data.columns if col[-1]=='D'
            if col not in self.negative_allowed_dates
        ]
        self.data = self.data.with_columns(
            [
                (
                    (
                        pl.col('date_decision') - pl.col(col)
                    )
                    .dt.total_days()
                    .cast(pl.Int32).alias(col)
                )
                for col in self.data.columns if col[-1]=='D'
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
                for col in self.calc_also_year_dates
            ]
        ).with_columns(
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
                        if col in self.calc_also_year_dates
                    ]
                )
            ]
        )
    
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

        n_rows_end = self._collect_item_utils(
            self.data.select(pl.count())
        )
        assert n_rows_begin == n_rows_end

