import os
import sys
sys.path.append(os.getcwd())

import time
import psutil
import warnings

import polars as pl

from contextlib import redirect_stdout

from src.utils.import_utils import import_config
from src.preprocess.pipeline import PreprocessPipeline

def track_memory_usage(processor: PreprocessPipeline) -> None:
        
    processor.inference = False

    def run_query(processor: PreprocessPipeline):
        processor.import_all()
        processor.create_feature()
        processor.merge_all()
        processor.add_additional_feature()

    def init_tracker():
        start_ = time.time()
        pid = os.getpid()
        process = psutil.Process(pid)
        return start_, process
    
    def get_usage(process):
        current_memory_usage = process.memory_info().rss
        max_memory_usage = max(0, current_memory_usage)
        gb_used = max_memory_usage/(1024**3)
        return gb_used

    def test_base_dataset(processor: PreprocessPipeline, dataset: str, streaming: bool=True):
        start_, process = init_tracker()
        processor.import_all()

        try:
            getattr(processor, dataset).collect(streaming=streaming)
            gb_used = get_usage(process=process)

            print(f"Dataset: {dataset}; Streaming {streaming}; Peak memory usage: {gb_used:2f} GB; Time: {(time.time()-start_)/60} min")
        except Exception as e:
            print(f'Dataset: {dataset}', str(e))

    def test_single_dataset_memory(processor: PreprocessPipeline, dataset: str, streaming: bool=True):
        start_, process = init_tracker()
        run_query(processor=processor)

        try:
            getattr(processor, dataset).collect(streaming=streaming)
            gb_used = get_usage(process=process)

            print(f"Dataset: {dataset}; Streaming {streaming}; Peak memory usage: {gb_used:2f} GB; Time: {(time.time()-start_)/60} min")
        except Exception as e:
            print(f'Dataset: {dataset}', str(e))


    def test_collect(processor: PreprocessPipeline, streaming: bool):
        start_, process = init_tracker()
        run_query(processor=processor)
        
        processor.data = processor.data.collect(streaming=streaming)

        gb_used = get_usage(process=process)

        print(f"Streaming: {streaming}; Peak memory usage: {gb_used:2f} GB; Time: {(time.time()-start_)/60} min")

    def save_downcast_and_test(processor: PreprocessPipeline, dataset: str):
        #downcast
        processor.import_all()

        filter_dict = {
            'applprev_1': pl.col('num_group1')==0,
            'credit_bureau_a_1': pl.col('num_group1')==0,
            'credit_bureau_a_2': (
                (pl.col('num_group1')==0) &
                (pl.col('num_group2')==0)
            ),
            'credit_bureau_b_1': pl.col('num_group1')==0,
            'credit_bureau_b_2': (pl.col('num_group1')==0),
            'debitcard_1': pl.col('num_group1')==0,
            'deposit_1': pl.col('num_group1')==0,
            'tax_registry_a_1': pl.col('num_group1')==0,
            'tax_registry_b_1': pl.col('num_group1')==0,
            'tax_registry_c_1': pl.col('num_group1')==0,
        } 

        data: pl.LazyFrame = getattr(processor, dataset)
        
        if dataset in filter_dict.keys():
            data = data.filter(filter_dict[dataset])
        
        if dataset[-1]=='S':
            data.sink_parquet(
                os.path.join(
                    'testing_data',
                    'downcasted', dataset + '.parquet'
                )
            )
        else:
            
            data.collect().write_parquet(
                os.path.join(
                    'testing_data',
                    'downcasted', dataset + '.parquet'
                )
            )
            
        start_, process = init_tracker()
        setattr(
            processor, dataset,
            pl.scan_parquet(
                os.path.join(
                    'testing_data',
                    'downcasted', dataset + '.parquet' 
                )
            )
        )
        processor.create_feature()
        getattr(processor, dataset).collect()
        
        gb_used = get_usage(process=process)
        print(f"Dataset: {dataset}; Peak memory usage: {gb_used:2f} GB; Time: {(time.time()-start_)/60} min")

    def downcasted_dataset_collect(processor: PreprocessPipeline):
        processor.import_all()

        for dataset in processor.used_dataset:
            setattr(
                processor, dataset,
                pl.scan_parquet(
                    os.path.join(
                        'testing_data',
                        'downcasted', dataset + '.parquet'
                    )
                )
            )
            
        start_, process = init_tracker()
        
        processor.create_feature()
        processor.merge_all()
        processor.add_additional_feature()

        processor.data.collect()
        gb_used = get_usage(process=process)

        print(f"Peak memory usage: {gb_used:2f} GB; Time: {(time.time()-start_)/60} min")
        
    #save downcast version and try to import and fe them
    # print('\n\nDowncast dataset\n\n')
    # for dataset in processor.used_dataset:
    #     save_downcast_and_test(processor=processor, dataset=dataset)

    # print('\n\nEntire Dataset Downcasted\n\n')
    # downcasted_dataset_collect(processor=processor)
    
    print('\n\nFE dataset\n\n')
    for dataset in ['base_data'] + processor.used_dataset:
        test_single_dataset_memory(processor=processor, dataset=dataset, streaming=False)
        sys.stdout.flush()

    print('\n\nEntire Dataset\n\n')
    test_collect(processor=processor, streaming=False)          
    sys.stdout.flush()

    # print('\n\nStarting on base dataset\n\n')
    # for dataset in processor.used_dataset:
    #     test_base_dataset(processor=processor, dataset=dataset, streaming=False)
        

if __name__=='__main__':
    import os
    import sys
    sys.path.append(os.getcwd())

    config_dict = import_config()
    
    #filter useless warning
    warnings.simplefilter(action='ignore', category=UserWarning)

    home_credit_preprocessor = PreprocessPipeline(
        config_dict=config_dict, 
        embarko_skip=6
    )
    print('Creating log report about memory usage')
    with open('log/log_mem_usage.txt', 'w') as f:
        with redirect_stdout(f):
            track_memory_usage(processor=home_credit_preprocessor)
    