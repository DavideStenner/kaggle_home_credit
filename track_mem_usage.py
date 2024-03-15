import os
import time
import psutil
import warnings
from contextlib import redirect_stdout

from src.utils.import_utils import import_config
from src.preprocess.pipeline import PreprocessPipeline

def track_memory_usage(processor: PreprocessPipeline) -> None:
        
    processor.inference = False

    def run_query():
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

    def test_base_dataset(dataset, streaming: bool=True):
        start_, process = init_tracker()
        processor.import_all()

        try:
            getattr(processor, dataset).collect(streaming=streaming)
            gb_used = get_usage(process=process)

            print(f"Dataset: {dataset}; Streaming {streaming}; Peak memory usage: {gb_used:2f} GB; Time: {(time.time()-start_)/60} min")
        except Exception as e:
            print(f'Dataset: {dataset}', str(e))

    def test_single_dataset_memory(dataset, streaming: bool=True):
        start_, process = init_tracker()
        run_query()

        try:
            getattr(processor, dataset).collect(streaming=streaming)
            gb_used = get_usage(process=process)

            print(f"Dataset: {dataset}; Streaming {streaming}; Peak memory usage: {gb_used:2f} GB; Time: {(time.time()-start_)/60} min")
        except Exception as e:
            print(f'Dataset: {dataset}', str(e))


    def test_collect(streaming: bool):
        start_, process = init_tracker()
        run_query()
        
        processor.data = processor.data.collect(streaming=streaming)

        gb_used = get_usage(process=process)

        print(f"Streaming: {streaming}; Peak memory usage: {gb_used:2f} GB; Time: {(time.time()-start_)/60} min")

    print('\n\nFE dataset\n\n')
    for dataset in processor.used_dataset:
        test_single_dataset_memory(dataset=dataset, streaming=False)

    print('\n\nEntire Dataset\n\n')
    test_collect(streaming=False)          

    print('\n\nStarting on base dataset\n\n')
    for dataset in processor.used_dataset:
        test_base_dataset(dataset=dataset, streaming=False)
        

if __name__=='__main__':
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
    