import os
import sys
sys.path.append(os.getcwd())

import time
import psutil
import logging
import warnings

from typing import Dict, Any
from src.utils.import_utils import import_config
from src.preprocess.pipeline import PreprocessPipeline

def track_memory_usage(processor: PreprocessPipeline, config_dict: Dict[str, Any]) -> None:
        
    processor.inference = False
    logger: logging.Logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(
        os.path.join(
            'log', 'log_mem_usage.txt'
        ), mode='w'
    )
    console_handlare = logging.StreamHandler()

    logger.addHandler(file_handler)
    logger.addHandler(console_handlare)
    
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

    def test_single_dataset_memory(processor: PreprocessPipeline, dataset: str, streaming: bool=True):
        start_, process = init_tracker()
        run_query(processor=processor)

        try:
            getattr(processor, dataset).collect(streaming=streaming)
            gb_used = get_usage(process=process)

            logger.info(f"Dataset: {dataset}; Streaming {streaming}; Peak memory usage: {gb_used:2f} GB; Time: {(time.time()-start_)/60} min")
        except Exception as e:
            logger.info(f'Dataset: {dataset}', str(e))


    def test_collect(processor: PreprocessPipeline, streaming: bool):
        start_, process = init_tracker()
        run_query(processor=processor)
        
        processor.data = processor.data.collect(streaming=streaming)

        gb_used = get_usage(process=process)

        logger.info(f"Streaming: {streaming}; Peak memory usage: {gb_used:2f} GB; Time: {(time.time()-start_)/60} min")
        
    logger.info('\n\nFE dataset\n\n')
    for dataset in ['base_data'] + processor.used_dataset:
        test_single_dataset_memory(processor=processor, dataset=dataset, streaming=False)

    logger.info('\n\nEntire Dataset\n\n')
    test_collect(processor=processor, streaming=False)        

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
    track_memory_usage(processor=home_credit_preprocessor, config_dict=config_dict)
    