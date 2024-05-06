import os
import sys
sys.path.append(os.getcwd())

if __name__=='__main__':
    import logging

    from src.utils.import_utils import import_config
    from src.utils.dtype import get_mapping_info
    
    logger: logging.Logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(
        os.path.join(
            'log', 'log_mapper.txt'
        ), mode='w'
    )
    console_handlare = logging.StreamHandler()

    logger.addHandler(file_handler)
    logger.addHandler(console_handlare)
    
    config = import_config()
    get_mapping_info(config=config, logger=logger)
    