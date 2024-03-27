import os
import sys
sys.path.append(os.getcwd())

if __name__=='__main__':
    from contextlib import redirect_stdout
    from src.utils.import_utils import import_config
    from src.utils.dtype import get_mapping_info
    
    config = import_config()
    with open('log/log_mapper.txt', 'w') as f:
        with redirect_stdout(f):
            get_mapping_info(config=config)
    