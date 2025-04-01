import sys
import argparse
import logging
import subprocess
from importlib import metadata

from  blm.utils.helpers1 import logging_config
logger = logging.getLogger(__name__)

def some_function():
    print("Hello")
    logging_config("processing.log")
    #logger.info(f"Total training examples: {len(ds['train'])}")

def main(args):
    some_function()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])