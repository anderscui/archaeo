# coding=utf-8
__version__ = "0.0.1"

import logging
import os
import sys

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

# Logger Config
LOG_LEVEL = os.getenv('ARCHAEO_LOG_LEVEL', 'DEBUG')
logger = logging.getLogger('archaeo')

# TODO: logger level setting
if LOG_LEVEL == 'DEBUG':
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s')
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
logger.propagate = False
