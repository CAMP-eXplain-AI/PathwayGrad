import datetime
import logging
import os
import sys
from inspect import getframeinfo, stack

from src.utils.ColoredFormatter import ColoredFormatter

# Set BasicConfig to log to console till output directory is created
stream_handler = logging.StreamHandler(stream=sys.stdout)

stream_handler.setFormatter(
    ColoredFormatter('%(log_color)s%(asctime)s | %(processName)15s | %(threadName)10s | %(message)s'))
logging.basicConfig(level=logging.INFO,
                    handlers=[stream_handler])


def init(log_path, log_level=logging.INFO):
    """Initialize logger
    Arguments:
     log_path:
         Path to save the logs
     log_level:
         Logging level
    """
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    log_file_path_name = os.path.join(log_path, 'log_' + timestamp + '.log')

    # Clear the previous basicConfig
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    file_handler = logging.FileHandler(datetime.datetime.now().strftime(log_file_path_name))
    handlers = [file_handler, stream_handler]
    logging.basicConfig(format="%(asctime)s | %(processName)15s | %(threadName)10s | %(message)s",
                        level=log_level,
                        handlers=handlers)


def error(message="", *args):
    caller = getframeinfo(stack()[1][0])
    message = '{:5s} | {:50s} | {}'.format('ERROR',
                                           os.path.basename(caller.filename) + '::' + caller.function + '():' + str(
                                               caller.lineno), message)
    logging.error(message, *args)


def info(message="", *args):
    caller = getframeinfo(stack()[1][0])
    message = '{:5s} | {:50s} | {}'.format('INFO',
                                           os.path.basename(caller.filename) + '::' + caller.function + '():' + str(
                                               caller.lineno), message)
    logging.info(message, *args)


def warn(message="", *args):
    caller = getframeinfo(stack()[1][0])
    message = '{:5s} | {:50s} | {}'.format('WARN',
                                           os.path.basename(caller.filename) + '::' + caller.function + '():' + str(
                                               caller.lineno), message)
    logging.warn(message, *args)


def debug(message, *args):
    caller = getframeinfo(stack()[1][0])
    message = '{:5s} | {:50s} | {}'.format('DEBUG',
                                           os.path.basename(caller.filename) + '::' + caller.function + '():' + str(
                                               caller.lineno), message)
    logging.debug(message, *args)


def exception(message, *args):
    caller = getframeinfo(stack()[1][0])
    message = '{:5s} | {:50s} | {}'.format('DEBUG',
                                           os.path.basename(caller.filename) + '::' + caller.function + '():' + str(
                                               caller.lineno), message)
    logging.exception(message, *args)
