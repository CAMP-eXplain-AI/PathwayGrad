import os
import shutil
import pickle as pkl
import time
from datetime import datetime

class Logger(object):
    def __init__(self):
        self._logger = None

    def init(self, logdir, name='log'):
        if self._logger is None:
            import logging
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file)
            self._logger = logging.getLogger()
            self._logger.setLevel('INFO')
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

    def info(self, str_info):
        now = datetime.now()
        display_now = str(now).split(' ')[1][:-3]
        self.init(os.path.expanduser('~/tmp_log'), 'tmp.log')
        self._logger.info('[' + display_now + ']' + ' ' + str_info)

logger = Logger()


def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)


def load_pickle(path):
    begin_st = time.time()
    with open(path, 'rb') as f:
        print("Loading pickle object from {}".format(path))
        v = pkl.load(f)
    print("=> Done ({:.4f} s)".format(time.time() - begin_st))
    return v


def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        print("Dumping pickle object to {}".format(path))
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


def prepare_logging(args):
    args.logdir = os.path.join('./logs', args.logdir)

    logger.init(args.logdir, 'log')

    ensure_dir(args.logdir)
    logger.info("=================FLAGS==================")
    for k, v in args.__dict__.items():
        logger.info('{}: {}'.format(k, v))
    logger.info("========================================")

