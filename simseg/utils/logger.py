from datetime import datetime
import sys
import traceback

from .context import ENV

DEBUG = -1
INFO = 0
EMPH = 1
WARNING = 2
ERROR = 3
FATAL = 4

log_level = INFO
line_seg = ''.join(['*'] * 65)


class LoggerFatalError(SystemExit):
    pass


def _format(level, messages, rank=None):
    timestr = datetime.strftime(datetime.now(), '%Y-%m-%d  %H:%M:%S')
    father = traceback.extract_stack()[-4]
    func_info = f'{father[0].split("/")[-1]}:{str(father[1]).ljust(4, " ")}'
    m = ' '.join(map(str, messages))
    msg = f'{level} {timestr} {func_info} #{rank}] {m}'
    return msg


_log_file = None
_log_buffer = []
_RED = '\033[0;31m'
_GREEN = '\033[1;32m'
_LIGHT_RED = '\033[1;31m'
_ORANGE = '\033[0;33m'
_YELLOW = '\033[1;33m'
_NC = '\033[0m'  # No Color


@ENV.root_only
def set_file(fname):
    global _log_file
    global _log_buffer
    if _log_file is not None:
        warning("Change log file to %s" % fname)
        _log_file.close()
    _log_file = open(fname, 'w')
    if len(_log_buffer):
        for s in _log_buffer:
            _log_file.write(s)
        _log_file.flush()


def debug(*messages, file=None, root_only=True):
    if log_level > DEBUG:
        return
    if root_only is True and ENV.rank != 0:
        return
    msg = _format('D', messages, ENV.rank)

    if file is None:
        sys.stdout.write(_YELLOW + msg + _NC + '\n')
        sys.stdout.flush()
    else:
        with open(file, 'a+') as f:
            print(msg, file=f)


def info(*messages, file=None, root_only=True, with_format=True):
    if log_level > INFO:
        return
    if root_only is True and ENV.rank != 0:
        return
    msg = _format('I', messages, ENV.rank) if with_format else ' '.join(map(str, messages))
    if file is None:
        sys.stdout.write(msg + '\n')
        sys.stdout.flush()
    else:
        with open(file, 'a+') as f:
            print(msg, file=f)


def emph(*messages, file=None, root_only=True):
    if log_level > EMPH:
        return
    if root_only is True and ENV.rank != 0:
        return
    msg = _format('EM', messages, ENV.rank)
    if file is None:
        sys.stdout.write(_GREEN + msg + _NC + '\n')
        sys.stdout.flush()
    else:
        with open(file, 'a+') as f:
            print(msg, file=f)


def warning(*messages, file=None, root_only=True):
    if log_level > WARNING:
        return
    if root_only is True and ENV.rank != 0:
        return
    msg = _format('W', messages, ENV.rank)
    if file is None:
        sys.stderr.write(_ORANGE + msg + _NC + '\n')
        sys.stderr.flush()
    else:
        with open(file, 'a+') as f:
            print(msg, file=f)


def error(*messages, file=None, root_only=True):
    if log_level > ERROR:
        return
    if root_only is True and ENV.rank != 0:
        return
    msg = _format('E', messages, ENV.rank)
    if file is None:
        sys.stderr.write(_RED + msg + _NC + '\n')
        sys.stderr.flush()
    else:
        with open(file, 'a+') as f:
            print(msg, file=f)


def fatal(*messages, file=None, root_only=True):
    if log_level > FATAL:
        return
    if root_only is True and ENV.rank != 0:
        return
    msg = _format('F', messages, ENV.rank)
    if file is None:
        sys.stderr.write(_LIGHT_RED + msg + _NC + '\n')
        sys.stderr.flush()
    else:
        with open(file, 'a+') as f:
            print(msg, file=f)

    raise LoggerFatalError(-1)
