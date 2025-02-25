import ast
import logging

IMPORTANT_LEVEL = 21

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    purple = "\x1b[35;1m"
    blue = "\x1b[34;1m"
    cyan = "\x1b[1;96m"
    reset = "\x1b[0m"
    log_format = "%(levelname)s | %(asctime)s.%(msecs)03d | %(message)s"

    FORMATS = {
        logging.DEBUG: purple + log_format + reset,
        logging.INFO: blue + log_format + reset,
        IMPORTANT_LEVEL: cyan + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def important(self, message, *args, **kwargs):
    if self.isEnabledFor(IMPORTANT_LEVEL):
        self._log(IMPORTANT_LEVEL, message, args, **kwargs) # pylint: disable=protected-access

def initializeLogging(level=logging.INFO):
    logging.addLevelName(IMPORTANT_LEVEL, 'IMPORTANT')
    logging.Logger.important = important
    logger = logging.getLogger('__name__')
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    handler.setFormatter(CustomFormatter())
    logger.setLevel(level)
    return logger

def arg_int_list(arg):
    try:
        # safely evaluate the string as a python literal (like [12, 24])
        parsed = ast.literal_eval(arg)
        if isinstance(parsed, list) and all(isinstance(i, int) for i in parsed):
            return parsed
        raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid format. Expected a list of integers, e.g. '[12, 24]'.")