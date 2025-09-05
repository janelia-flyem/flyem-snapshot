""" aging_deletion.py
    Delete files according to aging.
"""

import argparse
from datetime import datetime
import json
import os
import re
import shutil
import sys
from types import SimpleNamespace
import colorlog
import requests

#pylint: disable=broad-exception-caught
TEMPLATE = "An exception of type %s occurred. Arguments:\n%s"

# ------------------------------------------------------------------------------

def terminate_program(msg=None):
    """ Log an optional error to output, close files, and exit
        Keyword arguments:
          err: error message
        Returns:
           None
    """
    if msg:
        LOGGER.critical(msg)
    sys.exit(-1 if msg else 0)


def setup_logging(arg):
    """ Set up colorlog logging
        Keyword arguments:
          arg: argparse arguments
        Returns:
          colorlog handler
    """
    logger = colorlog.getLogger()
    if arg.DEBUG:
        logger.setLevel(colorlog.DEBUG)
    elif arg.VERBOSE:
        logger.setLevel(colorlog.INFO)
    else:
        logger.setLevel(colorlog.WARNING)
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter())
    logger.addHandler(handler)
    return logger


def _call_config_responder(endpoint):
    ''' Get a configuration from the configuration system
        Keyword arguments:
          endpoint: REST endpoint
        Returns:
          JSON response
    '''
    if not os.environ.get('CONFIG_SERVER_URL'):
        raise ValueError("Missing environment variable CONFIG_SERVER_URL")
    url = os.environ.get('CONFIG_SERVER_URL') + endpoint
    try:
        req = requests.get(url, timeout=10)
    except requests.exceptions.RequestException as err:
        raise err
    if req.status_code == 200:
        try:
            jstr = req.json()
        except Exception as err:
            raise requests.exceptions.JSONDecodeError("Could not decode response from " \
                                                      + f"{url} : {err}")
        return jstr
    raise ConnectionError(f"Could not get response from {url}: {req.text}")


def get_config(config):
    """ Convert the JSON received from a configuration to an object
        Keyword arguments:
          config: configuration name
        Returns:
          Configuration namespace object
    """
    try:
        data = (_call_config_responder(f"config/{config}"))["config"]
    except Exception as err:
        raise err
    return json.loads(json.dumps(data), object_hook=lambda dat: SimpleNamespace(**dat))


def simplenamespace_to_dict(nspace):
    """ Convert a simplenamespace to a dict recursively
        Keyword arguments:
          nspace: simplenamespace to convert
        Returns:
          The converted dict
    """
    result = {}
    for key, value in nspace.__dict__.items():
        if isinstance(value, SimpleNamespace):
            result[key] = simplenamespace_to_dict(value)
        else:
            result[key] = value
    return result


def find_old_files(dir_rec):
    """ Find old files for one aging entry
        Keyword arguments:
          dir_rec: aging entry record
        Returns:
          List of files (or directories) to delete
    """
    checked = selected = 0
    LOGGER.info("Checking %s", dir_rec['dir'])
    prog = re.compile(dir_rec['match'])
    current_time = datetime.now()
    for root, dirs, files in os.walk(dir_rec['dir']):
        if root != dir_rec['dir']:
            continue
        if 'type' in dir_rec and dir_rec['type'] == 'dir':
            for sdir in dirs:
                checked += 1
                days_old = (current_time - \
                            datetime.fromtimestamp(os.path.getmtime(os.path.join(root, sdir)))).days
                if prog.search(sdir) and days_old > dir_rec['aging']:
                    LOGGER.debug("%s %d", sdir, days_old)
                    selected += 1
                    yield os.path.join(root, sdir + '/')
        else:
            for file in files:
                checked += 1
                days_old = (current_time - \
                            datetime.fromtimestamp(os.path.getmtime(os.path.join(root, file)))).days
                if prog.search(file) and days_old > dir_rec['aging']:
                    LOGGER.debug("%s %d", file, days_old)
                    selected += 1
                    yield os.path.join(root, file)
        break
    print(f"{dir_rec['dir']}: {checked:,} checked, {selected:,} selected")


def process_directories():
    """ Process entries in aging dict to remove files/directories
        Keyword arguments:
          None
        Returns:
          None
    """
    for val in AGING.values():
        for file in find_old_files(val):
            if ARG.WRITE:
                if file.endswith('/'):
                    try:
                        shutil.rmtree(file)
                    except Exception as err:
                        msg = f"Could not rmtree {file}\n" \
                              + TEMPLATE % (type(err).__name__, err.args)
                        terminate_program(msg)
                else:
                    try:
                        os.remove(file)
                    except Exception as err:
                        msg = f"Could not remove {file}\n" \
                              + TEMPLATE % (type(err).__name__, err.args)
                        terminate_program(msg)


# ------------------------------------------------------------------------------

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Delete files by aging')
    PARSER.add_argument('--write', action='store_true', dest='WRITE',
                        default=False, help='Write to config system')
    PARSER.add_argument('--verbose', action='store_true', dest='VERBOSE',
                        default=False, help='Turn on verbose output')
    PARSER.add_argument('--debug', action='store_true', dest='DEBUG',
                        default=False, help='Turn on debug output')
    ARG = PARSER.parse_args()
    LOGGER = setup_logging(ARG)
    try:
        AGING = simplenamespace_to_dict(get_config("aging_deletion_flyem"))
    except Exception as gerr: # pylint: disable=broad-exception-caught
        terminate_program(gerr)
    process_directories()
    terminate_program()
