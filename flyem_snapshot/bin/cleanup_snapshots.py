''' cleanup_snapshots.py
    Clean up snapshot directories (minus reports) from
    /groups/flyem/data/snapshots.
    Aging is controlled with --days, and nothing is actually deleted unless
    --write is specified.
'''

import argparse
import collections
from datetime import datetime, timedelta
import logging
import os
import re
import shutil
import sys

# pylint: disable=broad-exception-caught,logging-fstring-interpolation
__version__ = "0.0.1"

# Configuration
ARG = LOGGER = None
BASE_DIR =  '/groups/flyem/data/snapshots/'
INSTANCES = ['fish2', 'yakuba']
EXCLUDE = ['reports']
# Counters
COUNT = collections.defaultdict(lambda: 0, {})

def terminate_program(msg=None):
    ''' Terminate the program gracefully
        Keyword arguments:
          msg: error message or object
        Returns:
          None
    '''
    if msg:
        if not isinstance(msg, str):
            msg = f"An exception of type {type(msg).__name__} occurred. Arguments:\n{msg.args}"
        LOGGER.critical(msg)
    sys.exit(-1 if msg else 0)


def cleanup_single_instance(instance):
    ''' Cleanup files and directories for a single instance
        Keyword arguments:
          None
        Returns:
          None
    '''
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}")
    threshold_date = datetime.now() - timedelta(days=ARG.DAYS)
    old_dirs = []
    for entry in os.listdir(os.path.join(BASE_DIR, instance)):
        full_path = os.path.join(BASE_DIR, instance, entry)
        match = pattern.match(entry)
        if os.path.isdir(full_path) and match:
            # uuid = entry.split('-')[3] # In case we need the UUID
            try:
                dir_date_str = match.group(0)
                dir_date = datetime.strptime(dir_date_str, "%Y-%m-%d")
                if dir_date <= threshold_date:
                    old_dirs.append(full_path)
            except ValueError:
                continue
    if not old_dirs:
        return
    # Save at least one full directory
    _ = old_dirs.pop()
    for target in old_dirs:
        for entry in os.listdir(target):
            if entry in EXCLUDE:
                continue
            full_path = os.path.join(target, entry)
            COUNT['found'] += 1
            if not ARG.WRITE:
                LOGGER.info(full_path)
                COUNT['deleted'] += 1
                continue
            try:
                if os.path.isdir(full_path):
                    LOGGER.warning(f"{full_path}")
                    shutil.rmtree(full_path)
                    LOGGER.warning(f"Deleted directory: {full_path}")
                elif os.path.isfile(full_path):
                    os.remove(full_path)
                    LOGGER.info(f"Deleted file: {full_path}")
                COUNT['deleted'] += 1
            except Exception as err:
                LOGGER.warning(f"Error deleting {full_path}: {err}")


def processing():
    ''' Cleanup files and directories
        Keyword arguments:
          None
        Returns:
          None
    '''
    for instance in INSTANCES:
        cleanup_single_instance(instance)
    print(f"Files/directories found:   {COUNT['found']}")
    print(f"Files/directories deleted: {COUNT['deleted']}")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Perform health checks on a NeuPrint server")
    PARSER.add_argument('--days', dest='DAYS', action='store',
                        default=4, type=int, help='Minimum days old')
    PARSER.add_argument('--write', dest='WRITE', action='store_true',
                        default=False, help='Write to filesystem')
    PARSER.add_argument('--verbose', dest='VERBOSE', action='store_true',
                        default=False, help='Flag, Chatty')
    PARSER.add_argument('--debug', dest='DEBUG', action='store_true',
                        default=False, help='Flag, Very chatty')
    ARG = PARSER.parse_args()
    LOGGER = logging.getLogger(__name__)
    if ARG.VERBOSE:
        LOGGER.setLevel(logging.INFO)
    if ARG.DEBUG:
        LOGGER.setLevel(logging.DEBUG)
    LOGGER.addHandler(logging.StreamHandler())
    processing()
    terminate_program()
