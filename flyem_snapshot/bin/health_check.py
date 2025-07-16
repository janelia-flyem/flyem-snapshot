''' health_check.py
    Perform health checks on a NeuPrint server. If all is well, it exits with a zero status code.
    If there's a problem, it exits with a non-zero status code.
'''

import argparse
import logging
import os
import sys
import requests

# pylint: disable=broad-exception-caught,logging-fstring-interpolation
__version__ = "0.0.1"

# Configuration
ARG = LOGGER = None

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

def perform_checks():
    ''' Perform health checks on a NeuPrint server
        Keyword arguments:
          None
        Returns:
          None
    '''
    if not ARG.SERVER:
        url = "https://neuprint.janelia.org"
    else:
        url = f"https://neuprint-{ARG.SERVER}.janelia.org"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ['NEUPRINT_JWT']}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            LOGGER.critical(f"Failed to connect to {url} ({response.status_code})")
            terminate_program()
        LOGGER.info(f"Connected to {url}")
        url += "/api/dbmeta/database"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            terminate_program(f"Failed to connect to {url} ({response.status_code})")
        LOGGER.info(f"Connected to {url}")
        data = response.json()
        LOGGER.info(f"Location: {data['Location']}")
        LOGGER.info(f"Description: {data['Description']}")
        url = url.replace("database", "datasets")
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            terminate_program(f"Failed to connect to {url} ({response.status_code})")
        data = response.json()
        for key, value in data.items():
            LOGGER.info(f"{key}: {value['uuid']}")
    except requests.exceptions.RequestException as err:
        LOGGER.critical(f"Failed to connect to {url}")
        terminate_program(err)
    except Exception as err:
        LOGGER.critical(f"An unexpected error occurred when connecting to {url}")
        terminate_program(err)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Perform health checks on a NeuPrint server")
    PARSER.add_argument('--server', dest='SERVER', action='store',
                        default="", help='NeuPrint server')
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
    perform_checks()
    terminate_program()
