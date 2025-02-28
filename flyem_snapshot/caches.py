import os
import copy
import inspect
import logging
import functools
from abc import abstractmethod
from itertools import chain

import numpy as np
import pyarrow.feather as feather

from neuclease.util import Timer

from .util.checksum import checksum

logger = logging.getLogger(__name__)


# FIXME:
#   For testing, it would be nice if there were a convenient way to
#   disable caching globally, either from the command-line or via the config.

def cached(serializer, cache_dir='cache'):
    def decorator(f):
        if serializer.enforce_matching_signature:
            assert inspect.signature(serializer.get_cache_key) == inspect.signature(f), (
                "Mismatched signatures in @cached() decorator: "
                f"{type(serializer)}.get_cache_key(...) does not match {f.__name__}(...)"
            )

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            os.makedirs(cache_dir, exist_ok=True)
            key = serializer.get_cache_key(*args, **kwargs)
            result_path = f'{cache_dir}/{key}'
            if os.path.exists(result_path):
                try:
                    logger.info(f"Loading {serializer.name} from cache: {result_path}")
                    return serializer.load_from_file(result_path)
                except Exception as ex:  # pylint: disable=broad-exception-caught
                    logger.error(f"Ignoring {serializer.name} cache due to error: {ex}")

            result = f(*args, **kwargs)
            logger.info(f"Storing {serializer.name} to cache: {result_path}")
            serializer.save_to_file(result, result_path)
            return result
        return wrapper
    decorator.serializer = serializer
    return decorator


class SerializerBase:

    def __init__(self, name, enforce_matching_signature=True):
        """
        If enforce_matching_signature is True, then the @cached decorator will assert
        that the signature of the serializer's get_cache_key() method matches the
        signature of the function it decorates.
        """
        self.name = name
        self.enforce_matching_signature = enforce_matching_signature

    # @abstractmethod
    # def get_cache_key(self, *args, **kwargs):
    #    raise NotImplementedError

    @abstractmethod
    def save_to_file(self, result, path):
        raise NotImplementedError

    @abstractmethod
    def load_from_file(self, path):
        raise NotImplementedError


class SentinelSerializer(SerializerBase):
    """
    A 'serializer' for functions with no return value, such as export functions.
    Instead of storing any data to disk, it merely writes an empty 'sentinel' file
    to disk to indicate whether the inputs have changed since the last invocation.

    This implementation works for functions whose arguments are each one of the following:
        - int/float
        - string
        - json-serializable dict
        - pd.DataFrame

    As a special convenience, if the first argument appears to be a config dict,
    then its 'processes' key is set to 0 to avoid incorporating that in
    the sentinel checksum.
    """
    def __init__(self, name, enforce_matching_signature=False):
        super().__init__(name, enforce_matching_signature)

    def get_cache_key(self, cfg, *args, **kwargs):
        if isinstance(cfg, dict) and 'processes' in cfg:
            cfg = copy.copy(cfg)
            cfg['processes'] = 0

        with Timer("Computing data checksum", logger, log_start=False):
            csum = checksum([cfg, *chain(args, kwargs.values())])

        return f'{self.name}-{hex(csum)}.sentinel'

    def save_to_file(self, result, path):
        assert result is None
        open(path, 'wb').close()

    def load_from_file(self, path):
        open(path, 'rb').close()


def cache_dataframe(df, path):
    """
    A wrapper around feather.write_feather with an extra check to ensure
    that NaN data will not change after a round-trip of write/read.
    Raises an error if the dataframe doesn't conform.
    """
    # Standardize on None as the null value (instead of NaN or "").
    # https://stackoverflow.com/questions/46283312/how-to-proceed-with-none-value-in-pandas-fillna
    FAKENULL = '__cache_dataframe_nullval__'
    bad_columns = []
    for col, dtype in df.dtypes.items():
        if dtype != object:
            continue

        if (df[col].replace([np.nan], [FAKENULL]) != df[col].replace([None], FAKENULL)).any():
            bad_columns.append(col)
    if bad_columns:
        msg = (
            "DataFrame cannot be cached because it contains column(s) of 'object' dtype "
            "that contain np.nan. write_feather() will convert them to None, giving them "
            "a different checksum.  Convert them to None yourself using Series.replace([np.nan], [None])) "
            "before returning the dataframe, so it can be cached.\n"
            f"The nan-containing columns are: [{bad_columns}]"
        )
        raise RuntimeError(msg)
    feather.write_feather(df, path)
