import os
import logging
import functools
from abc import abstractmethod

import pandas as pd
import pyarrow.feather as feather

logger = logging.getLogger(__name__)


def cached(serializer, name, cache_dir='tables'):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            key = serializer.get_cache_key(*args, **kwargs)
            result_path = f'{cache_dir}/{key}'
            if os.path.exists(result_path):
                try:
                    logger.info(f"Loading {name} from cache: {result_path}")
                    return serializer.load_from_file(result_path)
                except Exception as ex:  # pylint: disable=broad-exception-caught
                    logger.error(f"Ignoring {name} cache due to error: {ex}")

            result = f(*args, **kwargs)
            logger.info(f"Storing {name} to cache: {result_path}")
            serializer.save_to_file(result, result_path)
            return result
        return wrapper
    decorator.serializer = serializer
    return decorator


class SerializerBase:

    @abstractmethod
    def get_cache_key(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_to_file(self, result, path):
        raise NotImplementedError

    @abstractmethod
    def load_from_file(self, path):
        raise NotImplementedError


class DataFrameSerializer(SerializerBase):

    def __init__(self, save_index=True):
        self.save_index = save_index

    @abstractmethod
    def save_to_file(self, result, path):
        if self.save_index:
            assert not isinstance(result.index, pd.MultiIndex), \
                "Saving DataFrames with MultiIndex is not supported"
            feather.write_feather(result.reset_index(), path)
        else:
            feather.write_feather(result, path)

    @abstractmethod
    def load_from_file(self, path):
        if self.save_index:
            df = feather.read_feather(path)
            df = df.set_index(df.columns[0])
            return df
        else:
            return feather.read_feather(path)

