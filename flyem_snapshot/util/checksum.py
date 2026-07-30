from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from cityhash import CityHash64


def dataframe_checksum(df):
    checksums = []
    checksums.append(series_checksum(df.index))
    for c in df.columns:
        checksums.append(series_checksum(df[c]))
    return CityHash64(np.array(checksums))


def series_checksum(s):
    if s.dtype == 'category':
        return CityHash64(s.cat.codes.to_numpy())
    if s.dtype == 'object':
        # An object array holds pointers, so it can't be hashed directly.
        # Stringify it first to get a real contiguous buffer.
        return CityHash64(s.astype(str).to_numpy().astype(str))
    if not isinstance(s.dtype, np.dtype):
        # A pandas extension dtype (Int64, boolean, string, ...).
        # CityHash64 can't consume the extension array itself, and to_numpy()
        # may hand back an object array (i.e. pointers), which would hash
        # differently on every process. Stringify, as with object dtype.
        return CityHash64(s.astype(str).to_numpy().astype(str))
    else:
        return CityHash64(s.to_numpy())


def checksum(data):
    if data is None:
        return 999999999
    if isinstance(data, str):
        return CityHash64(data.encode('utf-8'))
    if isinstance(data, (int, float)) or np.issubdtype(type(data), np.number):
        return CityHash64(str(data).encode('utf-8'))
    if isinstance(data, np.ndarray):
        return CityHash64(data)
    if isinstance(data, pd.DataFrame):
        return dataframe_checksum(data)
    if isinstance(data, pd.Series):
        return dataframe_checksum(data.to_frame())
    if isinstance(data, Sequence):
        if all(isinstance(d, (int, float)) for d in data):
            return CityHash64(np.array(data))
        return CityHash64(np.array([checksum(e) for e in data]))
    if isinstance(data, Mapping):
        csums = []
        for k,v in sorted(data.items()):
            csums.append(checksum(k))
            csums.append(checksum(v))
        return CityHash64(np.array(csums))

    # Doesn't match any supported types, but we can see if the object
    # can be checksummed directly (i.e. supports buffer protocol)
    return CityHash64(data)