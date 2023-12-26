import logging

import pandas as pd
import pyarrow.feather as feather

from neuclease import PrefixFilter

logger = logging.getLogger(__name__)

LandmarksSchema = {
    "description":
        "Landmarks are arbitrary points which can be used by reports and/or neuprint exports.\n"
        "Landmarks have an xyz coordinate, and optionally ROIs or other properties.",
    "default": {},
    "additionalProperties": False,
    "required": "path",
    "properties": {
        "path": {
            "description": "File path to landmark table, either .csv or .feather.",
            "type": "string",
            "default": ""
        },
        "roi-set-names": {
            "description":
                "The list of ROI sets to include as columns in the landmark table.\n"
                "If nothing is listed here, all ROI sets are used.",
            "default": None,
            "oneOf": [
                {
                    "type": "array",
                    "items": {"type": "string"}
                },
                {
                    "type": "null"
                }
            ]
        }
    }
}


@PrefixFilter.with_context('landmarks')
def load_landmarks(cfg):
    path = cfg['path']
    if not path:
        return None

    if path.endswith('.csv'):
        landmark_df = pd.read_csv(path)

    if path.endswith('.feather'):
        landmark_df = feather.read_feather(path)

    return landmark_df
