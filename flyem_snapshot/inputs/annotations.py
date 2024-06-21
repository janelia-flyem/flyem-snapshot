import logging
from pathlib import Path

import hvplot.pandas
import numpy as np
import pandas as pd
import holoviews as hv
import pyarrow.feather as feather


from neuclease import PrefixFilter
from neuclease.dvid.keyvalue import fetch_body_annotations
from neuclease.dvid.annotation import fetch_all_elements
from neuclease.dvid.labelmap import fetch_labels_batched

from ..util import export_bokeh

_ = hvplot.pandas  # for linting

logger = logging.getLogger(__name__)

PointAnnotationSchema = {
    "description": "Settings to describe a source of point annotations in DVID which should be associated with each body.",
    "type": "object",
    "default": {},
    "required": ['instance', 'column-name'],
    "additionalProperties": False,
    "properties": {
        "instance": {
            "description":
                "Name of the DVID annotation instance which contains the points.\n"
                "Note: If more than one annotation point falls on the same body,\n"
                "      only one will be used, and the others simply dropped.\n"
                "Example: 'nuclei-centroids'\n",
            "type": "string"
            # no default
        },
        "column-name": {
            "description":
                "The name of the column in the annotations dataframe and feather file.\n"
                "(If you plan to export to neuprint, you can override this name specifically\n"
                "for neuprint in the neuprint portion of the config settings.)\n",
            "type": "string",
            "default": ""
        },
        "extract-properties": {
            "description":
                "DVID point annotations contain a 'Props' dictionary of properties.\n"
                "To extract those properties and list them as body annotation columns,\n"
                "list them here, mapped to the column name to give them.\n",
            "type": "object",
            "default": {},
            "additionalProperties": {
                "type": "string"
            }
        }
    }
}

AnnotationsSchema = {
    "description": "",
    "type": "object",
    "default": {},
    "required": [],
    "additionalProperties": False,
    "properties": {
        "body-annotations-table": {
            "type": "string",
            "default": ""  # By default, we read from DVID.
        },
        "point-annotations": {
            "description":
                "Point annotation datasets to use for populating properties on the :Neuron nodes.\n"
                "Each item should look something like this:\n"
                "- instance: <dvid-instance-name>\n"
                "  column-name: <name>\n",
            "type": "array",
            "items": PointAnnotationSchema,
            "default": [
                {
                    "instance": "nuclei-centroids",
                    "column-name": "soma_position",
                    "extract-properties": {
                        "radius": "nucleus_radius"
                    }
                }
            ]
        },
        "processes": {
            "description":
                "How many processes should be used to fetch body labels for point annotations?\n"
                "If not specified, default to the top-level config setting.\n",
            "type": ["integer", "null"],
            "default": None
        },
    }
}


@PrefixFilter.with_context('annotations')
def load_annotations(cfg, pointlabeler, snapshot_tag):
    """
    Load body annotations, either from a feather file or from DVID.
    If from a file, it MUST contain a 'body' column.

    If a list of 'point-annotations' instances are listed,
    they will be used to add (or overwrite) columns in the

    Note:
        We don't cache the annotation results because there's not really
        a faster way to determine if the annotations in DVID have changed
        that is much faster than just loading them from scratch anyway.
    """
    table_path = Path(cfg['body-annotations-table'])
    if not cfg['body-annotations-table']:
        ann = fetch_body_annotations(
            pointlabeler.dvidseg.server,
            pointlabeler.dvidseg.uuid,
            pointlabeler.dvidseg.instance + '_annotations'
        )
        # Feather seems to have a hard time if empty strings are in otherwise int columns.
        # Currently, it's legitimate to replace '' with None for all neuprint properties
        # we have so far, except for 'status', since that would mess up the category dtype!
        nonstatus_cols = [c for c in ann.columns if c != 'status']
        ann[nonstatus_cols] = ann[nonstatus_cols].replace([''], [None])

        # The result includes the original json as an extra column,
        # but that's not necessary for anything in this code.
        del ann['json']

    elif table_path.suffix == '.csv':
        logger.info(f"Reading body annotations CSV file: {table_path.name}")
        ann = pd.read_csv(table_path).set_index('body')
    else:
        logger.info(f"Reading body annotations feather file: {table_path.name}")
        ann = feather.read_feather(table_path).set_index('body')

    # This is ugly, but it's easier than a real fix.
    # The 'group_old' column should't exist, but it does and it has screwy types.
    # We don't want to deal with it.
    if 'group_old' in ann.columns:
        logger.info("Discarding obsolete column 'group_old'")
        del ann['group_old']

    feather.write_feather(
        ann.reset_index(),
        f'tables/body-annotations-{snapshot_tag}.feather'
    )

    vc = ann['status'].value_counts().sort_index(ascending=False)
    vc = vc[vc > 0]
    vc = vc[vc.index != ""]
    vc.to_csv(f'tables/status-counts-{snapshot_tag}.csv', index=True, header=True)

    try:
        title = f'body status counts ({snapshot_tag})'
        p = vc.hvplot.barh(flip_yaxis=True, title=title)
        export_bokeh(
            hv.render(p),
            f"body-status-counts-{snapshot_tag}.html",
            title
        )
    except RuntimeError as ex:
        if 'geckodriver' in str(ex):
            logger.warning(f"Not exporting body-status-counts graph: {str(ex)}")
        else:
            raise

    if cfg['point-annotations'] and not pointlabeler:
        raise RuntimeError("Can't read point-annotations without a dvid segmentation.")

    # Anything mentioned in the point-annotations config
    # will override what's in Clio if the name conflicts.
    for pa in cfg['point-annotations']:
        col = pa['column-name'] or pa['instance']
        with PrefixFilter.context(pa['instance']):
            df = fetch_all_elements(
                pointlabeler.dvidseg.server,
                pointlabeler.dvidseg.uuid,
                pa['instance'],
                format='pandas'
            )
            df = df.sort_values([*'zyx'])
            df['body'] = fetch_labels_batched(
                *pointlabeler.dvidseg,
                df[[*'zyx']].values,
                processes=cfg['processes']
            )
            df[col] = df[[*'xyz']].values.tolist()

            # Append this column to the annotation DataFrame, overwriting the column if necessary.
            # (Even if the column exists in Clio, we're overriding it with the data from DVID.)
            # Note: If more than one point lands on the same body, we drop duplicates.
            df = df.drop_duplicates('body').set_index('body')
            ann.drop(columns=[col], errors='ignore', inplace=True)
            ann[col] = df[col]

            # Repeat for the extracted point properties.
            for prop, propcol in pa['extract-properties'].items():
                if prop.lower() not in df:
                    logger.warning(f"Annotation instance {pa['instance']} contains no properties named '{prop}'")
                    continue

                # DVID stores annotation properties as strings, even if they're int or float
                # Attempt to convert to float if possible.
                try:
                    df[prop.lower()] = df[prop.lower()].astype(np.float32)
                except ValueError:
                    continue

                ann.drop(columns=[propcol], errors='ignore', inplace=True)
                ann[propcol] = df[prop.lower()]

    return ann
