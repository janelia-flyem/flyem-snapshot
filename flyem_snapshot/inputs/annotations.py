import os
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
        "replace-values": {
            "description":
                "Sometimes the annotations in one column may be derived from the annotations\n"
                "in a different column, in simple lookup-table fashion.\n"
                "This setting allows you to populate a column via mapping of translations.\n",
            "type": "array",
            "default": [],
            "items": {
                "type": "object",
                "default": {},
                "source-column": {
                    "type": "string",
                    "default": "",
                },
                "target-column": {
                    "type": "string",
                    "default": "",
                },
                "replacements": {
                    "description":
                        "A mapping of {old_term: new_term}.\n"
                        "Any values in the source which are not listed here will remain unmodified.\n",
                    "type": "object",
                    "additionalProperties": {
                        "type": "string"
                    }
                }
            }
        },
        "rename-annotation-columns": {
            "description":
                "Optionally rename columns of the annotation table immediately after it is loaded.\n"
                "There is a similar option in the neuprint config (annotation-property-names), but that is applied only for neuprint.\n"
                "This setting applies to the annotations globally, and the new column names are used for all outputs.\n",
            "additionalProperties": {
                "type": "string"
            },
            "default": {},
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
    os.makedirs('tables', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    ann = _load_raw_annotations(cfg, pointlabeler)
    for pa_cfg in cfg['point-annotations']:
        _append_dvid_point_annotations(pa_cfg, pa_cfg['instance'], ann, pointlabeler, cfg['processes'])

    _apply_value_replacements(cfg, ann)

    if renames := cfg['rename-annotation-columns']:
        # Drop anything that was renamed to ""
        drop_keys = [k for k,v in renames.items() if not v]
        ann = ann.drop(columns=drop_keys, errors='ignore')

        # Rename the others
        renames = {k:v for k,v in renames.items() if v}
        ann = ann.rename(columns=renames)

    # This is ugly, but it's simpler than a more general fix.
    # The 'group_old' column should't exist, but it does, and it has screwy types.
    # We don't want to deal with it.
    if 'group_old' in ann.columns:
        logger.info("Discarding obsolete column 'group_old'")
        del ann['group_old']

    feather.write_feather(
        ann.reset_index(),
        f'tables/body-annotations-{snapshot_tag}.feather'
    )

    _export_body_status_counts(ann, snapshot_tag)

    return ann


def _load_raw_annotations(cfg, pointlabeler):
    table_path = Path(cfg['body-annotations-table'])
    if not cfg['body-annotations-table']:
        ann_name = ' / '.join(pointlabeler.dvidseg)
        logger.info(f"Reading body annotations from DVID: {ann_name}")
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

    return ann


@PrefixFilter.with_context('{instance}')
def _append_dvid_point_annotations(pa_cfg, instance, ann, pointlabeler, processes):
    if not pointlabeler:
        raise RuntimeError("Can't read point-annotations without a dvid segmentation.")

    # Anything mentioned in the point-annotations config
    # will override what's in Clio if the name conflicts.
    col = pa_cfg['column-name'] or instance
    df = fetch_all_elements(
        pointlabeler.dvidseg.server,
        pointlabeler.dvidseg.uuid,
        instance,
        format='pandas'
    )
    df = df.sort_values([*'zyx'])
    df['body'] = fetch_labels_batched(
        *pointlabeler.dvidseg,
        df[[*'zyx']].values,
        processes=processes
    )
    df[col] = df[[*'xyz']].values.tolist()

    # Append this column to the annotation DataFrame, overwriting the column if necessary.
    # (Even if the column exists in Clio, we're overriding it with the data from DVID.)
    # Note: If more than one point lands on the same body, we drop duplicates.
    df = df.drop_duplicates('body').set_index('body')
    ann.drop(columns=[col], errors='ignore', inplace=True)
    ann[col] = df[col]

    # Repeat for the extracted point properties.
    for prop, propcol in pa_cfg['extract-properties'].items():
        if prop.lower() not in df:
            logger.warning(f"Annotation instance {instance} contains no properties named '{prop}'")
            continue

        # DVID stores annotation properties as strings, even if they're int or float
        # Attempt to convert to float if possible.
        try:
            df[prop.lower()] = df[prop.lower()].astype(np.float32)
        except ValueError:
            continue

        ann.drop(columns=[propcol], errors='ignore', inplace=True)
        ann[propcol] = df[prop.lower()]


def _apply_value_replacements(cfg, ann):
    for rpl in cfg['replace-values']:
        if not (src_col := rpl['source-column']):
            raise RuntimeError("replace-values config lists no source-column")
        tgt_col = rpl['target-column'] or src_col
        logger.info(f"Applying replacement values ({src_col} -> {tgt_col}])")
        ann[tgt_col] = ann[src_col].astype(object).replace(rpl['replacements'])


def _export_body_status_counts(ann, snapshot_tag):
    vc = ann['status'].value_counts().sort_index(ascending=False)
    vc = vc[vc > 0]
    vc = vc[vc.index != ""]
    vc.to_csv(f'tables/body-status-counts-{snapshot_tag}.csv', index=True, header=True)

    try:
        title = f'body status counts ({snapshot_tag})'
        p = vc.hvplot.barh(flip_yaxis=True, title=title)
        export_bokeh(
            hv.render(p),
            f"body-status-counts-{snapshot_tag}.html",
            title,
            "reports"
        )
    except RuntimeError as ex:
        if 'geckodriver' in str(ex):
            logger.warning(f"Not exporting body-status-counts graph: {str(ex)}")
        else:
            raise
