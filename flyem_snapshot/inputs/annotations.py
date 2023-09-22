import logging
import hvplot.pandas
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
                    "column-name": "soma_position"
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
def load_annotations(cfg, dvid_seg, snapshot_tag):
    """
    Load body annotations, either from a feather file or from DVID.
    If from a file, it MUST contain a 'body' column.

    If a list of 'point-annotations' instances are listed,
    they will be used to add (or overwrite) columns in the
    """
    if cfg['body-annotations-table']:
        logger.info("Reading body annotations table from disk INSTEAD of reading from DVID.")
        ann = feather.read_feather(cfg['body-annotations-table']).set_index('body')
    else:
        ann = fetch_body_annotations(*dvid_seg[:2], dvid_seg[2] + '_annotations')
        # The result includes the original json as an extra column,
        # but that's not necessary for anything in this code.
        del ann['json']

    feather.write_feather(
        ann.reset_index(),
        f'tables/body-annotations-{snapshot_tag}.feather'
    )

    vc = ann['status'].value_counts().sort_index(ascending=False)
    vc = vc[vc > 0]
    vc = vc[vc.index != ""]
    vc.to_csv(f'tables/status-counts-{snapshot_tag}.csv', index=True, header=True)

    title = f'body status counts ({snapshot_tag})'
    p = vc.hvplot.barh(flip_yaxis=True, title=title)
    export_bokeh(
        hv.render(p),
        f"body-status-counts-{snapshot_tag}.html",
        title
    )

    if cfg['point-annotations'] and not dvid_seg:
        raise RuntimeError("Can't read point-annotations without a dvid segmentation.")

    # Anything mentioned in the point-annotations config
    # will override what's in Clio if the name conflicts.
    for pa in cfg['point-annotations']:
        col = pa['column-name'] or pa['instance']
        with PrefixFilter.context(pa['instance']):
            df = fetch_all_elements(*dvid_seg[:2], pa['instance'], format='pandas')
            df = df.sort_values([*'zyx'])
            df['body'] = fetch_labels_batched(
                *dvid_seg,
                df[[*'zyx']].values,
                processes=cfg['processes']
            )
            df[col] = df[[*'xyz']].values.tolist()

            # Append this column to the annotation DataFrame, overwriting the column if necessary.
            # (Even if the column exists in Clio, we're overriding it with the data from DVID.)
            # Note: If more than one point lands on the same body, we drop duplicates.
            ann.drop(columns=[col], errors='ignore', inplace=True)
            ann[col] = df.drop_duplicates('body').set_index('body')[col]

    return ann
