"""
Generate connectome exports, neuprint databases and reports from flat files and DVID checkpoints.
"""
import os
import sys
import logging
import argparse
from collections.abc import Mapping

from confiddler import dump_default_config, load_config, dump_config
from neuclease import configure_default_logging
from neuclease.util import Timer, switch_cwd
from neuclease.dvid import set_default_dvid_session_timeout
from neuclease.dvid.labelmap import resolve_snapshot_tag

from .synapses import SnapshotSynapsesSchema, load_synapses
from .annotations import AnnotationsSchema, load_annotations
from .rois import RoisSchema, load_rois
from .flat import FlatConnectomeSchema, export_flat_connectome
from .sizes import BodySizesSchema, load_body_sizes
from .neuprint import NeuprintSchema, export_neuprint
from .reports import ReportsSchema, export_reports

logger = logging.getLogger(__name__)

ConfigSchema = {
    "description": "Configuration for exporting connectomic denormalizations from DVID",
    "default": {},
    "required": ["snapshot", "synapse-points", "synapse-partners"],
    "additionalProperties": False,
    "properties": {
        "snapshot-tag": {
            "description":
                "A suffix to add to export filenames.\n"
                "By default, a tag is automatically chosen which incorporates the\n"
                "snapshot date, uuid, and uuid commit status.",
            "type": "string",
            "default": "",
        },
        "output-dir": {
            "description":
                "Where to write output files.\n"
                "Relative paths here are interpreted from the directory in which this config file is stored.\n"
                "If not specified, a reasonable default is chosen IN THE SAME DIRECTORY AS THIS CONFIG FILE.\n",
            "type": "string",
            "default": "",
        },
        "synapses": SnapshotSynapsesSchema,
        "annotations": AnnotationsSchema,
        "rois": RoisSchema,
        "body-sizes": BodySizesSchema,
        "flat-connectome": FlatConnectomeSchema,
        "neuprint": NeuprintSchema,
        "reports": ReportsSchema,
        "processes": {
            "description":
                "For steps which benefit from multiprocessing, how many processes should be used?",
            "type": "integer",
            "default": 16,
        },
        "dvid-timeout": {
            "description": "Timeout for dvid requests, in seconds. Used for both 'connect' and 'read' timeout.",
            "type": "number",
            "default": 180.0,
        }
    }
}


def main():
    DEBUG = False
    if DEBUG and len(sys.argv) == 1:
        sys.stderr.write("DEBUGGING WITH ARTIFICAL ARGS\n")
        sys.argv.extend(['-c', 'neuprint-small-test.yaml'])

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--config')
    parser.add_argument('-y', '--dump-default-yaml', action='store_true')
    parser.add_argument('-v', '--dump-verbose-yaml', action='store_true')
    args = parser.parse_args()

    if sum((bool(args.config), bool(args.dump_default_yaml), bool(args.dump_verbose_yaml))) > 1:
        sys.exit("You can't provide more than one of these options: -c, -y, -v")

    if args.dump_default_yaml:
        dump_default_config(ConfigSchema, sys.stdout, 'yaml')
        return

    if args.dump_verbose_yaml:
        dump_default_config(ConfigSchema, sys.stdout, 'yaml-with-comments')
        return

    configure_default_logging()
    cfg = load_config(args.config, ConfigSchema)
    config_dir = os.path.dirname(args.config)

    with Timer("Exporting snapshot denormalizations", logger):
        export_all(cfg, config_dir)
    logger.info("DONE")


def export_all(cfg, config_dir):
    set_default_dvid_session_timeout(cfg['dvid-timeout'], cfg['dvid-timeout'])

    # Output dir is created in the cwd (if output-dir is a relative path).
    _finalize_config_and_output_dir(cfg, config_dir)

    dvid_seg = (
        cfg['synapses']['update-to']['server'],
        cfg['synapses']['update-to']['uuid'],
        cfg['synapses']['update-to']['instance'],
    )
    snapshot_tag = cfg['snapshot-tag']
    min_conf = cfg['synapses']['min-confidence']

    # All subsequent processing occurs from within the output-dir
    with switch_cwd(cfg['output-dir']):
        # Load inputs
        point_df, partner_df = load_synapses(cfg['synapses'], snapshot_tag)
        ann = load_annotations(cfg['annotations'], dvid_seg, snapshot_tag)
        point_df, partner_df = load_rois(cfg['rois'], point_df, partner_df)
        body_sizes = load_body_sizes(cfg['body-sizes'], dvid_seg, point_df, snapshot_tag)

        # Produce exports
        export_neuprint(cfg['neuprint'], point_df, partner_df, ann, body_sizes)
        export_reports(cfg['reports'], point_df, partner_df, ann)
        export_flat_connectome(cfg, point_df, partner_df, ann, snapshot_tag, min_conf)


def _finalize_config_and_output_dir(cfg, config_dir):
    uuid, snapshot_tag = resolve_snapshot_tag(
        cfg['snapshot']['server'],
        cfg['snapshot']['uuid'],
        cfg['snapshot']['instance']
    )
    cfg['snapshot']['uuid'] = uuid
    snapshot_tag = cfg['snapshot-tag'] = (cfg['snapshot-tag'] or snapshot_tag)

    # Some portions of the pipeline have their own setting for process count,
    # but they all default to the top-level config setting if the user didn't specify.
    for subcfg in cfg.values():
        if isinstance(subcfg, Mapping) and 'processes' in subcfg and subcfg['processes'] is None:
            subcfg['processes'] = cfg['processes']

    # Convert synapse and size paths to absolute (if necessary).
    # Relative paths are interpreted w.r.t. to the config file, not the cwd.
    with switch_cwd(config_dir):
        cfg['syndir'] = os.path.abspath(cfg['syndir'])
        if not cfg['synapse-points'].startswith('{syndir}'):
            cfg['synapse-points'] = os.path.abspath(cfg['synapse-points'])
        if not cfg['synapse-partners'].startswith('{syndir}'):
            cfg['synapse-partners'] = os.path.abspath(cfg['synapse-partners'])
        if cfg['body-sizes']['file']:
            cfg['body-sizes']['file'] = os.path.abspath(cfg['body-size-cache']['file'])

    for report in cfg['reports']:
        if report['name']:
            continue
        if cfg['zone'] == 'brain':
            report['name'] = 'brain'
        if cfg['zone'] == 'vnc':
            report['name'] = 'vnc'
        elif not report['rois']:
            report['name'] = 'all'
            continue
        else:
            report['name'] = '-'.join(report['rois'])

    output_dir = cfg['output-dir'] = os.path.abspath(cfg['output-dir'] or snapshot_tag)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    os.makedirs(f"{output_dir}/png", exist_ok=True)
    os.makedirs(f"{output_dir}/html", exist_ok=True)
    os.makedirs(f"{output_dir}/volumes", exist_ok=True)
    dump_config(cfg, f"{output_dir}/final-config.yaml")


if __name__ == "__main__":
    main()
