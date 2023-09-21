import os
import sys
import logging
from collections.abc import Mapping

from confiddler import dump_default_config, load_config, dump_config
from neuclease import configure_default_logging
from neuclease.util import Timer, switch_cwd
from neuclease.dvid import set_default_dvid_session_timeout
from neuclease.dvid.labelmap import resolve_snapshot_tag

from .inputs.synapses import SnapshotSynapsesSchema, load_synapses
from .inputs.annotations import AnnotationsSchema, load_annotations
from .inputs.rois import RoisSchema, load_rois
from .inputs.sizes import BodySizesSchema, load_body_sizes

from .outputs.flat import FlatConnectomeSchema, export_flat_connectome
from .outputs.neuprint import NeuprintSchema, export_neuprint
from .outputs.neuprint.meta import NeuprintMetaSchema
from .outputs.reports import ReportsSchema, export_reports

logger = logging.getLogger(__name__)

ConfigSchema = {
    "description": "Configuration for exporting connectomic denormalizations from DVID",

    # Note:
    #   Throughout this schema, we use 'default: {}' for complex objects.
    #   When using jsonschema/confiddler, this essentially means "don't override my children's defaults."
    #   This enables confiddler to construct fully-nested default values for objects with nested sub-objects.
    #   See the 'hint' in the jsonschema FAQ on this point:
    #   https://python-jsonschema.readthedocs.io/en/latest/faq/#why-doesn-t-my-schema-s-default-property-set-the-default-on-my-instance
    "default": {},

    "required": ["inputs", "outputs"],
    "additionalProperties": False,
    "properties": {
        "inputs": {
            "description": "Input files and specs to use for constructing the snapshot denormalizations and reports.",
            "default": {},
            "properties": {
                "synapses": SnapshotSynapsesSchema,
                "annotations": AnnotationsSchema,
                "rois": RoisSchema,
                "body-sizes": BodySizesSchema,
            }
        },
        "outputs": {
            "description": "Specs for the exports/reports to produce from the snapshot data.\n",
            "default": {},
            "properties": {
                "flat-connectome": FlatConnectomeSchema,
                "neuprint": NeuprintSchema,
                "connectivity-reports": ReportsSchema,
            }
        },
        "job-settings": {
            "description": "General settings.\n",
            "default": {},
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
                "processes": {
                    "description":
                        "For steps which benefit from multiprocessing, how many processes should be used?\n"
                        "This can be overridden for certain steps of the pipeline.  See the config subsections for inputs/outputs.",
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
    }
}


def main(args):
    # See argument definitions in bin/flyem_snapshot_entrypoint.py
    if args.dump_default_yaml:
        dump_default_config(ConfigSchema, sys.stdout, 'yaml')
        return

    if args.dump_verbose_yaml:
        dump_default_config(ConfigSchema, sys.stdout, 'yaml-with-comments')
        return

    if args.dump_neuprint_default_meta:
        dump_default_config(NeuprintMetaSchema, sys.stdout, 'yaml')
        return

    if args.dump_verbose_neuprint_default_meta:
        dump_default_config(NeuprintMetaSchema, sys.stdout, 'yaml-with-comments')
        return

    configure_default_logging()
    cfg = load_config(args.config, ConfigSchema)
    config_dir = os.path.dirname(args.config)

    with Timer("Exporting snapshot denormalizations", logger):
        export_all(cfg, config_dir)
    logger.info("DONE")


def export_all(cfg, config_dir):
    timeout = cfg['job-settings']['dvid-timeout']
    set_default_dvid_session_timeout(timeout, timeout)

    _finalize_config_and_output_dir(cfg, config_dir)

    # These config settings are needed by stages other than
    # the stage that "owns" the config setting,
    # so we pass them via args instead of via the config.
    snapshot_tag = cfg['job-settings']['snapshot-tag']
    min_conf = cfg['inputs']['synapses']['min-confidence']
    update_seg = cfg['inputs']['synapses']['update-to']
    dvid_seg = (update_seg['server'], update_seg['uuid'], update_seg['instance'])

    # All subsequent processing occurs from within the output-dir
    with switch_cwd(cfg['job-settings']['output-dir']):
        # Load inputs
        ann = load_annotations(cfg['inputs']['annotations'], dvid_seg, snapshot_tag)
        point_df, partner_df, last_mutation = load_synapses(cfg['inputs']['synapses'], snapshot_tag)
        point_df, partner_df = load_rois(cfg['inputs']['rois'], snapshot_tag, point_df, partner_df)
        body_sizes = load_body_sizes(cfg['inputs']['body-sizes'], dvid_seg, point_df, snapshot_tag)

        # Produce outputs
        export_neuprint(cfg['outputs']['neuprint'], point_df, partner_df, ann, body_sizes, last_mutation)
        export_reports(cfg['outputs']['connectivity-reports'], point_df, partner_df, ann, snapshot_tag)
        export_flat_connectome(cfg['outputs']['flat-connectome'], point_df, partner_df, ann, snapshot_tag, min_conf)


def _finalize_config_and_output_dir(cfg, config_dir):
    syncfg = cfg['inputs']['synapses']
    roicfg = cfg['inputs']['rois']
    neuprintcfg = cfg['outputs']['neuprint']
    jobcfg = cfg['job-settings']

    snapshot_tag = None
    if syncfg['update-to']:
        uuid, snapshot_tag = resolve_snapshot_tag(
            syncfg['update-to']['server'],
            syncfg['update-to']['uuid'],
            syncfg['update-to']['instance']
        )
        # Overwrite config UUID ref with the resolved (explicit) UUID
        syncfg['update-to']['uuid'] = uuid
        snapshot_tag = jobcfg['snapshot-tag'] = (jobcfg['snapshot-tag'] or snapshot_tag)

        # By default, the ROI config uses the same server/uuid as the synapses.
        roicfg['dvid']['server'] = roicfg['dvid']['server'] or syncfg['update-to']['server']
        roicfg['dvid']['uuid'] = roicfg['dvid']['server'] or uuid

    if not snapshot_tag:
        msg = (
            "Since your synapses config does not refer to a DVID "
            "snapshot UUID in the 'update-to' setting, you must supply "
            "an explicit snapshot-tag to use in output file names."
        )
        raise RuntimeError(msg)

    # Some portions of the pipeline have their own setting for process count,
    # but they all default to the top-level config setting if the user didn't specify.
    for subcfg in [*cfg['inputs'].values(), *cfg['outputs'].values()]:
        if isinstance(subcfg, Mapping) and 'processes' in subcfg and subcfg['processes'] is None:
            subcfg['processes'] = jobcfg['processes']

    # Convert file paths to absolute (if necessary).
    # Relative paths are interpreted w.r.t. to the config file, not the cwd.
    # Overwrite the paths with their absolute versions so subsequent functions
    # don't have to worry about relative paths.
    with switch_cwd(config_dir):
        syncfg['syndir'] = os.path.abspath(syncfg['syndir'])
        if not syncfg['synapse-points'].startswith('{syndir}'):
            syncfg['synapse-points'] = os.path.abspath(syncfg['synapse-points'])
        if not syncfg['synapse-partners'].startswith('{syndir}'):
            syncfg['synapse-partners'] = os.path.abspath(syncfg['synapse-partners'])

        bscfg = cfg['inputs']['body-sizes']
        if bscfg['cache-file']:
            bscfg['cache-file'] = os.path.abspath(bscfg['cache-file'])

        neuprintcfg['meta'] = os.path.abspath(neuprintcfg['meta'])
        output_dir = jobcfg['output-dir'] = os.path.abspath(jobcfg['output-dir'] or snapshot_tag)

    # If the user didn't specify an explicit subset
    #  of roi-sets to include in neuprint, include them all.
    if neuprintcfg['export-neuprint-snapshot']:
        if neuprintcfg['roi-set-names'] is None:
            neuprintcfg['roi-set-names'] = list(roicfg['roi-sets'].keys())

    # If any report is un-named, auto-name it
    # according to the zone and/or ROI list.
    for report in cfg['outputs']['connectivity-reports']['reports']:
        if report['name']:
            continue
        if report['rois']:
            report['name'] = '-'.join(report['rois'])
        elif syncfg['zone'] == 'brain':
            report['name'] = 'brain'
        elif syncfg['zone'] == 'vnc':
            report['name'] = 'vnc'
        else:
            report['name'] = 'all'

    # Ensure output directories exist.
    os.makedirs(f"{output_dir}/tables", exist_ok=True)
    os.makedirs(f"{output_dir}/png", exist_ok=True)
    os.makedirs(f"{output_dir}/html", exist_ok=True)

    # Dump the updated config so it's clear what modifications
    # we made and how the UUID was resolved.
    dump_config(cfg, f"{output_dir}/final-config.yaml")

    # We load the neuprint :Meta node configuraton in a separate file,
    # but we insert its loaded contents into the main config for all neuprint steps to use.
    if neuprintcfg['export-neuprint-snapshot']:
        metacfg = load_config(neuprintcfg['meta'], NeuprintMetaSchema)
        p = os.path.basename(neuprintcfg['meta'])
        dump_config(metacfg, f"{output_dir}/{p}")
        neuprintcfg['meta'] = metacfg
