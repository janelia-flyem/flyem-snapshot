import os
import sys
import pickle
import logging
from collections.abc import Mapping

from confiddler import dump_default_config, load_config, dump_config
from neuclease import configure_default_logging
from neuclease.util import Timer, switch_cwd
from neuclease.dvid import set_default_dvid_session_timeout
from neuclease.dvid.labelmap import resolve_snapshot_tag

from .inputs.dvidseg import DvidSegSchema, load_dvidseg
from .inputs.elements import ElementTablesSchema, load_elements
from .inputs.synapses import SnapshotSynapsesSchema, load_synapses, export_synapse_cache
from .inputs.annotations import AnnotationsSchema, load_annotations
from .inputs.rois import RoisSchema, load_point_rois, merge_partner_rois
from .inputs.sizes import BodySizesSchema, load_body_sizes
from .inputs.neurotransmitters import NeurotransmittersSchema, load_neurotransmitters

from .outputs.flat import FlatConnectomeSchema, export_flat_connectome
from .outputs.neurotransmitters import NeurotransmiterExportSchema, export_neurotransmitters
from .outputs.neuprint import NeuprintSchema, export_neuprint
from .outputs.neuprint.meta import NeuprintMetaSchema
from .outputs.reports import ReportsSchema, export_reports

from .util import log_lsf_details

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
            "additionalProperties": False,
            "properties": {
                "dvid-seg": DvidSegSchema,
                "elements": ElementTablesSchema,
                "synapses": SnapshotSynapsesSchema,
                "annotations": AnnotationsSchema,
                "rois": RoisSchema,
                "body-sizes": BodySizesSchema,
                "neurotransmitters": NeurotransmittersSchema,
            }
        },
        "outputs": {
            "description": "Specs for the exports/reports to produce from the snapshot data.\n",
            "default": {},
            "additionalProperties": False,
            "properties": {
                # TODO: BigQuery exports
                "neuprint": NeuprintSchema,
                "connectivity-reports": ReportsSchema,
                "flat-connectome": FlatConnectomeSchema,
                "neurotransmitters": NeurotransmiterExportSchema,
            }
        },
        "job-settings": {
            "description": "General settings.\n",
            "default": {},
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
                        "If not specified, a reasonable default is chosen and placed in the CURRENT DIRECTORY AT RUNTIME.\n",
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
                    "default": 300.0,
                }
            }
        }
    }
}


def main(args):
    """
    Main function.
    Handles everything except CLI argument parsing,
    which is located in a separate file to avoid importing
    lots of packages when the user just wants --help.

    For argument definitions and explanations, see bin/flyem_snapshot_entrypoint.py
    """
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

    if not args.config:
        sys.stderr.write("No config provided.\n")
        return 1

    cfg = load_config(args.config, ConfigSchema)
    config_dir = os.path.dirname(args.config)

    if args.print_snapshot_tag:
        _, snapshot_tag, _ = determine_snapshot_tag(cfg, config_dir)
        print(snapshot_tag)
        return
    elif args.print_output_directory:
        _, _, output_dir = determine_snapshot_tag(cfg, config_dir)
        print(output_dir)
        return

    configure_default_logging()
    log_lsf_details(logger)

    with Timer("Exporting snapshot denormalizations", logger, log_start=False):
        export_all(cfg, config_dir)
    logger.info("DONE")


def export_all(cfg, config_dir):
    timeout = cfg['job-settings']['dvid-timeout']
    set_default_dvid_session_timeout(timeout, timeout)

    _finalize_config_and_output_dir(cfg, config_dir)

    # All subsequent processing occurs from within the output-dir
    output_dir = cfg['job-settings']['output-dir']
    logger.info(f"Working in {output_dir}")
    with switch_cwd(output_dir):
        (last_mutation, ann, element_tables, point_df, partner_df,
            syn_roisets, element_roisets, body_sizes, tbar_nt, body_nt) = load_inputs(cfg)

        produce_outputs(cfg, last_mutation, ann, element_tables, point_df, partner_df,
                        syn_roisets, element_roisets, body_sizes, tbar_nt, body_nt)


def load_inputs(cfg):
    snapshot_tag = cfg['job-settings']['snapshot-tag']

    #
    # Segmentation parameters
    #
    dvidseg, last_mutation, pointlabeler = load_dvidseg(cfg['inputs']['dvid-seg'], snapshot_tag)

    #
    # Annotations
    #
    ann = load_annotations(
        cfg['inputs']['annotations'],
        dvidseg,
        snapshot_tag
    )

    point_df, partner_df = load_synapses(
        cfg['inputs']['synapses'],
        snapshot_tag,
        pointlabeler
    )

    #
    # Synapses
    #
    point_df, syn_roisets = load_point_rois(
        cfg['inputs']['rois'],
        point_df,
        cfg['inputs']['synapses']['roi-set-names']
    )

    partner_df = merge_partner_rois(
        cfg['inputs']['rois'],
        point_df,
        partner_df
    )

    export_synapse_cache(point_df, partner_df, snapshot_tag)

    #
    # Elements
    #
    element_tables = load_elements(cfg['inputs']['elements'], pointlabeler)
    element_roisets = {}
    for elm_name in list(element_tables.keys()):
        elm_points, elm_distances = element_tables[elm_name]
        if elm_points is None:
            continue
        elm_points, elm_roisets = load_point_rois(
            cfg['inputs']['rois'],
            elm_points,
            cfg['inputs']['synapses']['roi-set-names']
        )
        element_tables[elm_name] = (elm_points, elm_distances)
        element_roisets[elm_name] = elm_roisets

    with open('tables/element_tables.pkl', 'wb') as f:
        pickle.dump(element_tables, f)

    #
    # Body sizes (for synaptic bodies only)
    #
    body_sizes = load_body_sizes(cfg['inputs']['body-sizes'], dvidseg, point_df, snapshot_tag)

    #
    # Neurotransmitters
    #
    tbar_nt, body_nt = load_neurotransmitters(cfg['inputs']['neurotransmitters'], point_df)

    return (last_mutation, ann, element_tables, point_df, partner_df,
            syn_roisets, element_roisets, body_sizes, tbar_nt, body_nt)


def produce_outputs(cfg, last_mutation, ann, element_tables, point_df, partner_df,
                    syn_roisets, element_roisets, body_sizes, tbar_nt, body_nt):

    snapshot_tag = cfg['job-settings']['snapshot-tag']
    min_conf = cfg['inputs']['synapses']['min-confidence']

    export_neurotransmitters(cfg['outputs']['neurotransmitters'], tbar_nt, body_nt, point_df)

    export_neuprint(cfg['outputs']['neuprint'], point_df, partner_df, element_tables, ann, body_sizes,
                    tbar_nt, body_nt, syn_roisets, element_roisets, last_mutation)

    export_flat_connectome(cfg['outputs']['flat-connectome'], point_df, partner_df, ann, snapshot_tag, min_conf)
    export_reports(cfg['outputs']['connectivity-reports'], point_df, partner_df, ann, snapshot_tag)


def determine_snapshot_tag(cfg, config_dir):
    """
    Determine the snapshot tag and UUID from the user's config.
    If the user didn't provide a snapshot tag, construct one using a
    date from the segmentation mutation log.

    Also return the final output-dir.

    FIXME: Ideally, we should also consult the annotation instance
            to see if it has even newer dates.
    """
    dvidcfg = cfg['inputs'].get('dvid-seg', {"server": None})
    uuid = None
    if dvidcfg['server']:
        uuid, snapshot_tag = resolve_snapshot_tag(
            dvidcfg['server'],
            dvidcfg['uuid'],
            dvidcfg['instance']
        )

    # If user supplied an explicit snapshot-tag in the config, use that.
    snapshot_tag = (cfg['job-settings']['snapshot-tag'] or snapshot_tag)

    if not snapshot_tag:
        msg = (
            "Since your config does not refer to a dvid-seg, you must "
            "supply an explicit snapshot-tag to use in output file names."
        )
        raise RuntimeError(msg)

    jobcfg = cfg['job-settings']
    if jobcfg['output-dir']:
        # If an output-dir WAS specified, then a relative output dir
        # is relative to the config file.
        with switch_cwd(config_dir):
            output_dir = os.path.abspath(jobcfg['output-dir'])
    else:
        # If no output-dir was specified, then we place it in the
        # current (execution) directory, named according to the snapshot_tag
        output_dir = os.path.abspath(snapshot_tag)

    return uuid, snapshot_tag, output_dir


def _finalize_config_and_output_dir(cfg, config_dir):
    """
    This function encapsulates all logic related to overwriting the user's
    config after it has been loaded and populated with default values from the schema.

    - All relative file paths throughout the config are converted to absolute paths,
      assuming the paths were relative to the directory of the original config file.

    - In some cases, we fill in values that the user declined to set explicitly.
        - snapshot-tag
        - output-dir

    - In some cases, those default values come from OTHER sections in the config.
        - processes
        - uuid
        - roi server
        - roi-set-names
        - report names

    FIXME: This function has gotten a little out of hand.
           It was nice to have all config-manipulation in one place, but at this point
           we should probably move some of this logic into the relevant pipeline steps.
    """
    jobcfg = cfg['job-settings']
    dvidcfg = cfg['inputs']['dvid-seg'] = cfg['inputs'].get('dvid-seg', {"server": None})
    syncfg = cfg['inputs']['synapses']
    roicfg = cfg['inputs']['rois']
    neuprintcfg = cfg['outputs']['neuprint']

    uuid, snapshot_tag, output_dir = determine_snapshot_tag(cfg, config_dir)
    jobcfg['snapshot-tag'] = snapshot_tag
    jobcfg['output-dir'] = output_dir

    if uuid:
        uuid, snapshot_tag = resolve_snapshot_tag(
            dvidcfg['server'],
            dvidcfg['uuid'],
            dvidcfg['instance']
        )
        # Overwrite config UUID ref with the resolved (explicit) UUID
        dvidcfg['uuid'] = uuid

        # By default, the ROI config uses the main server/uuid.
        roicfg['dvid']['server'] = roicfg['dvid']['server'] or dvidcfg['server']
        roicfg['dvid']['uuid'] = roicfg['dvid']['uuid'] or uuid

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

        # The 'syndir' keyword is also respected in the neurotransmitter filepath
        ntcfg = cfg['inputs']['neurotransmitters']
        if ntcfg['synister-feather'].startswith('{syndir}'):
            ntcfg['synister-feather'] = ntcfg['synister-feather'].format(syndir=syncfg['syndir'])
        elif ntcfg['synister-feather']:
            ntcfg['synister-feather'] = os.path.abspath(ntcfg['synister-feather'])

        bscfg = cfg['inputs']['body-sizes']
        if bscfg['cache-file']:
            bscfg['cache-file'] = os.path.abspath(bscfg['cache-file'])

        neuprintcfg['meta'] = os.path.abspath(neuprintcfg['meta'])
        neuprintcfg['neuroglancer']['json-state'] = os.path.abspath(neuprintcfg['neuroglancer']['json-state'])

    # If the user didn't specify an explicit subset of roi-sets
    # to insert into to the synapse table, insert them all.
    if not syncfg['roi-set-names']:
        syncfg['roi-set-names'] = list(roicfg['roi-sets'].keys())

    # If the user didn't specify an explicit subset
    # of roi-sets to include in neuprint, include them all.
    if neuprintcfg['export-neuprint-snapshot'] and not neuprintcfg['roi-set-names']:
        neuprintcfg['roi-set-names'] = syncfg['roi-set-names']

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

    # We load the neuprint :Meta node configuraton from a separate file,
    # but we insert its loaded contents into the main config for all neuprint steps to use.
    if neuprintcfg['export-neuprint-snapshot']:
        metacfg = load_config(neuprintcfg['meta'], NeuprintMetaSchema)
        p = os.path.basename(neuprintcfg['meta'])
        dump_config(metacfg, f"{output_dir}/{p}")
        neuprintcfg['meta'] = metacfg
