import os
import sys
import copy
import json
import pickle
import logging.config
from collections.abc import Mapping

from confiddler import dump_default_config, load_config, dump_config
from neuclease import PrefixFilter
from neuclease.util import Timer, switch_cwd, dump_json
from neuclease.dvid import set_default_dvid_session_timeout
from neuclease.dvid.labelmap import resolve_snapshot_tag

from . import __version__
from .inputs.dvidseg import DvidSegSchema, load_dvidseg
from .inputs.elements import ElementTablesSchema, load_elements
from .inputs.synapses import SnapshotSynapsesSchema, load_synapses, RawSynapseSerializer
from .inputs.annotations import AnnotationsSchema, load_annotations
from .inputs.rois import RoisSchema, load_point_rois, merge_partner_rois
from .inputs.sizes import BodySizesSchema, load_body_sizes
from .inputs.neurotransmitters import NeurotransmittersSchema, load_neurotransmitters

from .outputs.flat import FlatConnectomeSchema, export_flat_connectome
from .outputs.neurotransmitters import NeurotransmitterExportSchema, export_neurotransmitters
from .outputs.neuprint import NeuprintSchema, export_neuprint
from .outputs.neuprint.meta import NeuprintMetaSchema
from .outputs.reports import ReportsSchema, export_reports

from .caches import cached, SerializerBase
from .util.lsf import log_lsf_details
from .util.checksum import checksum

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
                "synapses": SnapshotSynapsesSchema,
                "elements": ElementTablesSchema,
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
                "neurotransmitters": NeurotransmitterExportSchema,
                "flat-connectome": FlatConnectomeSchema,
                "neuprint": NeuprintSchema,
                "connectivity-reports": ReportsSchema,
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
                },
                "logging-config": {
                    "description": "Incremental logging configuration updates as described in:\n"
                                   "https://docs.python.org/3/library/logging.config.html#logging-config-dictschema\n",
                    "type": "object",
                    "default": {
                        "loggers": {
                            "flyem_snapshot": {
                                "level": "INFO",
                            }
                        },
                        "incremental": True
                    }
                }
            }
        }
    }
}


def main(args):
    """
    Main function.

    Handles everything except CLI argument parsing.
    For argument definitions, see bin/flyem_snapshot_entrypoint.py

    (CLI parsing is implemented in a separate entrypoint
    script to avoid importing lots of packages when the
    user just wants to see the --help message.)
    """
    cfg, config_dir = process_args_and_parse_config(args)

    with Timer("Processing snapshot", logger, log_start=False):
        main_setup(cfg, config_dir)
        main_impl(cfg)

    logger.info("DONE")


def process_args_and_parse_config(args):
    """
    Process the (already-parsed) command-line args and parse the config (if given).
    Normally, this just returns the parsed config data and the config's directory.
    But if the user specified one of the command-line options to dump out info instead of
    running a complete export, then this dumps out the requested info and exits immediately.
    """
    if args.dump_default_yaml:
        dump_default_config(ConfigSchema, sys.stdout, 'yaml')
        sys.exit(0)

    if args.dump_verbose_yaml:
        dump_default_config(ConfigSchema, sys.stdout, 'yaml-with-comments')
        sys.exit(0)

    if args.dump_neuprint_default_meta:
        dump_default_config(NeuprintMetaSchema, sys.stdout, 'yaml')
        sys.exit(0)

    if args.dump_verbose_neuprint_default_meta:
        dump_default_config(NeuprintMetaSchema, sys.stdout, 'yaml-with-comments')
        sys.exit(0)

    if not args.config:
        sys.stderr.write("No config provided.\n")
        sys.exit(1)

    cfg = load_config(args.config, ConfigSchema)
    config_dir = os.path.dirname(args.config)

    if args.print_snapshot_tag:
        _, snapshot_tag, _ = determine_snapshot_tag(cfg, config_dir)
        print(snapshot_tag)
        sys.exit(0)

    if args.print_output_directory:
        _, _, output_dir = determine_snapshot_tag(cfg, config_dir)
        print(output_dir)
        sys.exit(0)

    return cfg, config_dir


def main_setup(cfg, config_dir):
    init_logging(cfg['job-settings']['logging-config'])

    logger.info(f"Running with flyem-snapshot {__version__}")
    log_lsf_details(logger)

    timeout = cfg['job-settings']['dvid-timeout']
    set_default_dvid_session_timeout(timeout, timeout)

    standardize_config(cfg, config_dir)
    initialize_output_dir(cfg)


def main_impl(cfg):
    output_dir = cfg['job-settings']['output-dir']
    logger.info(f"Working in {output_dir}")
    with switch_cwd(output_dir):
        #
        # Load inputs
        #
        snapshot_tag = cfg['job-settings']['snapshot-tag']
        pointlabeler = load_dvidseg(cfg['inputs']['dvid-seg'], snapshot_tag)
        ann = load_annotations(cfg['inputs']['annotations'], pointlabeler, snapshot_tag)
        point_df, partner_df, syn_roisets = load_synapses_and_rois(cfg, pointlabeler)
        element_tables, element_roisets = load_elements_and_rois(cfg, pointlabeler)
        all_bodies = [
            ann.index.values,
            point_df['body'].values,
            *(p['body'].values for p, _ in element_tables.values())
        ]
        body_sizes = load_body_sizes(cfg['inputs']['body-sizes'], pointlabeler, all_bodies, snapshot_tag)
        tbar_nt, body_nt, nt_confusion = load_neurotransmitters(cfg['inputs']['neurotransmitters'], point_df, partner_df, ann)

        #
        # Produce outputs
        #
        min_conf = cfg['inputs']['synapses']['min-confidence']
        export_neurotransmitters(cfg['outputs']['neurotransmitters'], tbar_nt, body_nt, nt_confusion, point_df)
        export_flat_connectome(cfg['outputs']['flat-connectome'], point_df, partner_df, ann, snapshot_tag, min_conf)
        export_neuprint(cfg['outputs']['neuprint'], point_df, partner_df, element_tables, ann, body_sizes,
                        tbar_nt, body_nt, syn_roisets, element_roisets, pointlabeler)
        export_reports(cfg['outputs']['connectivity-reports'], point_df, partner_df, ann, snapshot_tag)


class SynapsesWithRoiSerializer(SerializerBase):

    def get_cache_key(self, cfg, pointlabeler):
        snapshot_tag = cfg['job-settings']['snapshot-tag']
        cfg = copy.deepcopy(cfg)
        del cfg['job-settings']
        cfg['inputs']['synapses']['processes'] = 0
        cfg['inputs']['rois']['processes'] = 0
        syn_hash = hex(checksum(cfg['inputs']['synapses']))
        roi_hash = hex(checksum(cfg['inputs']['rois']))

        if pointlabeler is None:
            return f'{snapshot_tag}-syn-{syn_hash}-roi-{roi_hash}'

        mutid = pointlabeler.last_mutation["mutid"]
        return f'{snapshot_tag}-seg-{mutid}-syn-{syn_hash}-roi-{roi_hash}'

    def save_to_file(self, result, path):
        point_df, partner_df, syn_roisets = result
        os.makedirs(path, exist_ok=True)
        RawSynapseSerializer('').save_to_file((point_df, partner_df), path)
        dump_json(syn_roisets, f'{path}/roisets.json')

    def load_from_file(self, path):
        point_df, partner_df = RawSynapseSerializer('').load_from_file(path)
        with open(f'{path}/roisets.json', 'r') as f:
            syn_roisets = json.load(f)
        return point_df, partner_df, syn_roisets


@PrefixFilter.with_context('synapses')
@cached(SynapsesWithRoiSerializer('labeled-synapses-with-rois'))
def load_synapses_and_rois(cfg, pointlabeler):
    """
    Load the synapse table and append columns for each ROI set.

    Although :Synapse is a special case of :Element,
    we can't just use load_elements_and_rois() for synapses,
    for the following reasons:

      - Since we expect the synapse table to be huge,
        we take special care to use optimal dtypes for each column.
      - We support special options for filtering by 'zone' and/or confidence.
      - We also create a special table of pre->post relationships
        (partner_df).  We merge ROI columns onto that table
        (based on the 'post' side).
    """
    snapshot_tag = cfg['job-settings']['snapshot-tag']
    point_df, partner_df = load_synapses(
        cfg['inputs']['synapses'],
        snapshot_tag,
        pointlabeler
    )

    point_df, syn_roisets = load_point_rois(
        cfg['inputs']['rois'],
        point_df,
        cfg['inputs']['synapses']['roi-set-names']
    )

    partner_df = merge_partner_rois(
        cfg['inputs']['rois'],
        point_df,
        partner_df,
        cfg['inputs']['synapses']['roi-set-names']
    )

    return point_df, partner_df, syn_roisets


class ElementsWithRoiSerializer(SerializerBase):

    def get_cache_key(self, cfg, pointlabeler):
        snapshot_tag = cfg['job-settings']['snapshot-tag']
        cfg = copy.deepcopy(cfg)
        del cfg['job-settings']
        cfg['inputs']['elements']['processes'] = 0
        cfg['inputs']['rois']['processes'] = 0
        elm_hash = hex(checksum(cfg['inputs']['elements']))
        roi_hash = hex(checksum(cfg['inputs']['rois']))

        if pointlabeler is None:
            return f'{snapshot_tag}-elm-{elm_hash}-roi-{roi_hash}'

        mutid = pointlabeler.last_mutation["mutid"]
        return f'{snapshot_tag}-seg-{mutid}-elm-{elm_hash}-roi-{roi_hash}'

    def save_to_file(self, result, path):
        os.makedirs(path, exist_ok=True)
        element_tables, element_roisets = result
        with open(f'{path}/element-cache.pkl', 'wb') as f:
            pickle.dump((element_tables, element_roisets), f)

    def load_from_file(self, path):
        with open(f'{path}/element-cache.pkl', 'rb') as f:
            element_tables, element_roisets = pickle.load(f)
        return element_tables, element_roisets


@PrefixFilter.with_context('elements')
@cached(ElementsWithRoiSerializer('labeled-elements-with-rois'))
def load_elements_and_rois(cfg, pointlabeler):
    element_tables = load_elements(cfg['inputs']['elements'], pointlabeler)
    element_roisets = {}
    for elm_name in list(element_tables.keys()):
        elm_points, elm_distances = element_tables[elm_name]
        if elm_points is None:
            continue
        with PrefixFilter.context(elm_name):
            elm_points, elm_roisets = load_point_rois(
                cfg['inputs']['rois'],
                elm_points,
                cfg['inputs']['elements'][elm_name]['roi-set-names']
            )
        element_tables[elm_name] = (elm_points, elm_distances)
        element_roisets[elm_name] = elm_roisets

    return element_tables, element_roisets


def determine_snapshot_tag(cfg, config_dir):
    """
    Determine the snapshot tag and UUID from the user's config.
    If the user didn't provide a snapshot tag, construct one using a
    date from the segmentation mutation log.

    Also return the final output-dir.

    FIXME: Ideally, we should also consult the annotation instance
           to see if it has even newer dates.

    Example default snapshot tags:

        - 2023-11-03-abc123
        - 2023-11-04-def456-unlocked

    """
    uuid, snapshot_tag = None, None
    dvidcfg = cfg['inputs'].get('dvid-seg', {"server": None})
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


def init_logging(log_cfg):
    logging.captureWarnings(True)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(PrefixFilter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    assert log_cfg.setdefault('version', 1) == 1, \
        "Python logging config version should be 1"
    assert log_cfg.setdefault('incremental', True), \
        "Only incremenetal logging configuration is supported via the config file."
    logging.config.dictConfig(log_cfg)


def standardize_config(cfg, config_dir):
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
    anncfg = cfg['inputs']['annotations']
    elmcfg = cfg['inputs']['elements']
    syncfg = cfg['inputs']['synapses']
    roicfg = cfg['inputs']['rois']
    neuprintcfg = cfg['outputs']['neuprint']
    output_ntcfg = cfg['outputs']['neurotransmitters']

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

        # By default, the neurotransmitter dvid backport goes to the main dvid server/uuid.
        output_ntcfg['dvid']['server'] = output_ntcfg['dvid']['server'] or dvidcfg['server']
        output_ntcfg['dvid']['uuid'] = output_ntcfg['dvid']['uuid'] or uuid
        output_ntcfg['dvid']['neuronjson_instance'] = output_ntcfg['dvid']['neuronjson_instance'] or f"{dvidcfg['instance']}_annotations"

    # Some portions of the pipeline have their own setting for process count,
    # but they all default to the top-level config setting if the user didn't specify.
    for subcfg in [*cfg['inputs'].values(), *cfg['outputs'].values()]:
        if isinstance(subcfg, Mapping) and 'processes' in subcfg and subcfg['processes'] is None:
            subcfg['processes'] = jobcfg['processes']

    # Convert file paths to absolute (if necessary).
    # Relative paths are interpreted w.r.t. to the config file, not the cwd.
    # Overwrite the paths with their absolute versions so subsequent functions
    # don't have to worry about relative paths.
    def make_abspath(d, key):
        if d[key]:
            d[key] = os.path.abspath(d[key])

    with switch_cwd(config_dir):
        make_abspath(anncfg, 'body-annotations-table')
        make_abspath(syncfg, 'syndir')

        if not syncfg['synapse-points'].startswith('{syndir}'):
            make_abspath(syncfg, 'synapse-points')
        if not syncfg['synapse-partners'].startswith('{syndir}'):
            make_abspath(syncfg, 'synapse-partners')

        # The 'syndir' keyword is also respected in the neurotransmitter filepath
        ntcfg = cfg['inputs']['neurotransmitters']
        make_abspath(ntcfg, 'ground-truth')
        make_abspath(ntcfg, 'experimental-groundtruth')
        if ntcfg['synister-feather'].startswith('{syndir}'):
            ntcfg['synister-feather'] = ntcfg['synister-feather'].format(syndir=syncfg['syndir'])
        else:
            make_abspath(ntcfg, 'synister-feather')

        make_abspath(cfg['inputs']['body-sizes'], 'cache-file')

        make_abspath(neuprintcfg, 'meta')
        make_abspath(neuprintcfg['neuroglancer'], 'json-state')

        for elm_name, c in elmcfg.items():
            if elm_name == 'processes':
                continue
            make_abspath(c, 'point-table')
            make_abspath(c, 'distance-table')

        for rs in roicfg['roi-sets'].values():
            if isinstance(rs['rois'], str) and rs['rois'].endswith('.json'):
                make_abspath(rs, 'rois')

    # If the user didn't specify an explicit subset of roi-sets
    # to insert into to the synapse table, insert them all.
    if not syncfg['roi-set-names']:
        syncfg['roi-set-names'] = list(roicfg['roi-sets'].keys())

    # Same for all generic Element sets.
    for elm_name, c in elmcfg.items():
        if elm_name == 'processes':
            continue
        if not c['roi-set-names']:
            c['roi-set-names'] = list(roicfg['roi-sets'].keys())

    # If the user didn't specify an explicit subset
    # of roi-sets to include in neuprint, include them all.
    if neuprintcfg['export-neuprint-snapshot'] and not neuprintcfg['roi-set-names']:
        neuprintcfg['roi-set-names'] = syncfg['roi-set-names']

    # If any report is un-named, auto-name it
    # according to the zone and/or ROI list.
    for reportset_cfg in cfg['outputs']['connectivity-reports']:
        for report in reportset_cfg['reports']:
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

    # We load the neuprint :Meta node configuraton from a separate file,
    # but we insert its loaded contents into the main config for all neuprint steps to use.
    if neuprintcfg['export-neuprint-snapshot'] and isinstance(neuprintcfg['meta'], str):
        metacfg = load_config(neuprintcfg['meta'], NeuprintMetaSchema)
        with switch_cwd(os.path.dirname(neuprintcfg['meta'])):
            def fix_hierarchy_paths(rh):
                if isinstance(rh, str) and rh.endswith(('.json', '.yaml')):
                    # Right now we don't tackle the case where a part of the hierarchy
                    # included via a file and that file itself includes others files
                    # which are not absolute paths.
                    return os.path.abspath(rh)
                elif isinstance(rh, dict):
                    return {k: fix_hierarchy_paths(v) for k,v in rh.items()}
                else:
                    return rh

            metacfg['roiHierarchy'] = fix_hierarchy_paths(metacfg['roiHierarchy'])

        neuprintcfg['meta'] = metacfg


def initialize_output_dir(cfg):
    """
    Create the empty output directory and dump the
    fully-standardized config into it (just for future reference).
    """
    output_dir = cfg['job-settings']['output-dir']
    os.makedirs(output_dir, exist_ok=True)

    # Dump the updated config so it's clear what the
    # settings are after "standardization"
    # (absolute paths, resolved UUID, etc.)
    dump_config(cfg, f"{output_dir}/final-config.yaml")

    # Also dump out the neuprint meta config.
    neuprintcfg = cfg['outputs']['neuprint']
    if neuprintcfg['export-neuprint-snapshot']:
        dump_config(neuprintcfg['meta'], f"{output_dir}/neuprint-meta.yaml")
