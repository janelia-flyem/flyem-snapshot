"""
NeuprintExplorer contains an embedded neuroglancer view.
That view is initialized using an ordinary neuroglancer JSON state file,
which we store in a special place within the neuprint docker container.
In a few places, the state includes config-dependent strings, such as the
dataset name/tag, and the DVID UUID which should be used by the layers in
the view.

Here, we allow the user to provide that state JSON as a template file in
which we'll replace certain key words with the appropriate values from our
snapshot config.
"""
import json
import logging
from neuclease.util import dump_json

logger = logging.getLogger(__name__)

NeuroglancerSettingsSchema = {
    "description": "Configuration for neuprint's embedded neuroglancer view.\n",
    "default": {},
    "additionalProperties": False,
    "properties": {
        "json-state": {
            "description": "A path to a JSON file containing a (template) neuroglancer link state.",
            "type": "string",
            "default": ""
        },
        ## FIXME: I should just use a jinja template with a few predefined variables.
        "replacements": {
            "type": "object",
            "description":
                "A set of keys and values to replace in the raw text of the neuroglancer settings JSON.\n"
                "DVID_UUID and DATASET_AND_TAG are available with appropriate default values (unless you set them explicitly).\n",
            "additionalProperties": {
                "type": "string"
            },
            "default": {
                "DATASET_AND_TAG": "",
                "DVID_UUID": "",
            }
        }
    }
}


def export_neuroglancer_json_state(cfg, last_mutation):
    if not cfg['neuroglancer']['json-state']:
        logger.info("No neuroglancer state provided.  Skipping.")
        return

    dset = cfg['meta']['dataset']
    tag = cfg['meta']['tag']
    dset_tag = f"{dset}:{tag}" if tag else dset

    repl = cfg['neuroglancer']['replacements']
    repl['DATASET_AND_TAG'] = repl['DATASET_AND_TAG'] or dset_tag
    if last_mutation and not repl['DVID_UUID']:
        repl['DVID_UUID'] = last_mutation['uuid']

    state_text = open(cfg['neuroglancer']['json-state'], 'r').read()
    for k,v in repl.items():
        state_text = state_text.replace(k, v)

    # First, write out a temporary version of the file before attempting to load as JSON.
    # If we fail to load the JSON for some reason, we can debug the temp file.
    output_name = f'neuprint/{dset_tag}.json'
    with open(output_name, 'w') as f:
        f.write(state_text)

    state = json.loads(state_text)

    # Now write the final version.
    logger.info(f"Writing {output_name}")
    dump_json(state, output_name, indent=2, unsplit_number_lists=True)
