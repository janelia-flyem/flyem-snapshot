"""
Compare neuronjson annotations from DVID/Clio with the Neuron/Segment
properties from Neuprint, and update the Neuprint properties to match
DVID/Clio if differences are found.

Only bodies which already exist in neuprint and still exist in Clio will be updated.
Others are ignored.

Requires admin access to the neuprint server.
Set NEUPRINT_APPLICATION_CREDENTIALS in your environment before running this script.

Example:

    update-neuprint-annotations emdata6.int.janelia.org:9000 ':master' segmentation_annotations neuprint-cns.janelia.org cns

TODO: This doesn't yet modify Meta.lastDatabaseEdit when making updates.
"""
import re
import os
import logging
import argparse
from collections import namedtuple

logger = logging.getLogger(__name__)
DvidDetails = namedtuple('DvidInstance', 'server uuid instance')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--output-directory', '-o',
        help="Optional. Export summary files to an output directory. "
             "If the directory already contains summary files from a prior run, they'll be overwritten.")
    parser.add_argument(
        '--dry-run', action='store_true',
        help="If given, generate the output files (including cypher commands) for the update, but don't "
             "actually execute the transaction to update neuprint."
    )
    parser.add_argument('dvid_server')
    parser.add_argument('dvid_uuid')
    parser.add_argument('dvid_instance')
    parser.add_argument('neuprint_server')
    parser.add_argument('neuprint_dataset')
    args = parser.parse_args()

    from neuclease import configure_default_logging
    configure_default_logging()

    dvid_details = DvidDetails(
        args.dvid_server,
        args.dvid_uuid,
        args.dvid_instance
    )

    from neuprint import Client
    neuprint_client = Client(
        args.neuprint_server,
        args.neuprint_dataset
    )

    clio_df, neuprint_df, changemask, commands = update_neuprint_annotations(dvid_details, args.dry_run, neuprint_client)

    if len(changemask) > 0 and (d := args.output_directory):
        logger.info(f"Writing summary files to {d}")
        os.makedirs(d, exist_ok=True)
        clio_df.to_csv(f'{d}/from-dvid.csv', header=True, index=True)
        neuprint_df.to_csv(f'{d}/neuprint-original.csv', header=True, index=True)
        changemask.to_csv(f'{d}/changemask.csv', header=True, index=True)
        with open(f'{d}/cypher-update-commands.cypher', 'w') as f:
            f.write('\n'.join(commands))

    logger.info("DONE")


def update_neuprint_annotations(dvid_details, dry_run=False, client=None):
    """
    Compare neuronjson annotations from DVID/Clio with the Neuron/Segment
    properties from Neuprint, and update the Neuprint properties to match
    DVID/Clio if differences are found.

    Only bodies which already exist in neuprint and still exist in Clio will be updated.
    Others are ignored.

    TODO: Update Meta.lastDatabaseEdit when making updates.

    Args:
        dvid_details:
            tuple (server, uuid, instance)
        client:
            neuprint Client
            Requires admin access to the neuprint server.

    Returns:
        clio_df, neuprint_df, changemask, cypher_commands
        Summaries of the commands which were sent to neuprint and
        the relevant DataFrames that led to those commands.
    """
    # Late imports to speed up --help message.
    from neuclease.util import Timer
    from neuprint.client import default_client

    if client is None:
        client = default_client()

    clio_df, neuprint_df, clio_segments = _fetch_comparison_dataframes(dvid_details, client)
    changemask = _compute_changemask(clio_df, neuprint_df)
    commands = _generate_commands(clio_df, changemask, clio_segments)

    if dry_run:
        msg = f"Dry run: Skipping update of {len(changemask)} Segments with {changemask.sum().sum()} out-of-date properties."
        logger.info(msg)
    elif len(changemask) == 0:
        logger.info("No out-of-date properties found.")
    else:
        msg = f"Updating {len(changemask)} Segments with {changemask.sum().sum()} out-of-date properties."
        with Timer(msg, logger):
            _post_commands(commands, client)

    # Restrict summary DataFrames to the minimal subset
    # of bodies/columns containing changes.
    clio_df = clio_df.loc[changemask.index, changemask.columns]
    neuprint_df = neuprint_df.loc[changemask.index, changemask.columns]
    return clio_df, neuprint_df, changemask, commands


def _fetch_comparison_dataframes(dvid_details, client):
    """
    Fetch all annotations from DVID/Clio and all of the corresponding Neuron (or Segment)
    properties from Neuprint.  Ensure that both DataFrames are indexed by 'body',
    and align them to each other. Discard columns that don't exist in DVID/Clio.
    Discard bodies (rows) that are present in only one of the two DataFrames.

    Returns:
        clio_df, neuprint_df, clio_segments
        Where clio_df and neuprint_df have the same columns and index,
        and clio_segments is the list of body IDs from DVID/Clio which are not labeled
        in Neuprint as :Neuron, so they must be matched via the :Segment label in Cypher queries.

    Notes:
        - Neurotransmitter columns are discarded.
    """
    import numpy as np
    import pandas as pd
    from neuclease.dvid import fetch_all
    from neuclease.util import Timer
    from neuprint import fetch_neurons, NeuronCriteria as NC
    from flyem_snapshot.outputs.neuprint.annotations import neuprint_segment_annotations

    with Timer("Fetching neuronjson body annotations from DVID", logger):
        clio_df = fetch_all(*dvid_details).drop(columns=['json'])

        # Convert to neuprint column names and values.
        # BTW, we don't yet support annotation property mappings with
        # config-defined column name mappings.
        # (That would require reading the snapshot config.)
        cfg = {'annotation-property-names': {}}
        clio_df = neuprint_segment_annotations(cfg, clio_df)

    # Fetch all Segment annotations that are also in Clio.
    # For a compact query, we first fetch all Neurons,
    # and then follow up with a query to fetch whatever Clio bodies that didn't
    # catch, which must be mere Segments in neuprint.
    with Timer("Fetching Neuprint Neuron properties", logger):
        neuprint_df, _syndist = fetch_neurons(NC(client=client), client=client)
        clio_segments = set(clio_df.index) - set(neuprint_df['bodyId'])
        segment_df, _ = fetch_neurons(sorted(clio_segments), client=client)
        neuprint_df = pd.concat((neuprint_df, segment_df))

    bodies = neuprint_df['bodyId'].rename('body')
    neuprint_df = neuprint_df.set_index(bodies).sort_index()

    # In some cases (such as spatial points), what neuprint
    # returns isn't what we need to WRITE.
    # So process the values just like we did with clio_df.
    neuprint_df = neuprint_segment_annotations(cfg, neuprint_df)

    # We handle all columns from clio, but only bodies which are in both.
    missing_cols = set(clio_df.columns) - set(neuprint_df.columns)
    neuprint_df = neuprint_df.assign(**{col: None for col in missing_cols})

    # Now align the two dataframes so they can be compared.
    neuprint_df, clio_df = neuprint_df.align(clio_df, 'inner', axis=None)

    # Special handling for neurotransmitters:
    # Neurotransmitters are supposed to be computed anew for each neuprint snapshot,
    # not loaded into clio.  But for historical reasons, we did upload NT predictions into the MANC clio.
    # We do NOT want to use those values to overwite neuprint.
    nt_cols = [col for col in clio_df.columns if col.startswith('nt') or col.startswith('predictedNt')]
    clio_df = clio_df.drop(columns=nt_cols)
    neuprint_df = neuprint_df.drop(columns=nt_cols)

    # Standardize on None as the null value (instead of NaN or "").
    # https://stackoverflow.com/questions/46283312/how-to-proceed-with-none-value-in-pandas-fillna
    clio_df = clio_df.fillna(np.nan).replace([np.nan, ""], [None, None])
    neuprint_df = neuprint_df.fillna(np.nan).replace([np.nan, ""], [None, None])
    return clio_df, neuprint_df, clio_segments


def _compute_changemask(clio_df, neuprint_df):
    import numpy as np

    # Find the positions with different values.
    changemask = (neuprint_df != clio_df) & (~neuprint_df.isnull() | ~clio_df.isnull())

    # This will ensure that all-float columns have float dtype.
    clio_df = clio_df.fillna(np.nan)
    neuprint_df = neuprint_df.fillna(np.nan)

    # Special handling for float columns: We don't demand exact equality.
    float_cols = (clio_df.dtypes == float) & (neuprint_df.dtypes == float)
    changemask.loc[:, float_cols] &= ~np.isclose(
        clio_df.loc[:, float_cols],
        neuprint_df.loc[:, float_cols]
    )

    # Filter for rows and columns which contain at least one change.
    changemask = changemask.loc[changemask.any(axis=1), changemask.any(axis=0)]
    return changemask


def _generate_commands(clio_df, changemask, clio_segments):
    commands = []
    for body, ischanged in changemask.T.items():
        props = ischanged[ischanged].index
        propvals = clio_df.loc[body, props].items()
        updates = ', '.join(f'n.`{prop}` = {_cypher_literal(val)}' for prop, val in propvals)

        # It turns out it is *much* faster to perform this SET
        # using :Neuron instead of :Segment if possible.
        if body in clio_segments:
            label = 'Segment'
        else:
            label = 'Neuron'
        commands.append(f"MATCH (n:{label} {{bodyId: {body}}}) SET {updates}")

    return commands


def _post_commands(commands, client):
    from neuclease.util import tqdm_proxy
    from neuprint.admin import Transaction

    # Is it best to send all of our Cypher commands within a
    # single big transaction like this, or many small transactions?
    with Transaction(client.dataset, client=client) as t:
        for q in tqdm_proxy(commands):
            t.query(q)


POINT_PATTERN = re.compile(r'{x:\d+, y:\d+, z:\d+}')


def _cypher_literal(x):
    """
    Represent a Python value as a Cypher literal.
    This implementation is not at all comprehensive, but it
    works for None, strings, and neo4j spatial 3d points,
    which is good enough for now.
    """
    if x is None:
        return 'NULL'
    if isinstance(x, str) and POINT_PATTERN.match(x):
        return f'point({x})'
    return repr(x)


if __name__ == "__main__":
    main()
