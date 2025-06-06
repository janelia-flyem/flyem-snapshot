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
"""
import re
import os
import sys
import logging
import traceback
import argparse
from collections import namedtuple

from requests import HTTPError

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
             "If the directory already contains summary files from a prior run, they'll be overwritten."
    )
    parser.add_argument(
        '--default-transaction-size', '-b',
        type=int,
        default=200,
        help="Neuron properties will be updated in batches. "
             "This is the batch size of each transaction, unless some transactions fail due to timeout errors, "
             "in which case they will be retried in smaller batches."
    )
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

    try:
        update_neuprint_annotations(dvid_details, args.dry_run, args.output_directory, args.default_transaction_size, neuprint_client)
    except BaseException:
        if d := args.output_directory:
            os.makedirs(d, exist_ok=True)
            with open(f'{d}/error.txt', 'w') as f:
                traceback.print_exception(*sys.exc_info(), file=f)
        raise

    logger.info("DONE")


def update_neuprint_annotations(dvid_details, dry_run=False, log_dir=None, default_transaction_size=100, client=None):
    """
    Compare neuronjson annotations from DVID/Clio with the Neuron/Segment
    properties from Neuprint, and update the Neuprint properties to match
    DVID/Clio if differences are found.

    Only bodies which already exist in neuprint and still exist in Clio will be updated.
    Others are ignored.

    Args:
        dvid_details:
            tuple (server, uuid, instance)
        dry_run:
            If True, generate the output files (including cypher commands) for the update,
            but don't actually execute the transactions to update neuprint.
        log_dir:
            Optional.  If given, write summary files to this directory.
        default_transaction_size:
            The default batch size for each transaction.
            If any transaction fails due to a timeout error,
            then it and all subsequent transactions will be retried
            using smaller transactions.
        client:
            neuprint Client
            Requires admin access to the neuprint server.

    Returns:
        clio_df, neuprint_df, changemask, cypher_commands
        Summaries of the commands which were sent to neuprint and
        the relevant DataFrames that led to those commands.
    """
    # Late imports to speed up --help message.
    from neuprint.client import default_client

    if client is None:
        client = default_client()

    clio_df, neuprint_df, clio_segments, timestamp = _fetch_comparison_dataframes(dvid_details, client)
    changemask = _compute_changemask(clio_df, neuprint_df)
    commands = _generate_commands(clio_df, changemask, clio_segments)
    _dump_summary_files(clio_df, neuprint_df, changemask, commands, log_dir)

    clio_df, neuprint_df, changemask, commands = _apply_neuprint_annotation_updates(
        clio_df, neuprint_df, changemask, commands, timestamp, dry_run, default_transaction_size, client)

    return clio_df, neuprint_df, changemask, commands


def _dump_summary_files(clio_df, neuprint_df, changemask, commands, log_dir):
    if len(changemask) == 0 or not log_dir:
        return
    d = log_dir
    logger.info(f"Writing summary files to {d}")
    os.makedirs(d, exist_ok=True)
    clio_df.to_csv(f'{d}/from-dvid.csv', header=True, index=True)
    neuprint_df.to_csv(f'{d}/neuprint-original.csv', header=True, index=True)
    changemask.to_csv(f'{d}/changemask.csv', header=True, index=True)
    with open(f'{d}/cypher-update-commands.cypher', 'w') as f:
        f.write('\n'.join(commands))
        f.write('\n')


def _apply_neuprint_annotation_updates(clio_df, neuprint_df, changemask, commands, timestamp, dry_run, default_transaction_size, client):
    from neuclease.util import Timer
    _update_meta_neuron_properties(clio_df, dry_run, client)

    if dry_run:
        msg = f"Dry run: Skipping update of {len(changemask)} Segments with {changemask.sum().sum()} out-of-date properties."
        logger.info(msg)
    elif len(changemask) == 0:
        logger.info("No out-of-date properties found.")
    else:
        msg = f"Updating {len(changemask)} Segments with {changemask.sum().sum()} out-of-date properties."
        with Timer(msg, logger):
            _post_commands(commands, default_transaction_size, client)
        _update_meta_timestamp(timestamp, client)

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
    from neuclease.util import Timer, snakecase_to_camelcase
    from neuprint import fetch_neurons, NeuronCriteria as NC
    from flyem_snapshot.outputs.neuprint.annotations import neuprint_segment_annotations

    with Timer("Fetching neuronjson body annotations from DVID", logger):
        clio_df = fetch_all(*dvid_details, show='time').drop(columns=['json'], errors='ignore')

        # Note: 'birthtime' is a legit column!
        time_cols = [c for c in clio_df.columns if c.endswith('_time')]
        timestamp = clio_df[time_cols].stack().max()
        clio_df = clio_df.drop(columns=time_cols)

        # Convert to neuprint column names and values.
        # BTW, we don't yet support annotation property mappings with
        # config-defined column name mappings.
        # (That would require reading the snapshot config.)
        cfg = {'annotation-property-names': {}}
        clio_df = neuprint_segment_annotations(cfg, clio_df)

    # Fetch all Segment annotations that are also in Clio.
    # For a compact query, we first fetch all Neurons,
    # and then follow up with a query to fetch whatever Clio bodies
    # that query didn't catch, which must be mere Segments in neuprint.
    with Timer("Fetching Neuprint Neuron properties", logger):
        neuprint_df = fetch_neurons(NC(client=client), omit_rois=True, client=client)
        clio_segments = set(clio_df.index) - set(neuprint_df['bodyId'])
        segment_df = fetch_neurons(sorted(clio_segments), omit_rois=True, client=client)

        # Drop empty columns before concat to avoid pandas warning.
        neuprint_df = neuprint_df.dropna(axis='columns', how='all')
        segment_df = segment_df.dropna(axis='columns', how='all')
        neuprint_df = pd.concat((neuprint_df, segment_df))

    bodies = neuprint_df['bodyId'].rename('body')
    neuprint_df = neuprint_df.set_index(bodies).sort_index()

    # In some cases (such as spatial points), what neuprint
    # RETURNS isn't in the format that we need to WRITE.
    # So process the values just like we did with clio_df.
    # (First, name 'statusLabel' to 'status' to hit the same code path that the clio annotations use.)
    neuprint_df = neuprint_df.drop(columns=['status']).rename(columns={'statusLabel': 'status'})
    neuprint_df = neuprint_segment_annotations(cfg, neuprint_df)

    # We handle all *columns* from clio, but only *bodies* which are in both clio and neuprint.
    missing_cols = set(clio_df.columns) - set(neuprint_df.columns)
    neuprint_df = neuprint_df.assign(**{col: None for col in missing_cols})

    # Now align the two dataframes so they can be compared.
    neuprint_df, clio_df = neuprint_df.align(clio_df, 'inner', axis=None)

    # Special handling for neurotransmitters:
    # Neurotransmitters are supposed to be computed anew for each neuprint snapshot,
    # not loaded into clio.  But for historical reasons, we did upload NT predictions into the MANC clio.
    # We do NOT want to use those clio NT values to overwite neuprint NT values.
    nt_cols = [
        'total_nt_predictions', 'predicted_nt', 'predicted_nt_confidence',
        'celltype_total_nt_predictions', 'celltype_predicted_nt', 'celltype_predicted_nt_confidence',
        'consensus_nt'
    ]
    nt_cols = [*map(snakecase_to_camelcase, nt_cols)]
    nt_cols = [col for col in clio_df.columns if col.startswith('nt') or col in nt_cols]
    clio_df = clio_df.drop(columns=nt_cols)
    neuprint_df = neuprint_df.drop(columns=nt_cols)

    # Standardize on None as the null value (instead of NaN or "").
    # https://stackoverflow.com/questions/46283312/how-to-proceed-with-none-value-in-pandas-fillna
    clio_df = clio_df.replace(["", np.nan, pd.NaT, pd.NA], [None, None, None, None])
    neuprint_df = neuprint_df.replace(["", np.nan, pd.NaT, pd.NA], [None, None, None, None])
    return clio_df, neuprint_df, clio_segments, timestamp


def _compute_changemask(clio_df, neuprint_df):
    """
    Compare the two DataFrames (which must already have
    identical indexes and columns), and return a boolean DataFrame
    indicating which positions in the original dataframes differ.

    The result will be reduced to the minimal set of rows and columns
    to represent the differing positions.  Only rows and columns with
    at least one True entry will be preserved in the output; other
    rows/columns will be dropped.

    Special cases:
        - Floating-point values will be compared with np.isclose()
          instead of pure equality.
        - Two null values are treated as equal to each other
          (unlike default NaN behavior).
    """
    import numpy as np

    # Find the positions with different values.
    # If both are NaN/None, they're considered equal.
    changemask = (neuprint_df != clio_df) & (neuprint_df.notnull() | clio_df.notnull())

    # This will ensure that all-float columns have float dtype.
    clio_df = clio_df.infer_objects(copy=False)
    neuprint_df = neuprint_df.infer_objects(copy=False)

    # Special handling for float columns: We don't demand exact equality.
    float_cols = (clio_df.dtypes == float) & (neuprint_df.dtypes == float)  # noqa: E721
    changemask.loc[:, float_cols] &= ~np.isclose(
        clio_df.loc[:, float_cols],
        neuprint_df.loc[:, float_cols]
    )

    # Filter for rows and columns which contain at least one change.
    changemask = changemask.loc[changemask.any(axis=1), changemask.any(axis=0)]
    return changemask


def _update_meta_neuron_properties(clio_df, dry_run, client):
    """
    In the neuprint :Meta node, we maintain a dict of
    all :Neuron properties and their types.

    If the annotation sync will include NEW properties from DVID,
    we need to ensure that those new properties are added to
    the :Meta info.  Client libraries/functions such as
    neuprint.fetch_neurons() depend on it.
    """
    import json
    from neuprint.admin import Transaction
    from flyem_snapshot.outputs.neuprint.util import neo4j_type_suffix

    q = "MATCH (m:Meta) RETURN m.neuronProperties"
    property_types = client.fetch_custom(q).iloc[0, 0]
    if property_types is None:
        logger.warning("Could not find Meta.neuronProperties in the database")
        return

    property_types = json.loads(property_types)
    new_prop_cols = {*clio_df.columns} - {*property_types.keys()}
    if not new_prop_cols:
        return

    if dry_run:
        logger.info(f"Dry run: Not updating Meta.neuronProperties to add {new_prop_cols}")
        return

    logger.info(f"Updating Meta.neuronProperties to add {new_prop_cols}")
    new_prop_types = {p: neo4j_type_suffix(clio_df[p]) for p in new_prop_cols}
    property_types.update(new_prop_types)
    with Transaction(client.dataset, client=client) as t:
        q = f"""\
            MATCH (m:Meta)
            SET m.neuronProperties = '{json.dumps(property_types)}'
        """
        t.query(q)

    client.fetch_neuron_keys.cache_clear()


def _update_meta_timestamp(timestamp, client):
    """
    The :Meta node stores the lastDatabaseEdit, a timestamp which
    is displayed in neuprintExplorer.

    To distinguish between the timestamp of the original snapshot
    and the timestamp of the most recently updated annotation property,
    we don't REPLACE the last database edit.
    Instead, we append a second timestamp to the string.
    """
    from neuprint.admin import Transaction

    q = "MATCH (m: Meta) RETURN m.lastDatabaseEdit"
    last_edit = client.fetch_custom(q).iloc[0, 0]
    last_edit = last_edit or ''
    last_edit = last_edit.split(' / ')[0]
    last_edit += f' / {timestamp} (segment property update)'

    cmd = f"MATCH (m: Meta) SET m.lastDatabaseEdit = '{last_edit}'"
    logger.info(cmd)
    with Transaction(client.dataset, client=client) as t:
        t.query(cmd)


def _generate_commands(clio_df, changemask, clio_segments):
    """
    Generate a list of Cypher commands to update the properties
    highlighted in the given changemask.
    This function just generates the Cypher commands;
    it doesn't send them to the server.

    Args:
        clio_df:
            DataFrame holding the new property values
        changemask:
            Boolean DataFrame indicating which positions in
            clio_df hold the properties to send.
        clio_segments:
            A list of body IDs which must be matched via :Segment instead of :Neuron.
            We prefer to use "MATCH (:Neuron)" by default, since that's much faster
            than "MATCH (:Segment)". But if the node to alter is labeled only with
            :Segment (not also :Neuron), we have no choice but to use "MATCH (:Segment)".
    Returns:
        list of str
    """
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


def _post_commands(commands, batch_size, client):
    """
    Send the list of cypher commands to the neuprint
    server using a series of Transactions.

    At first, each transaction will include a batch of commands (of the given size),
    but if one transaction fails, the batch size is halved and the commands are re-sent
    with the new batch size.  This continues until the batch size is 1.
    If any transaction still fails using batch size 1, we give up and raise an exception.
    """
    from itertools import chain
    from neuclease.util import tqdm_proxy, tqdm_proxy_config, iter_batches
    from neuprint.admin import Transaction

    # I want logged progress bars, no matter what.
    tqdm_proxy_config['output_file'] = 'logger'

    with tqdm_proxy(total=len(commands)) as progress:
        while True:
            try:
                batch_size = min(batch_size, len(commands))
                batches = iter_batches(commands, batch_size)
                batch_iter = iter(batches)

                for batch in batch_iter:
                    with Transaction(client.dataset, client=client) as t:
                        for command in batch:
                            t.query(command)
                    progress.update(len(batch))

                # Done: All (remaining) batches succeeded without error.
                break
            except HTTPError:
                if batch_size == 1:
                    logger.error("Transaction failed even with batch size 1.  Giving up.")
                    logger.error("Failed cypher command was:")
                    logger.error(batch[0])
                    raise
                logger.warning(f"Transaction failed with batch size: {batch_size}")
                logger.warning("Moving the failed commands to the end of the queue.")
                logger.warning(f"Resuming with smaller batch size: {batch_size // 2}")
                commands = [*chain(*batch_iter), *batch]
                batch_size //= 2


POINT_PATTERN = re.compile(r'{x:-?\d+, y:-?\d+, z:-?\d+}')


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
    # sys.argv.extend(['-o', '/tmp/debug'])
    # sys.argv.extend(['--dry-run'])
    # sys.argv.extend(['--default-transaction-size', '200'])
    # sys.argv.extend("emdata6.int.janelia.org:9000 :master segmentation_annotations neuprint-cns.janelia.org cns".split())

    # sys.argv.extend(['-o', '/tmp/debug'])
    # sys.argv.extend(['--dry-run'])
    # sys.argv.extend(['--default-transaction-size', '200'])
    # sys.argv.extend("emdata5.int.janelia.org:8400 :master segmentation_annotations neuprint-pre.janelia.org vnc".split())

    # sys.argv.extend(['-o', '/tmp/debug'])
    # sys.argv.extend(['--dry-run'])
    # sys.argv.extend(['--default-transaction-size', '200'])
    # sys.argv.extend("emdata7.int.janelia.org:8700 :new-agglo segmentation_annotations neuprint-pre.janelia.org yakuba-vnc".split())

    main()
