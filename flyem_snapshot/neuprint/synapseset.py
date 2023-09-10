@PrefixFilter.with_context("SynapseSet")
@timed
def _export_synapsesets(cfg, partner_df, connectome):
    body_pairs_df = connectome.reset_index()[['body_pre', 'body_post']]
    batches = iter_batches(body_pairs_df, 10_000)
    body_pairs_dfs = compute_parallel(_synset_ids, batches, processes=cfg['processes'], leave_progress=True)
    body_pairs_df = pd.concat(body_pairs_dfs, ignore_index=True)
    assert body_pairs_df.columns.tolist() == ['body_pre', 'body_post', 'synset_pre', 'synset_post']

    with Timer("Writing Neuprint_SynapseSet_to_SynapseSet.csv", logger):
        (
            body_pairs_df[['synset_pre', 'synset_post']]
            .rename(columns={
                'synset_pre': ':START_ID(SynSet-ID)',
                'synset_post': ':END_ID(SynSet-ID)'
            })
            .to_csv('neuprint/Neuprint_SynapseSet_to_SynapseSet.csv', index=False, header=True)
        )

    synset_pre = (
        body_pairs_df[['body_pre', 'synset_pre']]
        .rename(columns={
            'body_pre': 'body',
            'synset_pre': 'synset_id'
        })
    )
    synset_post = (
        body_pairs_df[['body_post', 'synset_post']]
        .rename(columns={
            'body_post': 'body',
            'synset_post': 'synset_id'
        })
    )
    synset_ids = pd.concat((synset_pre, synset_post), ignore_index=True)
    assert synset_ids.columns.tolist() == ['body', 'synset_id']

    with Timer("Writing Neuprint_SynapseSet.csv", logger):
        dataset = cfg['neuprint']['dataset']
        label = f"SynapseSet;{dataset}_SynapseSet"
        (
            synset_ids.assign(label=label)
            [['synset_id', 'label']]
            .rename(columns={
                'synset_id': ':ID(SynSet-ID)',
                'label': ':Label'
            })
            .to_csv('neuprint/Neuprint_SynapseSet.csv', index=False, header=True)
        )

    with Timer("Writing Neuprint_Neuron_to_SynapseSet.csv", logger):
        (
            synset_ids
            .rename(columns={
                'body': ':START_ID(Body-ID)',
                'synset_id': ':END_ID(SynSet-ID)'
            })
            .to_csv('neuprint/Neuprint_Neuron_to_SynapseSet.csv', index=False, header=True)
        )

    with Timer("Constructing Neuprint_SynapseSet_to_Synapses table", logger):
        partner_df = partner_df.merge(body_pairs_df, 'left', on=['body_pre', 'body_post'])

        ss_to_syn_pre = (
            partner_df
            .drop_duplicates(['pre_id', 'body_post'])
            [['synset_pre', 'pre_id']]
            .rename(columns={
                'synset_pre': 'synset',
                'pre_id': 'point_id'
            })
        )
        ss_to_syn_post = (
            partner_df
            [['synset_post', 'post_id']]
            .rename(columns={
                'synset_post': 'synset',
                'post_id': 'point_id'
            })
        )

    with Timer("Writing Neuprint_SynapseSet_to_Synapses.csv", logger):
        (
            pd.concat((ss_to_syn_pre, ss_to_syn_post), ignore_index=True)
            .rename(columns={
                'synset': ':START_ID(SynSet-ID)',
                'point_id': ':END_ID(Syn-ID)'
            })
            .to_csv('neuprint/Neuprint_SynapseSet_to_Synapses.csv', index=False, header=True)
        )


def _synset_ids(body_pairs_df):
    """
    Helper for _export_synapsesets.
    """
    # These SynapseSets will be 'contained' by the neuron on the pre side
    body_pairs_df['synset_pre'] = (
        body_pairs_df['body_pre'].astype(str) + '_' +  # noqa
        body_pairs_df['body_post'].astype(str) + '_pre'
    )
    # These SynapseSets will be 'contained' by the neuron on the post side
    body_pairs_df['synset_post'] = (
        body_pairs_df['body_pre'].astype(str) + '_' +  # noqa
        body_pairs_df['body_post'].astype(str) + '_post'
    )
    return body_pairs_df
