def _load_neuron_sizes(cfg, neuron_df):
    assert neuron_df.index.name == 'body'
    cache_file = cfg['body-size-cache']['file']
    cache_uuid = cfg['body-size-cache']['uuid']
    snapshot_uuid = cfg['snapshot']['uuid']
    if bool(cache_file) != bool(cache_uuid):
        logger.error("body-size-cache is not specified properly")
        cache_file = cache_uuid = ""

    if not cache_file:
        # No cache: Gotta fetch them all from DVID
        # (Takes ~1 hour for the full CNS -- would be worse if we had to also fetch sizes of NON-synaptic bodies.)
        logger.info("No body-size-cache provided.")
        with Timer("Loading all neuron sizes from DVID", logger):
            neuron_df['size'] = fetch_sizes(*cfg['dvid-seg'], neuron_df.index, processes=cfg['processes']).values
        snapshot_tag = cfg['snapshot-tag']
        feather.write_feather(
            neuron_df.reset_index(),
            f'neuprint/body-size-cache-{snapshot_tag}.feather')
        return

    # Note: Using 1 parenthesis and 1 bracket to indicate
    #       exclusive/inclusive mutation range: (a,b]
    delta_range = f"({cache_uuid}, {snapshot_uuid}]"
    muts = fetch_mutations(
        cfg['snapshot']['server'],
        delta_range,
        cfg['snapshot']['instance']
    )
    effects = compute_affected_bodies(muts)
    outofdate_bodies = np.concatenate((effects.changed_bodies, effects.new_bodies))
    old_sizes = feather.read_feather(cache_file)
    if len(outofdate_bodies) == 0:
        sizes = old_sizes
    else:
        with Timer("Fetching non-cached neuron sizes", logger):
            new_sizes = fetch_sizes(*cfg['dvid-seg'], outofdate_bodies, outofdate_bodies, processes=cfg['processes'])
        sizes = new_sizes.combine_first(old_sizes).astype(np.int64)

    neuron_df = neuron_df.merge(sizes, 'left', on='body')


