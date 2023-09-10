





def main():
    DEBUG = True
    if DEBUG and len(sys.argv) == 1:
        sys.stderr.write("DEBUGGING WITH ARTIFICAL ARGS\n")
        sys.argv.extend(['-c', 'neuprint-small-test.yaml'])

    parser = ArgumentParser()
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

    # All subsequent processing occurs from within the output-dir
    with switch_cwd(cfg['output-dir']):
        point_df, partner_df = _load_synapses(cfg)
        ann = _fetch_and_export_body_annotations(cfg)
        _export_flat_connectome(cfg, point_df, partner_df, ann)
        _export_neuprint(cfg, point_df, partner_df, ann)
        _export_reports(cfg, point_df, partner_df, ann)


def _finalize_config_and_output_dir(cfg, config_dir):
    uuid, snapshot_tag = resolve_snapshot_tag(
        cfg['snapshot']['server'],
        cfg['snapshot']['uuid'],
        cfg['snapshot']['instance']
    )
    cfg['snapshot']['uuid'] = uuid
    snapshot_tag = cfg['snapshot-tag'] = (cfg['snapshot-tag'] or snapshot_tag)

    # Convert synapse and size paths to absolute (if necessary).
    # Relative paths are interpreted w.r.t. to the config file, not the cwd.
    with switch_cwd(config_dir):
        cfg['syndir'] = os.path.abspath(cfg['syndir'])
        if not cfg['synapse-points'].startswith('{syndir}'):
            cfg['synapse-points'] = os.path.abspath(cfg['synapse-points'])
        if not cfg['synapse-partners'].startswith('{syndir}'):
            cfg['synapse-partners'] = os.path.abspath(cfg['synapse-partners'])
        if cfg['body-size-cache']['file']:
            cfg['body-size-cache']['file'] = os.path.abspath(cfg['body-size-cache']['file'])

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

    # Convenience item for other functions in this file.
    cfg['dvid-seg'] = (
        cfg['snapshot']['server'],
        cfg['snapshot']['uuid'],
        cfg['snapshot']['instance']
    )


if __name__ == "__main__":
    main()
