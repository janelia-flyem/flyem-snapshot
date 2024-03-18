"""
Generate connectome exports, neuprint databases and reports from flat files and DVID checkpoints.
"""
import sys
import argparse


def main():
    """
    This is a wrapper around flyem_snapshot.main.main().
    Command-line arguments are defined and parsed here,
    (before calling the real main) to enable quick response to --help.
    """
    DEBUG = True
    if DEBUG and len(sys.argv) == 1:
        # Providing fake CLI arguments this way is simpler than
        # than messing around with VSCode debugging configurations.
        import os
        sys.stderr.write("DEBUGGING WITH ARTIFICAL ARGS\n")
        os.chdir('/Users/bergs/workspace/snapshot-configs/test-configs')
        # p = 'small-ol-test/ol-small-test.yaml'
        # sys.argv.extend(['-c', p])
        sys.argv.extend(['-Y'])

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-c', '--config',
                        help='The input configuration yaml file')
    parser.add_argument('-t', '--print-snapshot-tag', action='store_true',
                        help='Print the snapshot tag and exit immediately.')
    parser.add_argument('-d', '--print-output-directory', action='store_true',
                        help='Print the full path to the directory which will be created or written to for results.')
    parser.add_argument('-y', '--dump-default-yaml', action='store_true',
                        help='Print out the default config file')
    parser.add_argument('-Y', '--dump-verbose-yaml', action='store_true',
                        help='Print out the default config file, with verbose comments above each setting.')
    parser.add_argument('-m', '--dump-neuprint-default-meta', action='store_true',
                        help='Print out the default neuprint "meta" parameters, which are specified via an auxiliary'
                             'yaml file and linked within the main config file.')
    parser.add_argument('-M', '--dump-verbose-neuprint-default-meta', action='store_true',
                        help='Print out the default neuprint "meta" parameters, with verbose comments above each setting.')
    args = parser.parse_args()

    config_args = (
        args.config,
        args.dump_default_yaml,
        args.dump_verbose_yaml,
    )
    if sum(bool(a) for a in config_args) > 1:
        sys.exit("You can't provide more than one of these options: -c, -y, -Y")

    printing_args = (
        args.dump_default_yaml,
        args.dump_verbose_yaml,
        args.print_snapshot_tag,
        args.print_output_directory,
        args.dump_neuprint_default_meta,
        args.dump_verbose_neuprint_default_meta,
    )
    if sum(bool(a) for a in printing_args) > 1:
        sys.exit("You can't provide more than one of these options: -t, -d, -y, -Y, -m, -M")

    import flyem_snapshot.main
    return flyem_snapshot.main.main(args)


if __name__ == "__main__":
    sys.exit(main())
