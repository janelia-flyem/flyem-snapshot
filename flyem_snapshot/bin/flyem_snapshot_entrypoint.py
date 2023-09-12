"""
Generate connectome exports, neuprint databases and reports from flat files and DVID checkpoints.
"""
import sys
import argparse


def main():
    DEBUG = False
    if DEBUG and len(sys.argv) == 1:
        sys.stderr.write("DEBUGGING WITH ARTIFICAL ARGS\n")
        sys.argv.extend(['-c', 'neuprint-small-test.yaml'])

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-c', '--config')
    parser.add_argument('-y', '--dump-default-yaml', action='store_true')
    parser.add_argument('-v', '--dump-verbose-yaml', action='store_true')
    args = parser.parse_args()

    if sum((bool(args.config), bool(args.dump_default_yaml), bool(args.dump_verbose_yaml))) > 1:
        sys.exit("You can't provide more than one of these options: -c, -y, -v")

    import flyem_snapshot.flyem_snapshot
    flyem_snapshot.flyem_snapshot.main(args)


if __name__ == "__main__":
    main()
