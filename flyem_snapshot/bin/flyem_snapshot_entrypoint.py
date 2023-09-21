"""
Generate connectome exports, neuprint databases and reports from flat files and DVID checkpoints.
"""
import sys
import argparse


def main():
    DEBUG = False
    if DEBUG and len(sys.argv) == 1:
        # This is simpler than messing around with VSCode debugging configurations.
        import os
        sys.stderr.write("DEBUGGING WITH ARTIFICAL ARGS\n")
        p = os.path.split(__file__)[0] + '/../../neuprint-small-test.yaml'
        sys.argv.extend(['-c', p])

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-c', '--config')
    parser.add_argument('-y', '--dump-default-yaml', action='store_true')
    parser.add_argument('-Y', '--dump-verbose-yaml', action='store_true')
    parser.add_argument('-m', '--dump-neuprint-default-meta', action='store_true')
    parser.add_argument('-M', '--dump-verbose-neuprint-default-meta', action='store_true')
    args = parser.parse_args()

    if sum((bool(args.config), bool(args.dump_default_yaml), bool(args.dump_verbose_yaml))) > 1:
        sys.exit("You can't provide more than one of these options: -c, -y, -v")

    import flyem_snapshot.main
    flyem_snapshot.main.main(args)


if __name__ == "__main__":
    main()
