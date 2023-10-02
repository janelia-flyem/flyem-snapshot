# flyem-snapshot

Generate connectome exports, neuprint databases and reports from flat files and DVID checkpoints.

Given a connectome in the following form:

- a table of synapse points
   - with a column for `body` (neuron)
   - one or more ROI columns (if not being sourced from DVID)
- synapse relationships (pre-post)

- a table neuron annotations (or a DVID neuronjson instance to load them from)
- Optionally, one or more ROI lists to load from DVID (instead of providing pre-loaded ROI columns in the input)

This tool can produce various outputs:

- a "flat" connectome table which is trivial to interpret but not efficient with RAM/disk space
- a neuprint neo4j database
- a series of connectivity reports illustrating various metrics in selected ROIs

The input is expected to be in Apache Feather format, but there is an option to update the input bodies and ROIs via

# Installation:

A conda recipe coming soon.  In the meantime:

```bash
conda create -n flyem neuclease
git clone ssh://git@github.com/janelia-flyem/flyem-snapshot
cd flyem-snapshot
python setup.py develop
```

# Documentation

Coming Soon.  In the mean time, see the following:

```bash
flyem-snapshot --help

# Dump a verbosely commented config file.
# Edit it to suit your needs.
flyem-snapshot -Y > snapshot-config.yaml

# If you're exporting a neuprint snapshot, you'll also need
# a separate config file with settings for the `:Meta` node.
flyem-snapshot -M > neuprint-meta-config.yaml

# From a Janelia cluster node, you can ingest
# a neuprint snapshot into a neo4j database.
ingest-neuprint-snapshot-using-apptainer --help
```
