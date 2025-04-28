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

The input is expected to be in Apache Feather format, but you can
optionally update the `body` and ROI columns from a DVID checkpoint.

# Installation:

A conda recipe coming soon.  In the meantime:

```bash
conda create -n flyem neuclease
git clone ssh://git@github.com/janelia-flyem/flyem-snapshot
cd flyem-snapshot
python setup.py develop
```

# Documentation

To get help on the apps in flyem-snapshot:
```
flyem-snapshot --help
ingest-neuprint-snapshot-using-apptainer --help
```

## Generating config file templates
1. Connect to *login1* or *login2*
2. Active the conda environment: `conda activate flyem-snapshot`

You can use the following command to dump a verbosely commented config file. Edit it to suit your needs:
`flyem-snapshot -Y > snapshot-config.yaml`

If you're exporting a neuprint snapshot, you'll also need  # a separate config file with settings for the `:Meta` node:
`flyem-snapshot -M > neuprint-meta-config.yaml`

## Taking a snapshot
1. Connect to *login1* or *login2*
2. Log in to a cluster node: `bsub -n 32 -W 8:00 -Is -Pflyem /bin/bash`
3. Active the conda environment: `conda activate flyem-snapshot`
4. Go to the snapshot directory: `cd workspace/snapshot-configs/manc/v1.2.1`
5. Start the job: `flyem-snapshot -c manc-v1.2.1-release.yaml`

If you didn't specify an output directory (job-settings:output-dir in the config file), one will be created using the snapshot tag and UUID. This output directory will be created in the current directory.
Inside the output directory, the following subdirectories will be created:
* cache: zero-byte sentinel files used to track run progress, as well as multiple directories containing pickle, feather, and JSON files
* flat-connectome: feather files for:
    * body-stats
    * connectome-weights (all and significant only)
    * syn-partners (all and significant only)
    * syn-points
* neuprint:
    * feather files for body elements, NeuPrint neuron connections, Neuprint neurons, and ROI elements
    * JSON files for the dataset and NeuPrint Meta debug data
    * CSV files that will be used to load a neo4j database
    * A cypher files to build indices
    * Neuprint_Neurons and Neuprint_Synapses directories containing CSV files
* nt
    * **body** and **tbar** feather files for neurotransmitters
* reports:
    * chart body-status-counts formatted as html and png:![Body status counts](https://github.com/janelia-flyem/flyem-snapshot/blob/master/documentation_images/body-status-counts-2024-02-01-3ddc3f.png?raw=true)
    * primary reports:
        * primary-all-status-stats CSV file and pickle files for primary-all_status_stats and primary-all_syncounts
        * primary: csv files, as well as html and png charts for traced and connectivity data. There will be eight charts in total, all variations on connectivity by status and ROI: ![Connectivity chart](https://github.com/janelia-flyem/flyem-snapshot/blob/master/documentation_images/primary-captured-connectivity-by-status-and-roi-sorted.png?raw=true)
        * reports: one subdirectory per region:
            * feather files for conn_df, downstream-capture, and syn_counts_df, as well as multiple HTML files and PNG images for charts: ![Cumulative connectivity](https://github.com/janelia-flyem/flyem-snapshot/blob/master/documentation_images/ANm-cumulative-connectivity-with-links-2024-02-01-3ddc3f.png?raw=true) ![Downstream capture](https://github.com/janelia-flyem/flyem-snapshot/blob/master/documentation_images/ANm-downstream-capture-2024-02-01-3ddc3f.png?raw=true)
* tables
    * **body-annotations** and **body-size-cache** feather files
    * body-status-counts CSV file
* volume: JSON files describing the volume:
    * primary-box-zyx.json, primary-ids.json, primary-vol.npy

## Ingesting a neuprint snapshot into a neo4j database
1. Connect to *login1* or *login2*
2. Log in to a cluster node: `bsub -n 32 -W 8:00 -Is -Pflyem /bin/bash`
3. Active the conda environment: `conda activate flyem-snapshot`
4. Go to the directory that contains the output from the snapshot: `cd workspace/snapshot-configs/manc/v1.2.1`
5. Ingest the snapshot into neo4j: `ingest-neuprint-snapshot-using-apptainer 2024-02-01-3ddc3f`
