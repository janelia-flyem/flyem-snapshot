# flyem-snapshot

This directory contains ancillary programs for flyem snapshots.

| Name | Description | Run frequency |
| -------------------------- | ------------------------------------------------------ | ---------------------- |
| aging_deletion_flyem.py | Delete unneeded files in /groups/flyem/home/flyem/bin | Every day during the 11 PM hour |
| cleanup_snapshots.py | Delete old snapshot subdirectories from /groups/flyem/data/snapshots | Every day during the 6 AM hour |
| health_check.py | Check to see if NeuPrint servers are up. If not, start them. | Every 15 minutes |

### aging_deletion_flyem.py

This program is run on Jenkins nightly in the
[flyem_aging_deletion](https://jenkins.int.janelia.org/view/FlyEM/job/flyem_aging_deletion/)
job.  It is installed in <code>/groups/scicompsoft/informatics/bin</code>.
Directories, file patterns, and aging times are specified in the
[aging_deletion_flyem](https://config.int.janelia.org/config/aging_deletion_flyem)
configuration. Files or directories in the specified location that match a
a provided pattern and are sufficiently aged will be deleted. Example:
```
aging: 7
dir: "/groups/flyem/home/flyem/bin/cns_update_annotations_neuprint"
match: "previous_cns_annotations.bak.\d+$"
```

In this example, files matching the *match* regex pattern in the *dir*
directory will be deleted if they are older then 7 days.

### cleanup_snapshots.py

This program is run on Jenkins nightly in the
[flyem_snapshot_deletion](https://jenkins.int.janelia.org/view/FlyEM/job/flyem_snapshot_deletion/)
job. It is installed in <code>/groups/flyem/data/snapshots</code>.
It searches for subdirectories under specified snapshot directories under
<code>/groups/flyem/data/snapshots</code>. The subdiectories are in the
following format:

DATESTAMP - UUID - unlocked

Example: 2025-09-04-8af84b-unlocked

If the subdirectory datestamp is of sufficient age (default is > 4 days), then
every subdirectory under that snapshot (unless excluded) is deleted. Currently,
only the "reports" subdirectory is retained.

### health_check.py

This program is run on Jenkins every 15 minutes in the
[flyem_neuprint_health_check](https://jenkins.int.janelia.org/view/FlyEM/job/flyem_neuprint_health_check/)
job. It is installed in <code>/groups/flyem/data/snapshots</code>.
The specified neuprint server (currently
neuprint, neuprint-cns, and neuprint-fabg) is checked to see if it is up.
If it isn't, the following steps are taken by the
<code>/data15/app/neo4j/data/db/basic_restart_instance.sh</code> shell scipt on
*emdata5*:
1. Scale down the http server
2. Wait 10 seconds
3. Scale up the http server

An email is then sent to Slack for the appropriate channel indicating that the
server was restarted.

