{#
    This is a jijna template.
    The flyem-snapshot tool renders it into an actual Cypher script.
    (See flyem_snapshot/outputs/neuprint/indexes.py)

    The rendered script is later executed via the neo4j
    cypher-shell to create indexes on Segment properties.
    (See ingest-neuprint-snapshot-within-neo4j-container.sh)
#}

// These uniqueness constraints implicitly create indexes, too.
// https://neo4j.com/docs/cypher-manual/4.4/constraints/

// This syntax appears to be for neo4j 3.5, with newer versions using a different syntax.
// But this seems to still work for now, at least in neo4j 4.4.
RETURN datetime() as time, "Creating uniqueness constraint on bodyId" as message;
CREATE CONSTRAINT ON ( `{{dataset}}segment`:`{{dataset}}_Segment` ) ASSERT `{{dataset}}segment`.bodyId IS UNIQUE;
CREATE CONSTRAINT ON ( `{{dataset}}neuron`:`{{dataset}}_Neuron` ) ASSERT `{{dataset}}neuron`.bodyId IS UNIQUE;

// We used to enforce a uniqueness constraint on Element.location, but there is technically no need for
// that if the Synapse/Element point_id was provided directly by the user (via their feather files).
// And even if we calculated the point_ids ourselves, duplicate locations would result in duplicate Element-IDs
// and the neo4j node ingestion would fail long before this script is called.
// CREATE CONSTRAINT ON ( `{{dataset}}element`:`{{dataset}}_Element` ) ASSERT `{{dataset}}element`.location IS UNIQUE;

// Create spatial index for Element (including Synapse) location.
CREATE POINT INDEX `{{dataset}}ElementLocation` FOR (n:`{{dataset}}_Element`) ON (n.location);
RETURN datetime() as time, ":Element.location: Initiated index creation" as message;

// I have no idea what this DataModel node is, so it's possible this line
// is erroneously left over from an earlier neuprint prototype.
// CREATE CONSTRAINT ON ( datamodel:DataModel ) ASSERT datamodel.dataModelVersion IS UNIQUE;

// I don't know what this mutationUuidAndId property is.
// CREATE CONSTRAINT ON ( {{dataset}}segment:{{dataset}}_Segment ) ASSERT {{dataset}}segment.mutationUuidAndId IS UNIQUE;

// I'm not sure why we index `type` separately here
// for the bare :Segment/:Neuron/:Synapse labels.
CREATE INDEX ON :Segment(`type`);
RETURN datetime() as time, "Initiated index creation: :Segment(`type`)" as message;
CREATE INDEX ON :Neuron(`type`);
RETURN datetime() as time, "Initiated index creation: :Neuron(`type`)" as message;
CREATE INDEX ON :Synapse(`type`);
RETURN datetime() as time, "Initiated index creation: :Synapse(`type`)" as message;

CREATE INDEX ON :`{{dataset}}_Synapse`(`bodyId`);
RETURN datetime() as time, "Initiated index creation: `{{dataset}}_Synapse`(`bodyId`)" as message;

//
// Element properties
//
{% for label, rois in element_rois_to_index.items() %}
CREATE INDEX ON :`{{dataset}}_{{label}}`(`bodyId`);
RETURN datetime() as time, "Initiated index creation: `{{dataset}}_{{label}}`(`bodyId`)" as message;

{% for roi in rois %}
CREATE INDEX ON :`{{dataset}}_{{label}}`(`{{roi}}`);
RETURN datetime() as time, ":{{label}} annotation property {{loop.index}}/{{rois|count}}: Initiated index creation for '{{roi}}'" as message;
{% endfor %}
{% endfor %}

//
// Segment/Neuron properties (other than ROIs)
//
{% for prop in segment_properties %}
CREATE INDEX ON :`{{dataset}}_Segment`(`{{prop}}`);
CREATE INDEX ON :`{{dataset}}_Neuron`(`{{prop}}`);
RETURN datetime() as time, ":Segment/:Neuron annotation property {{loop.index}}/{{segment_properties|count}}: Initiated index creation for '{{prop}}'" as message;
{% endfor %}

//
// Segment/Neuron ROI properties
//
{% for roi in segment_rois %}
CREATE INDEX ON :`{{dataset}}_Segment`(`{{roi}}`);
CREATE INDEX ON :`{{dataset}}_Neuron`(`{{roi}}`);
RETURN datetime() as time, ":Segment/:Neuron ROI property {{loop.index}}/{{segment_rois|count}}: Initiated index creation for '{{roi}}'" as message;
{% endfor %}

// Indexing is performed in the background,
// but we don't want to exit until the indexes are all online.
RETURN datetime() as time, "Waiting for indexes to come online..." as message;
CALL db.awaitIndexes(86400);  // wait up to 24 hours!
RETURN datetime() as time, "All indexes are online!" as message;

SHOW DATABASES;
SHOW INDEXES;

RETURN datetime() as time, "DONE with create-indexes.cypher" as message;
