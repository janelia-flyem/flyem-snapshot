{#
    This is a jijna template.
    The flyem-snapshot tool renders it into an actual Cypher script.
    (See flyem_snapshot/outputs/neuprint/indexes.py)

    The rendered script is later executed via the neo4j
    cypher-shell to create indexes on Segment properties.
    (See ingest-neuprint-snapshot-within-neo4j-container.sh)
#}

// These uniqueness constraints implicitly create indexes, too.
// https://neo4j.com/docs/cypher-manual/current/constraints/
RETURN datetime() as time, "Creating uniqueness constraint on bodyId" as message;
CREATE CONSTRAINT `{{dataset}}segment_bodyId_unique` FOR ( `{{dataset}}segment`:`{{dataset}}_Segment` ) REQUIRE `{{dataset}}segment`.bodyId IS UNIQUE;
CREATE CONSTRAINT `{{dataset}}neuron_bodyId_unique` FOR ( `{{dataset}}neuron`:`{{dataset}}_Neuron` ) REQUIRE `{{dataset}}neuron`.bodyId IS UNIQUE;

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
CREATE INDEX FOR (n:Segment) ON (n.`type`);
RETURN datetime() as time, "Initiated index creation: :Segment(`type`)" as message;
CREATE INDEX FOR (n:Neuron) ON (n.`type`);
RETURN datetime() as time, "Initiated index creation: :Neuron(`type`)" as message;
CREATE INDEX FOR (n:Synapse) ON (n.`type`);
RETURN datetime() as time, "Initiated index creation: :Synapse(`type`)" as message;

CREATE INDEX FOR (n:`{{dataset}}_Synapse`) ON (n.`bodyId`);
RETURN datetime() as time, "Initiated index creation: `{{dataset}}_Synapse`(`bodyId`)" as message;

//
// Element properties
//
{% for label, rois in element_rois_to_index.items() %}
CREATE INDEX FOR (n:`{{dataset}}_{{label}}`) ON (n.`bodyId`);
RETURN datetime() as time, "Initiated index creation: `{{dataset}}_{{label}}`(`bodyId`)" as message;

{% for roi in rois %}
CREATE INDEX FOR (n:`{{dataset}}_{{label}}`) ON (n.`{{roi}}`);
RETURN datetime() as time, ":{{label}} annotation property {{loop.index}}/{{rois|count}}: Initiated index creation for '{{roi}}'" as message;
{% endfor %}
{% endfor %}

//
// Segment/Neuron properties (other than ROIs)
//
{% for prop in segment_properties %}
CREATE INDEX FOR (n:`{{dataset}}_Segment`) ON (n.`{{prop}}`);
CREATE INDEX FOR (n:`{{dataset}}_Neuron`) ON (n.`{{prop}}`);
RETURN datetime() as time, ":Segment/:Neuron annotation property {{loop.index}}/{{segment_properties|count}}: Initiated index creation for '{{prop}}'" as message;
{% endfor %}

//
// Segment/Neuron ROI properties
//
{% for roi in segment_rois %}
CREATE INDEX FOR (n:`{{dataset}}_Segment`) ON (n.`{{roi}}`);
CREATE INDEX FOR (n:`{{dataset}}_Neuron`) ON (n.`{{roi}}`);
RETURN datetime() as time, ":Segment/:Neuron ROI property {{loop.index}}/{{segment_rois|count}}: Initiated index creation for '{{roi}}'" as message;
{% endfor %}

// 
// Fulltext index for FindNeurons autocomplete query
// These properties can be quickly searched for substrings.
//
CREATE FULLTEXT INDEX find_neurons_fulltext_properties_index FOR (n:`{{dataset}}_Neuron`)
ON EACH [
{% for prop in find_neurons_fulltext_index_properties %}
    n.`{{prop}}`{{ ", " if not loop.last else "" }}{% endfor %}
];
RETURN datetime() as time, "Initiated index creation: find_neurons_fulltext_properties_index for '{{find_neurons_fulltext_index_properties|count}} properties'" as message;

// Indexing is performed in the background,
// but we don't want to exit until the indexes are all online.
RETURN datetime() as time, "Waiting for indexes to come online..." as message;
CALL db.awaitIndexes(86400);  // wait up to 24 hours!
RETURN datetime() as time, "All indexes are online!" as message;

SHOW DATABASES;
SHOW INDEXES;

RETURN datetime() as time, "DONE with create-indexes.cypher" as message;
