// Let's try warming up the cache before we start creating indexes.
// This might improve indexing speed, though we haven't actually benchmarked it.
RETURN datetime() as time, "Warming up page cache" as message;
CALL apoc.warmup.run();

// These uniqueness constraints implicitly create indexes, too.
// https://neo4j.com/docs/cypher-manual/4.4/constraints/

// This syntax appears to be for neo4j 3.5, with newer versions using a different syntax.
// But this seems to still work for now, at least in neo4j 4.4.
CREATE CONSTRAINT ON ( {{dataset}}segment:{{dataset}}_Segment ) ASSERT {{dataset}}segment.bodyId IS UNIQUE;
CREATE CONSTRAINT ON ( {{dataset}}neuron:{{dataset}}_Neuron ) ASSERT {{dataset}}neuron.bodyId IS UNIQUE;

// This is a good constraint to check, since the assumption that synapse
// points are unique is baked in to the neuprint data model.
// However, this will implicitly create an index on all synapse locations.
// I'm not sure how costly that is and what benefit that brings.
// We should consider dropping this unless we can think of queries that benefit from this index.
RETURN datetime() as time, ":Synapse.location: Requesting index creation" as message;
CREATE CONSTRAINT ON ( {{dataset}}synapse:{{dataset}}_Synapse ) ASSERT {{dataset}}synapse.location IS UNIQUE;
RETURN datetime() as time, ":Synapse.location: Initiated index creation" as message;

// I have no idea what this DataModel node is, so it's possible this line
// is erroneously left over from an earlier neuprint prototype.
// CREATE CONSTRAINT ON ( datamodel:DataModel ) ASSERT datamodel.dataModelVersion IS UNIQUE;

// I don't know what this mutationUuidAndId property is.
// CREATE CONSTRAINT ON ( {{dataset}}segment:{{dataset}}_Segment ) ASSERT {{dataset}}segment.mutationUuidAndId IS UNIQUE;

// I'm not sure why we index `type` separately here,
// for the bare :Segment/:Neuron/:Synapse labels.
CREATE INDEX ON :Segment(`type`);
CREATE INDEX ON :Neuron(`type`);
CREATE INDEX ON :Synapse(`type`);

//
// Segment/Neuron properties (other than ROIs)
//
{% for prop in segment_properties %}
CREATE INDEX ON :{{dataset}}_Segment(`{{prop}}`);
CREATE INDEX ON :{{dataset}}_Neuron(`{{prop}}`);
RETURN datetime() as time, ":Segment/:Neuron annotation property #{{loop.index}}: Initiated index creation for '{{prop}}'" as message;
{% endfor %}

//
// Segment/Neuron ROI properties
//
{% for roi in rois %}
CREATE INDEX ON :{{dataset}}_Segment(`{{roi}}`);
CREATE INDEX ON :{{dataset}}_Neuron(`{{roi}}`);
RETURN datetime() as time, ":Segment/:Neuron ROI property #{{loop.index}}: Initiated index creation for '{{roi}}'" as message;
{% endfor %}

// Indexing is performed in the background,
// but we don't want to exit until the indexes are all online.
RETURN datetime() as time, "Waiting for indexes to come online..." as message;
CALL db.awaitIndexes(86400);  // wait up to 24 hours!
RETURN datetime() as time, "All indexes are online!" as message;

SHOW DATABASES;
SHOW INDEXES;

// The next step documented in our original procedure is to warm up
// the page cache using the following command. But shouldn't this be
// executed only on the production deployment?
// What's the point of doing this on the build machine?
//
// CALL apoc.warmup.run();
//
