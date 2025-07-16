from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, FeatureService, ValueType
from feast.types import Float32, Int64

# Define athlete entity
athlete = Entity(
    name="athlete_id",
    join_keys=["athlete_id"],
    value_type=ValueType.INT64 
)

# Define data source
athletes_v1_source = FileSource(
    path="data/v1_features.parquet",
    timestamp_field="event_timestamp",
)

athletes_v2_source = FileSource(
    path="data/v2_features.parquet",
    timestamp_field="event_timestamp",
)

# Define feature views
athletes_v1_fv = FeatureView(
    name="athletes_v1",
    entities=[athlete],
    ttl=timedelta(days=1),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="weight", dtype=Float32),
        Field(name="height", dtype=Float32),
        Field(name="total_lift", dtype=Float32),
    ],
    online=True,
    source=athletes_v1_source,
)

athletes_v2_fv = FeatureView(
    name="athletes_v2",
    entities=[athlete],
    ttl=timedelta(days=1),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="weight", dtype=Float32),
        Field(name="height", dtype=Float32),
        Field(name="deadlift", dtype=Float32),
        Field(name="snatch", dtype=Float32),
        Field(name="backsq", dtype=Float32),
        Field(name="total_lift", dtype=Float32),
    ],
    online=True,
    source=athletes_v2_source,
)

# Define feature services
athletes_service_v1 = FeatureService(
    name="athletes_service_v1",
    features=[athletes_v1_fv],
)

athletes_service_v2 = FeatureService(
    name="athletes_service_v2",
    features=[athletes_v2_fv],
)