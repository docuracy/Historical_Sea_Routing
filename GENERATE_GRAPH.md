## Fetch Environment Data

_Not yet Documented_

## Create Blended-Resolution Hex Graph

- Define Area of Interest (AOI) in `process/config.py` by adding to the `AOIS` array.
- Also set the desired `COASTAL_SEA_RESOLUTION`: 7 may be sufficient depending on use case, but does not (for example) allow for paths through narrow straits or harbour mouths.
- Set the required AOI index in `process/sea_graph.py` (for example, `AOI = AOIS[0]`), and then run the script.

## Build Enriched Graph

- Set the required AOI index in `process/build_graph.py` (for example, `AOI = AOIS[0]`), and then run the script.
