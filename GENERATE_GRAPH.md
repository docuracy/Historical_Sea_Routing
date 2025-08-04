## Fetch Environment Data

_Not yet documented_

## Create Blended-Resolution Hex Graph

- Define Area of Interest (AOI) in `process/config.py` by adding to the `AOIS` array.
- Also set the desired `COASTAL_SEA_RESOLUTION`: 7 may be sufficient depending on use case, but does not (for example) allow for paths through narrow straits or harbour mouths.
- Set the required AOI index in `process/sea_graph.py` (for example, `AOI = AOIS[0]`), and then run the script.

## Build Enriched Graph

- Set the required AOI index in `process/graph_from_datasets.py`, and then run the script.
- Unpacking the generated `routing_graph.msgpack.gz` takes too long in Firefox browsers, so split the graph into chunks for faster parallel loading and so as to be small enough to store on GitHub: set the required AOI index in `process/split_json_graph.py`, and then run the script.

## Generate Map Tiles

- Set the required AOI index in `process/generate_graph_tiles.py`, and then run the script.

