import gzip
import json
from collections import deque, defaultdict

import msgpack
from pathlib import Path

from process.config import AOIS

AOI = AOIS[1]

docs_directory = Path(__file__).resolve().parent.parent / "docs"
geo_output_directory = docs_directory / "data" / AOI["name"]

input_file_path = geo_output_directory / "routing_graph.msgpack.gz"

# Step 1: Read the original gzipped msgpack file
try:
    with gzip.open(input_file_path, 'rb') as f: # Open in binary read mode ('rb')
        compressed_binary_data = f.read() # Read the compressed binary data

    # Decompress first, then unpack msgpack
    decompressed_binary_data = compressed_binary_data # gzip.open already handles decompression
    graph_data = msgpack.unpackb(decompressed_binary_data, raw=False) # Use msgpack.unpackb
    # raw=False decodes byte strings to Python strings where appropriate

    print("MessagePack.gz loaded successfully.")
    print(f"Top-level keys: {list(graph_data.keys())}")

except FileNotFoundError:
    print(f"Error: Input file not found at {input_file_path}")
    exit()
except Exception as e:
    print(f"An error occurred while loading or unpacking the MessagePack.gz file: {e}")
    exit()

print("\n--- Checking Canonical Order of Edges ---")

edges_list = graph_data.get("edges", [])
if not edges_list:
    print("No edges found in the graph data to check.")
else:
    non_canonical_edges = []
    total_edges = len(edges_list)

    for i, edge in enumerate(edges_list):
        source = edge.get("source")
        target = edge.get("target")

        if source is None or target is None:
            print(f"Warning: Edge {i} has missing source or target: {edge}")
            continue

        # Check if source is alphabetically greater than target
        if source > target:
            non_canonical_edges.append(edge)

    num_non_canonical = len(non_canonical_edges)

    if num_non_canonical == 0:
        print(f"✅ All {total_edges} edges are already in canonical (source < target) order.")
    else:
        print(f"❌ Found {num_non_canonical} out of {total_edges} edges NOT in canonical (source < target) order.")
        print("Sample of non-canonical edges (first 5):")
        for edge in non_canonical_edges[:5]:
            print(json.dumps(edge, indent=2))
    print("--- Canonical Order Check Complete ---")


# # Prettyprint three random edges and three random nodes
import random
print("Sample edges:")
random_edges = random.sample(graph_data.get("edges", []), 3)
for edge in random_edges:
    print(json.dumps(edge, indent=2))

print("Sample nodes:")
random_nodes = random.sample(graph_data.get("nodes", []), 3)
for node in random_nodes:
    print(json.dumps(node, indent=2))

# --- Path Connectivity Check Starts Here ---
print("\n--- Checking Path Connectivity Between Specific Nodes ---")

# Pick random start and end nodes for path connectivity check
start_node = random_nodes[0].get("key")
end_node = random_nodes[1].get("key")

# Build an adjacency list representation of the graph
# This allows for efficient lookup of neighbors
adj_list = defaultdict(list)
all_nodes_in_graph = set()

for node_obj in graph_data.get("nodes", []):
    all_nodes_in_graph.add(node_obj.get("key"))

if not graph_data.get("edges"):
    print("No edges found in the graph data. Cannot check connectivity.")
else:
    for edge in graph_data["edges"]:
        source = edge.get("source")
        target = edge.get("target")
        if source and target:
            # Add both directions assuming the graph can be traversed both ways
            # If your graph is strictly directed and you only want to check
            # connectivity along explicit directions, remove the second line below.
            adj_list[source].append(target)
            adj_list[target].append(source) # For undirected path finding

    # Check if start and end nodes exist in the graph at all
    if start_node not in all_nodes_in_graph:
        print(f"Start node {start_node} not found in the graph's nodes list.")
    elif end_node not in all_nodes_in_graph:
        print(f"End node {end_node} not found in the graph's nodes list.")
    else:
        # Perform Breadth-First Search (BFS)
        queue = deque([start_node])
        visited = {start_node}
        path_found = False

        while queue:
            current_node = queue.popleft()

            if current_node == end_node:
                path_found = True
                break

            for neighbor in adj_list[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if path_found:
            print(f"✅ A path exists between {start_node} and {end_node}.")
        else:
            print(f"❌ No path found between {start_node} and {end_node}.")

print("--- Path Connectivity Check Complete ---")