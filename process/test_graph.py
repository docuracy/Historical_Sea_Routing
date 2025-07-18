import gzip
import msgpack
import random
from collections import deque, defaultdict
from pathlib import Path

from process.config import AOIS

AOI = AOIS[0]

docs_directory = Path(__file__).resolve().parent.parent / "docs"
geo_output_directory = docs_directory / "data" / AOI["name"]

input_file_path = geo_output_directory / "routing_graph.msgpack.gz"

# Step 1: Read the original gzipped msgpack file
try:
    with gzip.open(input_file_path, 'rb') as f:
        # gzip.open decompresses automatically
        decompressed_binary_data = f.read()
    graph_data = msgpack.unpackb(decompressed_binary_data, raw=False)

    print("MessagePack.gz loaded successfully.")
    print(f"Graph top-level structure: type={type(graph_data)}, length={len(graph_data)}")
except FileNotFoundError:
    print(f"Error: Input file not found at {input_file_path}")
    exit()
except Exception as e:
    print(f"An error occurred while loading or unpacking the MessagePack.gz file: {e}")
    exit()

# Extract nodes and edges lists
nodes_list = graph_data[0] if len(graph_data) > 0 else []
edges_list = graph_data[1] if len(graph_data) > 1 else []

print(f"Loaded {len(nodes_list)} nodes and {len(edges_list)} edges.")

print("\n--- Checking Canonical Order of Edges ---")

if not edges_list:
    print("No edges found in the graph data to check.")
else:
    non_canonical_edges = []
    total_edges = len(edges_list)

    for i, edge in enumerate(edges_list):
        # edge structure: [key, source, target, attrs?, monthly?]
        if len(edge) < 3:
            print(f"Warning: Edge {i} has insufficient elements: {edge}")
            continue

        source = edge[1]
        target = edge[2]

        if source is None or target is None:
            print(f"Warning: Edge {i} has missing source or target: {edge}")
            continue

        if source > target:
            non_canonical_edges.append(edge)

    num_non_canonical = len(non_canonical_edges)

    if num_non_canonical == 0:
        print(f"✅ All {total_edges} edges are already in canonical (source < target) order.")
    else:
        print(f"❌ Found {num_non_canonical} out of {total_edges} edges NOT in canonical (source < target) order.")
        print("Sample of non-canonical edges (first 5):")
        for edge in non_canonical_edges[:5]:
            print(edge)
    print("--- Canonical Order Check Complete ---")

# Pretty-print three random edges and three random nodes
print("\nSample edges:")
if len(edges_list) >= 3:
    random_edges = random.sample(edges_list, 3)
else:
    random_edges = edges_list

for edge in random_edges:
    print(edge)

print("\nSample nodes:")
if len(nodes_list) >= 3:
    random_nodes = random.sample(nodes_list, 3)
else:
    random_nodes = nodes_list

for node in random_nodes:
    print(node)

# --- Path Connectivity Check Starts Here ---
print("\n--- Checking Path Connectivity Between Specific Nodes ---")

if len(random_nodes) < 2:
    print("Not enough nodes to perform connectivity check.")
else:
    start_node = random_nodes[0][0]  # node key
    end_node = random_nodes[1][0]

    adj_list = defaultdict(list)
    all_nodes_in_graph = {node[0] for node in nodes_list}

    if not edges_list:
        print("No edges found in the graph data. Cannot check connectivity.")
    else:
        for edge in edges_list:
            if len(edge) < 3:
                continue
            source = edge[1]
            target = edge[2]
            if source and target:
                # Add edges in both directions for undirected connectivity
                adj_list[source].append(target)
                adj_list[target].append(source)

        if start_node not in all_nodes_in_graph:
            print(f"Start node {start_node} not found in the graph's nodes list.")
        elif end_node not in all_nodes_in_graph:
            print(f"End node {end_node} not found in the graph's nodes list.")
        else:
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
