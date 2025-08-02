import gzip
import json
import logging

import msgpack

from process.config import AOIS, head_directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(json_size_limit: int = 50_000_000):
    AOI = AOIS[0]  # Use Europe bounding box
    bbox = list(AOI["bounds"])
    logger.info(f"Using AOI: {AOI['name']} with bounds {bbox}")

    docs_directory = head_directory / "docs"
    output_dir = docs_directory / "data" / AOI["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = output_dir / "routing_graph.msgpack.gz"
    graph_parts_dir = output_dir / "routing_graph_parts"
    graph_parts_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(graph_path, "rb") as f:
        graph = msgpack.unpackb(f.read(), strict_map_key=False)

    nodes, edges = graph

    def write_chunks_by_size(data, label, max_bytes):
        chunks_written = 0
        chunk = []
        current_size = 2  # For [ and ]
        max_chunk_kb = 0

        def write_chunk_to_file(chunk_data, index):
            path = graph_parts_dir / f"{label}-{index}.json"
            with open(path, "w") as f:
                json.dump(chunk_data, f, separators=(",", ":"))
            size_kb = path.stat().st_size / 1024
            logger.debug(f"Wrote {label}-{index}.json ({size_kb:.1f} KB)")
            return size_kb

        for item in data:
            item_json = json.dumps(item, separators=(",", ":"))
            item_size = len(item_json.encode("utf-8")) + 1  # +1 for comma or bracket

            if current_size + item_size > max_bytes:
                size_kb = write_chunk_to_file(chunk, chunks_written)
                max_chunk_kb = max(max_chunk_kb, size_kb)
                chunks_written += 1
                chunk = []
                current_size = 2

            chunk.append(item)
            current_size += item_size

        if chunk:
            size_kb = write_chunk_to_file(chunk, chunks_written)
            max_chunk_kb = max(max_chunk_kb, size_kb)
            chunks_written += 1

        return chunks_written, max_chunk_kb

    logger.info("Writing nodes to size-limited JSON chunks...")
    node_chunks, max_node_kb = write_chunks_by_size(nodes, "nodes", json_size_limit)

    logger.info("Writing edges to size-limited JSON chunks...")
    edge_chunks, max_edge_kb = write_chunks_by_size(edges, "edges", json_size_limit)

    manifest = {
        "nodes_chunks": node_chunks,
        "edges_chunks": edge_chunks,
        "max_chunk_size_kb": round(max(max_node_kb, max_edge_kb), 1),
        "file_format": "json"
    }

    manifest_path = output_dir / "graph_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"✔ Done. Wrote {node_chunks} node chunks and {edge_chunks} edge chunks to {output_dir}")
    return manifest


if __name__ == "__main__":
    main()
