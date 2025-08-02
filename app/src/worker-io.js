// worker-io.js - Used exclusively in the graph worker to load and cache AOI graphs from IndexedDB or network.

import graphology from "graphology";

let loadedGraph = null;

/**
 * Reconstructs the keyed graph structure from positional arrays with minimal keys.
 *
 * Positional structure:
 *   nodes: [key, [lat, lng, bath, clear, dist]]
 *   edges: [
 *     key, source, target,
 *     [len, dx, dy, ang],
 *     [
 *       visibility_m[], daylight_ratio[],
 *       forward[[wind_ang, wind_mag, curr_ang, curr_mag], ...12],
 *       reverse[[wind_ang, wind_mag, curr_ang, curr_mag], ...12]
 *     ]
 *   ]
 *
 * Key mappings used:
 *   node attributes: lat (l), lng (g), bathymetry (b), clear_land (c), dist_m (d)
 *   edge attributes: length_m (L), dx (x), dy (y), angle (a)
 *   months keys:
 *     visibility_m (v), daylight_ratio (D)
 *     forward (f), reverse (r)
 *       wind_angle (wA), wind_mag (wM), current_angle (cA), current_mag (cM)
 */
function reconstructGraph(positionalGraph) {
    const n = positionalGraph[0].length;
    const e = positionalGraph[1].length;
    const nodes = new Array(n);
    const edges = new Array(e);

    for (let i = 0; i < n; i++) {
        const node = positionalGraph[0][i];
        const attrs = node[1];
        nodes[i] = {
            key: node[0],
            attributes: {
                b: attrs[0],  // bathymetry
                c: attrs[1],  // clear_land
                d: attrs[2]   // dist_m
            }
        };
    }

    for (let i = 0; i < e; i++) {
        const edge = positionalGraph[1][i];
        const attr = edge[3];
        const months = edge[4];

        // Preallocate forward/reverse month arrays
        const fArr = new Array(12);
        const rArr = new Array(12);

        const fRaw = months[2];
        const rRaw = months[3];

        for (let m = 0; m < 12; m++) {
            fArr[m] = {
                wA: fRaw[m][0],
                wM: fRaw[m][1],
                cA: fRaw[m][2],
                cM: fRaw[m][3]
            };
            rArr[m] = {
                wA: rRaw[m][0],
                wM: rRaw[m][1],
                cA: rRaw[m][2],
                cM: rRaw[m][3]
            };
        }

        edges[i] = {
            key: edge[0],
            source: edge[1],
            target: edge[2],
            attributes: {
                L: attr[0],  // length_m
                a: attr[1]   // angle
            },
            months: {
                v: months[0],  // visibility_m
                D: months[1],  // daylight_ratio
                f: fArr,       // forward
                r: rArr        // reverse
            }
        };
    }

    return {'nodes': nodes, 'edges': edges};
}

/**
 * Convert all relevant string attributes in the ultra-lean graph
 * structure to numbers, in place, using short keys.
 *
 * Short keys used:
 *   Nodes attributes: l (lat), g (lng), b (bathymetry), c (clear_land), d (dist_m)
 *   Edge attributes: L (length_m), a (angle)
 *   Months: v (visibility_m), D (daylight_ratio),
 *           f/r (forward/reverse arrays of [wA, wM, cA, cM])
 *           where wM, cM need numeric conversion.
 */
function numberiseGraph(graph) {
    // Convert node attribute strings to numbers
    for (let i = 0; i < graph.nodes.length; i++) {
        const attrs = graph.nodes[i].attributes;
        for (const k in attrs) {
            attrs[k] = Number(attrs[k]);
        }
    }

    // Convert edge-level attributes and monthly values
    const directions = ['f', 'r'];
    for (let i = 0; i < graph.edges.length; i++) {
        const edge = graph.edges[i];
        const months = edge.months;

        // Scalar arrays
        months.v = months.v.map(Number);

        // Vector arrays (forward/reverse)
        for (let j = 0; j < 2; j++) {
            const arr = months[directions[j]];
            for (let k = 0; k < arr.length; k++) {
                arr[k].wM = Number(arr[k].wM);
                arr[k].cM = Number(arr[k].cM);
            }
        }
    }
}


function addReverseEdges(graph) {
    const originalEdges = graph.edges;
    const newEdges = [];

    for (let i = 0; i < originalEdges.length; i++) {
        const edge = originalEdges[i];
        const {source, target, attributes, months} = edge;
        const length = attributes.L;
        const angle = attributes.a;

        newEdges.push({
            key: `${source}_${target}`,
            source,
            target,
            attributes: {
                L: length,
                a: angle,
                v: months.v,
                D: months.D,
                f: months.f
            }
        });

        newEdges.push({
            key: `${target}_${source}`,
            source: target,
            target: source,
            attributes: {
                L: length,
                a: (angle + Math.PI) % (2 * Math.PI), // Reverse angle
                v: months.v,
                D: months.D,
                f: months.r
            }
        });
    }

    graph.edges = newEdges;
}

async function fetchJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
    return res.json();
}

// Fetch all chunks given a base URL and count, returning the concatenated array
async function fetchChunks(baseUrl, count) {
    const urls = Array.from({ length: count }, (_, i) => `${baseUrl}-${i}.json`);
    const chunkArrays = await Promise.all(urls.map(fetchJSON));
    return chunkArrays.flat();
}


export async function loadAOIGraph(payload) {
    const {aoi} = payload;
    const totalStart = performance.now();

    try {
        const basePath = import.meta.env.BASE_URL;
        const manifestUrl = `${basePath}/data/${aoi}/graph_manifest.json`;

        // 1. Fetch manifest giving chunk counts for nodes and edges
        const manifest = await fetchJSON(manifestUrl);

        // 2. Fetch nodes and edges chunks in sequence
        const nodes = await fetchChunks(`${basePath}/data/${aoi}/routing_graph_parts/nodes`, manifest.nodes_chunks);
        const edges = await fetchChunks(`${basePath}/data/${aoi}/routing_graph_parts/edges`, manifest.edges_chunks);

        // 3. Fetch metadata
        const metadata = await fetchJSON(`${basePath}/data/${aoi}/metadata.json`);

        // 4. Reconstruct graph object
        let graphObject = reconstructGraph([nodes, edges]);
        graphObject.metadata = metadata;

        // 5. Convert strings to numbers and add reverse edges
        numberiseGraph(graphObject);
        addReverseEdges(graphObject);

        // 6. Build graphology graph
        const {DirectedGraph} = graphology;
        loadedGraph = DirectedGraph.from(graphObject);

        return {
            success: true,
            graph: loadedGraph,
            result: {
                message: `✅ AOI graph for ${aoi} loaded in ${(performance.now() - totalStart).toFixed(2)}ms.`,
                graphStats: {
                    nodeCount: loadedGraph.order,
                    edgeCount: loadedGraph.size,
                },
            },
        };
    } catch (e) {
        console.error(`❌ Failed to load graph for AOI "${aoi}":`, e);
        return {
            success: false,
            error: e.message || "Unknown error",
            result: {
                message: `❌ Failed to load graph for AOI "${aoi}": ${e.message}`,
                totalTime: (performance.now() - totalStart).toFixed(2),
            },
        };
    }
}