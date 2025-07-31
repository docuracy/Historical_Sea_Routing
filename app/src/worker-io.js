// worker-io.js - Used exclusively in the graph worker to load and cache AOI graphs from IndexedDB or network.

import pako from 'pako';
import msgpack from "msgpack-lite";
import graphology from "graphology";

let loadedGraph = null;

function openGraphDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('GraphCacheDB', 6);

        request.onupgradeneeded = (event) => {
            const db = request.result;

            if (!db.objectStoreNames.contains('graphs')) {
                db.createObjectStore('graphs');
            } else {
                const tx = event.target.transaction;
                const store = tx.objectStore('graphs');
                store.clear();  // wipe all cached graph data
            }
        };

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}


export async function storeGraph(graphId, graph) {
    const {nodes, edges, ...metadata} = graph.export();
    const db = await openGraphDB();

    function storeChunks(store, prefix, items, chunkSize = 5000) {
        for (let i = 0; i < items.length; i += chunkSize) {
            const chunk = items.slice(i, i + chunkSize);
            store.put(chunk, `${prefix}:${i / chunkSize}`);
        }
    }

    return new Promise((resolve, reject) => {
        const tx = db.transaction('graphs', 'readwrite');
        const store = tx.objectStore('graphs');

        try {
            storeChunks(store, `${graphId}:nodes`, nodes);
            storeChunks(store, `${graphId}:edges`, edges);
            store.put(metadata, `${graphId}:metadata`);
        } catch (e) {
            return reject(e);
        }

        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(tx.error);
    });
}

async function loadCachedGraph(graphId) {
    const db = await openGraphDB();
    const tx = db.transaction('graphs', 'readonly');
    const store = tx.objectStore('graphs');

    function loadChunks(store, prefix) {
        return new Promise((resolve, reject) => {
            const result = [];
            const range = IDBKeyRange.bound(`${prefix}:0`, `${prefix}:\uffff`);
            const request = store.openCursor(range);

            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    const chunkIndex = parseInt(cursor.key.split(':').pop(), 10);
                    result[chunkIndex] = cursor.value;
                    cursor.continue();
                } else {
                    resolve(result.flat());
                }
            };

            request.onerror = () => reject(request.error);
        });
    }

    try {
        const [nodes, edges, metadata] = await Promise.all([
            loadChunks(store, `${graphId}:nodes`),
            loadChunks(store, `${graphId}:edges`),
            new Promise((resolve, reject) => {
                const req = store.get(`${graphId}:metadata`);
                req.onsuccess = () => resolve(req.result || {});
                req.onerror = () => reject(req.error);
            })
        ]);

        return {
            type: 'split',
            data: {nodes, edges, metadata}
        };
    } catch (e) {
        console.error(`Failed to load graph chunks for ${graphId}:`, e);
        return null;
    }
}


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


export async function loadAOIGraph(payload) {
    const {aoi} = payload;
    const totalStart = performance.now();
    let doStore = false;
    try {
        const graphId = `routing_graph_${aoi}`;
        const cached = await loadCachedGraph(graphId);

        if (
            cached?.type === 'split' &&
            Array.isArray(cached.data?.nodes) &&
            cached.data.nodes.length > 0 &&
            Array.isArray(cached.data?.edges) &&
            cached.data.edges.length > 0
        ) {
            const graphData = {
                nodes: cached.data.nodes,
                edges: cached.data.edges,
                attributes: cached.data.metadata?.attributes || {},
                options: cached.data.metadata?.options || {},
            };

            const {DirectedGraph} = graphology;
            loadedGraph = DirectedGraph.from(graphData);
            console.log(`[Cache] Loaded graph ${cached.type} for ${graphId} from IndexedDB.`);
        } else {
            console.log(`[Cache] No cached graph object found for ${graphId}. Fetching from network...`);
            const basePath = import.meta.env.BASE_URL;
            const graphFile = `${basePath}/data/${aoi}/routing_graph.msgpack.gz`; // http://localhost:5173/data/Europe/routing_graph.msgpack.gz

            const response = await fetch(graphFile);
            if (!response.ok) {
                console.error(`Failed to fetch graph file: ${response.status} ${response.statusText}`);
                return {
                    success: false,
                    error: new Error(`Failed to fetch graph file: ${response.status} ${response.statusText}`),
                    result: {
                        message: `❌ Failed to fetch graph file: ${response.status} ${response.statusText}`,
                        totalTime: (performance.now() - totalStart).toFixed(2),
                    }
                };
            }

            let dataToDecode = new Uint8Array(await response.arrayBuffer());

            // Depending on server context, files may be delivered pre-decompressed.
            const hasGzipMagicBytes = dataToDecode.length >= 2 && dataToDecode[0] === 0x1F && dataToDecode[1] === 0x8B;
            if (hasGzipMagicBytes) {
                try {
                    dataToDecode = pako.ungzip(dataToDecode);
                    console.debug('Decompressed graph data using pako');
                } catch (e) {
                    throw new Error(e); // Re-throw to halt process
                }
            }

            let graphObject;
            try {
                graphObject = msgpack.decode(dataToDecode);
                console.debug('Decoded graph object')
            } catch (e) {
                throw new Error(e);
            }

            graphObject = reconstructGraph(graphObject);
            numberiseGraph(graphObject);
            addReverseEdges(graphObject);

            console.debug(graphObject);

            const {DirectedGraph} = graphology;
            loadedGraph = DirectedGraph.from(graphObject);

            doStore = true;
        }

        return {
            success: true,
            graph: loadedGraph,
            result: {
                message: `✅ AOI graph for ${aoi} loaded in ${(performance.now() - totalStart).toFixed(2)}ms.`,
                graphStats: {
                    nodeCount: loadedGraph.order,
                    edgeCount: loadedGraph.size,
                },
                doStore: doStore,
            }
        };

    } catch (e) {
        // Consolidated error return path
        console.error(`❌ Failed to load graph for AOI "${aoi}":`, e);
        return {
            success: false,
            error: e.message || "Unknown error",
            result: {
                message: `❌ Failed to load graph for AOI "${aoi}": ${e.message}`,
                totalTime: (performance.now() - totalStart).toFixed(2),
            }
        };
    }
}