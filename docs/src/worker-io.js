// worker-io.js - Used exclusively in the graph worker to load and cache AOI graphs from IndexedDB or network.

import pako from 'pako';
import msgpack from "msgpack-lite";
import graphology from "graphology";

let loadedGraph = null;

function openGraphDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('GraphCacheDB', 4);

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
    const finalGraphObject = graph.export();
    const db = await openGraphDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction('graphs', 'readwrite');
        const store = tx.objectStore('graphs');

        const req = store.put(finalGraphObject, graphId);

        req.onerror = () => reject(req.error);

        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(tx.error);
    });
}

async function loadCachedGraph(graphId) {
    const db = await openGraphDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction('graphs', 'readonly');
        const store = tx.objectStore('graphs');

        const req = store.get(graphId);

        req.onsuccess = () => {
            if (req.result) {
                resolve({type: 'object', data: req.result});
            } else {
                resolve(null);
            }
        };

        req.onerror = () => {
            console.warn(`[Cache] Failed to load graph object for ${graphId} from IndexedDB:`, req.error);
            reject(req.error);
        }
    });
}


export async function loadAOIGraph(payload) {
    const {aoi} = payload;
    const totalStart = performance.now();
    let doStore = false;
    try {
        const graphId = `routing_graph_${aoi}`;
        const cached = await loadCachedGraph(graphId);

        if (cached?.type === 'object') {
            const {DirectedGraph} = graphology;
            loadedGraph = DirectedGraph.from(cached.data);
            console.log(`[Cache] Loaded graph object for ${graphId} from IndexedDB.`);
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

            const {DirectedGraph} = graphology;
            loadedGraph = DirectedGraph.from(graphObject);

            loadedGraph.forEachEdge((edge, attributes, source, target) => {
                if (!loadedGraph.hasEdge(target, source)) {
                    loadedGraph.addDirectedEdgeWithKey(`${edge}_rev`, target, source, {
                        ...attributes,
                        dx: -attributes.dx,
                        dy: -attributes.dy,
                        reverse: true,
                    });
                }
            });

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