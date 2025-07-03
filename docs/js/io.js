// io.js

function openGraphDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('GraphCacheDB', 1);
        request.onupgradeneeded = () => {
            request.result.createObjectStore('graphs');
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}


async function storeGraph(graphId, decompressedUint8Array, finalGraphObject) {
    const db = await openGraphDB();
    const tx = db.transaction('graphs', 'readwrite');
    const store = tx.objectStore('graphs');

    store.put(decompressedUint8Array, `binary_${graphId}`);
    store.put(finalGraphObject, `graphObject_${graphId}`);

    return tx.done || tx.complete;
}


async function loadCachedGraph(graphId) {
    const db = await openGraphDB();
    const tx = db.transaction('graphs', 'readonly');
    const store = tx.objectStore('graphs');

    const graphObjectReq = store.get(`graphObject_${graphId}`);
    const binaryReq = store.get(`binary_${graphId}`);

    return new Promise((resolve, reject) => {
        graphObjectReq.onsuccess = () => {
            if (graphObjectReq.result) {
                resolve({type: 'object', data: graphObjectReq.result});
            } else {
                binaryReq.onsuccess = () => {
                    resolve(binaryReq.result
                        ? {type: 'binary', data: binaryReq.result}
                        : null);
                };
                binaryReq.onerror = () => reject(binaryReq.error);
            }
        };
        graphObjectReq.onerror = () => reject(graphObjectReq.error);
    });
}


async function loadAOIGraph() {
    const totalStart = performance.now();
    try {
        await showSpinner("Fetching graph data…");
        graphId = `routing_graph_${aoi}`;

        const cacheStart = performance.now();
        const cached = await loadCachedGraph(graphId);
        const cacheTime = performance.now() - cacheStart;

        if (cached?.type === 'object') {
            console.log(`[Cache] Final graph object loaded in ${cacheTime.toFixed(2)} ms.`);
            const buildStart = performance.now();
            const { DirectedGraph } = graphology;
            graph = DirectedGraph.from(cached.data);
            const buildTime = performance.now() - buildStart;
            console.log(`[Build] Graph loaded from cached object in ${buildTime.toFixed(2)} ms.`);
        } else {
            console.log("[Network] No cache found. Fetching from network…");

            const fetchStart = performance.now();
            const response = await fetch(`data/${aoi}/routing_graph.msgpack.gz`);
            const compressed = await response.arrayBuffer();
            const fetchTime = performance.now() - fetchStart;
            console.log(`[Network] Fetch took ${fetchTime.toFixed(2)} ms.`);

            const decompressStart = performance.now();
            const decompressed = fflate.decompressSync(new Uint8Array(compressed));
            const decompressTime = performance.now() - decompressStart;
            console.log(`[Decompress] Completed in ${decompressTime.toFixed(2)} ms.`);

            const decodeStart = performance.now();
            const graphObject = msgpack.decode(decompressed);
            const decodeTime = performance.now() - decodeStart;
            console.log(`[Decode] Decoded raw graph in ${decodeTime.toFixed(2)} ms.`);

            const buildStart = performance.now();
            const { DirectedGraph } = graphology;
            graph = DirectedGraph.from(graphObject);

            graph.forEachEdge((edge, attributes, source, target) => {
                if (!graph.hasEdge(target, source)) {
                    graph.addDirectedEdgeWithKey(`${edge}_rev`, target, source, {
                        ...attributes,
                        dx: -attributes.dx,
                        dy: -attributes.dy,
                        reverse: true,
                    });
                }
            });

            const buildTime = performance.now() - buildStart;
            console.log(`[Build] Graph built and reversed in ${buildTime.toFixed(2)} ms.`);

            const exportStart = performance.now();
            const finalGraphObject = graph.export();
            await storeGraph(graphId, decompressed.buffer, finalGraphObject);
            const exportTime = performance.now() - exportStart;
            console.log(`[Cache] Stored finalised graph object in ${exportTime.toFixed(2)} ms.`);
        }

        console.info(`✅ AOI graph for ${aoi} loaded: ${graph.order.toLocaleString()} nodes, ${graph.size.toLocaleString()} edges.`);

    } catch (err) {
        console.error("❌ Failed to load graph:", err);
        console.warn("⚠️ Graph failed to load or is empty.");
    } finally {
        const totalTime = performance.now() - totalStart;
        console.log(`🔁 Total graph load time: ${totalTime.toFixed(2)} ms.`);
        await hideSpinner();
    }
}


async function loadMetadata(defaultAOI = "Europe") {
    const params = new URLSearchParams(window.location.search);
    aoi = params.get("aoi") || defaultAOI;
    let res;

    try {
        res = await fetch(`data/${aoi}/metadata.json`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
        console.warn(`Failed to load metadata for "${aoi}", falling back to "${defaultAOI}".`);
        aoi = defaultAOI;
        res = await fetch(`data/${aoi}/metadata.json`);
    }

    metadata = await res.json();

    H3_RESOLUTION = metadata.h3_resolution;
}


function downloadGeoJson() {
    stopMonthCycle(); // Stop any ongoing month cycle to ensure we package the current state
    const features = [];

    routeColours.forEach((colour, idx) => {
        const isReturn = (idx === 1 && vesselParameters.return);
        const originH3 = isReturn ? endPointH3 : startPointH3;
        const destinationH3 = isReturn ? startPointH3 : endPointH3;
        const direction = isReturn ? 'return' : 'outward';

        const originalSource = map.getSource(`original-route-${colour}`);
        const processedSource = map.getSource(`processed-route-${colour}`);

        if (originalSource && originalSource._data) {
            features.push({
                type: 'Feature',
                properties: {
                    type: 'h3 hex centres',
                    origin: originH3,
                    destination: destinationH3,
                    direction
                },
                geometry: originalSource._data.geometry
            });
        }

        if (processedSource && processedSource._data) {
            features.push({
                type: 'Feature',
                properties: {
                    type: 'spline curve',
                    origin: originH3,
                    destination: destinationH3,
                    direction
                },
                geometry: processedSource._data.geometry
            });
        }
    });

    const geojson = {
        type: 'FeatureCollection',
        properties: {
            metadata: structuredClone(metadata),
            vesselParameters: structuredClone(vesselParameters),
            routeLogs: structuredClone(routeLogs),
        },
        features
    };

    const blob = new Blob([JSON.stringify(geojson)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `route_${startPointH3}_${endPointH3}.geojson`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}