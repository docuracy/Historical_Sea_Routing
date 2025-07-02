let aoi;

let map;
let graph;
let metadata;
let H3_RESOLUTION; // To store the maximum resolution of H3 cells in the graph

// Variables for click-based point selection
let startPointH3 = null;
let endPointH3 = null;
let selectingStart = true;

// MapLibre markers
let startMarker = null;
let endMarker = null;

let monthCycleInterval = null;

let routeColours = ['orange', 'pink'];
let routeBounds = new maplibregl.LngLatBounds();
let routeLogs;

function h3ToLngLat(h3Index) {
    const [lat, lng] = h3.cellToLatLng(h3Index);
    return [lng, lat];
}

function interpolate(p0, p1, denom, w1, w2) {
    return [
        (w1 * p0[0] + w2 * p1[0]) / denom,
        (w1 * p0[1] + w2 * p1[1]) / denom
    ];
}

function chaikinSmooth(coords, iterations = 3) {
    if (coords.length < 3) return coords;

    let newCoords = coords;
    for (let it = 0; it < iterations; it++) {
        const smoothed = [newCoords[0]]; // keep first point
        for (let i = 0; i < newCoords.length - 1; i++) {
            const [x0, y0] = newCoords[i];
            const [x1, y1] = newCoords[i + 1];

            const Q = [(0.75 * x0 + 0.25 * x1), (0.75 * y0 + 0.25 * y1)];
            const R = [(0.25 * x0 + 0.75 * x1), (0.25 * y0 + 0.75 * y1)];

            smoothed.push(Q, R);
        }
        smoothed.push(newCoords[newCoords.length - 1]); // keep last point
        newCoords = smoothed;
    }

    return newCoords;
}

function drawRoute(path, colour) {

    function findLayerBefore(prefix) {
        const layers = map.getStyle().layers;
        if (!layers) return undefined;
        const match = layers.find(l => l.id.startsWith(prefix));
        return match?.id;
    }

    const beforeId = findLayerBefore('original-route-layer-');

    if (!path || path.length < 2) {
        console.warn("Path is too short to draw or invalid.");
        clearRoute();
        return;
    }

    const originalCoordinates = path.map(h3ToLngLat);

    const originalGeojson = {
        type: 'Feature',
        geometry: {
            type: 'LineString',
            coordinates: originalCoordinates,
        },
    };

    if (map.getSource(`original-route-${colour}`)) {
        map.getSource(`original-route-${colour}`).setData(originalGeojson);
    } else {
        map.addSource(`original-route-${colour}`, {type: 'geojson', data: originalGeojson});
        map.addLayer({
            id: `original-route-layer-${colour}`,
            type: 'line',
            source: `original-route-${colour}`,
            paint: {
                'line-color': '#00000033', // Semi-transparent black
                'line-width': 2,
                'line-dasharray': [2, 2]
            },
        }, beforeId);
    }

    // Step 1: simplify
    const simplified = turf.simplify(turf.lineString(originalCoordinates), {
        tolerance: 0.01,
        highQuality: true
    });

    const simplifiedCoords = simplified.geometry.coordinates;

    // Apply Chaikin smoothing
    const smoothedCoords = chaikinSmooth(simplifiedCoords, 3);

    const smoothedGeojson = {
        type: 'Feature',
        geometry: {
            type: 'LineString',
            coordinates: smoothedCoords
        }
    };

    if (map.getSource(`processed-route-${colour}`)) {
        map.getSource(`processed-route-${colour}`).setData(smoothedGeojson);
    } else {
        map.addSource(`processed-route-${colour}`, {type: 'geojson', data: smoothedGeojson});
        map.addLayer({
            id: `processed-route-layer-${colour}`,
            type: 'line',
            source: `processed-route-${colour}`,
            paint: {
                'line-color': colour,
                'line-width': 4,
            },
        }, beforeId);
    }

    originalCoordinates.forEach(coord => {
        routeBounds.extend(coord);
    });
    smoothedCoords.forEach(coord => {
        routeBounds.extend(coord);
    });
}

function clearRoute() {
    routeColours.forEach((colour) => {
        // Check and remove the original route layer and source
        if (map.getLayer(`original-route-layer-${colour}`)) {
            map.removeLayer(`original-route-layer-${colour}`);
        }
        if (map.getSource(`original-route-${colour}`)) {
            map.removeSource(`original-route-${colour}`);
        }
        // Check and remove the processed route layer and source
        if (map.getLayer(`processed-route-layer-${colour}`)) {
            map.removeLayer(`processed-route-layer-${colour}`);
        }
        if (map.getSource(`processed-route-${colour}`)) {
            map.removeSource(`processed-route-${colour}`);
        }
    })
}

// Function to add/update markers on the map
async function addPointMarker(lngLat, color = '#3BB2D0', isStart = true) {
    // Remove existing marker if it's there
    if (isStart) {
        if (startMarker) startMarker.remove();
        startMarker = new maplibregl.Marker({color: color})
            .setLngLat(lngLat)
            .addTo(map);
    } else {
        if (endMarker) endMarker.remove();
        endMarker = new maplibregl.Marker({color: color})
            .setLngLat(lngLat)
            .addTo(map);
    }
}

function clearPointMarkers() {
    if (startMarker) {
        startMarker.remove();
        startMarker = null;
    }
    if (endMarker) {
        endMarker.remove();
        endMarker = null;
    }
}

function showToast(message, duration = 3000) {
    const $toast = $('<div class="toast-message"></div>').text(message);
    $('body').append($toast);
    $toast.fadeIn(400);
    setTimeout(() => {
        $toast.fadeOut(400, () => $toast.remove());
    }, duration);
}

// --- Find Closest Graph Node ---
function findClosestGraphNode(clickedLngLat, graph, h3Resolution) {
    const [clickedLng, clickedLat] = clickedLngLat;

    if (!graph || Object.keys(graph).length === 0) {
        console.error("Graph is not loaded or empty.");
        return null; // No graph to search in
    }

    let closestNode = null;
    let minDistanceSq = Infinity;

    // Get the initial H3 cell for the clicked location at the desired resolution
    const initialH3 = h3.latLngToCell(clickedLat, clickedLng, h3Resolution);

    const maxK = 50; // Maximum search radius in k-rings

    for (let k = 0; k <= maxK; k++) {
        const kRingCells = h3.gridDisk(initialH3, k);

        let foundInThisRing = false;

        for (const cell of kRingCells) {
            if (graph.hasNode(cell)) {

                foundInThisRing = true;

                // Get cell lat/lng
                const [nodeLat, nodeLng] = h3.cellToLatLng(cell);

                // Squared Euclidean distance (avoid sqrt inside loop)
                const distSq = (clickedLng - nodeLng) ** 2 + (clickedLat - nodeLat) ** 2;

                if (distSq < minDistanceSq) {
                    minDistanceSq = distSq;
                    closestNode = cell;
                }
            }
        }

        // If found any node this ring and have a closestNode, stop expanding rings
        if (foundInThisRing && closestNode) {
            break;
        }
    }

    if (!closestNode) {
        console.warn("No graph nodes found within search radius.");
    }

    return closestNode;
}

function logPathAttributes(graph, path, month) {
    const METRES_TO_KM = 0.001;
    const METRES_TO_MILES = 0.000621371;
    const METRES_TO_NAUTICAL_MILES = 1 / 1852;
    const SECONDS_TO_HOURS = 1 / 3600;
    const SECONDS_TO_DAYS = 1 / (3600 * 24);

    let totalLength = 0;
    let totalTime = 0;

    for (let i = 0; i < path.length - 1; i++) {
        const sourceId = path[i];
        const targetId = path[i + 1];
        const source = graph.getNodeAttributes(sourceId);
        const target = graph.getNodeAttributes(targetId);
        const edgeAttrs = graph.getEdgeAttributes(sourceId, targetId);

        if (edgeAttrs && typeof edgeAttrs.length_m === "number") {
            totalLength += edgeAttrs.length_m;

            const time = estimateSailingTime(source, target, edgeAttrs, month, true);
            if (isFinite(time)) {
                totalTime += time;
            } else {
                console.warn(`Infinite time estimate for ${sourceId} -> ${targetId}`);
            }
        } else {
            console.warn(`Missing length_m for edge ${sourceId} -> ${targetId}`);
        }
    }

    const distanceKm = `${(totalLength * METRES_TO_KM).toFixed(0)} km`;
    const distanceMiles = `${(totalLength * METRES_TO_MILES).toFixed(0)} mi`;
    const distanceNautical = `${(totalLength * METRES_TO_NAUTICAL_MILES).toFixed(0)} nmi`;

    const totalHours = totalTime * SECONDS_TO_HOURS;
    const totalDays = Math.floor(totalTime * SECONDS_TO_DAYS);
    const remainingHours = Math.round(totalHours - totalDays * 24);

    const timeLabel = totalDays >= 1
        ? `${totalDays} day${totalDays > 1 ? 's' : ''} ${remainingHours} hr${remainingHours !== 1 ? 's' : ''}`
        : `${totalHours.toFixed(1)} hrs`;

    return {
        'Duration': timeLabel,
        'Distance': distanceKm,
        'Distance (mi)': distanceMiles,
        'Distance (nmi)': distanceNautical,
        'Nodes': path.length.toString()
    };
}

function computeOneWayRoute(source, target, month, colour) {
    try {
        console.log(`Computing route from ${source} to ${target} for month ${month}`);

        const weightFn = (edgeKey, attributes) => {
            let sourceAttrs = graph.getSourceAttributes(edgeKey);
            let targetAttrs = graph.getTargetAttributes(edgeKey);

            const temporalWeight = estimateSailingTime(sourceAttrs, targetAttrs, attributes, month);

            const bathymetry = (typeof targetAttrs.bathymetry === "number" && !isNaN(targetAttrs.bathymetry)) ? targetAttrs.bathymetry : 0;
            const draughtMultiplier = bathymetry < draughtWithTolerance ? 1000 : 1;

            return temporalWeight * draughtMultiplier;
        };

        let path;
        try {
            path = graphologyLibrary.shortestPath.dijkstra.bidirectional(graph, source, target, weightFn);
        } catch (error) {
            throw new Error(`Failed to compute path from ${source} to ${target}: ${error.message}`);
        }

        if (!path || path.length === 0) {
            throw new Error(`No path found between ${source} and ${target}.`);
        }

        drawRoute(path, colour);

        return logPathAttributes(graph, path, month - 1);

    } catch (error) {
        showToast(error.message);
        clearRoute();
        clearPointMarkers();
        return false;
    }
}

async function computeRoutes() {
    const computeStartTime = performance.now();

    if (!graph || !graph.hasNode(startPointH3) || !graph.hasNode(endPointH3)) {
        console.error("Graph not loaded or source/target not found in graph.", {
            startPointH3,
            endPointH3,
            graphLoaded: !!graph
        });
        clearRoute();
        showToast("Could not compute route. One or both selected points are not valid graph nodes or graph not loaded.");
        return;
    }

    routeBounds = new maplibregl.LngLatBounds();

    month = vesselParameters.month;
    draughtWithTolerance = vesselParameters.vesselDraughtWithTolerance;

    routeLogs = []

    await new Promise(requestAnimationFrame);

    // Add result to routeLogs
    routeLogs.push(computeOneWayRoute(startPointH3, endPointH3, month, routeColours[0]));
    updateRouteLegLogs('Outward Log', routeLogs[0], routeColours[0]);
    if (vesselParameters.return) {
        routeLogs.push(computeOneWayRoute(endPointH3, startPointH3, month, routeColours[1]));
        updateRouteLegLogs('Return Log', routeLogs[1], routeColours[1]);
    } else {
        updateRouteLegLogs('Return Log', false);
    }

    // Exit early if no logs were generated
    if (routeLogs.every(log => !log)) {
        return;
    }

    geoJsonButton = pane.addButton({title: 'Download GeoJSON'});
    const $icon = $('<span>').html(`
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" 
             width="14" height="14" style="margin-right:-6px; vertical-align:middle;">
            <path d="M5 20h14v-2H5v2zm7-18v12l4-4h-3V4h-2v6H8l4 4V2z" fill="currentColor"/>
        </svg>
    `);
    $(geoJsonButton.element)
        .attr('title', 'Includes all parameters, data source references, and route logs too!')
        .find('.tp-btnv_t').prepend($icon);
    geoJsonButton.on('click', downloadGeoJson);

    map.fitBounds(routeBounds, {
        padding: {top: 20, bottom: 20, left: pane.element.offsetWidth, right: 20},
        duration: 3000
    });

    console.log(`Total Route Computation took: ${(performance.now() - computeStartTime).toFixed(2)} ms`);
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


$(async () => {
    map = new maplibregl.Map({
        container: 'map',
        style: 'https://tiles.whgazetteer.org/styles/whg-basic-light/style.json',
        zoom: 6,
        attributionControl: {
            customAttribution: 'Map Tiles: <a target="_blank" href="https://whgazetteer.org">World Historical Gazetteer</a>',
            compact: true
        }
    });

    map.on('style.load', async () => {
        await loadMetadata();
        await polygons(
            './data/Viabundus-2-water-1500.geojson',
            map,
            'viabundus-water',
            'Water c.1500: Holterman/<a target="_blank" href="https://www.landesgeschichte.uni-goettingen.de/handelsstrassen/info.php">Viabundus</a>',
            '#c1dbea'
        );
        await loadAOIGraph();
        if (!(graph && Object.keys(graph).length > 0)) {
            console.error("Graph failed to load or is empty.");
            $('#button').prop('disabled', true).text('Graph Load Failed');
            return; // Stop execution if graph is not loaded
        }
        map.setProjection({type: 'globe'});
        $('#map').addClass('visible');
    });

    // Add map navigation controls
    map.addControl(new maplibregl.NavigationControl(), 'top-right');

    initDeck();

    // --- Map Click Listener for Point Selection ---
    map.on('click', async (e) => {
        if (!graph) return; // Ensure graph is loaded

        const clickedLngLat = [e.lngLat.lng, e.lngLat.lat];

        // Check if the clicked point is within the AOI bounds
        const {west, south, east, north} = metadata.bounds;
        if (clickedLngLat[0] < west || clickedLngLat[0] > east || clickedLngLat[1] < south || clickedLngLat[1] > north) {
            showToast("Please pick a point within the AOI bounds.");
            return;
        }

        const closestNodeH3 = findClosestGraphNode(clickedLngLat, graph, H3_RESOLUTION);

        if (!closestNodeH3) {
            showToast("Please pick a point nearer to the coast.");
            return;
        }

        const closestNodeLngLat = h3ToLngLat(closestNodeH3); // Convert back to LngLat for marker

        if (selectingStart) {
            stopMonthCycle();
            clearPointMarkers();
            updateRouteLegLogs();
            startPointH3 = closestNodeH3;
            await addPointMarker(closestNodeLngLat, '#00FF00', true); // Green for start
            selectingStart = false;
            clearRoute(); // Clear previous route
            $(instructions.element).find('button').text('Pick Destination');
        } else {
            endPointH3 = closestNodeH3;
            await addPointMarker(closestNodeLngLat, '#FF0000', false); // Red for end
            selectingStart = true; // Reset for next selection cycle
            await showSpinner("Plotting route…");
            computeRoutes();
            await hideSpinner();
            $(instructions.element).find('button').text('Pick two Points');
        }
    });
});

async function reComputeRouteIfReady() {
    if (startPointH3 && endPointH3) {
        clearRoute();
        await computeRoutes();
    }
}

async function loadAOIGraph() {
    try {
        await showSpinner("Fetching graph data…");

        const res = await fetch(`data/${aoi}/routing_graph.msgpack.gz`);
        if (!res.ok) {
            throw new Error(`Failed to fetch routing graph: ${res.status} ${res.statusText}`);
        }

        await updateSpinnerText(`Decompressing ${metadata.node_count.toLocaleString()} nodes and ${metadata.edge_count.toLocaleString()} edges…`);
        const arrayBuffer = await res.arrayBuffer();
        const compressedData = new Uint8Array(arrayBuffer);
        const decompressedData = fflate.decompressSync(compressedData);
        const graphObject = msgpack.decode(decompressedData);

        await updateSpinnerText("Building graph…");
        const {DirectedGraph} = graphology;
        graph = DirectedGraph.from(graphObject);
        // Add reversed edges
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

        console.info(`Loaded AOI graph for ${aoi} with ${graph.order} nodes and ${graph.size} edges.`);
    } catch (err) {
        console.error(e);
    } finally {
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

    const {west, south, east, north} = metadata.bounds;

    map.fitBounds([[west, south], [east, north]], {
        padding: {top: 20, bottom: 20, left: pane.element.offsetWidth, right: 20},
        duration: 3000
    });

    // Draw AOI bounds on the map
    const boundsGeojson = {
        type: 'Feature',
        geometry: {
            type: 'Polygon',
            coordinates: [[
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south]
            ]]
        }
    };

    if (map.getSource('aoi-bounds')) {
        map.getSource('aoi-bounds').setData(boundsGeojson);
    } else {
        map.addSource('aoi-bounds', {type: 'geojson', data: boundsGeojson});
        map.addLayer({
            id: 'aoi-bounds-layer',
            type: 'line',
            source: 'aoi-bounds',
            paint: {
                'line-color': '#FF000055',
                'line-width': 2,
                'line-dasharray': [2, 2]
            }
        });
    }
}
