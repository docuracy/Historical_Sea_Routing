let map;
let graph;
let H3_RESOLUTION; // To store the resolution of H3 cells in our graph

// Variables for click-based point selection
let startPointH3 = null;
let endPointH3 = null;
let selectingStart = true; // True if selecting start, false for end

// MapLibre markers
let startMarker = null;
let endMarker = null;

let monthCycleInterval = null;

let routeColours = ['orange', 'pink']
let routeBounds = new maplibregl.LngLatBounds();

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
function addPointMarker(lngLat, color = '#3BB2D0', isStart = true) {
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
            const sourceAttrs = graph.getSourceAttributes(edgeKey);
            const targetAttrs = graph.getTargetAttributes(edgeKey);
            const temporalWeight = estimateSailingTime(sourceAttrs, targetAttrs, attributes, month);

            const bathymetry = (typeof targetAttrs.bathymetry === "number" && !isNaN(targetAttrs.bathymetry)) ? targetAttrs.bathymetry : 0;
            const draughtMultiplier = bathymetry < draughtWithTolerance ? 1000 : 1;

            return temporalWeight * draughtMultiplier;
        };

        const path = graphologyLibrary.shortestPath.dijkstra.bidirectional(graph, source, target, weightFn);

        if (!path || path.length === 0) {
            console.warn(`No path found between ${source} and ${target}.`);
            clearRoute();
            alert("No path found between the selected points.");
            return null;
        }

        drawRoute(path, colour);

        return logPathAttributes(graph, path, month - 1);

    } catch (error) {
        console.error("Error computing one-way route:", error);
        clearRoute();
        alert("An error occurred while computing the route. Check console for details.");
        return null;
    }
}


function computeRoutes() {
    const computeStartTime = performance.now();

    if (!graph || !graph.hasNode(startPointH3) || !graph.hasNode(endPointH3)) {
        console.error("Graph not loaded or source/target not found in graph.", {
            startPointH3,
            endPointH3,
            graphLoaded: !!graph
        });
        clearRoute();
        alert("Could not compute route. One or both selected points are not valid graph nodes or graph not loaded.");
        console.log(`Route Computation (Invalid Input) took: ${(performance.now() - computeStartTime).toFixed(2)} ms`);
        return;
    }

    routeBounds = new maplibregl.LngLatBounds();

    month = vesselParameters.month;
    draughtWithTolerance = vesselParameters.vesselDraughtWithTolerance;

    updateRouteLegLogs('Outward Log', computeOneWayRoute(startPointH3, endPointH3, month, routeColours[0]), routeColours[0]);
    if (vesselParameters.return) {
        updateRouteLegLogs('Return Log', computeOneWayRoute(endPointH3, startPointH3, month, routeColours[1]), routeColours[1]);
    } else {
        updateRouteLegLogs('Return Log', false);
    }

    map.fitBounds(routeBounds, {
        padding: {top: 20, bottom: 20, left: pane.element.offsetWidth, right: 20},
        duration: 3000
    });

    console.log(`Total Route Computation took: ${(performance.now() - computeStartTime).toFixed(2)} ms`);
}


$(async () => {
    map = new maplibregl.Map({
        container: 'map',
        style: 'https://tiles.whgazetteer.org/styles/whg-basic-light/style.json',
        center: [-1.5, 52],
        zoom: 6,
    });

    // Add map navigation controls
    map.addControl(new maplibregl.NavigationControl(), 'top-right');

    initDeck();

    graph = await loadAOIGraph();
    if (graph && Object.keys(graph).length > 0) {
        H3_RESOLUTION = 7;
        console.log('Graph loaded and H3 Resolution inferred:', H3_RESOLUTION);
    } else {
        console.error("Graph failed to load or is empty.");
        $('#button').prop('disabled', true).text('Graph Load Failed');
        return; // Stop execution if graph is not loaded
    }

    // --- Map Click Listener for Point Selection ---
    map.on('click', async (e) => {
        if (!graph) return; // Ensure graph is loaded


        const clickedLngLat = [e.lngLat.lng, e.lngLat.lat];
        const closestNodeH3 = findClosestGraphNode(clickedLngLat, graph, H3_RESOLUTION);

        if (!closestNodeH3) {
            console.warn("Could not find a closest graph node for the clicked location.");
            alert("No nearby graph node found. Please click closer to a known route.");
            return;
        }

        const closestNodeLngLat = h3ToLngLat(closestNodeH3); // Convert back to LngLat for marker
        console.log("Closest graph node selected:", closestNodeH3, "at coordinates:", closestNodeLngLat);

        if (selectingStart) {
            stopMonthCycle();
            clearPointMarkers();
            updateRouteLegLogs();
            startPointH3 = closestNodeH3;
            addPointMarker(closestNodeLngLat, '#00FF00', true); // Green for start
            selectingStart = false;
            clearRoute(); // Clear previous route
            $(instructions.element).find('button').text('Pick Destination');
        } else {
            endPointH3 = closestNodeH3;
            addPointMarker(closestNodeLngLat, '#FF0000', false); // Red for end
            selectingStart = true; // Reset for next selection cycle
            computeRoutes();
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

async function loadAOIGraph(aoi = "UK-Eire") {
    const res = await fetch(`data/${aoi}/routing_graph.json.gz`);
    const compressedData = new Uint8Array(await res.arrayBuffer());
    const decompressedData = fflate.decompressSync(compressedData);
    const jsonText = new TextDecoder("utf-8").decode(decompressedData);
    const json = JSON.parse(jsonText);

    let graph = graphology.Graph.from(json);

    console.info(`Loaded AOI graph for ${aoi} with ${graph.order} nodes and ${graph.size} edges.`);
    return graph;
}
