// router.js

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

    if (!isMobileDevice()) {
        map.fitBounds(routeBounds, {
            padding: {top: 20, bottom: 20, left: pane.element.offsetWidth, right: 20},
            duration: 3000
        });
    }

    console.log(`Total Route Computation took: ${(performance.now() - computeStartTime).toFixed(2)} ms`);
}


async function reComputeRouteIfReady() {
    if (startPointH3 && endPointH3) {
        clearRoute();
        await computeRoutes();
    }
}
