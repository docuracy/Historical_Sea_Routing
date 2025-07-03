// map.js

function initMap() {
    map = new maplibregl.Map({
        container: 'map',
        style: 'https://tiles.whgazetteer.org/styles/whg-basic-light/style.json',
        zoom: 6,
        attributionControl: {
            customAttribution: 'Map Tiles: <a target="_blank" href="https://whgazetteer.org">World Historical Gazetteer</a>',
            compact: true
        },
        center: [metadata.bounds.west + (metadata.bounds.east - metadata.bounds.west) / 2,
            metadata.bounds.south + (metadata.bounds.north - metadata.bounds.south) / 2],
    });

    map.addControl(new maplibregl.NavigationControl(), 'top-right');

    map.on('style.load', async () => {

        if (!isMobileDevice()) {
            map.setProjection({type: 'globe'});
        }

        const {west, south, east, north} = metadata.bounds;

        map.fitBounds([[west, south], [east, north]], {
            padding: {top: 20, bottom: 20, left: pane.element.offsetWidth, right: 20},
        });

        await polygons(
            './data/Viabundus-2-water-1500.geojson',
            map,
            'viabundus-water',
            'Water c.1500: Holterman/<a target="_blank" href="https://www.landesgeschichte.uni-goettingen.de/handelsstrassen/info.php">Viabundus</a>',
            '#c1dbea'
        );

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

        $('#map').addClass('visible');
    });

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
}


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