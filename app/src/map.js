// map.js

import $ from 'jquery';
import maplibregl from 'maplibre-gl';
import {state} from "./state";
import {h3ToLngLat, showToast} from "./utils";
import {computeRoutes} from "./router";
import {stopMonthCycle, updateRouteLegLogs} from "./quarterdeck";
import {worker} from "./main";
import {clusterPoints, getPortsGeoJSON, initPortsWorker, polygons} from "./map-utils";

export function initMap() {
    state.map = new maplibregl.Map({
        container: 'map',
        style: 'https://tiles.whgazetteer.org/styles/whg-basic-light/style.json',
        zoom: 6,
        attributionControl: {
            customAttribution: 'Map Tiles: <a target="_blank" href="https://whgazetteer.org">World Historical Gazetteer</a>',
            compact: true
        },
        center: [state.metadata.bounds.west + (state.metadata.bounds.east - state.metadata.bounds.west) / 2,
            state.metadata.bounds.south + (state.metadata.bounds.north - state.metadata.bounds.south) / 2],
    });

    state.map.addControl(new maplibregl.NavigationControl(), 'top-right');

    state.map.on('style.load', async () => {

        const {west, south, east, north} = state.metadata.bounds;
        state.routeBounds = new maplibregl.LngLatBounds();

        if (!state.isMobileDevice) {
            state.map.setProjection({type: 'globe'});
        }
        state.map.fitBounds([[west, south], [east, north]], {
            padding: {top: 20, bottom: 20, left: state.isMobileDevice ? 20 : state.pane.element.offsetWidth, right: 20},
        });

        await polygons(
            './data/Viabundus-2-water-1500.geojson',
            state.map,
            'viabundus-water',
            'Water c.1500: Holterman/<a target="_blank" href="https://www.landesgeschichte.uni-goettingen.de/handelsstrassen/info.php">Viabundus</a>',
            '#c1dbea'
        );

        const queryString = await initPortsWorker();
        const wikidataQueryURL = `https://query.wikidata.org/#${encodeURIComponent(queryString)}`;

        const portsGeojson = await getPortsGeoJSON([west, south, east, north]);
        console.debug(`Loaded ${portsGeojson.features.length} ports in AOI bounds.`, portsGeojson);
        await clusterPoints(
            portsGeojson,
            state.map,
            'ports',
            `Ports: <a target="_blank" href="${wikidataQueryURL}">Wikidata</a>`,
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

        if (state.map.getSource('aoi-bounds')) {
            state.map.getSource('aoi-bounds').setData(boundsGeojson);
        } else {
            state.map.addSource('aoi-bounds', {type: 'geojson', data: boundsGeojson});
            state.map.addLayer({
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

    state.map.on('click', async (e) => {
        if (!state.graph) {
            showToast("Graph is not yet fully loaded. Please wait.");
            return;
        }

        const clickedLngLat = [e.lngLat.lng, e.lngLat.lat];

        // Check if the clicked point is within the AOI bounds
        const {west, south, east, north} = state.metadata.bounds;
        if (clickedLngLat[0] < west || clickedLngLat[0] > east || clickedLngLat[1] < south || clickedLngLat[1] > north) {
            showToast("Please pick a point within the AOI bounds.");
            return;
        }

        worker.postMessage({
            type: 'find-closest-node',
            payload: {
                clickedLngLat: clickedLngLat,
                h3Resolution: state.metadata.h3_resolution
            }
        });
    });
}


export function handleFindClosestNode(success, error, result) {
    if (success) {
        const closestNodeH3 = result
        const closestNodeLngLat = h3ToLngLat(closestNodeH3); // Convert back to LngLat for marker

        if (state.selectingStart) {
            stopMonthCycle();
            clearPointMarkers();
            updateRouteLegLogs();
            state.startPointH3 = closestNodeH3;
            addPointMarker(closestNodeLngLat, '#00FF00', true); // Green for start
            state.selectingStart = false;
            clearRoute(); // Clear previous route
            $(state.instructions.element).find('button').text('Pick Destination');
        } else {
            state.endPointH3 = closestNodeH3;
            addPointMarker(closestNodeLngLat, '#FF0000', false); // Red for end
            state.selectingStart = true; // Reset for next selection cycle
            $(state.instructions.element).find('button').text('Computing Route...');
            computeRoutes();
            $(state.instructions.element).find('button').text('Pick two Points');
        }
        state.closestNodeH3 = result;
    } else {
        showToast(result.message);
    }
}


export function addPointMarker(lngLat, color = '#3BB2D0', isStart = true) {
    if (isStart) {
        if (state.startMarker) state.startMarker.remove();
        state.startMarker = new maplibregl.Marker({color: color})
            .setLngLat(lngLat)
            .addTo(state.map);
    } else {
        if (state.endMarker) state.endMarker.remove();
        state.endMarker = new maplibregl.Marker({color: color})
            .setLngLat(lngLat)
            .addTo(state.map);
    }
}


export function clearPointMarkers() {
    if (state.startMarker) {
        state.startMarker.remove();
        state.startMarker = null;
    }
    if (state.endMarker) {
        state.endMarker.remove();
        state.endMarker = null;
    }
}

export function drawRoute(featureCollection, routeBounds) {

    function findLayerBefore(prefix) {
        const layers = state.map.getStyle().layers;
        if (!layers) return undefined;
        const match = layers.find(l => l.id.startsWith(prefix));
        return match?.id;
    }

    const beforeId = findLayerBefore('original-route-layer-');

    featureCollection.features.forEach((f) => {
        const sourceName = f.properties.layer.source;
        if (state.map.getSource(sourceName)) {
            state.map.getSource(sourceName).setData(f);
        } else {
            state.map.addSource(sourceName, {type: 'geojson', data: f});
            state.map.addLayer(f.properties.layer, beforeId);
        }
    });

    // Reconstruct maplibregl.LngLatBounds from the route bounds
    const {_sw, _ne} = routeBounds;
    state.routeBounds = new maplibregl.LngLatBounds(
        new maplibregl.LngLat(_sw.lng, _sw.lat),
        new maplibregl.LngLat(_ne.lng, _ne.lat)
    );
}


export function clearRoute() {
    state.routeColours.forEach((colour) => {
        // Check and remove the original route layer and source
        if (state.map.getLayer(`original-route-layer-${colour}`)) {
            state.map.removeLayer(`original-route-layer-${colour}`);
        }
        if (state.map.getSource(`original-route-${colour}`)) {
            state.map.removeSource(`original-route-${colour}`);
        }
        // Check and remove the processed route layer and source
        if (state.map.getLayer(`processed-route-layer-${colour}`)) {
            state.map.removeLayer(`processed-route-layer-${colour}`);
        }
        if (state.map.getSource(`processed-route-${colour}`)) {
            state.map.removeSource(`processed-route-${colour}`);
        }
    })
}