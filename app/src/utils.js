// utils.js

import * as h3 from "h3-js";
import {stopMonthCycle} from "./quarterdeck";
import {state} from "./state";
import {handleFindClosestNode} from "./map";
import {handleComputeRoute} from "./router";
import {worker} from "./main";


export function initWorker() {
    if (typeof Worker === "undefined") {
        showToast("Web Workers are not supported in this environment. Please use a modern browser.");
        return;
    }

    const worker = new Worker(new URL('./worker-main.js', import.meta.url), {type: 'module'});

    worker.onmessage = (event) => {
        const {type, success, error, result} = event.data;

        if (type === 'load-graph') {
            handleLoadGraph(success, error, result);
        } else if (type === 'store-graph') {
            if (success) {
                console.info(`Graph stored successfully.`);
            } else {
                console.error(`Failed to store graph: ${error}`);
            }
        } else if (type === 'find-closest-node') {
            handleFindClosestNode(success, error, result);
        } else if (type === 'compute-route') {
            handleComputeRoute(success, error, result);
        } else {
            console.warn(`Unknown message type from worker: ${type}`);
        }
    };

    worker.postMessage({
        type: 'load-graph',
        payload: {aoi: state.aoi}
    });

    return worker;
}


function handleLoadGraph(success, error, result) {
    if (success) {
        hideSpinner();
        state.graph = result.graphStats;
        if (result.doStore) {
            worker.postMessage({
                type: 'store-graph',
                payload: {
                    graphId: `routing_graph_${state.aoi}`,
                }
            });
        }
        console.info(result.message);
        console.info(`Graph has ${result.graphStats.nodeCount} nodes and ${result.graphStats.edgeCount} edges.`);
    } else {
        console.error(result, error);
    }
}

export function updateSpinnerText(message) {
    const text = document.getElementById("spinner-text");
    if (text) text.innerHTML = message;
}

export function showSpinner(message = "Loading…") {
    updateSpinnerText(message);
    document.getElementById("spinner-overlay")?.classList.add('visible');
}

export function hideSpinner() {
    document.getElementById("spinner-overlay")?.classList.remove('visible');
}


export function showToast(message, duration = 3000) {
    const toast = document.createElement('div');
    toast.className = 'toast-message';
    toast.textContent = message;
    document.body.appendChild(toast);

    // Trigger fade in
    requestAnimationFrame(() => {
        toast.classList.add('show');
    });

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 400); // match transition duration
    }, duration);
}


export function isMobileDevice() {
    return /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ||
        (window.innerWidth <= 768 && window.innerHeight <= 1024);
}


export function h3ToLngLat(h3Index) {
    const [lat, lng] = h3.cellToLatLng(h3Index);
    return [lng, lat];
}


export function polygons(url, map, sourceName, attribution, colour = 'red', opacity = 1, outline = false) {

    return fetch(url)
        .then(response => response.json())
        .then(data => {
            map.addSource(sourceName, {
                type: 'geojson',
                data,
                attribution: attribution
            });

            // Polygon layer
            map.addLayer({
                id: sourceName,
                type: 'fill',
                source: sourceName,
                paint: {
                    'fill-color': colour,
                    'fill-opacity': opacity
                }
            });

            // Outline layer
            if (outline) {
                map.addLayer({
                    id: `${sourceName}-outline`,
                    type: 'line',
                    source: sourceName,
                    paint: {
                        'line-color': '#004080',
                        'line-width': 2
                    }
                });
            }
        });
}


export async function loadMetadata(defaultAOI = "Europe") {
    const params = new URLSearchParams(window.location.search);
    state.aoi = params.get("aoi") || defaultAOI;
    let res;

    try {
        res = await fetch(`data/${state.aoi}/metadata.json`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
        console.warn(`Failed to load metadata for "${state.aoi}", falling back to "${defaultAOI}".`);
        state.aoi = defaultAOI;
        res = await fetch(`data/${state.aoi}/metadata.json`);
    }

    state.metadata = await res.json();
    state.isMobileDevice = isMobileDevice();

    updateSpinnerText(`Loading ${state.metadata.node_count.toLocaleString()} nodes for <i>${state.aoi}</i>...`);
}


export function downloadGeoJson() {
    stopMonthCycle(); // Stop any ongoing month cycle to ensure we package the current state
    const features = [];

    state.routeColours.forEach((colour, idx) => {
        const isReturn = (idx === 1 && state.vesselParameters.return);
        const originH3 = isReturn ? state.endPointH3 : state.startPointH3;
        const destinationH3 = isReturn ? state.startPointH3 : state.endPointH3;
        const direction = isReturn ? 'return' : 'outward';

        const originalSource = state.map.getSource(`original-route-${colour}`);
        const processedSource = state.map.getSource(`processed-route-${colour}`);

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

    const routeGeoJSON = structuredClone(state.routeGeoJSON);
    routeGeoJSON.properties = {
        metadata: structuredClone(state.metadata),
        vesselParameters: structuredClone(state.vesselParameters),
    }

    const blob = new Blob([JSON.stringify(routeGeoJSON)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `route_${state.startPointH3}_${state.endPointH3}.geojson`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
