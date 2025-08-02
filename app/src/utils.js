// utils.js

import * as h3 from "h3-js";
import {stopMonthCycle} from "./quarterdeck";
import {state} from "./state";
import {handleFindClosestNode} from "./map";
import {handleComputeRoute} from "./router";
import {runStarfield} from "starfield-webgl";


(() => {
    runStarfield({});
})();

let graphLoadResult = null;
let userProceeded = false;

function tryProceed() {
    if (graphLoadResult && userProceeded) {
        proceedWithSuccess(graphLoadResult);
    }
}


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
        } else if (type === 'find-closest-node') {
            handleFindClosestNode(success, error, result);
        } else if (type === 'compute-route') {
            handleComputeRoute(success, error, result);
        } else {
            console.warn(`Unknown message type from worker: ${type}`);
        }
    };

    // Show instructions now if needed
    const shouldShowInstructions = localStorage.getItem("hideInstructions") !== "true";

    if (shouldShowInstructions) {
        showInstructions(() => {
            userProceeded = true;
            tryProceed();
        });
    } else {
        userProceeded = true;
    }

    // Trigger graph loading regardless
    worker.postMessage({
        type: 'load-graph',
        payload: {aoi: state.aoi}
    });

    return worker;
}


function showInstructions(onProceed) {
    const message = `
        <div id="instructions">
            This tool estimates plausible historical sailing routes at different times of year for a variety of square-rigged vessel types.<br><br>
            <em>Initial loading time can be 10 seconds or fewer on Chrome and Edge browsers, but is noticeably longer on Firefox.</em><br><br>
            <b>NOTE:</b> Please read the <a href="https://github.com/docuracy/Historical_Sea_Routing?tab=readme-ov-file#erutter-historical-sea-routing" target="_blank">documentation</a> for an explanation of calibration and limitations of the software. <br>
            <button id="proceed" style="margin-top: 0.5em;">OK</button>
            <button id="dismiss-instructions" style="margin-top: 0.5em;">Don't show this again</button>
        </div>
    `;

    updateSpinnerText(message, true);

    // Delay wiring up buttons until DOM is ready
    requestAnimationFrame(() => {
        const instructionsDiv = document.getElementById("instructions");
        const proceedBtn = document.getElementById("proceed");
        const dismissBtn = document.getElementById("dismiss-instructions");

        proceedBtn?.addEventListener("click", () => {
            instructionsDiv?.remove();
            onProceed();
        });

        dismissBtn?.addEventListener("click", () => {
            localStorage.setItem("hideInstructions", "true");
            instructionsDiv?.remove();
            onProceed();
        });
    });
}


function handleLoadGraph(success, error, result) {
    if (!success) {
        console.error(result, error);
        updateSpinnerText("Failed to load graph. Please check the console for details.");
        return;
    }

    graphLoadResult = result;
    tryProceed();
}


function proceedWithSuccess(result) {
    hideSpinner();
    document.getElementById('map')?.classList.add('visible');
    document.getElementById('pane-container')?.classList.add('visible');
    state.graph = result.graphStats;
    console.info(result.message);
    console.info(`Graph has ${result.graphStats.nodeCount} nodes and ${result.graphStats.edgeCount} edges.`);
}


export function updateSpinnerText(message, append = false) {
    const text = document.getElementById("spinner-text");
    const logoHTML = document.getElementById("logo")?.outerHTML || '';

    if (text) {
        if (append) {
            text.innerHTML += `<br>${message}`;
        } else {
            text.innerHTML = `${logoHTML}<br>${message}`;
        }
    }

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

    let edges_and_nodes = state.metadata.edge_count * 2 + state.metadata.node_count;
    updateSpinnerText(`Loading ${edges_and_nodes.toLocaleString()} nodes & edges for <i>${state.aoi}</i>...`);
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
