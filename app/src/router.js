// router.js

import {clearPointMarkers, clearRoute, drawRoute} from "./map";
import {state} from "./state";
import {showToast} from "./utils";
import {updateRouteLegLogs} from "./quarterdeck";
import maplibregl from "maplibre-gl";
import {worker} from "./main";


export function handleComputeRoute(success, error, result) {
    if (success) {
        const {featureCollection, routeBounds} = result;
        drawRoute(featureCollection, routeBounds);
        updateRouteLegLogs(featureCollection);
    } else {
        console.error(`❌ Failed to compute route: ${error}`);
        showToast(`Could not compute route: ${error}`);
    }
    state.instructions.element.querySelector('button').textContent = 'Pick two Points';
}


export async function computeRoutes() {
    state.routeBounds = new maplibregl.LngLatBounds();
    const month = state.vesselParameters.month;

    try {
        worker.postMessage({
            type: 'compute-route',
            payload: {
                source: state.startPointH3,
                target: state.endPointH3,
                colours: state.routeColours,
                vesselParameters: state.vesselParameters // Includes month and return flag
            }
        });

    } catch (error) {
        showToast(error.message);
        clearRoute();
        clearPointMarkers();
        return false;
    }
}


export async function reComputeRouteIfReady() {
    if (state.startPointH3 && state.endPointH3) {
        clearRoute();
        await computeRoutes();
    }
}
