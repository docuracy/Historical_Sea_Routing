// main.js

import {initWorker, loadMetadata} from './utils'
import {initMap} from './map';
import {initDeck} from './quarterdeck';
import {initVesselPresets} from "./sailing_vessels";
import 'maplibre-gl/dist/maplibre-gl.css';

export let worker;

window.addEventListener('DOMContentLoaded', async () => {
    await loadMetadata();
    await initVesselPresets();
    await initDeck(); // Required by initMap
    worker = initWorker();
    initMap();
});