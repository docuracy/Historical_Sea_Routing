// main.js

import {initWorker, loadMetadata, showSpinner} from './utils'
import {initMap} from './map';
import {initDeck} from './quarterdeck';
import {getVesselConfig} from "./premodern-sailing";
import 'maplibre-gl/dist/maplibre-gl.css';

export let worker;

window.addEventListener('DOMContentLoaded', async () => {
    showSpinner();
    await loadMetadata();
    await getVesselConfig();
    await initDeck(); // Required by initMap
    worker = initWorker();
    initMap();
});