// main.js

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

$(async () => {

    await loadMetadata();
    loadAOIGraph();
    initMap();
    initDeck();
});