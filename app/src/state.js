// state.js

export const state = {
    // AOI and graph data
    aoi: null,
    graph: false,
    metadata: {},

    // Click-based selection
    startPointH3: null,
    endPointH3: null,
    selectingStart: true,

    // Quarterdeck parameters
    vesselPresets: {},
    vesselParameters: null,
    pane: null,
    monthInput: null,
    currentMonth: null,
    playInput: null,
    returnInput: null,
    instructions: null,
    routeLegFolders: {},
    geoJsonButton: null,
    openOptions: true,
    openLogs: true,

    // Map and route state
    map: null,
    startMarker: null,
    endMarker: null,
    routeBounds: null,
    routeColours: ['orange', 'pink'],
    routeGeoJSON: null,

    // UI
    monthCycleInterval: null,
    isProgrammaticChange: false,
    isMobileDevice: false,
};