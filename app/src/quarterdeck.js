// quarterdeck.js

import $ from 'jquery';
import {state} from "./state";
import {reComputeRouteIfReady} from "./router";
import {getVesselConfig} from "./sailing_vessels";
import * as Tweakpane from "tweakpane";
import {downloadGeoJson, isMobileDevice} from "./utils";

export function initDeck() {

    const defaultVessel = "default";
    const defaultVesselConfig = getVesselConfig(defaultVessel);

    if (state.isMobileDevice) {
        state.openOptions = false; // Close options pane on mobile
        state.openLogs = false; // Close logs pane on mobile
    }

    state.vesselParameters = {
        vesselType: defaultVessel,
        month: 1,
        play: false,
        return: false,
        ...defaultVesselConfig
    };

    state.pane = new Tweakpane.Pane({container: document.getElementById('pane-container')});

    const rootFolder = state.pane.addFolder({
        title: 'Options',
        expanded: state.openOptions,
    });
    const $gearIcon = $('<span class="folder-icon">').html(`
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
             width="14" height="14" style="margin-right:-6px; vertical-align:middle;">
            <path fill="currentColor" d="M19.14,12.94c0.04,-0.31 0.06,-0.63 0.06,-0.94s-0.02,-0.63 -0.06,-0.94l2.03,-1.58c0.18,-0.14 0.23,-0.41 0.12,-0.62l-1.92,-3.32c-0.11,-0.2 -0.35,-0.28 -0.56,-0.22l-2.39,0.96c-0.5,-0.38 -1.05,-0.7 -1.65,-0.94l-0.36,-2.54c-0.03,-0.22 -0.22,-0.39 -0.44,-0.39h-3.84c-0.22,0 -0.41,0.16 -0.44,0.39l-0.36,2.54c-0.6,0.24 -1.16,0.56 -1.65,0.94l-2.39,-0.96c-0.21,-0.06 -0.45,0.02 -0.56,0.22l-1.92,3.32c-0.11,0.21 -0.06,0.48 0.12,0.62l2.03,1.58c-0.04,0.31 -0.06,0.63 -0.06,0.94s0.02,0.63 0.06,0.94l-2.03,1.58c-0.18,0.14 -0.23,0.41 -0.12,0.62l1.92,3.32c0.11,0.2 0.35,0.28 0.56,0.22l2.39,-0.96c0.5,0.38 1.05,0.7 1.65,0.94l0.36,2.54c0.03,0.22 0.22,0.39 0.44,0.39h3.84c0.22,0 0.41,-0.16 0.44,-0.39l0.36,-2.54c0.6,-0.24 1.16,-0.56 1.65,-0.94l2.39,0.96c0.21,0.06 0.45,-0.02 0.56,-0.22l1.92,-3.32c0.11,-0.21 0.06,-0.48 -0.12,-0.62l-2.03,-1.58ZM12,15.5c-1.93,0 -3.5,-1.57 -3.5,-3.5s1.57,-3.5 3.5,-3.5s3.5,1.57 3.5,3.5s-1.57,3.5 -3.5,3.5Z"/>
        </svg>
    `);
    $(rootFolder.element).find('.tp-fldv_t').prepend($gearIcon);


    // Vessel type dropdown
    const vesselTypeInput = rootFolder.addInput(state.vesselParameters, 'vesselType', {
        options: Object.keys(state.vesselPresets).reduce((acc, key) => {
            const label = key.charAt(0).toUpperCase() + key.slice(1);
            acc[label] = key;
            return acc;
        }, {}),
        label: 'Vessel Type',
    });

    // Month dropdown
    const months = [...Array(12).keys()].map(i =>
        new Intl.DateTimeFormat(navigator.language || 'en-GB', {month: 'long'}).format(new Date(2025, i, 1))
    );

    const monthOptions = {};
    months.forEach((month, i) => {
        monthOptions[month] = i + 1;
    });

    state.monthInput = rootFolder.addInput(state.vesselParameters, 'month', {
        options: monthOptions,
        label: 'Month of Voyage',
    });

    // Play/Pause toggle
    state.playInput = rootFolder.addInput(state.vesselParameters, 'play', {
        label: 'Auto Cycle',
    });

    // Include return journey toggle
    state.returnInput = rootFolder.addInput(state.vesselParameters, 'return', {
        label: 'Include Return',
    });

    const portToggle = rootFolder.addInput(state, 'showPorts', {
        label: 'Show Ports',
    });

    // Create a folder with a title and make it collapsible
    const numericFolder = rootFolder.addFolder({
        title: 'Vessel & Voyage Parameters',
        expanded: false
    });

    // Add sliders/inputs for all numeric parameters
    const numericInputs = {};
    Object.keys(defaultVesselConfig).forEach((key) => {
        numericInputs[key] = numericFolder.addInput(state.vesselParameters, key);
    });

    state.instructions = state.pane.addButton({title: 'Pick two Points'});
    $(state.instructions.element).find('button').css({
        backgroundColor: '#ccff00', color: 'black',
        cursor: 'default',
    });

    // Listen for changes on vesselType to update numeric parameters
    vesselTypeInput.on('change', (ev) => {
        const selectedType = ev.value;
        const preset = getVesselConfig(selectedType);

        if (!preset) return;

        state.isProgrammaticChange = true;  // START suppressing recomputes

        // Update vesselParameters with new preset values
        Object.keys(preset).forEach((key) => {
            state.vesselParameters[key] = preset[key];
            // Update the corresponding input control in the pane
            if (numericInputs[key]) {
                numericInputs[key].value = preset[key];
                numericInputs[key].refresh(); // ensure UI updates visually
            }
        });

        state.pane.refresh();

        state.isProgrammaticChange = false; // STOP suppressing recomputes

        reComputeRouteIfReady(); // run once after all updates
    });

    state.monthInput.on("change", () => {
        if (!state.isProgrammaticChange && state.monthCycleInterval) {
            stopMonthCycle();
        }
        state.isProgrammaticChange = false;
        reComputeRouteIfReady();
    });

    state.playInput.on('change', (ev) => {
        if (ev.value) {
            if (state.startPointH3 && state.endPointH3) {
                startMonthCycle();
            } else {
                state.vesselParameters.play = false;
                state.playInput.value = false;  // update UI
                state.playInput.refresh();
            }
        } else {  // play toggled OFF
            stopMonthCycle();
        }
    });

    state.returnInput.on('change', (ev) => {
        state.vesselParameters.return = ev.value;
        reComputeRouteIfReady();
    });

    Object.values(numericInputs).forEach(input => {
        input.on('change', () => {
            if (!state.isProgrammaticChange) {
                reComputeRouteIfReady();
            }
        });
    });

    portToggle.on('change', (ev) => {
        const visible = ev.value ? 'visible' : 'none';

        const portLayerPrefix = 'ports-';
        const allLayers = state.map.getStyle().layers || [];

        allLayers
            .map(layer => layer.id)
            .filter(id => id.startsWith(portLayerPrefix))
            .forEach(layerId => {
                if (state.map.getLayer(layerId)) {
                    state.map.setLayoutProperty(layerId, 'visibility', visible);
                }
            });
    });

    const year = new Date().getFullYear();
    const $credit = $(`
      <span id="credit">
        © ${year} Stephen Gadd | 
        <a href="https://github.com/docuracy/Historical_Sea_Routing" target="_blank" rel="noopener noreferrer">
          About
        </a> 
      </span>
    `);
    $('#pane-container').append($credit);

}


function logPathAttributes(log) {
    const {totalLength, totalUnweightedTime, pathNodeKeys} = log;

    const METRES_TO_KM = 0.001;
    const METRES_TO_MILES = 0.000621371;
    const METRES_TO_NAUTICAL_MILES = 1 / 1852;
    const SECONDS_TO_HOURS = 1 / 3600;
    const SECONDS_TO_DAYS = 1 / (3600 * 24);


    const distanceKm = `${(totalLength * METRES_TO_KM).toFixed(0)} km`;
    const distanceMiles = `${(totalLength * METRES_TO_MILES).toFixed(0)} mi`;
    const distanceNautical = `${(totalLength * METRES_TO_NAUTICAL_MILES).toFixed(0)} nmi`;

    const totalHours = totalUnweightedTime * SECONDS_TO_HOURS;
    const totalDays = Math.floor(totalUnweightedTime * SECONDS_TO_DAYS);
    const remainingHours = Math.round(totalHours - totalDays * 24);

    const timeLabel = totalDays >= 1
        ? `${totalDays} day${totalDays > 1 ? 's' : ''} ${remainingHours} hr${remainingHours !== 1 ? 's' : ''}`
        : `${totalHours.toFixed(1)} hrs`;

    return {
        'Duration': timeLabel,
        'Distance': distanceKm,
        'Distance (mi)': distanceMiles,
        'Distance (nmi)': distanceNautical,
        'Nodes': pathNodeKeys.length.toString()
    };
}


export function updateRouteLegLogs(featureCollection = null) {

    // Remove the GeoJSON button if it exists
    if (state.geoJsonButton) {
        state.pane.remove(state.geoJsonButton);
        state.geoJsonButton = null;
    }

    // Remove all existing route leg folders
    if (state.routeLegFolders) {
        Object.values(state.routeLegFolders).forEach(folder => {
            state.pane.remove(folder);
        });
        state.routeLegFolders = {};
    }

    // Remove all existing route leg folders
    if (!featureCollection) {
        return
    }

    const logs = [...featureCollection.features]
        .reverse()
        .filter(f => {
            return f.properties?.layer?.id?.startsWith("original-route-layer-");
        });


    // Create folders for each log
    logs.forEach((log) => {
        const direction = log.properties?.layer?.direction
        const legTitle = direction.charAt(0).toUpperCase() + direction.slice(1);

        // Create a new folder
        const folder = state.pane.addFolder({
            title: legTitle,
            expanded: state.openLogs,
        });

        const $titleEl = $(folder.element).find('.tp-fldv_t');
        if ($titleEl.length) {
            const $swatch = $('<span>').css({
                display: 'inline-block',
                width: '12px',
                height: '12px',
                background: log.properties?.layer?.colour,
                'border-radius': '2px',
                'margin-right': '6px',
                'vertical-align': 'middle',
            });
            $titleEl.prepend($swatch);
        }

        $(folder.element).on('click', (e) => {
            state.openLogs = $(e.target).closest('.tp-fldv').hasClass('tp-fldv-expanded');
        });

        // Add logs as text monitors
        const formattedLog = logPathAttributes(log.properties?.layer);

        Object.entries(formattedLog).forEach(([key, value]) => {
            folder.addMonitor({[key]: value}, key, {view: 'text'});
        });

        // Store reference
        state.routeLegFolders[legTitle] = folder;
    });

    state.geoJsonButton = state.pane.addButton({title: 'Download GeoJSON'});
    const $icon = $('<span>').html(`
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
                 width="14" height="14" style="margin-right:-6px; vertical-align:middle;">
                <path d="M5 20h14v-2H5v2zm7-18v12l4-4h-3V4h-2v6H8l4 4V2z" fill="currentColor"/>
            </svg>
        `);
    $(state.geoJsonButton.element)
        .attr('title', 'Includes all parameters, data source references, and route logs too!')
        .find('.tp-btnv_t').prepend($icon);
    state.geoJsonButton.on('click', downloadGeoJson);
    state.routeGeoJSON = featureCollection; // Store the route GeoJSON in state

    if (!isMobileDevice()) {
        state.map.fitBounds(state.routeBounds, {
            padding: {
                top: 20,
                bottom: 20,
                left: state.isMobileDevice ? 20 : state.pane.element.offsetWidth,
                right: 20
            },
            duration: 3000
        });
    }


}

function startMonthCycle() {
    if (state.monthCycleInterval) return;

    state.currentMonth = state.monthInput.value || 1; // Default to January if not set
    state.monthCycleInterval = setInterval(() => {
        state.currentMonth = state.currentMonth === 12 ? 1 : state.currentMonth + 1;

        state.isProgrammaticChange = true;
        state.vesselParameters.month = state.currentMonth;
        state.monthInput.value = state.currentMonth;
        state.monthInput.refresh();
    }, 5000);
}

export function stopMonthCycle() {
    if (!state.monthCycleInterval) return;
    clearInterval(state.monthCycleInterval);
    state.monthCycleInterval = null;
    // Clear the cycle toggle
    state.vesselParameters.play = false;
    if (state.playInput) {
        state.playInput.value = false;  // update UI
        state.playInput.refresh();
    }
}
