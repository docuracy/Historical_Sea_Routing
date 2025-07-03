// sailing_deck.js

let isProgrammaticChange = false;
let vesselParameters;
let pane;
let monthInput;
let playInput;
let returnInput;
let instructions;
let routeLegFolders = {};
let geoJsonButton;


function initDeck() {

    const defaultVessel = "default";
    const defaultVesselConfig = getVesselConfig(defaultVessel);

    vesselParameters = {
        vesselType: defaultVessel,
        month: 1,
        play: false,
        return: false,
        ...defaultVesselConfig
    };

    pane = new Tweakpane.Pane({container: document.getElementById('pane-container')});

    // Vessel type dropdown
    const vesselTypeInput = pane.addInput(vesselParameters, 'vesselType', {
        options: Object.keys(vesselPresets).reduce((acc, key) => {
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

    monthInput = pane.addInput(vesselParameters, 'month', {
        options: monthOptions,
        label: 'Month of Voyage',
    });

    // Play/Pause toggle
    playInput = pane.addInput(vesselParameters, 'play', {
        label: 'Auto Cycle',
    });

    // Include return journey toggle
    returnInput = pane.addInput(vesselParameters, 'return', {
        label: 'Include Return',
    });

    // Separator
    pane.addSeparator();

    // Create a folder with a title and make it collapsible
    const numericFolder = pane.addFolder({
        title: 'Vessel & Voyage Parameters',
        expanded: false
    });

    // Add sliders/inputs for all numeric parameters
    const numericInputs = {};
    Object.keys(defaultVesselConfig).forEach((key) => {
        numericInputs[key] = numericFolder.addInput(vesselParameters, key);
    });

    instructions = pane.addButton({title: 'Pick two Points'});
    $(instructions.element).find('button').css({
        backgroundColor: '#ccff00', color: 'black',
        cursor: 'default',
    });

    // Listen for changes on vesselType to update numeric parameters
    vesselTypeInput.on('change', (ev) => {
        const selectedType = ev.value;
        const preset = getVesselConfig(selectedType);

        if (!preset) return;

        isProgrammaticChange = true;  // START suppressing recomputes

        // Update vesselParameters with new preset values
        Object.keys(preset).forEach((key) => {
            vesselParameters[key] = preset[key];
            // Update the corresponding input control in the pane
            if (numericInputs[key]) {
                numericInputs[key].value = preset[key];
                numericInputs[key].refresh(); // ensure UI updates visually
            }
        });

        pane.refresh();

        isProgrammaticChange = false; // STOP suppressing recomputes

        reComputeRouteIfReady(); // run once after all updates
    });

    monthInput.on("change", () => {
        if (!isProgrammaticChange && monthCycleInterval) {
            stopMonthCycle();
        }
        isProgrammaticChange = false;
        reComputeRouteIfReady();
    });

    playInput.on('change', (ev) => {
        if (ev.value) {
            if (startPointH3 && endPointH3) {
                startMonthCycle();
            } else {
                vesselParameters.play = false;
                playInput.value = false;  // update UI
                playInput.refresh();
            }
        } else {  // play toggled OFF
            stopMonthCycle();
        }
    });

    returnInput.on('change', (ev) => {
        vesselParameters.return = ev.value;
        reComputeRouteIfReady();
    });

    Object.values(numericInputs).forEach(input => {
        input.on('change', () => {
            if (!isProgrammaticChange) {
                reComputeRouteIfReady();
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


function logPathAttributes(graph, path, month) {
    const METRES_TO_KM = 0.001;
    const METRES_TO_MILES = 0.000621371;
    const METRES_TO_NAUTICAL_MILES = 1 / 1852;
    const SECONDS_TO_HOURS = 1 / 3600;
    const SECONDS_TO_DAYS = 1 / (3600 * 24);

    let totalLength = 0;
    let totalTime = 0;

    for (let i = 0; i < path.length - 1; i++) {
        const sourceId = path[i];
        const targetId = path[i + 1];
        const source = graph.getNodeAttributes(sourceId);
        const target = graph.getNodeAttributes(targetId);
        const edgeAttrs = graph.getEdgeAttributes(sourceId, targetId);

        if (edgeAttrs && typeof edgeAttrs.length_m === "number") {
            totalLength += edgeAttrs.length_m;

            const time = estimateSailingTime(source, target, edgeAttrs, month, true);
            if (isFinite(time)) {
                totalTime += time;
            } else {
                console.warn(`Infinite time estimate for ${sourceId} -> ${targetId}`);
            }
        } else {
            console.warn(`Missing length_m for edge ${sourceId} -> ${targetId}`);
        }
    }

    const distanceKm = `${(totalLength * METRES_TO_KM).toFixed(0)} km`;
    const distanceMiles = `${(totalLength * METRES_TO_MILES).toFixed(0)} mi`;
    const distanceNautical = `${(totalLength * METRES_TO_NAUTICAL_MILES).toFixed(0)} nmi`;

    const totalHours = totalTime * SECONDS_TO_HOURS;
    const totalDays = Math.floor(totalTime * SECONDS_TO_DAYS);
    const remainingHours = Math.round(totalHours - totalDays * 24);

    const timeLabel = totalDays >= 1
        ? `${totalDays} day${totalDays > 1 ? 's' : ''} ${remainingHours} hr${remainingHours !== 1 ? 's' : ''}`
        : `${totalHours.toFixed(1)} hrs`;

    return {
        'Duration': timeLabel,
        'Distance': distanceKm,
        'Distance (mi)': distanceMiles,
        'Distance (nmi)': distanceNautical,
        'Nodes': path.length.toString()
    };
}


function updateRouteLegLogs(legTitle = false, logs = false, colour = 'black') {
    // Remove the GeoJSON button if it exists
    if (geoJsonButton) {
        pane.remove(geoJsonButton);
        geoJsonButton = null;
    }

    // Remove all existing route leg folders if legTitle is not provided
    if (!legTitle) {
        Object.values(routeLegFolders).forEach(folder => {
            pane.remove(folder);
        });
        routeLegFolders = {};
        return;
    }

    // Remove existing folder if it exists
    if (routeLegFolders[legTitle]) {
        pane.remove(routeLegFolders[legTitle]);
        delete routeLegFolders[legTitle];
    }

    // If no logs, do not create a folder
    if (!logs || Object.keys(logs).length === 0) {
        return;
    }

    // Create a new folder
    const folder = pane.addFolder({
        title: legTitle,
        expanded: true,
    });

    const $titleEl = $(folder.element).find('.tp-fldv_t');
    if ($titleEl.length) {
        const $swatch = $('<span>').css({
            display: 'inline-block',
            width: '12px',
            height: '12px',
            background: colour,
            'border-radius': '2px',
            'margin-right': '6px',
            'vertical-align': 'middle',
        });
        $titleEl.prepend($swatch);
    }

    // Add logs as text monitors
    Object.entries(logs).forEach(([key, value]) => {
        folder.addMonitor({[key]: value}, key, {view: 'text'});
    });

    // Store reference
    routeLegFolders[legTitle] = folder;
}

function startMonthCycle() {
    if (monthCycleInterval) return;

    currentMonth = monthInput.value || 1; // Default to January if not set
    monthCycleInterval = setInterval(() => {
        currentMonth = currentMonth === 12 ? 1 : currentMonth + 1;

        isProgrammaticChange = true;
        vesselParameters.month = currentMonth;
        monthInput.value = currentMonth;
        monthInput.refresh();
    }, 5000);
}

function stopMonthCycle() {
    if (!monthCycleInterval) return;
    clearInterval(monthCycleInterval);
    monthCycleInterval = null;
    // Clear the cycle toggle
    vesselParameters.play = false;
    if (playInput) {
        playInput.value = false;  // update UI
        playInput.refresh();
    }
}
