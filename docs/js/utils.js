async function showSpinner(message = "Loading…") {
    $("#spinner-text").text(message);
    $("#spinner-overlay").fadeIn(200);
}

async function updateSpinnerText(message) {
    $("#spinner-text").text(message);
}

async function hideSpinner() {
    $("#spinner-overlay").fadeOut(200);
}

function polygons(url, map, sourceName, attribution, colour = 'red', opacity = 1, outline = false) {

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
