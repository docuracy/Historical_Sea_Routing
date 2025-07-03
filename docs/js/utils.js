// utils.js

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


function showToast(message, duration = 3000) {
    const $toast = $('<div class="toast-message"></div>').text(message);
    $('body').append($toast);
    $toast.fadeIn(400);
    setTimeout(() => {
        $toast.fadeOut(400, () => $toast.remove());
    }, duration);
}


function isMobileDevice() {
    return /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ||
           (window.innerWidth <= 768 && window.innerHeight <= 1024);
}


function h3ToLngLat(h3Index) {
    const [lat, lng] = h3.cellToLatLng(h3Index);
    return [lng, lat];
}


function interpolate(p0, p1, denom, w1, w2) {
    return [
        (w1 * p0[0] + w2 * p1[0]) / denom,
        (w1 * p0[1] + w2 * p1[1]) / denom
    ];
}


function chaikinSmooth(coords, iterations = 3) {
    if (coords.length < 3) return coords;

    let newCoords = coords;
    for (let it = 0; it < iterations; it++) {
        const smoothed = [newCoords[0]]; // keep first point
        for (let i = 0; i < newCoords.length - 1; i++) {
            const [x0, y0] = newCoords[i];
            const [x1, y1] = newCoords[i + 1];

            const Q = [(0.75 * x0 + 0.25 * x1), (0.75 * y0 + 0.25 * y1)];
            const R = [(0.25 * x0 + 0.75 * x1), (0.25 * y0 + 0.75 * y1)];

            smoothed.push(Q, R);
        }
        smoothed.push(newCoords[newCoords.length - 1]); // keep last point
        newCoords = smoothed;
    }

    return newCoords;
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
