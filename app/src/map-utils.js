// map-utils.js


const portsWorker = new Worker(new URL('./worker-ports.js', import.meta.url), {type: 'module'});


// Simple helper to send a message and await one response of expected type
export function sendMessage(type, payload) {
    return new Promise((resolve, reject) => {
        function handler(event) {
            const msg = event.data;
            if (msg.type === 'error') {
                portsWorker.removeEventListener('message', handler);
                reject(new Error(msg.message));
            } else if (
                (type === 'init' && msg.type === 'ready') ||
                (type === 'closest' && msg.type === 'closestResult') ||
                (type === 'all' && msg.type === 'allResult')
            ) {
                portsWorker.removeEventListener('message', handler);
                resolve(msg.data);
            }
        }

        portsWorker.addEventListener('message', handler);
        portsWorker.postMessage({type, payload});
    });
}


// Initialize worker and cache (call once)
export async function initPortsWorker() {
    return new Promise((resolve, reject) => {
        const handleMessage = (event) => {
            if (event.data.type === 'info') {
                console.log(event.data.message);
            } else if (event.data.type === 'ready') {
                portsWorker.removeEventListener('message', handleMessage);
                resolve(event.data.query);
            } else if (event.data.type === 'error') {
                portsWorker.removeEventListener('message', handleMessage);
                reject(new Error(event.data.message));
            }
        };

        portsWorker.addEventListener('message', handleMessage);
        sendMessage('init');
    });
}


// Fetch closest ports to a point (lon, lat)
export async function findClosestPorts(lon, lat, bufferDeg = 0.1, maxResults = 1) {
    return await sendMessage('closest', {point: [lon, lat], bufferDeg, maxResults});
}


// Fetch all ports optionally limited to bbox [west, south, east, north]
export async function getPortsGeoJSON(bbox) {
    return await sendMessage('all', bbox ? {bbox} : {});
}

// Example usage
// (async () => {
//     await initPortsWorker();
//
//     // Find closest port to London (approx coords)
//     const closest = await findClosestPorts(-0.1276, 51.5074);
//     console.log('Closest port(s):', closest);
//
//     // Get all ports within a bounding box (e.g., Europe roughly)
//     const bbox = [-10, 35, 40, 60];
//     const geojson = await getPortsGeoJSON(bbox);
//     console.log('Ports GeoJSON:', geojson);
// })();


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


export function clusterPoints(source, map, sourceName, attribution) {
    let dataPromise;

    if (typeof source === 'string') {
        // source is a URL - fetch GeoJSON
        dataPromise = fetch(source).then(res => res.json());
    } else if (typeof source === 'object' && source.type === 'FeatureCollection') {
        dataPromise = Promise.resolve(source);
    } else {
        return Promise.reject(new Error('Source must be a URL string or a GeoJSON FeatureCollection object'));
    }

    return dataPromise.then(data => {
        const features = data.features || [];
        const clusterZoom = 4;      // clusters start appearing at zoom 4
        const distinctZoom = 6;     // distinct points and labels appear at zoom 7 and above

        map.addSource(sourceName, {
            type: 'geojson',
            data: {
                type: 'FeatureCollection',
                features: features
            },
            cluster: true,
            clusterMaxZoom: distinctZoom,
            clusterRadius: 40,
            attribution
        });

        // Cluster circles
        map.addLayer({
            id: `${sourceName}-clusters`,
            type: 'circle',
            source: sourceName,
            filter: ['has', 'point_count'],
            minzoom: clusterZoom,
            maxzoom: distinctZoom,
            paint: {
                'circle-color': 'rgba(108,117,125,0.5)',
                'circle-radius': ['step', ['get', 'point_count'], 16, 10, 20, 50, 25],
                'circle-stroke-width': 1,
                'circle-stroke-color': '#fff'
            }
        });

        // Cluster counts
        map.addLayer({
            id: `${sourceName}-cluster-count`,
            type: 'symbol',
            source: sourceName,
            filter: ['has', 'point_count'],
            minzoom: clusterZoom,
            maxzoom: distinctZoom,
            layout: {
                'text-field': ['get', 'point_count'],
                'text-font': ['Noto Sans Regular'],
                'text-size': 12
            },
            paint: {
                'text-color': '#fff'
            }
        });

        // Unclustered point styling
        map.addLayer({
            id: `${sourceName}-unclustered-point`,
            type: 'circle',
            source: sourceName,
            filter: ['!', ['has', 'point_count']],
            minzoom: distinctZoom,
            paint: {
                'circle-color': [
                    'case',

                    ['==', ['at', 0, ['get', 'categories']], 'historical'],
                    'red', // red for historical category

                    ['==', ['at', 0, ['get', 'categories']], 'settlement'],
                    'orange', // orange for settlement (city, town, village, etc.)

                    ['==', ['at', 0, ['get', 'categories']], 'seaport'],
                    'green', // green for seaport category

                    ['==', ['at', 0, ['get', 'categories']], 'naval'],
                    '#003366', // dark blue for naval

                    ['==', ['at', 0, ['get', 'categories']], 'industrial'],
                    '#666666', // grey for industrial

                    ['==', ['at', 0, ['get', 'categories']], 'marina'],
                    '#008080', // teal for marina

                    '#004080'  // default blue
                ],

                'circle-radius': [
                    'case',

                    ['==', ['at', 0, ['get', 'categories']], 'historical'],
                    7,

                    ['==', ['at', 0, ['get', 'categories']], 'settlement'],
                    5,

                    ['==', ['at', 0, ['get', 'categories']], 'seaport'],
                    5,

                    4
                ],

                'circle-stroke-width': 1,
                'circle-stroke-color': '#fff'
            }
        });

        // Labels for unclustered points, visible only at zoom >= 7
        map.addLayer({
            id: `${sourceName}-labels`,
            type: 'symbol',
            source: sourceName,
            filter: ['!', ['has', 'point_count']],
            minzoom: distinctZoom,
            layout: {
                'text-field': ['coalesce', ['get', 'title'], ['get', 'name']],
                'text-font': ['Noto Sans Regular'],
                'text-size': 12,
                'text-offset': [0, 0.8],
                'text-anchor': 'top'
            },
            paint: {
                'text-color': '#000',
                'text-halo-color': '#fff',
                'text-halo-width': 1.5,
                'text-halo-blur': 0.5
            }
        });
    });
}