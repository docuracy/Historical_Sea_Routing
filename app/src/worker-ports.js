import wdk from 'wikidata-sdk';
import Dexie from 'dexie';
import * as turf from "@turf/turf";
import {filterAndClassifyTypes} from "./worker-ports-filter";

const SPARQL_QUERY = `
    SELECT ?port ?portLabel ?coord ?countryLabel ?typeLabel
    WHERE {
      ?port wdt:P31/wdt:P279* wd:Q44782.
      ?port wdt:P625 ?coord.
      ?port wdt:P17 ?country.
      ?port wdt:P31 ?type.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    LIMIT 10000
`;

// Dexie DB setup
const db = new Dexie('PortsDB');
db.version(3).stores({
    ports: 'wikidata, name, country, lat, lon, types, categories',
    metadata: 'key'
});

// Current data version, bump this to force refetch
const DATA_VERSION = '2025-07-05:09:32';

// Helper: save metadata
async function setMetadata(key, value) {
    await db.metadata.put({key, value});
}

async function getMetadata(key) {
    const record = await db.metadata.get(key);
    return record ? record.value : null;
}

// Fetch Wikidata ports, parse, store in DB
async function fetchAndStorePorts() {
    const url = wdk.sparqlQuery(SPARQL_QUERY);

    const res = await fetch(url);
    const data = await res.json();

    const portsMap = new Map();

    for (const item of data.results.bindings) {
        const uri = item.port.value;
        const m = item.coord.value.match(/Point\(([-\d.]+) ([-\d.]+)\)/);
        if (!m) continue;
        const [, lon, lat] = m;

        if (!portsMap.has(uri)) {
            portsMap.set(uri, {
                wikidata: uri,
                name: item.portLabel?.value ?? '',
                country: item.countryLabel?.value ?? '',
                lat: parseFloat(lat),
                lon: parseFloat(lon),
                types: new Set()
            });
        }

        if (item.typeLabel?.value) {
            portsMap.get(uri).types.add(item.typeLabel.value);
        }
    }

    const rawPorts = [...portsMap.values()].map(p => ({
        ...p,
        types: [...p.types] // convert Set to Array for Dexie
    }));

    console.debug(`Fetched ${rawPorts.length} ports from Wikidata.`);

    const ports = rawPorts
        .map(filterAndClassifyTypes)
        .filter(Boolean);

    console.debug(`Filtered and classified to ${ports.length} ports.`);

    await db.ports.clear();
    await db.ports.bulkPut(ports);
    await setMetadata('version', DATA_VERSION);
    return ports.length;
}

// Ensure data is loaded and up to date
async function ensureData() {
    const version = await getMetadata('version');
    if (version !== DATA_VERSION) {
        const count = await fetchAndStorePorts();
        postMessage({type: 'info', message: `Loaded ${count} ports from Wikidata.`});
    } else {
        postMessage({type: 'info', message: 'Using cached ports data.'});
    }

    return SPARQL_QUERY;
}

// Query ports in bbox (west,south,east,north)
async function queryPortsInBBox(west, south, east, north) {
    return await db.ports
        .where('lat').between(south, north, true, true)
        .and(port => port.lon >= west && port.lon <= east)
        .toArray();
}

// Helper: convert port object to GeoJSON Feature
function portToFeature(port) {
    return {
        type: 'Feature',
        geometry: {
            type: 'Point',
            coordinates: [port.lon, port.lat]
        },
        properties: {
            name: port.name,
            wikidata: port.wikidata,
            country: port.country,
            types: port.types,
            categories: port.categories,
        }
    };
}

// Get closest port(s) to a point with a buffer (in degrees)
async function getClosestPorts(point, bufferDeg = 0.1, maxResults = 1) {
    const [lon, lat] = point;
    const west = lon - bufferDeg;
    const east = lon + bufferDeg;
    const south = lat - bufferDeg;
    const north = lat + bufferDeg;

    // Query candidates in bounding box
    const candidates = await queryPortsInBBox(west, south, east, north);

    if (candidates.length === 0) return [];

    // Turf point for query
    const pt = turf.point([lon, lat]);

    // Map candidates to turf features with distance
    const candidatesWithDist = candidates.map(c => {
        const candidatePt = turf.point([c.lon, c.lat]);
        const dist = turf.distance(pt, candidatePt, {units: 'kilometers'});
        return {...c, dist};
    });

    // Sort by distance
    candidatesWithDist.sort((a, b) => a.dist - b.dist);

    // Return top maxResults
    return candidatesWithDist.slice(0, maxResults);
}

// Get all ports as GeoJSON FeatureCollection, optionally limited by bbox
async function getAllPortsGeoJSON(bbox) {
    if (bbox && bbox.length === 4) {
        const [west, south, east, north] = bbox;

        // Use indexed query inside instead of filtering all ports in memory
        const candidates = await queryPortsInBBox(west, south, east, north);

        return {
            type: 'FeatureCollection',
            features: candidates.map(portToFeature)
        };
    } else {
        // no bbox: return all ports
        const allPorts = await db.ports.toArray();
        return {
            type: 'FeatureCollection',
            features: allPorts.map(portToFeature)
        };
    }
}

// Listen for messages from main thread
self.addEventListener('message', async (event) => {
    const {type, payload} = event.data;

    try {
        switch (type) {
            case 'init':
                const queryString = await ensureData();
                postMessage({type: 'ready', query: queryString});
                break;

            case 'closest':
                // payload: { point: [lon, lat], bufferDeg, maxResults }
                const closest = await getClosestPorts(payload.point, payload.bufferDeg, payload.maxResults);
                postMessage({type: 'closestResult', data: closest});
                break;

            case 'all':
                // payload: { bbox: [west, south, east, north]? }
                const all = await getAllPortsGeoJSON(payload?.bbox);
                postMessage({type: 'allResult', data: all});
                break;

            default:
                postMessage({type: 'error', message: `Unknown message type: ${type}`});
        }
    } catch (err) {
        postMessage({type: 'error', message: err.message});
    }
});
