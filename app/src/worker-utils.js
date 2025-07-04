import * as h3 from "h3-js";
import {estimateSailingTime} from "./sailing";
import dijkstra from "graphology-shortest-path/dijkstra";
import * as turf from "@turf/turf";
import maplibregl from "maplibre-gl";


function checkGraph(graph) {
    if (!graph || Object.keys(graph).length === 0) {
        console.error("Graph is either not loaded or empty.");
        return {
            success: false,
            error: "Graph is either not loaded or empty."
        }
    }
}


export async function findClosestGraphNode(payload, graph) {
    const {clickedLngLat, h3Resolution} = payload;
    const [clickedLng, clickedLat] = clickedLngLat;

    checkGraph(graph);

    let closestNode = null;
    let minDistanceSq = Infinity;

    // Get the initial H3 cell for the clicked location at the desired resolution
    const initialH3 = h3.latLngToCell(clickedLat, clickedLng, h3Resolution);

    const maxK = 50; // Maximum search radius in k-rings

    for (let k = 0; k <= maxK; k++) {
        const kRingCells = h3.gridDisk(initialH3, k);

        let foundInThisRing = false;

        for (const cell of kRingCells) {
            if (graph.hasNode(cell)) {

                foundInThisRing = true;

                // Get cell lat/lng
                const [nodeLat, nodeLng] = h3.cellToLatLng(cell);

                // Squared Euclidean distance (avoid sqrt inside loop)
                const distSq = (clickedLng - nodeLng) ** 2 + (clickedLat - nodeLat) ** 2;

                if (distSq < minDistanceSq) {
                    minDistanceSq = distSq;
                    closestNode = cell;
                }
            }
        }

        // If found any node this ring and have a closestNode, stop expanding rings
        if (foundInThisRing && closestNode) {
            break;
        }
    }

    if (closestNode) {
        return {
            success: true,
            result: closestNode
        };
    } else {
        console.warn("No graph nodes found within search radius.");
        return {
            success: false,
            result: {
                message: "Please pick a point nearer to the coast.",
            }
        };
    }
}


function h3ToLngLat(h3Index) {
    const [lat, lng] = h3.cellToLatLng(h3Index);
    return [lng, lat];
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

function buildFeatureCollection(pathNodeKeys, colour, source, target, totalUnweightedTime, totalLength, direction) {
    const coordinates = pathNodeKeys.map(h3ToLngLat);
    const originalFeature = turf.lineString(coordinates, {
        name: 'Original Path',
        layer: {
            id: `original-route-layer-${colour}`,
            type: 'line',
            source: `original-route-${colour}`,
            paint: {
                'line-color': '#00000033', // Semi-transparent black
                'line-width': 2,
                'line-dasharray': [2, 2]
            },
            origin: source,
            destination: target,
            totalUnweightedTime: totalUnweightedTime,
            totalLength: totalLength,
            direction: direction,
            pathNodeKeys: pathNodeKeys,
            colour: colour,
        },
    });

    const simplified = turf.simplify(turf.lineString(coordinates), {
        tolerance: 0.01,
        highQuality: true
    });

    const simplifiedCoords = simplified.geometry.coordinates;
    const smoothedCoords = chaikinSmooth(simplifiedCoords, 3);
    const smoothedFeature = turf.lineString(smoothedCoords, {
        name: 'Smoothed Path',
        layer: {
            id: `processed-route-layer-${colour}`,
            type: 'line',
            source: `processed-route-${colour}`,
            paint: {
                'line-color': colour,
                'line-width': 4,
            },
        },
    });


    return turf.featureCollection([
        originalFeature,
        smoothedFeature
    ]);
}


function oneWayRoute(graph, source, target, vesselParameters, direction, colour) {

    const month = vesselParameters.month;

    const weightFn = (edgeKey, attributes) => {
        let sourceAttrs = graph.getSourceAttributes(edgeKey);
        let targetAttrs = graph.getTargetAttributes(edgeKey);

        const temporalWeight = estimateSailingTime({
            source: sourceAttrs,
            target: targetAttrs,
            edge: attributes,
            month: month,
            vesselParameters: vesselParameters,
        });

        const bathymetry = (typeof targetAttrs.bathymetry === "number" && !isNaN(targetAttrs.bathymetry)) ? targetAttrs.bathymetry : 0;
        const draughtMultiplier = bathymetry < vesselParameters.vesselDraughtWithTolerance ? 1000 : 1;

        return temporalWeight * draughtMultiplier;
    };

    let pathNodeKeys;
    try {
        pathNodeKeys = dijkstra.bidirectional(graph, source, target, weightFn);

        if (!pathNodeKeys || pathNodeKeys.length === 0) {
            return {
                success: false,
                error: `No path found between ${source} and ${target}.`
            };
        }
    } catch (error) {
        return {
            success: false,
            error: `Failed to compute path from ${source} to ${target}: ${error.message}`
        };
    }

    let totalUnweightedTime = 0;
    let totalLength = 0; // To accumulate total length of the path

    // Iterate through the found path (sequence of node keys)
    for (let i = 0; i < pathNodeKeys.length - 1; i++) {
        const sourceId = pathNodeKeys[i];
        const targetId = pathNodeKeys[i + 1];
        const segmentSourceAttrs = graph.getNodeAttributes(sourceId);
        const segmentTargetAttrs = graph.getNodeAttributes(targetId);
        const edgeAttrs = graph.getEdgeAttributes(sourceId, targetId);

        if (edgeAttrs && typeof edgeAttrs.length_m === "number") {
            totalLength += edgeAttrs.length_m;

            const segmentUnweightedTime = estimateSailingTime({
                source: segmentSourceAttrs,
                target: segmentTargetAttrs,
                edge: edgeAttrs,
                month: month,
                vesselParameters: vesselParameters,
                timeOnly: true
            });

            if (isFinite(segmentUnweightedTime)) {
                totalUnweightedTime += segmentUnweightedTime;
            } else {
                console.warn(`Infinite time estimate for ${sourceId} -> ${targetId}`);
            }
        } else {
            console.warn(`Missing length_m for edge ${sourceId} -> ${targetId}`);
        }
    }

    try {

        const featureCollection = buildFeatureCollection(pathNodeKeys, colour, source, target, totalUnweightedTime, totalLength, direction);

        const bbox = turf.bbox(featureCollection);
        const southWest = new maplibregl.LngLat(bbox[0], bbox[1]);
        const northEast = new maplibregl.LngLat(bbox[2], bbox[3]);

        return {
            success: true,
            result: {
                featureCollection: featureCollection,
                routeBounds: new maplibregl.LngLatBounds(southWest, northEast),
            }
        };

    } catch (error) {
        console.error(error);
        return {
            success: false,
            error: `Failed to build feature collection: ${error.message}`,
        };
    }

}


export async function findRoute(payload, graph) {

    // payload: {
    //     start: state.startPointH3,
    //     end: state.endPointH3,
    //     colours: state.routeColours,
    //     vesselParameters: state.vesselParameters // Includes month and return flag
    // }

    checkGraph(graph);

    const {source, target, colours, vesselParameters} = payload;
    const includeReturn = vesselParameters.return || false;

    const outwardRoute = oneWayRoute(graph, source, target, vesselParameters, "outward", colours[0]);

    if (!includeReturn || !outwardRoute.success) {
        return outwardRoute;
    }

    const returnRoute = oneWayRoute(graph, target, source, vesselParameters, "return", colours[1]);
    if (!returnRoute.success) {
        return returnRoute;
    }

    return {
        success: true,
        result: {
            featureCollection: turf.featureCollection([
                ...(returnRoute.result.featureCollection.features || []),
                ...(outwardRoute.result.featureCollection.features || []),
            ]),
            routeBounds: outwardRoute.result.routeBounds.extend(returnRoute.result.routeBounds)
        }
    }

}