// worker-main.js

import {findClosestGraphNode, findRoute} from './worker-utils.js';
import {loadAOIGraph, storeGraph} from "./worker-io";

export let graph = null;

const messageHandlers = {
    'load-graph': async (payload) => {
        const result = await loadAOIGraph(payload);
        if (result.success) {
            graph = result.graph;
        }
        return result;
    },
    'store-graph': async (payload) => {
        const {graphId} = payload;
        try {
            await storeGraph(graphId, graph);
            return {success: true, message: `Graph stored successfully with ID: ${graphId}`};
        } catch (error) {
            console.error(`Failed to store graph: ${error}`);
            return {success: false, error: error.message};
        }
    },
    'find-closest-node': async (payload) => {
        return await findClosestGraphNode(payload, graph);
    },
    'compute-route': async (payload) => {
        return await findRoute(payload, graph);
    },
};

self.onmessage = async (event) => {
    const {type, payload} = event.data;

    try {
        if (type !== 'load-graph' && !graph) {
            throw new Error('Graph not loaded. Please load the graph first.');
        }

        // console.debug(`Worker received "${type}": ${JSON.stringify(payload)}`);
        const handler = messageHandlers[type];

        if (handler) {
            const results = await handler(payload);
            self.postMessage({type, ...results});
        } else {
            throw new Error(`Unknown message type: ${type}`);
        }

    } catch (error) {
        self.postMessage({type, success: false, payload});
    }
};