// worker-ports-filter.js

export function filterAndClassifyTypes(port) {
    const knownCategories = {
        historical: new Set([
            'archaeological site', 'archaeological park', 'Roman archaeological site', 'Roman bridge',
            'ancient city', 'ancient monument', 'ancient port', 'historic district', 'historic site',
            'monument', 'ruins', 'Samian Ware Discovery Site', 'cultural heritage', 'cultural heritage ensemble',
            'cultural property', 'archaeological artifact museum', 'archaeological park',
            'architectural ensemble', 'architectural heritage monument', 'architectural landmark',
            'former neighbourhood', 'former settlement', 'ghost town',
            'historic district', 'historic site', 'old town', 'settlement site'
        ]),
        seaport: new Set([
            'port', 'commercial port', 'fishing port', 'military port', 'container terminal', 'cruise port',
            'dry marina', 'deep water port', 'guest harbor', 'inland port', 'historical port',
            'oil port', 'sea terminal', 'trust port', 'regional port', 'major port', 'priority port',
            'harbor', 'dock', 'wet dock', 'port city', 'port of entry', 'specified port',
            'port of refuge', 'port and harbour facilities (Japan)', 'international strategic port',
            'international hub port (Japan)'
        ]),
        settlement: new Set([
            'city', 'town', 'village', 'hamlet', 'neighborhood', 'suburb', 'urban area',
            'urban-type settlement in Russia', 'urban area in Sweden', 'quarter', 'district',
            'posyolok', 'frazione', 'ward', 'borough', 'civil parish', 'district of city', 'barrio',
            'gazetted locality of Victoria', 'locality', 'community', 'census-designated place in the United States',
            'city or town', 'city of oblast significance', 'city/town in Russia', 'neighborhood of Buenos Aires',
            'suburb/locality of Tasmania', 'village in Finland', 'village of Senegal', 'urban area in Sweden',
            'unparished area', 'large burgh'
        ]),
        naval: new Set([
            'naval base', 'naval station', 'naval air station', 'naval aeronautics base',
            'Royal Navy Dockyard', 'submarine base', 'submarine pen', 'naval arsenal'
        ]),
        industrial: new Set([
            'shipyard', 'marine oil terminal', 'oil depot', 'fuel depot',
            'liquefied natural gas terminal', 'floating regasification terminal', 'liquefaction terminal',
            'regasification terminal'
        ]),
        marina: new Set([
            'marina', 'berth', 'bathing site', 'overnight stay harbour'
        ]),
    };

    const label = port.name || '';

    // Exclude ports where label is only a Q-id, e.g. "Q12345"
    if (/^Q\d+$/.test(label.trim())) {
        return null;
    }

    const cleanedTypes = port.types.map(t => t.trim());

    const matchedCategories = Object.entries(knownCategories)
        .filter(([_, typeSet]) => cleanedTypes.some(t => typeSet.has(t)))
        .map(([category]) => category);

    if (matchedCategories.length === 0) {
        // Return null to filter out ports with no known categories
        return null;
    }

    // Define the desired order of categories by priority
    const priorityOrder = Object.keys(knownCategories);

    // Sort matchedCategories by the priority order
    matchedCategories.sort((a, b) => {
        return priorityOrder.indexOf(a) - priorityOrder.indexOf(b);
    });

    return {
        ...port,
        categories: matchedCategories,
        types: cleanedTypes
    };
}