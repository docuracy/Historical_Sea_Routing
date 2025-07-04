// sailing.js

// Function to estimate the sailing time of a medieval vessel between two nodes

// Example node structure:
// {
//     "lat": 54.30561669687364,
//     "lng": -0.3155303708465397,
//     "bathymetry": 41.359935435577654,
//     "daylight_ratio": 0.3323284313946748,
//     "clear_land": 4032.931542171909,
//     "env": {
//         "1": {
//             "current_x": 0.030925552628679534,
//             "current_y": 0.00430583491381745,
//             "swell_x": -0.9411165391641211,
//             "swell_y": 0.3380823268255046,
//             "swell_height": 0.9697182881842616,
//             "swell_period": 6.877967653106486,
//             "wave_x": 0.6276355794029418,
//             "wave_y": 0.7785072764383989,
//             "wave_height": 0.6166800666990234,
//             "wave_period": 2.7232393757505737,
//             "wind_x": 2.63,
//             "wind_y": 2.1681818181818184,
//             "visibility_m": 3414
//         },
//         "2": {...
//         }
//     }
// }

// Example edge structure:
// {
//     "w": [ // Precomputed weights for each month
//         70225.72657365588,
//         80619.78995224785,
//         ... (12 values)
//     ],
//     "length_m": 2212.8259734900244,
//     "dx": -0.9984838741323828,
//     "dy": -0.055045009742827625
// }

export function estimateSailingTime(payload) {
    const {source, target, edge, month, vesselParameters, timeOnly = false} = payload;

    // Helper to safely get number values, defaulting if not valid
    function safeValue(value, defaultValue) {
        return (typeof value === "number" && !isNaN(value)) ? value : defaultValue;
    }

    // Return infinity for missing or invalid edge length
    const length = safeValue(edge.length_m, null);
    if (length === null || length <= 0) return Infinity; // Also handle non-positive length

    const dx = safeValue(edge.dx, 0);
    const dy = safeValue(edge.dy, 0);
    // Use Math.hypot for robustness when calculating edge direction,
    // in case dx/dy are very small.
    const edgeLengthHypot = Math.hypot(dx, dy);
    const edgeDir = edgeLengthHypot > 0 ? {x: dx / edgeLengthHypot, y: dy / edgeLengthHypot} : {x: 0, y: 0};


    // Safely access environment data for the given month
    const se = source.env?.[month];
    const te = target.env?.[month];

    // Helper for averaging environment properties between source and target
    const avgEnv = (prop) => (safeValue(se?.[prop], 0) + safeValue(te?.[prop], 0)) / 2;

    // --- Environmental Factors ---

    // Wind Speed and Direction
    const windX = avgEnv('wind_x');
    const windY = avgEnv('wind_y');
    const windSpeed = Math.hypot(windX, windY);
    const windDir = windSpeed > 0 ? {x: windX / windSpeed, y: windY / windSpeed} : {x: 0, y: 0};

    // Angle between vessel's intended direction and wind direction
    const cosAngle = edgeDir.x * windDir.x + edgeDir.y * windDir.y;
    const windAngleRad = Math.acos(Math.min(Math.max(cosAngle, -1), 1)); // radians, clamped for robustness

    /**
     * Calculates sail efficiency based on wind angle and speed.
     * This multiplier is applied to the _potential_ speed generated by the wind.
     * @param {number} angleRad - Angle between vessel direction and wind in radians.
     * @param {number} windSpd - Wind speed in m/s.
     * @returns {number} Efficiency multiplier (0 to vesselParameters.maxEfficiency).
     */
    function calculateSailEfficiency(angleRad, windSpd) {
        if (windSpd < vesselParameters.minWindSpeed) return vesselParameters.minWindEfficiency; // Sailing is inefficient in calm

        const deg = angleRad * 180 / Math.PI; // Convert to degrees

        // Apply minimum efficiency for "no-go" angles (too close to wind or dead downwind)
        if (deg < vesselParameters.noSailAngleMin || deg > vesselParameters.noSailAngleMax) return vesselParameters.minEfficiency;

        // Calculate drop-off from peak efficiency (90 degrees)
        const angleFromPeak = Math.abs(vesselParameters.efficiencyPeakAngle - deg);
        const normalizedAngle = Math.min(angleFromPeak, vesselParameters.efficiencyDropRange); // Cap to prevent over-dropping

        // Linear interpolation for efficiency factor
        const efficiencyFactor = 1 - (normalizedAngle / vesselParameters.efficiencyDropRange);

        // Ensure result is within vesselParameters.minEfficiency and vesselParameters.maxEfficiency
        return Math.max(vesselParameters.minEfficiency, Math.min(
            vesselParameters.minEfficiency + (vesselParameters.maxEfficiency - vesselParameters.minEfficiency) * efficiencyFactor,
            vesselParameters.maxEfficiency
        ));
    }

    const sailEfficiency = calculateSailEfficiency(windAngleRad, windSpeed);

    // --- Deriving Vessel Speed from Environmental Inputs ---

    // 1. Speed generated directly by wind acting on sails
    // Use vesselParameters.maxPossibleWindSpeed to cap the wind's effective contribution to thrust
    const effectiveWindSpeed = Math.min(windSpeed, vesselParameters.maxPossibleWindSpeed);
    let windGeneratedSpeed = effectiveWindSpeed * vesselParameters.windSpeedToVesselSpeedRatio * sailEfficiency;

    // 2. Add a minimum speed for very calm conditions (e.g., rowing, minimal drift)
    // This ensures a vessel can always move, even without strong wind.
    windGeneratedSpeed = Math.max(windGeneratedSpeed, vesselParameters.calmWaterMinVesselSpeed);

    // 3. Apply wave/swell penalty to the wind-generated speed
    const swellHeight = avgEnv('swell_height');
    const waveHeight = avgEnv('wave_height');
    const combinedWaveHeight = swellHeight + waveHeight;
    const wavePenalty = Math.min(Math.max(vesselParameters.minWavePenalty, 1 - vesselParameters.waveImpactFactor * combinedWaveHeight), vesselParameters.maxWavePenalty);
    let totalPropulsiveSpeed = windGeneratedSpeed * wavePenalty;

    // 4. Add current component along the edge (can be positive or negative)
    const currentX = avgEnv('current_x');
    const currentY = avgEnv('current_y');
    const currentAlongEdge = currentX * edgeDir.x + currentY * edgeDir.y;

    // The final speed along the edge, factoring in all propulsion and resistance
    const finalVesselSpeed = Math.min(totalPropulsiveSpeed + currentAlongEdge, vesselParameters.maxVesselSpeed || Infinity);

    // If total speed is zero or negative, the path is impossible or effectively infinite time
    if (finalVesselSpeed <= 0) return Infinity;

    // --- Calculate Base Sailing Time ---
    const sailingTime = length / finalVesselSpeed;
    if (timeOnly) return sailingTime; // Return raw travel time if requested

    // --- Apply Additional Multipliers for Full Cost (not just time) ---

    // Visibility and Light Level Penalties
    const meteorological_visibility = safeValue(te?.visibility_m, Infinity);
    const topographical_visibility = safeValue(target.clear_land, Infinity);
    const diurnal_visibility = safeValue(target.daylight_ratio, 1);

    // Check if land is visually obstructed by weather (penalty if not)
    const landIsVisible = meteorological_visibility >= topographical_visibility;
    const weatherVisibilityPenalty = landIsVisible ? 1 : vesselParameters.invisibleLandPenalty;

    // Calculate the time penalty due to light levels (e.g., night travel is slower)
    // When diurnal_visibility is 1 (full daylight), light_level_penalty = 1
    // When diurnal_visibility is 0 (full darkness), light_level_penalty = vesselParameters.darknessPenaltyFactor
    const light_level_penalty = vesselParameters.darknessPenaltyFactor - (diurnal_visibility * (vesselParameters.darknessPenaltyFactor - 1));

    // Combine all visibility-related penalties (multiplicative)
    const totalVisibilityPenalty = weatherVisibilityPenalty * light_level_penalty;

    // Bathymetry penalty (target node's depth)
    const bathymetry = safeValue(target.bathymetry, 0);
    // Apply a penalty if depth is less than vessel's draught, otherwise no penalty (1)
    const draughtPenalty = (bathymetry < vesselParameters.vesselDraughtWithTolerance) ? vesselParameters.bathymetricPenalty : 1;

    // Final weighted time (cost) for the pathfinding algorithm
    return sailingTime * totalVisibilityPenalty * draughtPenalty;
}