import {state} from "./state";

const largeValue = Infinity // 1e12; // If Infinity causes problems in calculations, use a very large number instead

// --- 1. VESSEL HYDRODYNAMICS & GEOMETRY CALCULATIONS ---
// Contains functions to derive key physical properties of the vessel based on its input dimensions and load.
const VesselHydrodynamics = {
    RHO_SEAWATER: 1025, // kg/m^3 (average density of seawater)
    KINEMATIC_VISCOSITY_WATER: 1.188e-6, // m^2/s at 15C (kinematic viscosity of water)
    G: 9.81, // m/s^2 (acceleration due to gravity)

    /**
     * Calculates the actual draught of the vessel given its design parameters and current cargo.
     * @param {object} vesselParameters - Contains lengthOverall, beam, hullCoefficient, designDraught, lightshipMass, nominalBurthen.
     * @param {number} cargoWeight - Current cargo weight in tons.
     * @returns {number} Actual draught in meters.
     */
    getActualDraught: function (vesselParameters, cargoWeight) {
        const LoA = vesselParameters.lengthOverall;
        const Beam = vesselParameters.beam;
        const HullCoefficient = vesselParameters.hullCoefficient;
        const lightshipMass = vesselParameters.lightshipMass; // Assumed in tons for consistency with cargoWeight

        const totalMass_kg = (lightshipMass + cargoWeight) * 1000; // Convert tons to kg

        // Displacement Volume V = Total Mass / Density
        const actualDisplacementVolume = totalMass_kg / this.RHO_SEAWATER;

        // V = L * B * T * Cb => T = V / (L * B * Cb)
        // Add a small epsilon to denominator to prevent division by zero for invalid vessel params
        const denominator = LoA * Beam * HullCoefficient;
        if (denominator <= 0) return vesselParameters.designDraught; // Fallback to design draught or 0

        const calculatedDraught = actualDisplacementVolume / denominator;

        // Ensure draught is not excessively small or large compared to design,
        // though the mass calculation should keep it reasonable.
        return Math.max(0.1, calculatedDraught); // Minimum draught to avoid division by zero or unrealistic scenarios
    },

    /**
     * Estimates the wetted surface area of the hull.
     * Uses a simplified empirical formula.
     * @param {object} vesselParameters - Contains lengthOverall, beam.
     * @param {number} actualDraught - The calculated actual draught.
     * @returns {number} Wetted surface area in square meters.
     */
    getWettedSurfaceArea: function (vesselParameters, actualDraught) {
        const LoA = vesselParameters.lengthOverall;
        const actualDisplacementVolume = vesselParameters.actualDisplacementVolume; // Use the passed value

        if (LoA <= 0 || actualDisplacementVolume <= 0) return 0; // Prevent sqrt(negative) or div by zero

        const Cw = 2.8; // A typical coefficient for merchant ships.
        return Cw * Math.sqrt(actualDisplacementVolume * LoA);
    },

    /**
     * Calculates the Frictional Resistance Coefficient (Cf) using ITTC 1957 formula.
     * @param {number} vesselSpeed - Speed of the vessel through water (m/s).
     * @param {number} LoA - Length Overall of the vessel (m).
     * @returns {number} Frictional resistance coefficient.
     */
    getFrictionalResistanceCoefficient: function (vesselSpeed, LoA) {
        if (vesselSpeed <= 0 || LoA <= 0) return 0;
        const ReynoldsNumber = (vesselSpeed * LoA) / this.KINEMATIC_VISCOSITY_WATER;
        if (ReynoldsNumber < 1e5) return 0.001; // Avoid issues with very small Re
        return 0.075 / Math.pow(Math.log10(ReynoldsNumber) - 2, 2);
    },

    /**
     * Calculates the Froude Number.
     * @param {number} vesselSpeed - Speed of the vessel through water (m/s).
     * @param {number} LoA - Length Overall of the vessel (m).
     * @returns {number} Froude Number.
     */
    getFroudeNumber: function (vesselSpeed, LoA) {
        if (LoA <= 0) return 0;
        return vesselSpeed / Math.sqrt(this.G * LoA);
    }
};

// --- 2. SAILING MECHANICS (Square Sails Specific) ---
// Contains functions related to thrust generation from sails.
const SailingMechanics = {
    RHO_AIR: 1.225, // kg/m^3 (density of air)

    /**
     * Calculates sail thrust efficiency factor for SQUARE SAILS based on apparent wind angle.
     * This factor is applied to the maximum potential thrust.
     * @param {number} apparentWindAngleRad - Absolute angle between vessel's direction and apparent wind in radians.
     * 0 rad = directly against apparent wind, PI rad = directly with apparent wind.
     * @param {number} apparentWindSpeed - Apparent wind speed in m/s.
     * @param {object} vesselParameters - Contains MinWindSpeedForSailing, NoGoAngle, MaxSailEfficiencyAngle, SailEfficiencyFactor.
     * @returns {number} Efficiency factor (0 to 1).
     */
    calculateSquareSailEfficiency: function (apparentWindAngleRad, apparentWindSpeed, vesselParameters) {
        if (apparentWindSpeed < vesselParameters.minWindSpeedForSailing) {
            return 0; // No effective thrust from sails in calm conditions
        }

        const deg = apparentWindAngleRad * 180 / Math.PI;
        const effectiveNoGoAngleDeg = vesselParameters.noGoAngle * 180 / Math.PI;
        const peakEfficiencyAngleDeg = vesselParameters.maxSailEfficiencyAngle * 180 / Math.PI;

        let angleEfficiency;
        if (deg < effectiveNoGoAngleDeg) {
            // Too close to the apparent wind for square sails
            angleEfficiency = 0.2 * Math.cos(Math.PI * (effectiveNoGoAngleDeg - deg) / effectiveNoGoAngleDeg);
            angleEfficiency = Math.max(0, angleEfficiency);
        } else if (deg <= peakEfficiencyAngleDeg) {
            // Linear increase from no-go angle to peak efficiency angle
            angleEfficiency = (deg - effectiveNoGoAngleDeg) / (peakEfficiencyAngleDeg - effectiveNoGoAngleDeg);
        } else {
            // Taper slightly downwind beyond peak
            const downwindDrop = (180 - deg) / (180 - peakEfficiencyAngleDeg);
            angleEfficiency = Math.max(0.7, Math.pow(downwindDrop, 0.8)); // Maintain at least 70% efficiency
        }

        angleEfficiency = Math.min(1, Math.max(0, angleEfficiency)); // Clamp between 0 and 1

        return angleEfficiency * vesselParameters.sailEfficiencyFactor;
    },

    /**
     * Calculates the total thrust generated by the sails.
     * @param {object} vesselParameters - Contains sailArea, sailEfficiencyFactor, minWindSpeedForSailing.
     * @param {number} apparentWindSpeed - Speed of wind relative to vessel (m/s).
     * @param {number} apparentWindAngleRad - Angle of apparent wind relative to vessel's heading (radians).
     * @returns {number} Total thrust in Newtons.
     */
    getSailThrust: function (vesselParameters, apparentWindSpeed, apparentWindAngleRad) {
        const sailAngleEfficiency = this.calculateSquareSailEfficiency(
            apparentWindAngleRad,
            apparentWindSpeed,
            vesselParameters
        );

        // Basic thrust formula: 0.5 * rho_air * SailArea * V_apparent^2 * SailEfficiency
        // The SailEfficiency already incorporates a coefficient for sail force.
        return 0.5 * this.RHO_AIR * vesselParameters.sailArea *
            Math.pow(apparentWindSpeed, 2) * sailAngleEfficiency;
    }
};

// --- 3. MAIN ESTIMATE SAILING TIME FUNCTION ---
/**
 * Estimates the sailing time of a medieval vessel between two nodes,
 * incorporating vessel dimensions, cargo, and varying environmental conditions.
 */
export function estimateSailingTime(payload) {
    const {source, target, edge, month, vesselParameters, timeOnly = false} = payload;

    // --- Environmental conditions at the target node ---
    const {
        b: bathymetry,  // Depth below sea level (m)
        c: land_sight,  // Distance to nearest visible land (m)
        d: land_distance  // Distance to nearest land (m)
    } = target

    // --- Edge geometry and environment ---
    const {
        L: edge_length,
        a: edge_angle,
        v, // Meteorological visibility (m) per month
        D, // Diurnal visibility factor (percentage) per month
        f // Flow: wind and current conditions per month
    } = edge;

    if (!f || !f[month-1]) {
        console.warn(`Missing environmental data for edge ${source.id} → ${target.id} in month ${month}`);
        console.debug(`Edge data:`, edge);
    }


    const {
        wA: wind_angle,      // Wind direction (degrees)
        wM: wind_mag,        // Wind speed (m/s)
        cA: current_angle,   // Current direction (degrees)
        cM: current_mag      // Current speed (m/s)
    } = f[month-1];

    const visibility = v[month-1];  // Meteorological visibility (m)
    const daylight = D[month-1];  // Diurnal visibility factor (percentage)

    // Vessel parameters
    const {
        lengthOverall: LoA,
        beam: Beam,
        minWindSpeedForSailing: MinWindSpeedForSailing,
        maxStructuralSpeed: MaxStructuralSpeed,
        calmWaterMinSpeed: CalmWaterMinSpeed,
        waveImpactFactor: WaveImpactFactor,
        currentLateralDragFactor: CurrentLateralDragFactor,
        actualDraught,
        lightshipMass,
        cargoWeight,
        wettedSurfaceArea,
        invisibleLandPenalty,
        darknessPenaltyFactor,
        bathymetricPenalty
    } = vesselParameters;

    // --- Calculate vessel displacement volume (m³) from total mass ---
    const actualDisplacementVolume = (lightshipMass + (cargoWeight || 0)) * 1000 / VesselHydrodynamics.RHO_SEAWATER;

    // --- ITERATIVE SOLUTION FOR VESSEL SPEED THROUGH WATER ---
    const MAX_ITERATIONS = 15;
    const TOLERANCE = 0.005;

    let currentVesselSpeed = Math.max(CalmWaterMinSpeed, wind_mag * 0.1); // Initial guess

    for (let i = 0; i < MAX_ITERATIONS; i++) {
        // Compute vessel velocity vector
        const vesselVelocity = {
            x: currentVesselSpeed * Math.cos(edge_angle),
            y: currentVesselSpeed * Math.sin(edge_angle)
        };

        // Compute apparent wind vector and speed
        const apparentWindX = wind_mag * Math.cos(wind_angle) - vesselVelocity.x;
        const apparentWindY = wind_mag * Math.sin(wind_angle) - vesselVelocity.y;
        const apparentWindSpeed = Math.hypot(apparentWindX, apparentWindY);

        // Angle between apparent wind and vessel direction
        const cosApparentAngle = apparentWindSpeed > 0
            ? (Math.cos(edge_angle) * (apparentWindX / apparentWindSpeed) + Math.sin(edge_angle) * (apparentWindY / apparentWindSpeed))
            : 1;
        const apparentWindAngleRad = Math.acos(Math.min(Math.max(cosApparentAngle, -1), 1));

        // Estimate wind thrust from sails
        let windThrust = SailingMechanics.getSailThrust(vesselParameters, apparentWindSpeed, apparentWindAngleRad);

        // Minimum propulsion fallback
        if (windThrust < 1 && apparentWindSpeed < MinWindSpeedForSailing) {
            windThrust += 0.5 * VesselHydrodynamics.RHO_SEAWATER * wettedSurfaceArea * CalmWaterMinSpeed ** 2;
        }

        // --- Resistance components ---
        const Cf = VesselHydrodynamics.getFrictionalResistanceCoefficient(currentVesselSpeed, LoA);
        const frictionalResistance = 0.5 * VesselHydrodynamics.RHO_SEAWATER * wettedSurfaceArea *
            currentVesselSpeed ** 2 * Cf;

        const FroudeNumber = VesselHydrodynamics.getFroudeNumber(currentVesselSpeed, LoA);
        let waveMakingResistance = 0;
        if (FroudeNumber > 0.1) {
            waveMakingResistance = 0.5 * VesselHydrodynamics.RHO_SEAWATER * actualDisplacementVolume * VesselHydrodynamics.G *
                FroudeNumber ** 3 * WaveImpactFactor;
        }

        const effectiveFrontalArea = Beam * (LoA * 0.1);
        const airResistance = 0.5 * SailingMechanics.RHO_AIR * effectiveFrontalArea * apparentWindSpeed ** 2 * 0.8;

        // Current-induced lateral drag
        const currentDir = {x: Math.cos(current_angle), y: Math.sin(current_angle)};
        const cosAngleCurrentToVessel = current_mag > 0
            ? Math.cos(edge_angle) * currentDir.x + Math.sin(edge_angle) * currentDir.y
            : 1;
        const angleCurrentToVesselRad = Math.acos(Math.min(Math.max(cosAngleCurrentToVessel, -1), 1));
        const currentCrossComponent = current_mag * Math.sin(angleCurrentToVesselRad);
        const maxLateralDrag = 0.5 * VesselHydrodynamics.RHO_SEAWATER * LoA * actualDraught * current_mag ** 2;
        const currentAddedResistance = Math.min(
            CurrentLateralDragFactor * currentCrossComponent ** 2 *
            VesselHydrodynamics.RHO_SEAWATER * LoA * actualDraught,
            maxLateralDrag * 0.5
        );

        const totalResistance = frictionalResistance + waveMakingResistance + airResistance + currentAddedResistance;

        // --- Update speed estimate ---
        let nextVesselSpeed;
        if (totalResistance <= 0) {
            nextVesselSpeed = MaxStructuralSpeed;
        } else {
            const ratio = windThrust / totalResistance;
            nextVesselSpeed = currentVesselSpeed * Math.pow(ratio, 0.2);
        }

        // Clamp to physical bounds
        nextVesselSpeed = Math.min(nextVesselSpeed, MaxStructuralSpeed);
        nextVesselSpeed = Math.max(nextVesselSpeed, CalmWaterMinSpeed);

        if (Math.abs(nextVesselSpeed - currentVesselSpeed) < TOLERANCE) {
            currentVesselSpeed = nextVesselSpeed;
            break;
        }
        currentVesselSpeed = nextVesselSpeed;
    }

    // === FINAL SPEED OVER GROUND ===
    const edgeDir = {x: Math.cos(edge_angle), y: Math.sin(edge_angle)};
    const currentX = current_mag * Math.cos(current_angle);
    const currentY = current_mag * Math.sin(current_angle);
    const currentAlongEdge = currentX * edgeDir.x + currentY * edgeDir.y;

    let finalSpeedOverGround = currentVesselSpeed + currentAlongEdge;

    if (finalSpeedOverGround <= 0) {
        finalSpeedOverGround = 0.1; // 0.1 m/s minimal drift speed rather than completely stalled
    }

    const sailingTime = edge_length / finalSpeedOverGround;
    if (timeOnly) return sailingTime;

    // === VISIBILITY AND LIGHT PENALTIES ===
    const landIsVisible = visibility >= land_sight;
    const weatherVisibilityPenalty = landIsVisible ? 1 : invisibleLandPenalty;
    const daylightFraction = Math.max(0, Math.min(daylight, 1));  // Clamp to [0, 1]
    const darknessPenalty = 1 + (darknessPenaltyFactor - 1) * (1 - daylightFraction);
    const totalVisibilityPenalty = weatherVisibilityPenalty * darknessPenalty;

    // === SHORELINE PROXIMITY PENALTY ===
    const idealLandDistance = 5000;           // Ideal clearance from shore (metres)
    const maxLandPenaltyMultiplier = 3.0;     // Maximum penalty when directly adjacent to shore
    let landProximityPenalty = 1;
    if (land_distance < idealLandDistance) {
        const proximityFactor = 1 - (land_distance / idealLandDistance);  // 0 near ideal, 1 near shore
        landProximityPenalty = 1 + (maxLandPenaltyMultiplier - 1) * (proximityFactor ** 2);
    }

    // === DEPTH PENALTY (DRAUGHT CONSTRAINT) ===
    const depthTolerance = 1.1;
    const draughtPenalty = (bathymetry < (actualDraught * depthTolerance)) ? bathymetricPenalty : 1;

    return sailingTime * totalVisibilityPenalty * landProximityPenalty * draughtPenalty;
}


state.vesselPresets = {
    "cog": {
        lengthOverall: 20, // metres (Length Over All)
        beam: 6,           // metres (maximum width)
        designDraught: 2.5,  // metres (draught at nominal/design load)
        nominalBurthen: 80, // tons (e.g., 1 tun = 224 gallons wine, roughly 1 metric ton. So 80 tons burthen)
        hullCoefficient: 0.6, // Block coefficient (fuller hull shape for cargo)
        lightshipMass: 40, // tons (mass of empty vessel) - crucial for total displacement calculation

        sailArea: 250,     // square metres (estimated total sail area, e.g., one large square sail, or a few smaller ones)
        sailEfficiencyFactor: 0.6, // Overall efficiency of sails (lower for less refined medieval sails)
        minWindSpeedForSailing: 2.5, // m/s, minimum wind speed for effective sailing (around 5 knots)
        maxSailEfficiencyAngle: Math.PI, // Radians (180 degrees, dead downwind, where square sails are very efficient)
        noGoAngle: Math.PI / 2.5, // Radians (approx 72 degrees) - square riggers are very poor at sailing into wind

        waveImpactFactor: 0.1, // General factor for how much waves generate added resistance
        currentLateralDragFactor: 0.6, // Higher for medieval ships due to broad, deep hulls when hit sideways
        maxAddedWaveResistanceCoeff: 0.1, // Higher for blunter medieval bows in head seas

        calmWaterMinSpeed: 0.1, // m/s, minimum speed (e.g., rowing, minimal drift)
        maxStructuralSpeed: 5, // m/s, typical max speed for such vessels (around 10 knots)
        maxDraftTolerance: 0.3, // metres, how much deeper than designDraught is acceptable before major penalty.

        invisibleLandPenalty: 0.5, // Multiplier for speed when land is not visible
        darknessPenaltyFactor: 0.7, // Multiplier for speed in darkness
        bathymetricPenalty: 0.05, // Multiplier for speed if draught is exceeded significantly (very slow or stuck)

        nominalSpeed: 3, // m/s, nominal speed for hydrodynamics estimates (around 6 knots)
    },
    //////////////////////////////////////////////////////////////////
    // The following types may override the default cog parameters. //
    //////////////////////////////////////////////////////////////////
    "barge": { // Simple, often flat-bottomed river/coastal vessel, sometimes with simple sail
        lengthOverall: 15,
        beam: 4,
        designDraught: 1.0, // Very shallow draft
        nominalBurthen: 30,
        hullCoefficient: 0.75, // Very full and often boxy
        lightshipMass: 15,
        sailArea: 80, // Small, simple sail (often one square or sprit)
        sailEfficiencyFactor: 0.4, // Low efficiency, often auxiliary or for light winds
        minWindSpeedForSailing: 1.5, // Can operate in lighter winds due to simple rigging
        maxSailEfficiencyAngle: Math.PI,
        noGoAngle: Math.PI / 2.2, // Very wide no-go zone (~82 degrees) due to hull shape
        waveImpactFactor: 0.15, // High impact from waves on flat bottom
        currentLateralDragFactor: 0.8, // Very high due to flat sides
        maxAddedWaveResistanceCoeff: 0.2, // High for blunt forms in waves
        calmWaterMinSpeed: 0.05, // Can drift very slowly
        maxStructuralSpeed: 2.5, // Very slow top speed
        maxDraftTolerance: 0.2,
        bathymetricPenalty: 0.01,
    },// Very sensitive to depth, easily grounded
    "caravel": { // 15th-16th Century exploration ship, often with Lateen sails (but we'll model as square for consistency)
        // NOTE: If truly modeling Lateen, noGoAngle would be much smaller (e.g. PI/4) and
        // maxSailEfficiencyAngle closer to PI/2 (90 degrees / beam reach) with a strong drop-off downwind.
        // For this model, assuming primarily square-rigged caravels or simplification.
        lengthOverall: 25,
        beam: 7,
        designDraught: 2.8,
        nominalBurthen: 70,
        hullCoefficient: 0.58, // Finer than cogs/hulks, more agile
        lightshipMass: 35,
        sailArea: 280, // Good sail area, potentially multiple masts
        sailEfficiencyFactor: 0.7, // Generally more efficient than earlier square rigs due to design advances
        minWindSpeedForSailing: 2.0,
        maxSailEfficiencyAngle: Math.PI, // Sticking to square-rig peak for now
        noGoAngle: Math.PI / 2.8, // Slightly better (~64 degrees) due to finer lines
        waveImpactFactor: 0.09,
        currentLateralDragFactor: 0.55,
        maxAddedWaveResistanceCoeff: 0.09,
        maxStructuralSpeed: 6.0, // Good speed
        maxDraftTolerance: 0.3,
    },
    "carrack": {// 15th-17th Century large merchant/warship (e.g., Santa Maria, Mary Rose)
        lengthOverall: 35,
        beam: 10,
        designDraught: 4.0,
        nominalBurthen: 300, // Large capacity
        hullCoefficient: 0.65, // Substantial, full hull
        lightshipMass: 150,
        sailArea: 500, // Multiple masts, significant sail area
        sailEfficiencyFactor: 0.6, // Large, complex rigging can introduce inefficiencies
        minWindSpeedForSailing: 3.0, // Needs more wind to get moving due to size
        maxSailEfficiencyAngle: Math.PI,
        noGoAngle: Math.PI / 2.5, // Similar to cog (~72 degrees)
        waveImpactFactor: 0.1,
        currentLateralDragFactor: 0.65, // High due to large size
        maxAddedWaveResistanceCoeff: 0.1,
        maxStructuralSpeed: 5.0, // Slower due to size and bulk
        maxDraftTolerance: 0.5,
    },
    "corbita": { // Roman merchant vessel (round ship)
        lengthOverall: 25,
        beam: 8,
        designDraught: 3.0,
        nominalBurthen: 150,
        hullCoefficient: 0.7, // Very full and deep for cargo
        lightshipMass: 70,
        sailArea: 300, // Large main square sail, sometimes small artemon
        sailEfficiencyFactor: 0.65, // Reasonable for simple square rig
        minWindSpeedForSailing: 2.5,
        maxSailEfficiencyAngle: Math.PI,
        noGoAngle: Math.PI / 2.7, // (~66 degrees)
        waveImpactFactor: 0.12,
        currentLateralDragFactor: 0.7, // High due to beam and depth
        maxAddedWaveResistanceCoeff: 0.12, // High for blunt bows
        maxStructuralSpeed: 4.5, // Relatively slow
        maxDraftTolerance: 0.4,
    },
    "knarr": { // Viking merchant vessel, robust for North Atlantic
        lengthOverall: 18,
        beam: 5.5,
        designDraught: 2.0,
        nominalBurthen: 60,
        hullCoefficient: 0.65, // Fuller than longship, but less than hulk/cog
        lightshipMass: 30,
        sailArea: 220,
        sailEfficiencyFactor: 0.75, // Good efficiency for square sail
        minWindSpeedForSailing: 2.0,
        maxSailEfficiencyAngle: Math.PI,
        noGoAngle: Math.PI / 3.2, // Slightly better upwind than medieval cogs (~56 degrees)
        waveImpactFactor: 0.08,
        currentLateralDragFactor: 0.5,
        maxAddedWaveResistanceCoeff: 0.08,
        maxStructuralSpeed: 5.5, // Faster than cog/hulk
        maxDraftTolerance: 0.25,
    },
    "longship": { // Viking war/exploration vessel, fast under sail but also rowed
        // NOTE: This profile is for its SAILING characteristics only.
        // It's optimized for speed under sail, but would be slower than a dedicated merchant knarr for cargo.
        // If rowing is ever added, this would be a key candidate.
        lengthOverall: 30,
        beam: 5,
        designDraught: 1.0, // Very shallow draft
        nominalBurthen: 20, // Low cargo capacity, but can carry people/supplies
        hullCoefficient: 0.45, // Sleek, fine hull for speed
        lightshipMass: 15,
        sailArea: 180, // Single large square sail
        sailEfficiencyFactor: 0.8, // Very efficient for a square sail, well-designed
        minWindSpeedForSailing: 2.0,
        maxSailEfficiencyAngle: Math.PI,
        noGoAngle: Math.PI / 3.2, // (~56 degrees)
        waveImpactFactor: 0.06, // Less impact from waves due to fine bow
        currentLateralDragFactor: 0.4, // Lower due to shallower draught and finer lines
        maxAddedWaveResistanceCoeff: 0.06, // Lower for fine bows
        calmWaterMinSpeed: 0.2, // Acknowledge potential for rowing/drift
        maxStructuralSpeed: 7, // Very fast (reconstructions have hit 15+ knots)
        maxDraftTolerance: 0.1, // Sensitive to shallow water due to design
        bathymetricPenalty: 0.02,
    },// Higher penalty for grounding risk
    "hulk": { // Early medieval merchant vessel, often very beamy and rounder than cogs
        lengthOverall: 18,
        beam: 7,
        designDraught: 2.8,
        nominalBurthen: 100,
        hullCoefficient: 0.7, // Even fuller hull than a cog
        lightshipMass: 50,
        sailArea: 260, // Often a single, very large sail
        sailEfficiencyFactor: 0.55, // Slightly lower, less refined
        minWindSpeedForSailing: 3.0, // May need more wind to get going due to bluntness
        noGoAngle: Math.PI / 2.3, // Wider no-go zone (~78 degrees)
        currentLateralDragFactor: 0.7, // Very high due to extreme beam
        maxAddedWaveResistanceCoeff: 0.15, // High for blunt bow
        maxStructuralSpeed: 4.0, // Slower top speed
        maxDraftTolerance: 0.4,
    },
}

// --- VESSEL PARAMETERS ---
export function getVesselConfig(type = "cog", cargoWeight = 100) {
    // Ensure vesselType is lowercase for consistent lookup
    const typeKey = type.toLowerCase();

    // Get default values (from 'cog' if no specific type, or if 'cog' is explicitly requested)
    const defaults = state.vesselPresets["cog"];
    const specific = state.vesselPresets[typeKey] || {};

    // Merge specific type parameters over defaults
    const merged = {...defaults, ...specific};

    merged.vesselType = typeKey; // Store the type for reference

    // Attach cargo weight
    merged.cargoWeight = cargoWeight; // Attach cargo weight for hydrodynamics calculations

    // These need to be calculated based on the merged parameters and cargoWeight
    merged.actualDraught = VesselHydrodynamics.getActualDraught(merged, cargoWeight);

    // wettedSurfaceArea depends on actualDraught and other merged dimensions
    const totalMass_kg = (merged.lightshipMass + cargoWeight) * 1000;
    merged.actualDisplacementVolume = totalMass_kg / VesselHydrodynamics.RHO_SEAWATER;
    merged.wettedSurfaceArea = VesselHydrodynamics.getWettedSurfaceArea({
        lengthOverall: merged.lengthOverall,
        beam: merged.beam,
        actualDisplacementVolume: merged.actualDisplacementVolume
    }, merged.actualDraught);

    merged.frictionalResistanceCoefficient = VesselHydrodynamics.getFrictionalResistanceCoefficient(merged.nominalSpeed, merged.lengthOverall);
    merged.froudeNumber = VesselHydrodynamics.getFroudeNumber(merged.nominalSpeed, merged.lengthOverall);

    console.debug(`Vessel Config for ${typeKey}:`, merged);

    return merged;
}