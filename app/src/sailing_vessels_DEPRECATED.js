import {state} from "./state";

export function initVesselPresets() {
    state.vesselPresets = {
        "default": {
            "windSpeedToVesselSpeedRatio": 0.6,
            "maxVesselSpeed": 4.0,
            "maxPossibleWindSpeed": 12.0,
            "calmWaterMinVesselSpeed": 0.6,
            "vesselDraughtWithTolerance": 1.5,
            "bathymetricPenalty": 100,
            "invisibleLandPenalty": 3,
            "darknessPenaltyFactor": 1.5,
            "minWindSpeed": 0.8,
            "minWindEfficiency": 0.6,
            "minEfficiency": 0.35,
            "maxEfficiency": 1.2,
            "efficiencyPeakAngle": 90,
            "efficiencyDropRange": 60,
            "noSailAngleMin": 30,
            "noSailAngleMax": 150,
            "waveImpactFactor": 0.07,
            "minWavePenalty": 0.7,
            "maxWavePenalty": 0.9,
        },
        "cog": {
            "windSpeedToVesselSpeedRatio": 0.55,
            "vesselDraughtWithTolerance": 1.8,
            "minWindEfficiency": 0.4,
            "minEfficiency": 0.3,
            "maxEfficiency": 1.0,
            "noSailAngleMin": 35,
            "noSailAngleMax": 145
        },
        "caravel": {
            "windSpeedToVesselSpeedRatio": 0.65,
            "vesselDraughtWithTolerance": 1.5,
            "minWindEfficiency": 0.5,
            "minEfficiency": 0.4,
            "maxEfficiency": 1.25,
            "noSailAngleMin": 25,
            "noSailAngleMax": 155
        },
        "galley": {
            "windSpeedToVesselSpeedRatio": 0.4,
            "calmWaterMinVesselSpeed": 1.2,
            "vesselDraughtWithTolerance": 1.2,
            "minWindEfficiency": 0.2,
            "minEfficiency": 0.2,
            "maxEfficiency": 0.9,
            "noSailAngleMin": 40,
            "noSailAngleMax": 140
        }
    }
}


export function getVesselConfig(vesselType) {
    const defaults = state.vesselPresets["default"];
    const specific = state.vesselPresets[vesselType] || {};
    return {...defaults, ...specific};
}
