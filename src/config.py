from __future__ import annotations

SERVICE_STATIONS = [
    (36.0666667, -86.4347222),
    (35.5883333, -86.4438888),
    (36.1950, -83.174722),
]

NUMERIC_FEATURES = [
    "faults_in_last_hr",
    "DistanceLtd",
    "EngineTimeLtd",
    "FuelLtd",
    "activeTransitionCount",
    "TurboBoostPressure",
    "AcceleratorPedal",
    "BarometricPressure",
    "CruiseControlSetSpeed",
    "EngineCoolantTemperature",
    "EngineLoad",
    "EngineOilPressure",
    "EngineOilTemperature",
    "EngineRpm",
    "FuelLevel",
    "FuelRate",
    "FuelTemperature",
    "IntakeManifoldTemperature",
    "Speed",
    "SwitchedBatteryVoltage",
    "Throttle",
]

CATEGORICAL_FEATURES = [
    "spn",
    "fmi",
    "LampStatus",
    "active",
    "CruiseControlActive",
    "IgnStatus",
    "ParkingBrake",
    "year",
    "weekday",
    "summer",
    "daytime",
    "nighttime",
    "month",
    "day",
    "hour",
]

TARGETS = ["derate_6_hr", "derate_12_hr", "derate_24_hr"]
DATE_SPLIT = "2019-01-01"