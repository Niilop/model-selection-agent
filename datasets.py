"""Definitions for the 5 benchmark regression datasets."""

DATASETS = {
    "california": {
        "description": "California housing prices — spatial, non-linear, 20k blocks",
        "target": "median_house_value",
        "n_rows": 2000,
        "notes": "Strong spatial signal (Lat/Lon), one dominant feature (MedInc), capped target at $500k",
    },
    "diabetes": {
        "description": "Diabetes disease progression — medical, small, linear-ish, 442 patients",
        "target": "target",
        "n_rows": None,  # use all 442 rows
        "notes": "Small dataset; regularised linear models are competitive; no spatial features",
    },
    "abalone": {
        "description": "Abalone age prediction — ecological, 4177 samples, ordinal ring count target",
        "target": "Rings",
        "n_rows": 2000,
        "notes": "Sex column (M/F/I) dropped as non-numeric; moderate non-linearity expected",
    },
    "energy": {
        "description": "Building energy efficiency — 768 samples, 8 features, strong non-linearity",
        "target": "Y1",
        "n_rows": None,  # use all 768 rows
        "notes": "Heating load prediction; compact/surface area interactions are highly non-linear",
    },
    "kin8nm": {
        "description": "Kinematics simulation — 8192 samples, 8 joint angle features, non-linear",
        "target": "y",
        "n_rows": 2000,
        "notes": "Robot arm end-effector position; highly non-linear angular relationships",
    },
}
