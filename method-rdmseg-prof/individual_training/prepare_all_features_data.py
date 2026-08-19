from pathlib import Path
import argparse

import numpy as np
import pandas as pd


ALL_FEATURES = [
    "age",
    "gender",
    "residence",
    "enculturation",
    "language",
    "genre",
    "instrument",
    "training",
    "duration",
]


def main(source_dir: Path, output_dir: Path) -> None:
    annotations = pd.read_pickle(
        source_dir / "exps_ready3.pkl"
    )

    profiles = pd.read_pickle(
        source_dir / "pinfo_numero.pkl"
    )

    all_profiles = (
        profiles[
            ["workerid"] + ALL_FEATURES
        ]
        .drop_duplicates(subset="workerid")
    )

    data = annotations.merge(
        all_profiles,
        on="workerid",
        how="left",
        validate="many_to_one",
    )

    data["profile"] = data[ALL_FEATURES].apply(
        lambda row: np.asarray(
            row.to_numpy(dtype=np.float32)
        ),
        axis=1,
    )

    valence_data = data[
        [
            "workerid",
            "songurl",
            "profile",
            "valences",
        ]
    ].rename(
        columns={"valences": "labels"}
    )

    arousal_data = data[
        [
            "workerid",
            "songurl",
            "profile",
            "arousals",
        ]
    ].rename(
        columns={"arousals": "labels"}
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    valence_data.to_pickle(
        output_dir
        / "exps_individual_valence_all_features.pkl"
    )

    arousal_data.to_pickle(
        output_dir
        / "exps_individual_arousal_all_features.pkl"
    )

    print("Valence shape:", valence_data.shape)
    print("Arousal shape:", arousal_data.shape)
    print("Songs:", data["songurl"].nunique())
    print("Listeners:", data["workerid"].nunique())

    print(
        "Missing profile values:",
        data[ALL_FEATURES].isna().sum().to_dict(),
    )

    print(
        "Example profile:",
        valence_data.iloc[0]["profile"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
    )

    args = parser.parse_args()

    main(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )