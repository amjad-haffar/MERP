from pathlib import Path
import argparse

import pandas as pd


def main(source_dir: Path, output_dir: Path) -> None:
    annotations = pd.read_pickle(
        source_dir / "exps_ready3.pkl"
    )

    valence_data = annotations[
        [
            "workerid",
            "songurl",
            "valences",
        ]
    ].rename(
        columns={"valences": "labels"}
    )

    arousal_data = annotations[
        [
            "workerid",
            "songurl",
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
        / "exps_individual_valence_audio.pkl"
    )

    arousal_data.to_pickle(
        output_dir
        / "exps_individual_arousal_audio.pkl"
    )

    print("Valence shape:", valence_data.shape)
    print("Arousal shape:", arousal_data.shape)
    print("Songs:", annotations["songurl"].nunique())
    print("Listeners:", annotations["workerid"].nunique())


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