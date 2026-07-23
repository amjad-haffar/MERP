from pathlib import Path
import argparse
import pandas as pd


def main(source_dir: Path, output_dir: Path) -> None:
    # Load the already-cleaned MERP files
    annotations = pd.read_pickle(source_dir / "exps_ready3.pkl")
    profiles = pd.read_pickle(source_dir / "pinfo_numero.pkl")

    # Keep one age value per listener
    age_profiles = (
        profiles[["workerid", "age"]]
        .drop_duplicates(subset="workerid")
    )

    # Attach age to every individual annotation
    data = annotations.merge(
        age_profiles,
        on="workerid",
        how="left",
        validate="many_to_one",
    )

    # Create the format expected by dataset.py
    valence_data = data[
        ["workerid", "songurl", "age", "valences"]
    ].rename(
        columns={
            "age": "profile",
            "valences": "labels",
        }
    )

    arousal_data = data[
        ["workerid", "songurl", "age", "arousals"]
    ].rename(
        columns={
            "age": "profile",
            "arousals": "labels",
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    valence_data.to_pickle(
        output_dir / "exps_individual_valence_age.pkl"
    )

    arousal_data.to_pickle(
        output_dir / "exps_individual_arousal_age.pkl"
    )

    # Minimal final checks
    print("Valence shape:", valence_data.shape)
    print("Arousal shape:", arousal_data.shape)
    print("Songs:", data["songurl"].nunique())
    print("Listeners:", data["workerid"].nunique())
    print("Missing ages:", data["age"].isna().sum())


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