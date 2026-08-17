from pathlib import Path
import argparse

import numpy as np
import pandas as pd


def average_exps_by_songurl(exps, affect_type):

    ave_labels = {}

    for songurl, group in exps.groupby("songurl"):

        ave = group[affect_type].mean()

        ave_labels[songurl] = ave

    return ave_labels


def ave_exps_by_cluster(
    exps,
    cluster_info,
    affect_type,
    n_clusters=4,
):

    combined_songurl_list = []
    combined_labels_list = []
    combined_cluster_list = []

    for cluster_id, group_p in cluster_info.groupby("cluster"):

        wids_in_cluster = (
            group_p["workerid"]
            .to_numpy()
        )

        cluster_exps = exps[
            exps["workerid"].isin(
                wids_in_cluster
            )
        ]

        ave_cluster_labels = (
            average_exps_by_songurl(
                cluster_exps,
                affect_type
            )
        )

        cluster_vector = np.eye(
            n_clusters,
            dtype=np.float32
        )[int(cluster_id)]

        profile_col = [
            cluster_vector.copy()
            for _ in range(
                len(ave_cluster_labels)
            )
        ]

        combined_songurl_list += list(
            ave_cluster_labels.keys()
        )

        combined_labels_list += list(
            ave_cluster_labels.values()
        )

        combined_cluster_list += profile_col

    df = pd.DataFrame()

    df["songurl"] = combined_songurl_list
    df["labels"] = combined_labels_list
    df["profile"] = combined_cluster_list

    return df


def main(
    source_dir: Path,
    output_dir: Path,
):

    exps = pd.read_pickle(
        source_dir / "exps_ready3.pkl"
    )

    clusters = pd.read_pickle(
        source_dir / "profile_clusters_k4.pkl"
    )

    print(
        "Annotation rows:",
        len(exps)
    )

    print(
        "Listeners:",
        exps["workerid"].nunique()
    )

    print(
        "Cluster listeners:"
    )

    print(
        clusters["cluster"]
        .value_counts()
        .sort_index()
    )

    valence_data = ave_exps_by_cluster(
        exps,
        clusters,
        "valences",
        n_clusters=4,
    )

    arousal_data = ave_exps_by_cluster(
        exps,
        clusters,
        "arousals",
        n_clusters=4,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    valence_data.to_pickle(
        output_dir
        / "exps_cluster_valence_k4.pkl"
    )

    arousal_data.to_pickle(
        output_dir
        / "exps_cluster_arousal_k4.pkl"
    )

    print(
        "Valence shape:",
        valence_data.shape
    )

    print(
        "Arousal shape:",
        arousal_data.shape
    )

    print(
        "Valence songs:",
        valence_data["songurl"].nunique()
    )

    print(
        "Arousal songs:",
        arousal_data["songurl"].nunique()
    )

    print(
        "Example profile:",
        valence_data.iloc[0]["profile"]
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
        required=True,
    )

    args = parser.parse_args()

    main(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )