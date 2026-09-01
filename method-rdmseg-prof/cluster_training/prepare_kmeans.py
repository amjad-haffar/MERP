from pathlib import Path
import argparse

import numpy as np
import pandas as pd


def average_exps_by_songurl(exps, affect_type):

    ave_labels = {}

    for songurl, group in exps.groupby("songurl"):
        ave_labels[songurl] = group[affect_type].mean()

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

        wids_in_cluster = group_p["workerid"].to_numpy()

        cluster_exps = exps[
            exps["workerid"].isin(wids_in_cluster)
        ]

        ave_cluster_labels = average_exps_by_songurl(
            cluster_exps,
            affect_type,
        )

        cluster_vector = np.eye(
            n_clusters,
            dtype=np.float32
        )[int(cluster_id)]

        profile_col = [
            cluster_vector.copy()
            for _ in range(len(ave_cluster_labels))
        ]

        combined_songurl_list += list(
            ave_cluster_labels.keys()
        )

        combined_labels_list += list(
            ave_cluster_labels.values()
        )

        combined_cluster_list += profile_col

    return pd.DataFrame({
        "songurl": combined_songurl_list,
        "labels": combined_labels_list,
        "profile": combined_cluster_list,
    })


def build_cluster_dataset(
    exps,
    cluster_path,
    output_dir,
    experiment_name,
    n_clusters=4,
):

    clusters = pd.read_pickle(cluster_path)

    print("\n========================================")
    print("Experiment:", experiment_name)
    print("========================================")

    print(
        "Cluster listeners:",
        clusters["workerid"].nunique()
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
        n_clusters=n_clusters,
    )

    arousal_data = ave_exps_by_cluster(
        exps,
        clusters,
        "arousals",
        n_clusters=n_clusters,
    )

    valence_path = (
        output_dir
        / f"exps_{experiment_name}_valence_k4.pkl"
    )

    arousal_path = (
        output_dir
        / f"exps_{experiment_name}_arousal_k4.pkl"
    )

    valence_data.to_pickle(valence_path)
    arousal_data.to_pickle(arousal_path)

    print(
        "Valence shape:",
        valence_data.shape
    )

    print(
        "Arousal shape:",
        arousal_data.shape
    )

    print(
        "Example profile:",
        valence_data.iloc[0]["profile"]
    )

    print("Saved:")
    print(valence_path)
    print(arousal_path)


def main(
    source_dir: Path,
    output_dir: Path,
):

    exps = pd.read_pickle(
        source_dir / "exps_ready3.pkl"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # EXP 1 — Behaviour + Profile Direct K-means
    # ========================================================

    build_cluster_dataset(
        exps=exps,
        cluster_path=(
            source_dir
            / "behaviour_profile_kmeans_clusters_k4.pkl"
        ),
        output_dir=output_dir,
        experiment_name="behaviour_profile_kmeans",
    )

    # ========================================================
    # EXP 2 — Behaviour MDS
    # ========================================================

    build_cluster_dataset(
        exps=exps,
        cluster_path=(
            source_dir
            / "behaviour_mds_clusters_k4.pkl"
        ),
        output_dir=output_dir,
        experiment_name="behaviour_mds",
    )

    # ========================================================
    # EXP 3 — Behaviour + Profile MDS
    # ========================================================

    build_cluster_dataset(
        exps=exps,
        cluster_path=(
            source_dir
            / "behaviour_profile_mds_clusters_k4.pkl"
        ),
        output_dir=output_dir,
        experiment_name="behaviour_profile_mds",
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