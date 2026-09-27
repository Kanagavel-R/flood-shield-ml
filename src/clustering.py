"""
clustering.py – Unsupervised learning: K-Means clustering + PCA.
Internal academic module — NOT exposed in the Streamlit frontend.

Run directly:
    python src/clustering.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    FEATURE_COLUMNS,
    GRAPHS_DIR,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
)


def run_clustering(n_clusters: int = 3, random_state: int = 42):
    """Run K-Means clustering and PCA on the flood dataset."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(RAW_DATA_PATH)
    X = df[FEATURE_COLUMNS].values

    # Scale for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Elbow method ─────────────────────────────────────────────────────────
    print("[INFO] Running elbow method (k=2 to 8)...")
    inertias = []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), inertias, marker="o", color="steelblue")
    plt.title("K-Means Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "kmeans_elbow.png", dpi=150)
    plt.close()
    print(f"[INFO] Elbow chart saved: outputs/graphs/kmeans_elbow.png")

    # ── K-Means with chosen k ─────────────────────────────────────────────────
    print(f"[INFO] Fitting K-Means with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    df["Cluster"] = cluster_labels
    print(f"[INFO] Cluster distribution:\n{pd.Series(cluster_labels).value_counts().sort_index()}")

    # ── PCA for visualization ─────────────────────────────────────────────────
    print("[INFO] Running PCA (2 components)...")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    print(f"[INFO] PCA explained variance: PC1={explained[0]:.4f}, PC2={explained[1]:.4f} "
          f"(total={sum(explained):.4f})")

    # PCA scatter plot coloured by cluster
    plt.figure(figsize=(8, 6))
    colors = ["steelblue", "coral", "seagreen", "gold", "mediumpurple"]
    for c in range(n_clusters):
        mask = cluster_labels == c
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            label=f"Cluster {c}",
            s=5, alpha=0.5,
            color=colors[c % len(colors)],
        )
    plt.title(f"PCA — K-Means Clusters (k={n_clusters})")
    plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    plt.legend(markerscale=4)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "kmeans_pca_scatter.png", dpi=150)
    plt.close()
    print("[INFO] PCA scatter saved: outputs/graphs/kmeans_pca_scatter.png")

    # PCA scatter coloured by FloodProbability
    fp = df[TARGET_COLUMN].values
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=fp, cmap="RdYlGn_r", s=5, alpha=0.5)
    plt.colorbar(sc, label="FloodProbability")
    plt.title("PCA — Coloured by FloodProbability")
    plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "pca_flood_probability.png", dpi=150)
    plt.close()
    print("[INFO] PCA flood prob chart saved: outputs/graphs/pca_flood_probability.png")

    # ── Cluster summary report ────────────────────────────────────────────────
    cluster_summary = df.groupby("Cluster")[FEATURE_COLUMNS + [TARGET_COLUMN]].mean()
    cluster_summary.to_csv(REPORTS_DIR / "cluster_summary.csv")
    print(f"[INFO] Cluster summary saved: outputs/reports/cluster_summary.csv")

    print("\n=== CLUSTER ANALYSIS ===")
    print(cluster_summary[[TARGET_COLUMN]])

    return cluster_labels, X_pca


if __name__ == "__main__":
    run_clustering()
    print("\n[DONE] Clustering complete.")
