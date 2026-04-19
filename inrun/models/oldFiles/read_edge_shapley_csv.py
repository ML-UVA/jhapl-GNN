import pandas as pd

CSV_FILE = "graph_edge_shapley_undirected_normalized.csv"

# --------------------------------------------------
# Load CSV
# --------------------------------------------------
df = pd.read_csv(CSV_FILE)

print("\nLoaded CSV:")
print(df.head())

# --------------------------------------------------
# Basic sanity checks
# --------------------------------------------------
print("\nBasic stats:")
print(f"Number of edges: {len(df)}")
print(f"Min shapley: {df['shapley_normalized'].min():.6e}")
print(f"Max shapley: {df['shapley_normalized'].max():.6e}")
print(f"Sum shapley: {df['shapley_normalized'].sum():.6f}")

# --------------------------------------------------
# Top-k edges
# --------------------------------------------------
TOP_K = 10
print(f"\nTop {TOP_K} most important edges:")

topk = df.sort_values("shapley_normalized", ascending=False).head(TOP_K)
for _, row in topk.iterrows():
    print(
        f"{int(row.src)} -- {int(row.dst)} | "
        f"score = {row.shapley_normalized:.6e}"
    )

# --------------------------------------------------
# Threshold-based filtering (optional)
# --------------------------------------------------
THRESHOLD = 0.001  # change as needed
important = df[df["shapley_normalized"] >= THRESHOLD]

print(f"\nEdges with shapley >= {THRESHOLD}: {len(important)}")

# --------------------------------------------------
# Save filtered subgraph (optional)
# --------------------------------------------------
OUT_FILE = "edge_shapley_top_edges.csv"
important.to_csv(OUT_FILE, index=False)
print(f"Saved filtered edges to {OUT_FILE}")
