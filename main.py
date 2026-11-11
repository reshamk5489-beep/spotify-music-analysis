"""
What Makes a Song Popular?
An exploratory data analysis and machine-learning project using Spotify track data.

Run from this project folder:
    python main.py

Expected input:
    data/dataset.csv

Generated outputs:
    figures/   PNG visualizations
    results/   CSV tables and model metrics
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Save plots without opening GUI windows (helps avoid Thonny/Tkinter cleanup issues).
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance


# ---------------------------------------------------------------------
# 1. Paths and settings
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "dataset.csv"
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"

FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# 2. Load and inspect the dataset
# ---------------------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}.\n"
        "Place the Spotify CSV file at data/dataset.csv inside this project folder."
    )

df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 70)
print("WHAT MAKES A SONG POPULAR? — SPOTIFY DATA ANALYSIS")
print("=" * 70)

print("\nDATASET OVERVIEW")
print("Original dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nMissing values by column:")
print(df.isnull().sum())
print("\nExact duplicate rows:", df.duplicated().sum())

required_columns = [
    "track_id", "track_name", "artists", "popularity", "track_genre",
    "danceability", "energy", "valence", "tempo", "acousticness",
    "instrumentalness", "speechiness", "liveness", "loudness",
    "duration_ms", "explicit", "key", "mode", "time_signature"
]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"The dataset is missing required columns: {missing_columns}")

# Rows without an ID or target cannot be used for track-level analysis.
df = df.dropna(subset=["track_id", "popularity"]).copy()

# ---------------------------------------------------------------------
# 3. Understand repeated track IDs and create one row per track
# ---------------------------------------------------------------------

print("\nTRACK ID CHECK")
print("Rows after removing missing track IDs/popularity:", len(df))
print("Unique track IDs:", df["track_id"].nunique())
print("Rows with track IDs appearing more than once:",
      df.duplicated(subset=["track_id"], keep=False).sum())

duplicate_id_counts = df["track_id"].value_counts()
print("Number of track IDs appearing multiple times:",
      int((duplicate_id_counts > 1).sum()))
print("Maximum genre entries for one track:", int(duplicate_id_counts.max()))

# The same track can appear under multiple genre labels, sometimes with
# different popularity values. Audio features are identical for repeated IDs
# in this dataset. Average popularity creates one target value per track.
aggregation = {
    "track_name": "first",
    "artists": "first",
    "popularity": "mean",
    "duration_ms": "first",
    "explicit": "first",
    "danceability": "first",
    "energy": "first",
    "key": "first",
    "loudness": "first",
    "mode": "first",
    "speechiness": "first",
    "acousticness": "first",
    "instrumentalness": "first",
    "liveness": "first",
    "valence": "first",
    "tempo": "first",
    "time_signature": "first",
}

songs = df.groupby("track_id").agg(aggregation).reset_index()

print("\nTRACK-LEVEL DATASET")
print("Shape after grouping by track_id:", songs.shape)
print("Missing values in track-level data:")
print(songs.isnull().sum())

songs.to_csv(RESULTS_DIR / "track_level_data.csv", index=False)


# ---------------------------------------------------------------------
# 4. Explore popularity
# ---------------------------------------------------------------------

print("\nPOPULARITY SUMMARY")
print(songs["popularity"].describe())

zero_count = int((songs["popularity"] == 0).sum())
zero_percentage = zero_count / len(songs) * 100
print(f"Tracks with zero popularity: {zero_count:,} ({zero_percentage:.2f}%)")

top_tracks = songs.nlargest(10, "popularity")[
    ["track_name", "artists", "popularity"]
]
print("\nTop 10 tracks by average recorded popularity:")
print(top_tracks.to_string(index=False))
top_tracks.to_csv(RESULTS_DIR / "top_10_tracks.csv", index=False)

plt.figure(figsize=(10, 6))
sns.histplot(songs["popularity"], bins=20, kde=True)
plt.title("Distribution of Spotify Track Popularity")
plt.xlabel("Popularity score")
plt.ylabel("Number of tracks")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_popularity_distribution.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 5. Audio features and popularity correlations
# ---------------------------------------------------------------------

audio_features = [
    "danceability", "energy", "valence", "tempo", "acousticness",
    "instrumentalness", "speechiness", "liveness", "loudness",
    "duration_ms"
]

feature_correlations = (
    songs[audio_features + ["popularity"]]
    .corr(numeric_only=True)["popularity"]
    .drop("popularity")
    .sort_values()
)

print("\nAUDIO FEATURE CORRELATIONS WITH POPULARITY")
print(feature_correlations)
feature_correlations.rename("correlation_with_popularity").to_csv(
    RESULTS_DIR / "audio_feature_correlations.csv"
)

plt.figure(figsize=(10, 7))
feature_correlations.sort_values().plot(kind="barh")
plt.title("Correlation Between Audio Features and Popularity")
plt.xlabel("Pearson correlation")
plt.ylabel("Audio feature")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_audio_feature_correlations.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=songs.sample(min(5000, len(songs)), random_state=RANDOM_STATE),
    x="danceability",
    y="popularity",
    alpha=0.35,
    s=20
)
plt.title("Danceability vs. Track Popularity")
plt.xlabel("Danceability")
plt.ylabel("Popularity score")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_danceability_vs_popularity.png", dpi=300)
plt.close()

heatmap_features = [
    "danceability", "energy", "valence", "tempo", "acousticness",
    "instrumentalness", "speechiness", "liveness", "loudness", "popularity"
]
plt.figure(figsize=(11, 8))
sns.heatmap(
    songs[heatmap_features].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)
plt.title("Correlation Heatmap of Audio Features and Popularity")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_correlation_heatmap.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 6. Compare popularity across genres
# ---------------------------------------------------------------------

# Genre comparisons use the original genre-tagged rows. A track can appear
# under more than one genre, so genre rows are not independent observations.
genre_summary = (
    df.groupby("track_genre")
    .agg(
        entry_count=("track_id", "size"),
        unique_tracks=("track_id", "nunique"),
        mean_popularity=("popularity", "mean"),
        median_popularity=("popularity", "median"),
        popularity_std=("popularity", "std")
    )
    .sort_values("mean_popularity", ascending=False)
)

print("\n15 GENRES WITH HIGHEST AVERAGE POPULARITY")
print(genre_summary.head(15).round(2).to_string())
print("\n15 GENRES WITH LOWEST AVERAGE POPULARITY")
print(genre_summary.tail(15).sort_values("mean_popularity").round(2).to_string())

genre_summary.to_csv(RESULTS_DIR / "genre_popularity_summary.csv")

top_genres = genre_summary.head(15).sort_values("mean_popularity")
bottom_genres = genre_summary.tail(15).sort_values("mean_popularity")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
top_genres["mean_popularity"].plot(kind="barh", ax=axes[0])
axes[0].set_title("15 Highest-Average-Popularity Genres")
axes[0].set_xlabel("Average popularity")
axes[0].set_ylabel("Genre")

bottom_genres["mean_popularity"].plot(kind="barh", ax=axes[1])
axes[1].set_title("15 Lowest-Average-Popularity Genres")
axes[1].set_xlabel("Average popularity")
axes[1].set_ylabel("Genre")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_genre_popularity_comparison.png", dpi=300)
plt.close()

# Calculate danceability/popularity correlations within each genre.
genre_danceability_corr = (
    df.groupby("track_genre")
    .apply(lambda group: group["danceability"].corr(group["popularity"]))
    .dropna()
    .sort_values()
)
genre_danceability_corr.rename(
    "danceability_popularity_correlation"
).to_csv(RESULTS_DIR / "genre_danceability_correlations.csv")

strongest_negative = genre_danceability_corr.head(15)
strongest_positive = genre_danceability_corr.tail(15)
selected_genre_corr = pd.concat([strongest_negative, strongest_positive]).sort_values()

plt.figure(figsize=(11, 10))
selected_genre_corr.plot(kind="barh")
plt.axvline(0, linewidth=1)
plt.title("Genres with Strongest Positive and Negative Danceability Correlations")
plt.xlabel("Correlation between danceability and popularity")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_genre_danceability_correlations.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 7. Compare audio features across popularity groups
# ---------------------------------------------------------------------

bins = [-1, 25, 50, 75, 100]
labels = ["Low (0–25)", "Moderate (26–50)", "High (51–75)", "Very High (76–100)"]
songs["popularity_group"] = pd.cut(
    songs["popularity"], bins=bins, labels=labels
)

group_counts = songs["popularity_group"].value_counts().reindex(labels)
group_means = (
    songs.groupby("popularity_group", observed=False)[
        ["danceability", "energy", "valence", "liveness",
         "loudness", "tempo", "acousticness", "instrumentalness",
         "speechiness"]
    ]
    .mean()
)

print("\nSONG COUNTS BY POPULARITY GROUP")
print(group_counts)
print("\nAVERAGE AUDIO FEATURES BY POPULARITY GROUP")
print(group_means.round(3))

group_counts.rename("track_count").to_csv(
    RESULTS_DIR / "popularity_group_counts.csv"
)
group_means.to_csv(RESULTS_DIR / "popularity_group_feature_means.csv")

features_to_plot = ["danceability", "energy", "valence", "liveness", "acousticness"]
group_means[features_to_plot].plot(
    kind="bar",
    figsize=(12, 7)
)
plt.title("Average Audio Features Across Popularity Groups")
plt.xlabel("Popularity group")
plt.ylabel("Average feature value")
plt.xticks(rotation=20, ha="right")
plt.legend(title="Audio feature")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "07_features_by_popularity_group.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 8. Machine learning: predict popularity from audio features
# ---------------------------------------------------------------------

model_features = [
    "danceability", "energy", "valence", "tempo", "acousticness",
    "instrumentalness", "speechiness", "liveness", "loudness",
    "duration_ms", "explicit", "key", "mode", "time_signature"
]

X = songs[model_features].copy()
y = songs["popularity"].copy()

# Convert Boolean explicit values to numeric 0/1 if needed.
X["explicit"] = X["explicit"].astype(int)

# Ensure all model columns are numeric and finite.
X = X.apply(pd.to_numeric, errors="coerce")
finite_rows = np.isfinite(X.to_numpy(dtype=float)).all(axis=1) & np.isfinite(y.to_numpy(dtype=float))
X = X.loc[finite_rows].copy()
y = y.loc[finite_rows].copy()

print("\nMODEL INPUT CHECK")
print("Rows used for modeling:", len(X))
print("Missing values remaining:", int(X.isnull().sum().sum()))
print("All model inputs finite:", bool(np.isfinite(X.to_numpy(dtype=float)).all()))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

models = {
    "Baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

model_results = []
fitted_models = {}
predictions_by_model = {}

print("\nMODEL PERFORMANCE")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r_squared = r2_score(y_test, predictions)

    fitted_models[name] = model
    predictions_by_model[name] = predictions
    model_results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R_squared": r_squared
    })

    print(f"\n{name}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R-squared: {r_squared:.3f}")

model_results_df = pd.DataFrame(model_results)
model_results_df.to_csv(RESULTS_DIR / "model_performance.csv", index=False)

# Model performance comparison plot.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ["MAE", "RMSE", "R_squared"]
titles = ["Mean Absolute Error", "Root Mean Squared Error", "R-squared"]

for ax, metric, title in zip(axes, metrics, titles):
    ax.bar(model_results_df["model"], model_results_df[metric])
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=25)

plt.suptitle("Comparison of Popularity Prediction Models")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "08_model_performance.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 9. Random Forest feature importance
# ---------------------------------------------------------------------

rf_model = fitted_models["Random Forest"]

rf_importance = pd.Series(
    rf_model.feature_importances_,
    index=model_features,
    name="impurity_importance"
).sort_values(ascending=False)

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print(rf_importance.round(4).to_string())
rf_importance.to_csv(RESULTS_DIR / "random_forest_feature_importance.csv")

plt.figure(figsize=(10, 7))
rf_importance.sort_values().plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Impurity-based importance")
plt.ylabel("Audio feature")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "09_random_forest_feature_importance.png", dpi=300)
plt.close()

# Permutation importance measures the drop in test-set R-squared when a
# feature is shuffled. It offers a complementary test-set perspective.
permutation = permutation_importance(
    rf_model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=RANDOM_STATE,
    scoring="r2",
    n_jobs=-1
)

permutation_importance_series = pd.Series(
    permutation.importances_mean,
    index=model_features,
    name="mean_r2_decrease"
).sort_values(ascending=False)

print("\nRANDOM FOREST PERMUTATION IMPORTANCE (TEST SET)")
print(permutation_importance_series.round(4).to_string())
permutation_importance_series.to_csv(
    RESULTS_DIR / "random_forest_permutation_importance.csv"
)

plt.figure(figsize=(10, 7))
permutation_importance_series.sort_values().plot(kind="barh")
plt.title("Random Forest Permutation Importance")
plt.xlabel("Mean decrease in test R-squared after shuffling")
plt.ylabel("Audio feature")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "10_random_forest_permutation_importance.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 10. Final summary
# ---------------------------------------------------------------------

best_model_row = model_results_df.loc[model_results_df["R_squared"].idxmax()]

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)
print(f"Original dataset rows: {len(df):,}")
print(f"Unique tracks analyzed: {len(songs):,}")
print(f"Best model by R-squared: {best_model_row['model']}")
print(f"Best model R-squared: {best_model_row['R_squared']:.3f}")
print("\nAll charts were saved in the figures/ folder.")
print("All summary tables were saved in the results/ folder.")
print("Analysis complete.")
