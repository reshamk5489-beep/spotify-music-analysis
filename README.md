# What Makes a Song Popular? 🎵

An independent data science project exploring how Spotify audio characteristics relate to track popularity, using exploratory data analysis, statistical comparisons, visualizations, and regression models.

## Research question

**How are Spotify audio features associated with a track's popularity, and how well can those features predict popularity?**

This project focuses on associations and prediction. It does not claim that an audio feature causes a song to become popular.

## Project overview

The analysis uses a Spotify tracks dataset containing 114,000 genre-tagged rows and 21 columns. Because the same track ID can appear under multiple genre labels, the project creates a track-level dataset with one row per unique `track_id`.

The track-level dataset contains **89,741 unique tracks**. For repeated track IDs, audio features are retained from the first record because the repeated entries had the same audio-feature values in the checks performed during this project. Popularity is averaged across the genre-tagged records for that track. This means the target represents the **average recorded popularity across the track's genre entries**, not necessarily one single canonical Spotify popularity value.

## Questions explored

- How is popularity distributed across the tracks?
- How strongly are audio features such as danceability, energy, loudness, valence, and acousticness correlated with popularity?
- How does average popularity vary across genre labels?
- Do audio features differ across low-, moderate-, high-, and very-high-popularity groups?
- How well can a baseline model, Linear Regression, and Random Forest predict popularity from audio features?
- Which features are most useful to the Random Forest model?

## Dataset

Input file expected at:

```text
data/dataset.csv
```

The CSV used in the analysis has 114,000 rows and 21 columns, including track identifiers, artist and track names, popularity, genre labels, and audio features.

**Dataset source:** Add the original dataset page or download URL here before publishing. Keep the dataset's original attribution and license information. Do not upload the CSV to GitHub unless its license and the dataset provider's terms allow redistribution.

## Methods

### 1. Data inspection and cleaning

- Inspected the dataset's dimensions, columns, missing values, and exact duplicate rows.
- Checked repeated `track_id` values.
- Grouped records by `track_id` to create one observation per unique track.
- Averaged popularity across repeated genre-tagged records for each track.
- Retained one set of audio features per track.

### 2. Exploratory data analysis

The project creates visualizations and summary tables for:

- Popularity distribution
- Audio-feature correlations with popularity
- Danceability versus popularity
- Audio-feature correlation heatmap
- Highest- and lowest-average-popularity genres
- Genre-specific danceability/popularity correlations
- Average audio features across popularity groups

### 3. Predictive modeling

The target variable is `popularity`. The predictors are:

- danceability
- energy
- valence
- tempo
- acousticness
- instrumentalness
- speechiness
- liveness
- loudness
- duration_ms
- explicit
- key
- mode
- time_signature

The data is split into training and test sets using an 80/20 split and `random_state=42`.

Three regression models are compared:

1. **Baseline:** predicts the mean popularity from the training set.
2. **Linear Regression:** uses standardized input features.
3. **Random Forest Regressor:** uses 100 trees, maximum depth 15, and minimum leaf size 5.

Evaluation metrics:

- **MAE:** average absolute prediction error, in popularity points.
- **RMSE:** prediction error that gives larger errors more weight, in popularity points.
- **R²:** proportion of test-set target variation explained by the model relative to a mean-prediction baseline.

Random Forest feature importance is examined using both built-in impurity importance and permutation importance on the test set.

## Findings

The completed model run produced the following test-set results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline | 17.24 | 20.55 | approximately 0.000 |
| Linear Regression | 16.71 | 20.24 | 0.030 |
| Random Forest | **14.79** | **18.54** | **0.186** |

Random Forest performed best among the three models. Its R² of 0.186 means it explained about 18.6% of the variation in popularity in the held-out test set. Its average absolute prediction error was 14.79 popularity points. This is an improvement over the baseline, but most of the variation remained unexplained.

### Random Forest feature importance

The model's built-in impurity-based importance ranked the features as follows:

| Feature | Importance |
|---|---:|
| acousticness | 0.1323 |
| duration_ms | 0.1122 |
| instrumentalness | 0.1017 |
| danceability | 0.0987 |
| valence | 0.0961 |
| speechiness | 0.0923 |
| loudness | 0.0860 |
| energy | 0.0844 |
| tempo | 0.0783 |
| liveness | 0.0687 |
| key | 0.0282 |
| explicit | 0.0084 |
| mode | 0.0079 |
| time_signature | 0.0048 |

Permutation importance on the test set gave a complementary ranking. The largest mean decreases in R² after shuffling a feature were instrumentalness (0.0990), acousticness (0.0970), duration_ms (0.0827), valence (0.0771), and danceability (0.0669).

Both methods place acousticness, instrumentalness, and duration among the more useful predictors in this model. However, feature importance describes how the fitted model uses the available variables; it does not show that changing a feature would cause popularity to change. Correlated features can also share or affect importance scores.

### Correlation analysis

The track-level correlation analysis showed generally weak linear relationships between individual audio features and popularity. For example, instrumentalness had a negative correlation of approximately -0.128, while loudness and danceability had small positive correlations of approximately 0.072 and 0.064. These are associations, not evidence of causation.

### Interpretation

Taken together, the results suggest that audio characteristics contain some information about popularity, but they are not enough to predict it accurately on their own. Factors not included in this dataset—such as artist recognition, marketing, playlist placement, release timing, and social media exposure—may also matter.

**Numerical warning note:** During the Linear Regression run, NumPy emitted divide-by-zero, overflow, and invalid-value warnings during matrix multiplication. The reported predictions were finite, and a manual calculation matched the scikit-learn predictions exactly in the diagnostic check. The warning's underlying cause has not been conclusively identified, so it should not be described as fixed. The Random Forest comparison does not depend on that Linear Regression calculation.

## Limitations

- **Popularity is a snapshot:** Spotify popularity is a platform-specific score that can change over time. It should not be treated as a permanent measure of a song's quality or cultural importance.
- **Genre-tagged rows are repeated:** A track may appear in multiple genre categories. Genre-level comparisons use the original genre-tagged rows, so observations across genres are not fully independent.
- **Popularity aggregation:** Repeated track IDs can have different popularity values across genre entries. Averaging those values creates a consistent track-level target but can hide genre-specific differences.
- **Audio features are not the whole story:** Marketing, artist recognition, release timing, playlist placement, social media, geography, and listener behavior are not represented by the selected audio features.
- **Prediction is limited:** The initial Random Forest model explained 18.6% of test-set popularity variation. Audio features alone therefore do not provide a highly accurate prediction.
- **Correlation is not causation:** Correlations and feature importance do not show that changing an audio feature would change a song's popularity.
- **Feature importance has limitations:** Impurity-based importance can favor certain kinds of variables and can be affected by correlated predictors. Permutation importance provides a complementary view but can also be affected by correlated features.

## Repository structure

```text
Spotify-Music-Analysis/
├── data/
│   ├── dataset.csv              # Add the dataset locally if license permits
│   └── .gitkeep
├── figures/                     # Generated PNG charts
├── results/                     # Generated CSV tables
├── main.py                      # Complete analysis script
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

1. Install Python 3.10 or later.
2. Download the dataset and place it at `data/dataset.csv`.
3. Open a terminal in this project folder.
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Run the analysis:

   ```bash
   python main.py
   ```

The script saves charts in `figures/` and tables in `results/`.

## Reproducibility

The train/test split and randomized models use `random_state=42`. Model performance can still vary if the dataset changes, preprocessing changes, or library versions differ.

## Tools

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

---

*This project was completed as an independent exploration of data cleaning, exploratory data analysis, visualization, regression modeling, model evaluation, and interpretation.*
