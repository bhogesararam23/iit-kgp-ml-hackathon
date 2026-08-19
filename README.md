# IIT KGP ML Hackathon — Reactor Yield Prediction

A machine-learning solution for the IIT Kharagpur Predictive Modeling Optimization Challenge.

## Project Overview

The objective of this project is to predict the final yield of desired product **B** in a non-isothermal continuous-flow chemical reactor.

The reactor contains two competing reactions:

```
A → B   Desired reaction
B → C   Side reaction
```

The model learns the relationship between reactor operating conditions and the final `overall_yield` of Product B. This provides a fast data-driven surrogate for computationally expensive reactor simulations.

## Dataset

The project uses two CSV files:

| File | Description | Rows |
| --- | --- | --- |
| `train_dataset.csv` | Input features and the known target `overall_yield` | 150 |
| `test_dataset.csv` | Input features without the target value | 50 |

### Input Features

| Feature | Description |
| --- | --- |
| `flow_rate_L_min` | Volumetric flow rate of the reactant mixture in litres per minute |
| `concentration_mol_L` | Inlet concentration of Reactant A in moles per litre |
| `inlet_temperature_K` | Temperature of the feed entering the reactor in Kelvin |
| `length_m` | Reactor length in metres |
| `jacket_temperature_K` | External heating-jacket temperature in Kelvin |

### Target Variable

```
overall_yield
```

This is the final percentage yield of Product B at the reactor exit.

## Methodology

The workflow was implemented in Google Colab using Python and scikit-learn.

### 1. Data validation

The training and test datasets were checked for:

- Correct column names and dimensions.

- Missing values.

- Duplicate rows.

- Numeric data types.

- Valid target ranges.

The training dataset contains 150 rows and the test dataset contains 50 rows. No missing values or duplicate rows were found.

### 2. Feature engineering

In addition to the original reactor measurements, the following physics-inspired features were created:

```
residence_time_proxy = length_m / flow_rate_L_min

temperature_difference = jacket_temperature_K - inlet_temperature_K

mean_temperature = (jacket_temperature_K + inlet_temperature_K) / 2

thermal_residence = mean_temperature * residence_time_proxy

concentration_residence = concentration_mol_L * residence_time_proxy

absolute_temperature_difference = abs(temperature_difference)
```

The residence-time proxy represents the combined effect of reactor length and flow rate. Temperature-based features help the model capture the nonlinear thermal behavior of the reactor and the possibility of undesired conversion of B into C at excessive thermal severity.

### 3. Model comparison

Several regression models were compared using shuffled 5-fold cross-validation. The evaluation metric was Root Mean Squared Error (RMSE), where a lower value is better.

| Model | Average validation RMSE |
| --- | --- |
| Extra Trees | **17.57** |
| Random Forest | 19.76 |
| Gradient Boosting | 22.33 |
| Linear Regression | 32.17 |
| Baseline Average Predictor | 38.25 |

Extra Trees achieved the best validation result among the tested models and was selected for the final predictions.

## Final Model

The final Extra Trees model used the following configuration:

```python
ExtraTreesRegressor(
    n_estimators=400,
    max_features=0.9,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
```

After model selection, the model was retrained using all 150 labeled training rows. It then generated predictions for all 50 rows in the test dataset.

The final predictions were clipped to the physically meaningful yield range of 0 to 100 and rounded to three decimal places.

## Repository Contents

```
IIT_KGP_ML_Hackathon.ipynb   Complete Google Colab notebook
train_dataset.csv            Training data, if distribution is permitted
 test_dataset.csv             Test data, if distribution is permitted
hirenbvala321.csv             Final prediction submission file
README.md                     Project documentation
```

> If the submission CSV has a different filename in this repository, replace `hirenbvala321.csv` above with the exact filename shown in the repository.

## Submission Format

The final submission file contains exactly one column:

```
overall_yield
```

It contains exactly 50 predictions in the original order of `test_dataset.csv`.

Example:

```
overall_yield
18.526
82.257
25.412
82.070
0.776
```

## How to Reproduce the Results

1. Open `IIT_KGP_ML_Hackathon.ipynb` in Google Colab.

1. Upload `train_dataset.csv` and `test_dataset.csv` when prompted.

1. Run the notebook cells from top to bottom.

1. Inspect the data and engineered features.

1. Run the cross-validation model comparison.

1. Train the final Extra Trees model on the complete training dataset.

1. Generate the 50 test predictions.

1. Download the final submission CSV.

## Key Engineering Insight

The training data shows a strong nonlinear relationship between temperature and yield. Higher jacket and inlet temperatures are often associated with near-zero yields. This is consistent with a competing-reaction system in which increased thermal severity may accelerate both the desired reaction `A → B` and the undesired side reaction `B → C`.

Therefore, maximizing a single operating variable is not sufficient. The model must learn interactions between temperature, flow rate, reactor length, concentration, and residence time.

## Tools Used

- Python

- Google Colab

- pandas

- NumPy

- Matplotlib

- Seaborn

- scikit-learn

- GitHub

## Author

**Team NOMOS**

Created for the IIT Kharagpur ML Hackathon.
