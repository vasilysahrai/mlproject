# Combined Cycle Power Plant Data Analysis

This repository contains Python code to train a Linear Regression model on the Combined Cycle Power Plant dataset to predict the net hourly electrical energy output (PE).

## Dataset

The dataset (`combined_cycle_power_plant.csv`) consists of hourly averaged variables from a combined cycle power plant. These variables include:

- **AT**: Ambient Temperature (in °C)
- **V**: Exhaust Vacuum (in cm Hg)
- **AP**: Ambient Pressure (in mbar)
- **RH**: Relative Humidity (%)
- **PE**: Net Hourly Electrical Energy Output (in MW)

## Libraries Used

- `pandas` for data manipulation
- `sklearn` for machine learning tasks (model selection, preprocessing, metrics)
- `matplotlib` for data visualization

## Steps

1. **Loading and Exploring the Data**:
   - The dataset is loaded using `pandas.read_csv()`.
   - First few rows of the dataset are printed to understand its structure.

2. **Data Preprocessing**:
   - The features (`X`) are extracted by dropping the target variable (`PE`).
   - The target variable (`y`) is isolated.
   - The dataset is split into training and testing sets using `train_test_split()`.

3. **Feature Scaling**:
   - Features (`X`) are scaled using `StandardScaler()` to standardize the dataset.

4. **Model Training**:
   - A Linear Regression model is instantiated and trained on the scaled training data (`X_train_scaled`, `y_train`).

5. **Model Evaluation**:
   - Predictions are made on the scaled test data (`X_test_scaled`) using the trained model.
   - Metrics such as Mean Squared Error (MSE) and R-squared score are computed to evaluate the model's performance.

6. **Visualization**:
   - A scatter plot is generated to visualize the actual vs predicted values of the target variable (`PE`).

## Usage

To run the code:

1. Clone this repository:
   ```bash
   git clone <https://github.com/vasilysahrai/mlproject>
   cd <mlproject>

2. Install the required libraries if not already installed:
pip install pandas scikit-learn matplotlib

3.Run the Python script:
python analysis.py

The script will execute the steps outlined above and display the evaluation metrics and a visualization of the model's predictions.

