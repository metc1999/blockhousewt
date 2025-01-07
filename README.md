# Order Flow Imbalance (OFI) Analysis

This project investigates the relationship between Order Flow Imbalance (OFI) and price changes in equity markets, focusing on both contemporaneous and lagged effects. The analysis uses high-frequency data and explores the explanatory and predictive power of OFI metrics for price dynamics.

## Features

1. **Compute OFI Metrics**:
   - Derive multi-level OFI metrics for up to 5 levels of the order book.
   - Integrate these metrics into a single explanatory variable using Principal Component Analysis (PCA).

2. **Cross-Impact Analysis**:
   - Examine the contemporaneous impact of OFI on price changes for individual stocks.
   - Evaluate the predictive power of lagged OFI features (e.g., 1-minute and 5-minute lags) for future price changes.

3. **Visualization**:
   - Generate heatmaps, scatter plots, and bar charts to illustrate the relationships between OFI metrics and price changes.

4. **Quantification**:
   - Use regression models to calculate coefficients, intercepts, and \(R^2\) values for OFI’s impact on price changes.
   - Compare self-impact (within stocks) and cross-impact (across stocks).

## Requirements

The required Python libraries are listed in the `requirements.txt` file. Install them using:

```bash
pip install -r requirements.txt
```

## Usage

1. **Setup**:
   - Clone this repository.
   - Ensure the dataset is available and formatted correctly.

2. **Run Analysis**:
   - Execute the Python script to compute OFI metrics and perform regression analyses.
   - Generate visualizations and summaries of the results.

3. **Outputs**:
   - Regression results, including coefficients and \(R^2\) values, for each stock.
   - Visualizations saved as `.png` files, such as:
     - Heatmaps showing correlations.
     - Scatter plots of OFI vs. price changes.
     - Bar charts of regression coefficients.

## File Structure

- `main.py`: The main script for running the analysis.
- `requirements.txt`: List of dependencies.
- `data/`: Directory to store input datasets.
- `outputs/`: Directory to store visualizations and results.

## Example Outputs

- **Regression Results**:
  ```
  Stock: SOUN
    Coefficients (Lagged OFI): [-0.01515399 -0.00587307]
    Intercept: 11.341050103573735
    R-squared: 0.0025
  ```

- **Visualizations**:
  - Heatmaps of correlations.
  - Scatter plots showing relationships between OFI and price changes.
  - Bar charts of regression coefficients for individual stocks.

## Next Steps

- Incorporate additional market features (e.g., trading volume, volatility).
- Explore advanced machine learning models for improved predictive power.
- Analyze cross-stock impacts to identify interdependencies among stocks.

## License

This project is open-source and licensed under the MIT License.

## Contact

For questions or feedback, please contact:
- **Name**: Muhammad Essa Tabish Chawla
- **Email**: mchawla4@dons.usfca.edu
