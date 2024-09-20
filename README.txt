## Project Overview

This project explores the prediction of blood glucose (BG) levels in individuals with Type 1 Diabetes (T1D) using a combination of physiological models and machine learning methods. The goal is to improve the accuracy of BG predictions, which are critical for effective diabetes management and preventing complications. By integrating insights from the Hovorka physiological model with advanced machine learning models like Long Short-Term Memory (LSTM) networks and Transformers, we aim to provide more personalized and reliable predictions.

### Key Highlights
- **Dataset**: OhioT1DM dataset & synthetic data from the Simglucose simulator.
- **Physiological Model**: Hovorka Model.
- **Machine Learning Models**: LSTM and Transformer.
- **Prediction Horizons**: 15, 30, and 60 minutes.
- **Key Metrics**: RMSE, MAPE, CG-EGA accuracy, benign and critical error rates.

### Abstract
Type 1 Diabetes (T1D) is a chronic condition requiring continuous BG monitoring to prevent complications. Accurate BG prediction is essential for timely interventions. This project compares hybrid models combining physiological insights with machine learning techniques against models relying solely on historical data. The analysis reveals that hybrid models outperform purely data-driven approaches, especially at shorter prediction horizons (15-30 minutes), achieving lower error rates and more reliable predictions.

### Research Objectives:
1. **Hybrid Model Development**: Can combining machine learning and physiological modeling improve BG prediction accuracy? This research tests various hybrid architectures integrating the Hovorka model and machine learning approaches.
2. **Model Evaluation**: After selecting the best-performing model, further experimentation is conducted by adjusting parameters such as prediction horizons, historical data length, and applying smoothing techniques like Kalman filtering to improve performance.

## Repository Structure

The repository contains two main parts, each contributing to the overall research objectives:

### Part 1: Model Selection
This part focuses on choosing the best-performing model by comparing hybrid and data-driven approaches. The following experiments are conducted:
- **Experiment 1-6**: Comparison of hybrid models (Hovorka + ML) vs data-based models using LSTM and Transformer architectures created in the research.
  
Each experiment has its own folder containing:
- `.ipynb` file with the model code.
- Result `.pdf` files showing model performance metrics.

### Part 2: Further Experimentation
Once the best model is identified, additional experiments are performed to refine predictions:
- **Experimenting with Prediction Horizons (PH)**: Testing 15, 30, and 60-minute horizons.
- **Data History Length**: Using glucose history windows of 2, 3, and 4 hours.
- **Data Smoothing**: Applying Kalman Smoothing to improve the prediction reliability.

Contents:
- A single `.ipynb` notebook containing code for the experiments.
- A folder with results for all 4 experiments.

## Getting Started

### Prerequisites
- **Python 3.x**
- **Libraries**: TensorFlow (Keras), NumPy, Pandas, Matplotlib, Seaborn, OhioT1DM dataset, OhioT1DM Viewer, Simglucose simulator.


### Usage
1. Run the Jupyter notebooks in the `Part 1` or `Part 2` folders to replicate the experiments.
2. Adjust hyperparameters in the notebook to experiment with different settings such as prediction horizons or data history windows.

## Results Summary
- **Best Hybrid Model**: Achieved an RMSE of 22.24 mg/dL (1.23 mmol/L) and CG-EGA accuracy of 81.12%.
- **Data-Based Models**: Showed higher error rates, especially during rapid BG fluctuations.
- **Impact of Parameters**: Shorter prediction horizons (15-30 minutes) and incorporating physiological data improved accuracy. Data smoothing further reduced prediction errors.


## Contact
For questions or feedback, feel free to reach out at [narayaniverma98@gmail.com].