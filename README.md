Confidence-Aware Grasp Success Prediction

Predicting whether a robotic grasp will hold, and how confident we can be about it, from the joint sensor readings of a three-finger Shadow Robot hand.

This project is an end-to-end machine learning pipeline built on the Shadow Robot Grasping Dataset (~1M sensor readings, about 30 columns). Each grasp attempt is turned into a single feature vector. The pipeline then trains and compares classical classifiers, ensemble methods, regressors, clustering, and a small reinforcement-learning agent. One command runs everything and saves all figures, metrics and a comparison report.

Why "confidence-aware"?
A robot needs more than a yes/no answer before it lifts an object. This project gives three kinds of output:

Output	What it tells the robot	How it's produced
Stable / Unstable label	Will this grasp hold?	Classifiers + ensembles
Success probability	How sure is the model?	predict_proba scores, measured with ROC-AUC
Robustness score	How stable is the grasp (continuous)?	Linear and polynomial regression
The probability and robustness score let a robot rank candidate grasps or re-plan when confidence is low, instead of acting on a hard prediction alone.

Pipeline Overview
Raw CSV (~1M rows of joint readings)
        │
        ▼
1. Preprocessing ─ clean → aggregate per grasp → engineer features → split & scale
        │
        ├──► 2. Classification   KNN · SVM (RBF) · Decision Tree
        ├──► 3. Ensembles        Bagging · Random Forest · AdaBoost · Gradient Boosting · Soft Voting
        ├──► 4. Regression       Linear · Polynomial (degree 2 + Ridge)
        ├──► 5. Clustering       K-Means (k chosen by silhouette, plus k = 2 vs. label)
        │
6. Q-Learning ─ grid-world navigation agent (separate from the grasp data)
        │
        ▼
7. Comparative Analysis ─ master results table, comparison charts, Markdown summary
1. Preprocessing (preprocessing.py)
Loads the CSV with float32 downcasting to keep memory use down.
Removes duplicate rows, fills missing values, and drops non-finite robustness values.
Aggregates per experiment: each grasp has many time-steps. These are reduced to mean, std and last values for the position, velocity and effort of all 9 joints (3 fingers × 3 joints).
Derived features: overall torque, velocity and position magnitude; torque and velocity per finger; and the spread of torque across fingers (uneven loading often means an unstable grasp).
Labels: robustness is split at the median into stable (1) and unstable (0), which gives balanced classes. The raw robustness value is kept as the regression target.
Stratified 80/20 train/test split and StandardScaler, giving 91 features per grasp.
2. Classification (classification.py)
Model	Key settings
K-Nearest Neighbors	k = 7
Support Vector Machine	RBF kernel, C = 1.0, trained on up to 15k samples for speed
Decision Tree	max_depth = 15
Each model is scored on accuracy, precision, recall, F1, ROC-AUC, and training and prediction time.

3. Ensemble Learning (ensemble.py)
Bagging (15 trees), Random Forest (100 trees), AdaBoost (100), Gradient Boosting (60, depth 4), and a soft-voting ensemble that averages the probabilities of KNN, SVM and the Decision Tree.

4. Regression (regression.py)
Predicts the continuous robustness score. Robustness has a very long right tail (from about 0 to 2,900+). The models are therefore trained on log1p(robustness). Predictions are clipped to the training range, converted back with expm1, and scored on the original scale.

Linear Regression
Polynomial Regression: degree-2 expansion of the 20 highest-variance features, regularised with Ridge (α = 1.0)
Scored on RMSE, MAE and R².

5. Clustering (clustering.py)
K-Means searches k = 2 … 8 and picks the best k by silhouette score (Davies–Bouldin is also reported). It then runs k = 2 on its own to check how well unsupervised clusters line up with the stable/unstable label.

6. Reinforcement Learning (qlearning.py)
A tabular Q-Learning agent learns to cross a 7×7 grid with about 20% obstacles (ε-greedy, α = 0.1, γ = 0.95, 1,500 episodes). Grasp prediction covers deciding to grasp; this part covers deciding where to move.

7. Comparative Analysis (comparative_analysis.py)
Collects the metrics from every module into one master table. It plots base models against ensembles and compares training cost, then writes comparative_summary.md.

Project Structure
.
├── data/
│   └── shadow_robot_dataset.csv      ← you download this (see Step 3)
├── src/
│   ├── main.py                       # runs the full pipeline
│   ├── preprocessing.py
│   ├── classification.py
│   ├── ensemble.py
│   ├── regression.py
│   ├── clustering.py
│   ├── qlearning.py
│   └── comparative_analysis.py
└── results/                          # created automatically
    ├── figures/
    ├── metrics/
    ├── models/
    └── comparative_summary.md
Getting Started
Prerequisites
Python 3.9+
About 4 GB of free RAM for the full dataset (use --sample on smaller machines)
About 600 MB of disk space for the dataset
Step 1: Clone the repository
git clone https://github.com/MuzammilSHK/Confidence-Aware-Grasp-Success-Prediction-ML-Model-.git
cd Confidence-Aware-Grasp-Success-Prediction-ML-Model-
Step 2: Create a virtual environment and install dependencies
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install numpy pandas scikit-learn matplotlib seaborn
scikit-learn >= 1.2 is required (the Bagging ensemble uses the estimator= argument).

Step 3: Download the dataset
Download the Shadow Robot Grasping Dataset from Kaggle: https://www.kaggle.com/datasets/ugocupcic/grasping-dataset
Rename the CSV to shadow_robot_dataset.csv if needed. It is about 508 MB, with 992,641 rows and 30 columns.
Put it in a data/ folder at the repository root, next to src/:
data/shadow_robot_dataset.csv
Step 4: Run the full pipeline
cd src
python main.py
For a faster run on a smaller machine, use only part of the grasp experiments:

python main.py --sample 0.3      # 30% of experiments
Step 5 (optional): Run a single module
Each module can also run on its own from inside src/:

python preprocessing.py          # dataset summary + shapes
python classification.py         # KNN, SVM, Decision Tree only
python ensemble.py               # ensembles (uses a 20% sample by default)
python qlearning.py              # Q-Learning only (no dataset needed)
Outputs
Everything is written to results/:

Folder / file	Contents
figures/classification_*.png	Confusion matrices, metric bar chart, ROC curves
figures/ensemble_*.png	Ensemble confusion matrices, base vs. ensemble comparison
figures/regression_*.png	Predicted vs. actual robustness, residual histograms
figures/cluster_*.png	k-search plot, PCA projection, cluster-vs-label heatmap
figures/qlearning_*.png	Learning curves, learned policy and path, Q-value heatmap
figures/comparison_*.png	Overall comparison, training-time cost chart
metrics/*.json	Raw metrics for each module
metrics/master_results.csv	All models in one table
comparative_summary.md	Written summary of the results
Tech Stack
Python · NumPy · pandas · scikit-learn · Matplotlib · seaborn

Dataset Credit
Shadow Robot Grasping Dataset by Ugo Cupcic (Kaggle). The median-split stable/unstable label follows his original setup.

Author
Muzammil · GitHub @MuzammilSHK
