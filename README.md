# 🏌️ Golf Performance Modeling with Regression & Machine Learning

This project analyzes curated professional golf performance data to identify key factors driving low scores. By leveraging regression techniques and machine learning models, it quantifies how environmental conditions, course characteristics, and player profiles influence outcomes. The goal is to develop a reliable end-to-end modeling pipeline—starting with Ordinary Least Squares (OLS) regression as a baseline—before exploring more advanced predictive techniques.

I built this project when I started playing a lot more golf. My goal was to understand, through data, what factors drive low scores for professionals, and hopefully transfer those insights to my game to lower my handicap.

---

## 📁 Dataset Overview

The dataset is a structured CSV file containing:

- Player-level statistics (e.g., driving accuracy, strokes gained)
- Course metadata (e.g., length, par, rating, slope)
- Weather conditions (e.g., wind, precipitation, temperature)
- Event and round-level data

---

## 🧩 Project Components

### 1. 📊 Data Integration
- Parsed and cleaned raw CSV using `pandas`
- Standardized identifiers across players, courses, and events
- Handled missing values and ensured type consistency

### 2. 🔍 Feature Engineering
- Created domain-specific features such as:
  - *Weather-adjusted Strokes Gained*
  - *Course Aggressiveness Index*
  - *Player Risk-Reward Profiles*
- Normalized course difficulty metrics (e.g., slope, hole complexity)

### 3. 📈 OLS Benchmark Modeling
- Built an Ordinary Least Squares regression model using `statsmodels`
- Included key predictors: weather, course design, player attributes
- Used interaction terms (e.g., wind × aggressiveness) to model non-linear relationships
- Interpreted coefficients for insight into performance drivers

### 4. 🤖 Machine Learning Extension
- Applied models including:
  - Lasso & Ridge Regression (`scikit-learn`)
  - Random Forest & XGBoost
  - K-Means for player archetype clustering
- Evaluated using cross-validation, RMSE, and model interpretability tools like feature importance

---

## 📊 Example Visualizations (Planned/Future)
- Radar plots comparing player profiles
- Heatmaps of weather sensitivity by hole
- Scenario-based score simulations under hypothetical conditions

---

## 🛠️ Tools & Libraries
- Python (Pandas, NumPy, Scikit-learn, Statsmodels, Matplotlib, Seaborn, XGBoost)
- Jupyter Notebook
- Git/GitHub for version control

---

## 📌 Future Directions
- Integrate real-time data sources (e.g., API-based weather updates)
- Expand unsupervised learning to cluster course types and player styles
- Build an interactive dashboard (e.g., with Plotly or Streamlit)
