# 🏏 Sports Performance Analysis Dashboard

## 📌 Overview

The **Sports Performance Analysis Dashboard** is an interactive web application built using **Streamlit**. It provides in-depth insights into player performance, team statistics, and machine learning model comparisons using sports (cricket) data.

This dashboard enables users to:

* Analyze batting and bowling performance
* Explore team-level statistics and visualizations
* Generate heatmaps for match insights
* Train and compare multiple machine learning models

---

## 🚀 Features

### 👤 Player Analysis

* Select a **batter** and view:

  * Total Runs
  * Batting Average
  * Strike Rate
* Select a **bowler** and view:

  * Total Wickets
  * Bowling Average
  * Strike Rate
  * Economy Rate

### 📊 Team Statistics

* Most wins by team (bar chart)
* Top 10 run scorers
* Top 10 wicket takers
* Match participation distribution (pie chart)
* Runs distribution (violin plot)
* Histogram of wickets per match

### 🔥 Heatmaps

* Runs scored by teams across overs
* Wickets taken by teams across overs

### 🤖 Machine Learning Models

Compare multiple models:

* Logistic Regression
* Gaussian Naive Bayes
* Decision Tree
* Random Forest
* XGBoost
* AdaBoost
* MLP Classifier

Metrics displayed:

* Accuracy
* ROC AUC Score
* Training Time
* Classification Report
* Confusion Matrix

---

## 📂 Project Structure

```
├── app.py                     # Main Streamlit application
├── deliveries.csv             # Raw dataset
├── processed_data.parquet     # Cached processed dataset
├── README.md                  # Project documentation
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd sports-performance-dashboard
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 📦 Dependencies

* streamlit
* pandas
* plotly
* matplotlib
* seaborn
* scikit-learn
* xgboost

---

## 📈 Data Processing

* Data is loaded from `deliveries.csv`
* Team names are cleaned and standardized
* Processed data is cached as `processed_data.parquet` for faster loading

---

## 🧠 Model Training

* Uses `total_runs` as the target variable
* Uses `batsman_runs` as feature input
* Data is split into training and testing sets
* Standard scaling is applied before training

---

## 🎨 UI Highlights

* Custom CSS styling for better UI/UX
* Sidebar for player and model selection
* Interactive Plotly charts
* Clean layout using Streamlit columns

---

## ⚠️ Notes

* Ensure dataset (`deliveries.csv`) is present in the root directory
* XGBoost may require additional installation:

```bash
pip install xgboost
```

---

## 📌 Future Improvements

* Add more features for model training
* Include match-level predictions
* Enhance UI with filters and comparisons
* Deploy on cloud (Streamlit Cloud / AWS / Heroku)

---

## 👨‍💻 Author

Developed as a **Sports Analytics Project** using Streamlit and Machine Learning.

---

## 📜 License

This project is for educational purposes and can be modified or extended.

---
