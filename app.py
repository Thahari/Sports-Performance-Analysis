import streamlit as st
import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import time
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Sports Performance Analysis Dashboard", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color:#f0f8ff;  
        color:#333333;  
    }
    .stTitle, .stHeader {
        color:#333333;  
    }
    .stSidebar .sidebar-content {
        background-color:#ffffff;  
        color:#333333;  
    }
    .stSidebar .sidebar .sidebar-header {
        color:#333333;  
    }
    .css-1d391kg {
        color:#333333;  
    }
    .stMetric {
        background-color:#ffeb3b;  
        color:#333333;  
        border-radius: 10px;
        padding: 10px;
    }
    .stPlotlyChart {
        margin-top: 20px;
    }
    .stColumn {
        padding: 10px;
    }
    .css-1kyxreq {
        padding: 5px;
    }
    .stTextInput, .stSelectbox, .stButton {
        background-color:#ffcc00; 
        color:#333333; 
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
    }
    .stTextInput input, .stSelectbox select {
        color:#333333;  
        background-color:#ffcc00; 
    }
    .stSelectbox select {
        font-weight: bold;
        color: #333333;  /* Color of the selected text */
        background-color: #ffcc00;  /* Color of the select box */
        border: 2px solid #ffcc00;  /* Border color */
    }
    .stTextInput:focus, .stSelectbox:focus, .stButton:focus {
        border: 2px solid#1e90ff;  
    }
    /* Add hover effects */
    .stTextInput:hover, .stSelectbox:hover, .stButton:hover {
        background-color:#ffb300;  
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("Sports Performance Analysis Dashboard")

@st.cache_resource
def load_data():
    try:
        df = pd.read_parquet("processed_data.parquet")
    except FileNotFoundError:
        df = pd.read_csv("deliveries.csv")
        df['batting_team'] = df['batting_team'].str.strip().str.lower()
        df['bowling_team'] = df['bowling_team'].str.strip().str.lower()

        team_replacements = {
            "rising pune supergiant": "rising pune supergiants",
            "pune warriors": "rising pune supergiants",
            "kings xi punjab": "punjab kings",
            "royal challengers bangalore": "royal challengers bengaluru",
            "gujarat lions": "gujarat titans",
            "delhi daredevils": "delhi capitals",
            "deccan chargers": "sunrisers hyderabad"
        }
        df.replace({"batting_team": team_replacements, "bowling_team": team_replacements}, inplace=True)

        df.to_parquet("processed_data.parquet")
    return df

df = load_data()

st.sidebar.header("Player Selection")
selected_batter = st.sidebar.selectbox("Select a Batter", df['batter'].unique(), key="batter_selectbox")
selected_bowler = st.sidebar.selectbox("Select a Bowler", df['bowler'].unique(), key="bowler_selectbox")

st.subheader(f"Player Performance: {selected_batter}")
batter_data = df[df['batter'] == selected_batter]
total_runs = batter_data['batsman_runs'].sum()
times_dismissed = batter_data['player_dismissed'].notna().sum()  
batting_avg = total_runs / times_dismissed if times_dismissed > 0 else total_runs
strike_rate = (total_runs / batter_data['ball'].count()) * 100 if batter_data['ball'].count() > 0 else 0

st.markdown("Batting Key Player Metrics")
col1, col2, col3 = st.columns(3)
col1.metric(label="Total Runs", value=total_runs)
col2.metric(label="Batting Average", value=round(batting_avg, 2))
col3.metric(label="Strike Rate", value=round(strike_rate, 2))

st.subheader(f"Bowler Performance: {selected_bowler}")
bowler_data = df[df['bowler'] == selected_bowler]
total_wickets = bowler_data['player_dismissed'].count()
bowling_avg = bowler_data['total_runs'].sum() / total_wickets if total_wickets > 0 else 0
strike_rate_bowler = (bowler_data['ball'].count() / total_wickets) if total_wickets > 0 else 0
economy_rate = (bowler_data['total_runs'].sum() / bowler_data['ball'].count()) * 6 if bowler_data['ball'].count() > 0 else 0

st.markdown("Bowling Key Player Metrics")
col4, col5, col6, col7 = st.columns(4)
col4.metric(label="Total Wickets", value=total_wickets)
col5.metric(label="Bowling Average", value=round(bowling_avg, 2))
col6.metric(label="Strike Rate", value=round(strike_rate_bowler, 2))
col7.metric(label="Economy Rate", value=round(economy_rate, 2))

st.header("Team Statistics")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Most Wins by Team")
    wins = df['batting_team'].value_counts().reset_index()
    wins.columns = ['team', 'wins']
    fig = px.bar(wins, x='team', y='wins', title='Most Wins by Team', labels={'team': 'Team', 'wins': 'Wins'}, color='wins', color_continuous_scale='Purples')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Players with Most Runs")
    player_runs = df.groupby('batter')['batsman_runs'].sum().reset_index()
    top_players = player_runs.nlargest(10, 'batsman_runs')
    fig = px.bar(top_players, x='batter', y='batsman_runs', title='Most Runs by Players', labels={'batter': 'Players', 'batsman_runs': 'Runs'}, color='batsman_runs', color_continuous_scale='sunsetdark')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Histogram of Wickets per Match")
    wickets_in_match = df[df['is_wicket'] == 1].groupby('match_id').size().reset_index(name='wickets_in_match')
    fig = px.histogram(wickets_in_match, x='wickets_in_match', nbins=10, title='Most Wickets Taken in a Match', labels={'wickets_in_match': 'Wickets per Match'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Players with Most Wickets")
    player_wickets = df.groupby('bowler')['player_dismissed'].count().reset_index()
    top_wicket_takers = player_wickets.nlargest(10, 'player_dismissed')
    fig = px.bar(top_wicket_takers, x='bowler', y='player_dismissed', title='Most Wickets by Players', labels={'bowler': 'Players', 'player_dismissed': 'Wickets'}, color='player_dismissed', color_continuous_scale='Greys')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Team Participation in Matches")
    teams_in_matches = pd.concat([df['batting_team'], df['bowling_team']])
    team_counts = teams_in_matches.value_counts()
    fig = px.pie(names=team_counts.index, values=team_counts, title="Participation Percentage of Teams in Matches")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution of Runs Scored by Teams")
    fig = px.violin(df, x='batting_team', y='total_runs', box=True, points='all', title='Distribution of Runs by Teams')
    st.plotly_chart(fig, use_container_width=True)

st.header("Heatmaps of Team Performance")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Heatmap: Runs Scored by Teams")
    heatmap_data = df.pivot_table(index='over', columns='batting_team', values='total_runs', aggfunc='sum')
    fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, colorscale='Reds'))
    fig.update_layout(title="Runs Scored by Teams Over Each Over", xaxis_title="Team", yaxis_title="Over")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Heatmap: Wickets Taken by Teams")
    heatmap_data = df.pivot_table(index='over', columns='bowling_team', values='player_dismissed', aggfunc='count')
    fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, colorscale='Blues'))
    fig.update_layout(title="Wickets Taken by Teams Over Each Over", xaxis_title="Team", yaxis_title="Over")
    st.plotly_chart(fig, use_container_width=True)

st.header("Model Comparison for Sports Data")
st.sidebar.header("Model Selection")
models = [
    ('Logistic Regression', LogisticRegression()),
    ('Gaussian Naive Bayes', GaussianNB()),
    ('Decision Tree', DecisionTreeClassifier(max_depth=12, max_features='sqrt')),
    ('Random Forest', RandomForestClassifier()),
    ('XGBoost', xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)),
    ('AdaBoost', AdaBoostClassifier()),
    ('MLP Classifier', MLPClassifier(max_iter=500))
]

target_column = "total_runs"
feature_columns = ["batsman_runs"]

X = df[feature_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

selected_model = st.sidebar.selectbox("Select model to train", [model[0] for model in models])

def run_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    time_taken = time.time() - start_time
    st.write(f"**Accuracy:** {accuracy:.5f}")
    st.write(f"**ROC Curve:** {roc:.5f}")
    st.write(f"**Time Taken:** {time_taken:.4f}s")
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred, digits=5))

    fig, ax = plt.subplots(figsize=(6, 6))
    y_pred = model.predict(X_test_scaled)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues, normalize='all', ax=ax) #type: ignore
    st.pyplot(fig)

if st.sidebar.button("Run Selected Model"):
    model = dict(models)[selected_model]
    run_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

results = []
for name, model in models:
    start_time = time.time()
    model.fit(X_train_scaled, y_train)  
    training_time = time.time() - start_time
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled), multi_class='ovr') 
    results.append({'Model': name, 'Time (s)': training_time, 'ROC_AUC': roc_auc})

st.header("Model Evaluation")
data = pd.DataFrame(results).sort_values('ROC_AUC', ascending=False)
fig = go.Figure()
fig.add_trace(go.Bar(
    x=data['Model'], 
    y=data['Time (s)'],
    name='Training Time (s)',
    marker_color='violet',
    yaxis='y1'
))
fig.add_trace(go.Scatter(
    x=data['Model'], 
    y=data['ROC_AUC'],
    mode='lines+markers',
    name='ROC AUC',
    line=dict(color='red'),
    yaxis='y2'
))
fig.update_layout(
    title="Model Comparison: ROC AUC and Time",
    xaxis_title="Model",
    yaxis_title="Time (s)",
    yaxis2=dict(
        title="ROC AUC",
        overlaying='y',
        side='right'
    ),
    xaxis_tickangle=-45,
    template="plotly",
    showlegend=True
)
st.plotly_chart(fig)