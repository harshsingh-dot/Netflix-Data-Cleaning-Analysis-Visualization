import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('netflix1.csv')

# Check and display basic info
print("Initial shape:", data.shape)
print("Columns:", data.columns)

# Drop duplicates
data.drop_duplicates(inplace=True)

# Drop rows with missing director or country (cast column doesn't exist)
data.dropna(subset=['director', 'country'], inplace=True)

# Convert date
data['date_added'] = pd.to_datetime(data['date_added'], errors='coerce')
data.dropna(subset=['date_added'], inplace=True)

# === Visualizations ===
plt.figure(figsize=(6,4))
sns.countplot(x='type', data=data)
plt.title("Content Type Distribution")
plt.show()

# Genre extraction
data['genres'] = data['listed_in'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
all_genres = sum(data['genres'], [])
pd.Series(all_genres).value_counts().head(10).plot(kind='barh')
plt.title("Top 10 Genres")
plt.show()

# Content added by year
data['year_added'] = data['date_added'].dt.year
plt.figure(figsize=(10,4))
sns.countplot(x='year_added', data=data, order=sorted(data['year_added'].dropna().unique()))
plt.xticks(rotation=45)
plt.title("Content Added Over Years")
plt.show()

# Top directors
top_directors = data['director'].value_counts().head(10)
sns.barplot(x=top_directors.values, y=top_directors.index)
plt.title("Top 10 Directors")
plt.show()

# WordCloud
titles = ' '.join(data[data['type'] == 'Movie']['title'].dropna())
wordcloud = WordCloud(background_color='black', width=800, height=400).generate(titles)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud of Movie Titles")
plt.show()

# Country distribution
data['country'].value_counts().head(10).plot(kind='bar', title='Top 10 Countries by Content')
plt.ylabel("Count")
plt.show()

# Ratings distribution
data['rating'].value_counts().plot(kind='bar', title="Ratings Distribution")
plt.ylabel("Count")
plt.show()

# Monthly trend
data['month'] = data['date_added'].dt.month
monthly_counts = data['month'].value_counts().sort_index()
monthly_counts.plot(kind='line', title="Monthly Content Addition Trend")
plt.xlabel("Month")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# === Feature Engineering ===
data['num_genres'] = data['listed_in'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
data['duration_minutes'] = data['duration'].str.extract('(\d+)').astype(float)
top_directors = data['director'].value_counts().head(10).index
data['famous_director'] = data['director'].apply(lambda x: 1 if x in top_directors else 0)
current_year = pd.Timestamp.now().year
data['content_age'] = current_year - data['release_year']

# === Machine Learning: Movie vs TV Show ===
le = LabelEncoder()
data['country_encoded'] = le.fit_transform(data['country'].astype(str))
data['rating_encoded'] = le.fit_transform(data['rating'].astype(str))

features = ['release_year', 'num_genres', 'duration_minutes', 'country_encoded', 'rating_encoded']
target = data['type'].apply(lambda x: 1 if x == 'Movie' else 0)

X = data[features].fillna(0)
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n🎯 Classification Report:\n")
print(classification_report(y_test, y_pred))

# === Plotly Visualizations ===
fig = px.histogram(data, x='release_year', color='type', title='Content Release by Year')
fig.show()

country_counts = data['country'].value_counts().reset_index()
country_counts.columns = ['country', 'count']
fig = px.choropleth(country_counts, locations='country', locationmode='country names', color='count',
                    title='Netflix Content by Country')
fig.show()

animated_df = data.groupby(['year_added', 'type']).size().reset_index(name='count')
fig = px.bar(animated_df, x='type', y='count', color='type', animation_frame='year_added',
             title='Content Growth Over Time')
fig.show()

# === Recommendation ===
def recommend(title):
    if title not in data['title'].values:
        print("❌ Title not found.")
        return pd.DataFrame()
    genres = data[data['title'] == title]['listed_in'].values[0]
    if not isinstance(genres, str):
        return pd.DataFrame()
    first_genre = genres.split(',')[0]
    similar = data[data['listed_in'].str.contains(first_genre, na=False)]
    return similar[['title', 'listed_in']].drop_duplicates().head(5)

# Try a sample recommendation
print("\n📺 Recommendations based on 'Breaking Bad':\n")
print(recommend('Breaking Bad'))
