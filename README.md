# Netflix-Data-Cleaning-Analysis-Visualization
The project aims to explore and analyze Netflix’s dataset to uncover patterns in its content library, visualize key insights, predict content types using machine learning, and provide a simple recommendation system based on genres.
Objective:
The project aims to explore and analyze Netflix’s dataset to uncover patterns in its content library, visualize key insights, predict content types using machine learning, and provide a simple recommendation system based on genres.

Dataset:

Source: netflix1.csv

Key columns: title, type, director, country, date_added, release_year, rating, duration, listed_in (genres), description

Key Steps:

Data Cleaning & Preprocessing

Removed duplicates and irrelevant missing values (director, country)

Converted date_added to datetime format

Extracted year_added and month

Created new features: num_genres, duration_minutes, famous_director, content_age

Exploratory Data Analysis (EDA)

Content type distribution (Movies vs TV Shows)

Top genres and directors

Country-wise content distribution

Ratings breakdown

Temporal trends in content addition

WordCloud of movie titles

Interactive visualizations (Plotly) – choropleth maps, animated bar charts

Machine Learning Model

Model: Random Forest Classifier

Target: Classify content as Movie or TV Show

Features: release_year, num_genres, duration_minutes, country_encoded, rating_encoded

Achieved ~96% accuracy

Recommendation System

Genre-based filtering: Suggests titles similar to a given show/movie based on primary genre

Key Insights:

Dominant Genres: International Movies & Dramas

Top Content Producers: USA and India

Trend: Increasing TV show releases post-2015; movie additions peaked earlier

Common Ratings: TV-MA and TV-14

Conclusion:
The project demonstrates how data cleaning, visualization, and basic ML can generate valuable insights from entertainment datasets. The recommendation system enhances content discovery, while the classification model accurately predicts content type.

Future Scope:

Incorporate collaborative filtering for personalized recommendations

Sentiment analysis on descriptions

Predict popularity scores for titles
