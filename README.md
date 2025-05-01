---
title: "Final Project"
format: html
author: Nicole Sanchez Flores, Vanessa Pliego, Debora Camacho, Kimberly By Goytia, Ari Cross
date: "`r format(Sys.Date(), '%B %d, %Y')`"
---

# Overview

In this project, we perform an exploratory data analysis (EDA) and model the **Billboard Top Songs** dataset. This dataset contains information about 5,000 songs, including features such as song title, artist, streams, danceability, and more. We explore the relationships between different song features, perform unsupervised learning (e.g., hierarchical clustering and K-means), and build supervised learning models to predict whether a song will make the Top 10 chart.

# Exploratory Data Analysis

We begin by loading the dataset and performing some initial cleaning. We use the `janitor` package to clean the dataset column names and perform basic exploratory tasks such as examining artist distribution and correlations between numeric features. We use visualizations like bar plots and heatmaps to analyze trends.

```r
library(readr)
library(tidyverse)
library(janitor)
library(pheatmap)
library(corrplot)

music_dataset <- read_csv("music_dataset.csv")
```

We create a plot that shows the distribution of songs by artist and identify the top and bottom 5 artists in the dataset.

The dataset contains 13 variables such as song, artist, streams, daily_streams, genre, release_year, peak_position, and more. Below is a brief description of some of the key variables:

- **song**: Character data - Title of the song.
- **artist**: Character data - Name of the artist.
- **streams**: Numeric - Total number of streams.
- **daily_streams**: Numeric - Average daily streams.
- **danceability**: Numeric - How danceable the song is.
- **energy**: Numeric - Energy level of the song.

# Unsupervised Models

We applied Hierarchical Clustering and K-means clustering to group songs based on features such as streams, danceability, and energy.

```r
hc <- hclust(music_data, method = "complete")
plot(hc, main = "Hierarchical Clustering")
```

We used K-means clustering to further investigate how songs can be grouped into clusters based on their attributes:
![Cluster Plot](https://github.com/NicoleSanchezFlores/SDS-293-Billboard-Top-100/blob/main/Clusterplot.png?raw=true)

```r
km_result <- kmeans(billboard_scaled, centers = 2, nstart = 25)
fviz_cluster(km_result, data = billboard_scaled)
```

# Supervised Learning

## Random Forest

We apply Random Forest to classify top 10 songs:

```r
# Prepare the data
tidy_music_data <- tidy_music_data %>%
  mutate(top10 = ifelse(peak_position <= 10, 1, 0)) %>%
  mutate(top10 = as.factor(top10))

# Split the data into training and testing sets
set.seed(100)
samp <- sample(nrow(tidy_music_data), 0.80 * nrow(tidy_music_data))
train <- tidy_music_data[samp, ]
test <- tidy_music_data[-samp, ]

# Train Random Forest model
rf_model <- randomForest(top10 ~ ., data = train, ntree = 1000, mtry = 5)

# Evaluate the model
rf_model$confusion

# Predict on test set
prediction <- predict(rf_model, newdata = test)
table(prediction, test$top10)

# Calculate accuracy
accuracy <- sum(prediction == test$top10) / nrow(test)
accuracy
```

![Random Forest](https://github.com/NicoleSanchezFlores/SDS-293-Billboard-Top-100/blob/main/Randomforest.png?raw=true)


## Gradient Boosting

We build a Gradient Boosting model for predicting top 10 status.

```r
# Train Gradient Boosting model
set.seed(123)
gbm_model <- gbm(top10 ~ ., data = train, distribution = "bernoulli", n.trees = 5000, interaction.depth = 4, shrinkage = 0.01, cv.folds = 5)

# Make predictions and evaluate RMSE
predictions <- predict(gbm_model, newdata = test, n.trees = gbm_model$n.trees)
rmse <- sqrt(mean((predictions - test$top10)^2))
rmse
```

# Conclusion

The Random Forest model performed well in predicting whether a song would make it to the top 10, with an accuracy of around 91%. We identified key predictors such as streams, danceability, and tik_tok_virality as influential features in determining chart success. Other models, such as Logistic Regression, did not perform as well, highlighting the complexity of predicting top 10 rankings.
