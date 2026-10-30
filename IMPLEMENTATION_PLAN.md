# SOCIAL MEDIA ENGAGEMENT ANALYSIS

## Complete Website Development & Machine Learning Implementation Plan

**Project Name:** EngageIQ – Social Media Engagement Analytics & Prediction System

**Project Type:** Full-Stack Web Application with Machine Learning
**Backend:** Python + Flask
**Frontend:** HTML, CSS, JavaScript
**Database:** SQLite
**Machine Learning:** Scikit-learn
**Datasets:** Real-world Facebook engagement datasets from the UCI Machine Learning Repository
**Deployment:** Excluded for now

The objective is to develop a **fully functional social media analytics website that uses real historical engagement data and trained machine learning models to analyze social media performance, identify patterns, and predict potential engagement for future posts.**

Unlike a basic college project that displays charts using hardcoded values, this system will use actual datasets, train models, store results, and generate predictions through the website.

---

# 1. PROJECT OVERVIEW

### 1.1 Problem Statement

Social media platforms generate large amounts of engagement data, including likes, comments, shares, reactions, and impressions.

However, content creators, businesses, and digital marketers often struggle to understand:

* Which content formats generate higher engagement?
* What is the best time to publish a post?
* Which days receive more user interactions?
* How does engagement change over time?
* Which posts perform significantly better than others?
* Can machine learning predict the engagement of a future post?

The proposed system will analyze historical engagement data and use machine learning to identify meaningful patterns and estimate the engagement potential of new posts.

### 1.2 Proposed Solution

Develop a web application where users can:

1. Explore social media engagement statistics.
2. View interactive analytical dashboards.
3. Compare the performance of different post types.
4. Identify high-performing posting times.
5. Examine likes, comments, shares, and reactions.
6. Upload their own engagement datasets.
7. Predict engagement using a trained ML model.
8. Identify patterns behind high-performing posts.
9. Download analytical reports.
10. Explore information about the trained models and their performance.

The system will initially focus on historical **Facebook data**, because we can obtain well-documented datasets containing real engagement measurements.

It will be designed so that support for Instagram, YouTube, or other platforms can be added later using appropriate platform-specific datasets and models.

---

# 2. DATASET SELECTION

This is one of the most important parts of the project.

Rather than generating random engagement values or creating synthetic training data, I recommend using **two real datasets**.

The reason for using two datasets is that one is much larger and better suited to descriptive analytics, while the other contains useful pre-publication features for machine learning predictions.

## DATASET 1: Facebook Live Sellers in Thailand

### Primary Dataset for Social Media Engagement Analysis

**Source:** UCI Machine Learning Repository

[Download Dataset – Facebook Live Sellers in Thailand](https://www.archive.ics.uci.edu/dataset/488/facebook%2Blive%2Bsellers%2Bin%2Bthailand?utm_source=chatgpt.com)

### Dataset Information

| Property          | Details                             |
| ----------------- | ----------------------------------- |
| Dataset name      | Facebook Live Sellers in Thailand   |
| Source            | UCI Machine Learning Repository     |
| Number of records | Approximately 7,050                 |
| Data type         | CSV                                 |
| Platform          | Facebook                            |
| Collection period | March 2012 – June 2018              |
| Original sources  | 10 Thai retail sellers              |
| Categories        | Fashion and cosmetics               |
| Data authenticity | Real historical Facebook engagement |
| License           | Creative Commons Attribution 4.0    |
| Access            | Publicly downloadable               |

The original research describes engagement measurements collected from Facebook posts, including images, videos, statuses, and links. These measurements include comments, shares, likes, and other reactions. ([ScienceDirect][1])

### Available Dataset Columns

| Feature            | Description                | Purpose                             |
| ------------------ | -------------------------- | ----------------------------------- |
| `status_id`        | Unique post identifier     | Data identification                 |
| `status_type`      | Video, Photo, Status, Link | Content analysis                    |
| `status_published` | Publication date and time  | Temporal analysis                   |
| `num_reactions`    | Total reactions            | Engagement analysis                 |
| `num_comments`     | Number of comments         | Interaction analysis                |
| `num_shares`       | Number of shares           | Virality analysis                   |
| `num_likes`        | Number of likes            | Reaction analysis                   |
| `num_loves`        | Love reactions             | Reaction analysis                   |
| `num_wows`         | Wow reactions              | Reaction analysis                   |
| `num_hahas`        | Haha reactions             | Reaction analysis                   |
| `num_sads`         | Sad reactions              | Sentiment-related reaction analysis |
| `num_angrys`       | Angry reactions            | Sentiment-related reaction analysis |

### How This Dataset Will Be Used

**A. Engagement Dashboard**

Calculate and display:

* Total posts analyzed
* Total reactions
* Total comments
* Total shares
* Average interactions per post
* Most common content type
* Highest-engagement content format

**B. Content Performance Analysis**

Compare engagement across:

* Videos
* Photos
* Status posts
* Links

**C. Time-Based Analysis**

Analyze engagement according to:

* Hour of publication
* Day of the week
* Month
* Historical engagement trends

**D. Reaction Analysis**

Compare the distribution of likes, loves, wows, hahas, sads, and angry reactions.

**E. Post Performance Segmentation**

Group posts into low-, medium-, and high-engagement categories for historical analysis.

### Important Dataset Limitation

This dataset does not provide impressions, reach, or follower counts.

Therefore, we can calculate average interactions per post, but we cannot calculate a genuine reach-based engagement rate from this dataset.

For example:

**Total Interactions = Reactions + Comments + Shares**

We must not add likes separately because likes are already included in total reactions.

---

## DATASET 2: Facebook Metrics Dataset

### Primary Dataset for Machine Learning Prediction

**Source:** UCI Machine Learning Repository

[Download Dataset – Facebook Metrics](https://www.archive.ics.uci.edu/dataset/368/facebook%2Bmetrics?utm_source=chatgpt.com)

### Dataset Information

| Property             | Details                             |
| -------------------- | ----------------------------------- |
| Dataset              | Facebook Metrics                    |
| Records              | 500                                 |
| Columns              | 19                                  |
| Platform             | Facebook                            |
| Data collection year | 2014                                |
| Source               | Cosmetics brand Facebook page       |
| Original purpose     | Social media performance prediction |
| Dataset type         | Regression                          |
| License              | CC BY 4.0                           |

The dataset was released alongside research on predicting social media performance. It contains seven features available before a post is published and additional measurements describing the post's actual performance. ([UCI Machine Learning Repository][2])

### Features Available Before Posting

These will become the input fields in our prediction interface.

| Feature          | Description                          |
| ---------------- | ------------------------------------ |
| Page Total Likes | Number of likes on the Facebook page |
| Type             | Photo, Video, Status, or Link        |
| Category         | Action, Product, or Inspiration      |
| Post Month       | Month of publication                 |
| Post Weekday     | Weekday of publication               |
| Post Hour        | Time of publication                  |
| Paid             | Whether the post is promoted         |

These seven features allow the model to make an estimate **before the post is published**.

### Target Variable

The primary prediction target will be:

**Total Interactions**

This represents the total number of likes, comments, and shares received by a post.

We can use a regression model to predict this numerical value.

For example, the system may receive:

| Input                | Example |
| -------------------- | ------- |
| Page followers/likes | 50,000  |
| Post type            | Photo   |
| Category             | Product |
| Month                | October |
| Day                  | Friday  |
| Posting hour         | 6 PM    |
| Paid promotion       | No      |

The trained model will estimate the expected interactions based on patterns learned from historical data.

**The displayed result will be an estimate, not a guaranteed number.** Because this dataset is small, historical, and from one brand, predictions may have considerable uncertainty.

---

# 3. COMPLETE SYSTEM ARCHITECTURE

The website will have two major workflows.

### Workflow A: Historical Engagement Analysis

```text
            USER OPENS WEBSITE
                    |
                    v
            ANALYTICS DASHBOARD
                    |
                    v
          LOAD HISTORICAL DATASET
                    |
                    v
           DATA PREPROCESSING
                    |
                    v
        ENGAGEMENT CALCULATIONS
                    |
                    v
            DATA ANALYSIS
                    |
          +---------+---------+
          |         |         |
          v         v         v
       CONTENT     TIME     REACTION
       ANALYSIS   ANALYSIS   ANALYSIS
          |         |         |
          +---------+---------+
                    |
                    v
          INTERACTIVE CHARTS
                    |
                    v
             FINAL INSIGHTS
```

### Workflow B: ML-Based Engagement Prediction

```text
         USER ENTERS POST DETAILS
                    |
                    v
              FLASK BACKEND
                    |
                    v
             INPUT VALIDATION
                    |
                    v
            FEATURE PREPARATION
                    |
                    v
             TRAINED ML MODEL
                    |
                    v
           ENGAGEMENT PREDICTION
                    |
                    v
              RESULT DISPLAY
                    |
                    v
          SAVE PREDICTION HISTORY
```

The two workflows will be connected through a common website and backend.

Importantly, the two datasets will be processed separately because they represent different populations and contain different features.

---

# 4. PHASE-WISE IMPLEMENTATION PLAN

I recommend dividing the development into **11 phases**, with deployment intentionally excluded.

---

## PHASE 1: Project Setup & Architecture

**Objective:** Establish the project structure and development environment.

### Technologies

| Component        | Technology    |
| ---------------- | ------------- |
| Frontend         | HTML5         |
| Styling          | CSS3          |
| Interactivity    | JavaScript    |
| Charts           | Chart.js      |
| Backend          | Flask         |
| Data processing  | Pandas, NumPy |
| Machine learning | Scikit-learn  |
| Database         | SQLite        |
| Model storage    | Joblib        |
| Development      | VS Code       |

### Implementation Tasks

1. Create a Python project.
2. Establish a Flask application structure.
3. Create separate frontend, backend, dataset, and model directories.
4. Configure Python dependencies.
5. Configure SQLite.
6. Implement basic routing.
7. Create a consistent API response format.
8. Establish centralized error handling and logging.

### Expected Folder Structure

```text
EngageIQ/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── backend/
│   ├── __init__.py
│   ├── routes.py
│   ├── analytics.py
│   ├── predictor.py
│   ├── database.py
│   └── validators.py
│
├── data/
│   ├── raw/
│   │   ├── facebook_live.csv
│   │   └── facebook_metrics.csv
│   │
│   └── processed/
│       ├── engagement_clean.csv
│       └── training_clean.csv
│
├── ml/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── clustering.py
│   └── feature_engineering.py
│
├── models/
│   ├── engagement_model.joblib
│   ├── clustering_model.joblib
│   ├── model_metrics.json
│   └── model_metadata.json
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── analytics.html
│   ├── predictor.html
│   ├── upload.html
│   └── model_performance.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   └── js/
│       ├── dashboard.js
│       ├── charts.js
│       ├── predictor.js
│       └── upload.js
│
├── scripts/
│   ├── download_datasets.py
│   ├── prepare_datasets.py
│   └── train_models.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_analytics.py
│   ├── test_prediction.py
│   └── test_api.py
│
└── docs/
    ├── implementation_plan.md
    ├── model_report.md
    └── dataset_sources.md
```

### Phase 1 Deliverable

A functioning Flask application structure with organized development files.

---

## PHASE 2: Real Dataset Collection & Preparation

**Objective:** Obtain, validate, clean, and prepare both datasets.

This phase is essential because data quality directly affects the credibility of the results.

### Step 1: Download the Datasets

The project will contain a reproducible dataset downloading script.

It will obtain the official UCI datasets using their repository identifiers:

```python
from ucimlrepo import fetch_ucirepo

# Historical engagement analysis
facebook_live = fetch_ucirepo(id=488)

# Machine learning prediction
facebook_metrics = fetch_ucirepo(id=368)
```

The downloaded datasets will be saved locally so that the application does not require an internet connection each time it starts.

### Step 2: Data Validation

Perform the following checks:

| Validation                    | Action                                                                  |
| ----------------------------- | ----------------------------------------------------------------------- |
| Missing values                | Detect and handle missing records                                       |
| Duplicate records             | Identify and remove inappropriate duplicates                            |
| Incorrect data types          | Convert numeric and date columns                                        |
| Negative interactions         | Flag invalid values                                                     |
| Invalid timestamps            | Remove or flag unusable dates                                           |
| Unexpected content categories | Standardize supported categories                                        |
| Extreme values                | Identify outliers without automatically removing legitimate viral posts |

### Step 3: Feature Engineering

Create additional fields for analytics.

**For the 7,050-post dataset:**

```text
total_interactions
publication_hour
publication_day
publication_month
publication_year
engagement_category
```

The interaction metric will be calculated as:

$$
\text{Total Interactions}
=
\text{Reactions}+\text{Comments}+\text{Shares}
$$

Other calculated metrics:

$$
\text{Average Interactions}
=
\frac{\text{Total Interactions}}{\text{Number of Posts}}
$$

Where available, engagement rate by reach can be calculated as:

$$
\text{Engagement Rate}
=
\frac{\text{Interactions}}{\text{Reach}}\times100
$$

The reach-based calculation will only be shown for datasets that contain appropriate reach measurements.

### Step 4: Save Cleaned Datasets

The processed data will be saved as:

```text
data/processed/engagement_clean.csv
data/processed/training_clean.csv
```

### Phase 2 Deliverable

Two cleaned and validated real datasets, plus a reproducible preprocessing script.

---

# PHASE 3: Exploratory Data Analysis

**Objective:** Understand engagement trends before implementing prediction models.

We will use Pandas, NumPy, and visualization libraries.

### Analysis 1: Overall Engagement

Calculate:

* Number of analyzed posts
* Total interactions
* Average and median interactions
* Reaction totals
* Comment totals
* Share totals
* Engagement distribution

Median values are important because a small number of viral posts can significantly increase average engagement.

### Analysis 2: Content-Type Performance

Compare the engagement generated by:

| Content Type | Metrics                             |
| ------------ | ----------------------------------- |
| Videos       | Average reactions, comments, shares |
| Photos       | Average reactions, comments, shares |
| Status posts | Average reactions, comments, shares |
| Links        | Average reactions, comments, shares |

Create a bar chart showing average interactions by content type.

### Analysis 3: Posting Time Analysis

Analyze engagement across different hours.

Example visualization:

```text
AVERAGE INTERACTIONS BY POSTING HOUR

12 AM  ██████
03 AM  ███
06 AM  ███████
09 AM  █████████
12 PM  ███████████
03 PM  █████████
06 PM  █████████████
09 PM  ████████
```

*Illustrative chart only. Actual results will come from the dataset.*

We will also calculate the number of posts in each time slot, so that a high average based on very few posts is not mistaken for a reliable pattern.

### Analysis 4: Engagement Trends

Create charts for:

* Monthly engagement
* Engagement over time
* Average comments by content type
* Share distribution
* Reaction distribution

### Analysis 5: Correlation Analysis

Analyze relationships between:

* Reactions and comments
* Reactions and shares
* Comments and shares
* Content characteristics and outcomes

These results describe associations, not necessarily causal relationships.

### Phase 3 Deliverable

A complete exploratory data analysis module providing the data required for the website's charts.

---

# PHASE 4: MACHINE LEARNING MODEL DEVELOPMENT

This will be the core technical component.

The goal is to create an actual trained model using the Facebook Metrics dataset, not a manually programmed scoring system.

## 4.1 Machine Learning Problem

**Primary Task:** Regression

**Input:** Post characteristics before publication.

**Output:** Predicted number of interactions.

### Why Regression?

Because engagement is represented as a numerical quantity.

For example:

```text
Predicted interactions: 185
```

The website can then interpret this estimate as relatively low, medium, or high compared with the training data.

These categories will be based on historical percentiles rather than arbitrary hardcoded numbers.

---

## 4.2 Model Inputs

The model will accept these seven features:

```python
features = [
    "Page total likes",
    "Type",
    "Category",
    "Post Month",
    "Post Weekday",
    "Post Hour",
    "Paid"
]
```

### Target

```python
target = "Total Interactions"
```

We will use the target directly from the dataset after checking its consistency.

### Important: Preventing Data Leakage

The model must not use variables such as:

* Lifetime post reach
* Lifetime impressions
* Actual engaged users
* Actual likes
* Actual comments
* Actual shares

These are available only after publication.

Including them would make the model appear more accurate during evaluation while preventing realistic predictions for future posts.

---

## 4.3 Data Splitting

The dataset contains only 500 records, so we must avoid wasting too much training data.

### Proposed Strategy

Use a chronological test set based on publication month where feasible.

| Dataset Portion | Purpose                        |
| --------------- | ------------------------------ |
| Earlier 80%     | Model development and training |
| Later 20%       | Final performance evaluation   |

Within the training portion, perform cross-validation for selecting model parameters.

The time split will be checked for sufficient observations and representative content types.

### Why Time-Based Evaluation?

It more closely represents the real application:

**Train using historical posts → Predict future posts.**

A randomized split may also be reported for comparison, but the primary evaluation should reflect real-world usage as closely as possible.

---

## 4.4 Data Preprocessing

The training pipeline will use Scikit-learn.

### Numerical Features

Example:

```text
Page total likes
```

Apply:

* Missing-value imputation
* Scaling where necessary

### Categorical Features

Example:

```text
Type
Category
Month
Weekday
Hour
Paid
```

Apply:

* Categorical encoding
* Missing-value handling
* Consistent transformation during training and prediction

We can implement these operations using:

```python
ColumnTransformer
Pipeline
OneHotEncoder
SimpleImputer
```

This ensures that the model receives exactly the same preprocessing during training and website predictions.

---

## 4.5 Algorithms to Train

We should compare multiple machine learning algorithms rather than assume one is the best.

### Model 1: Linear Regression / Ridge Regression

**Purpose:** Establish a simple baseline.

Advantages:

* Easy to understand
* Fast training
* Useful comparison model

Limitations:

* May struggle with nonlinear relationships

### Model 2: Decision Tree Regressor

**Purpose:** Learn nonlinear relationships between post characteristics and engagement.

Advantages:

* Interpretable
* Handles nonlinear patterns

Limitations:

* Can overfit small datasets

### Model 3: Random Forest Regressor

**Purpose:** Predict engagement using multiple decision trees.

Advantages:

* Captures nonlinear relationships
* Handles complex feature interactions
* Often more robust than an individual tree

Limitations:

* Can still overfit
* May not generalize well beyond the training data

### Model 4: Gradient Boosting Regressor

**Purpose:** Compare a more advanced ensemble learning approach.

Advantages:

* Models complex relationships
* Supports nonlinear prediction

Limitations:

* Requires careful tuning
* Can also overfit small datasets

### Model Comparison

| Algorithm         | Primary Purpose                |
| ----------------- | ------------------------------ |
| Ridge Regression  | Baseline                       |
| Decision Tree     | Interpretable nonlinear model  |
| Random Forest     | Ensemble prediction            |
| Gradient Boosting | Advanced regression comparison |

All four algorithms will be trained and evaluated.

The best-performing model will be selected based on validation results.

---

# PHASE 5: MODEL TRAINING & EVALUATION

This phase converts the prepared dataset into a trained model that the website can actually use.

## 5.1 Training Pipeline

```text
          FACEBOOK METRICS DATASET
                    |
                    v
              DATA CLEANING
                    |
                    v
             FEATURE SELECTION
                    |
                    v
             TRAIN / TEST SPLIT
                    |
                    v
             PREPROCESSING
                    |
                    v
          TRAIN MULTIPLE ALGORITHMS
                    |
          +---------+----------+
          |         |          |
          v         v          v
       LINEAR    DECISION    RANDOM
       MODEL       TREE      FOREST
          |         |          |
          +---------+----------+
                    |
                    v
             GRADIENT BOOSTING
                    |
                    v
             CROSS-VALIDATION
                    |
                    v
             MODEL COMPARISON
                    |
                    v
          SELECT BEST VALIDATED MODEL
                    |
                    v
            FINAL TEST EVALUATION
                    |
                    v
              SAVE MODEL
```

## 5.2 Evaluation Metrics

We will calculate:

### A. Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted engagement.

$$
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
$$

Lower is better.

### B. Root Mean Squared Error (RMSE)

Penalizes large prediction errors more heavily.

$$
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
$$

Lower is better.

### C. R² Score

Measures how much variation in engagement the model explains.

Higher is generally better, but negative scores are possible when a model performs worse than a simple baseline.

### D. Baseline Comparison

We will also evaluate a simple model that predicts the median training engagement for every post.

The trained model should outperform this baseline before we present it as a useful prediction system.

## 5.3 Handling Skewed Engagement Data

Social media engagement is often heavily skewed.

Most posts receive relatively few interactions, while a small number receive exceptionally high engagement.

Therefore, we will evaluate a `log1p` transformation of the target variable during training.

Predictions will be converted back to the original interaction-count scale.

All reported evaluation metrics will use the original scale.

## 5.4 Expected Model Output

The final training process will produce files such as:

```text
models/
│
├── engagement_model.joblib
├── model_metrics.json
├── model_metadata.json
└── training_report.json
```

### Example Training Report Structure

```json
{
  "dataset": "Facebook Metrics",
  "training_records": 400,
  "testing_records": 100,
  "target": "Total Interactions",
  "selected_model": "Random Forest Regressor",
  "metrics": {
    "mae": null,
    "rmse": null,
    "r2": null
  }
}
```

This is an example of the report structure, **not a claim about actual training results**. The final record counts, selected model, and metric values will be generated by the real training run.

### Important Requirement

The completed project should include:

* The actual trained model file
* Reproducible training code
* Real dataset used for training
* Evaluation metrics
* Actual-versus-predicted visualization
* Model comparison results
* Clear model limitations

The user should not need to train the model manually just to open the website.

The training script will still be provided so the model can be retrained.

### Phase 5 Deliverable

A saved and evaluated machine learning model ready to integrate with Flask.

---

# PHASE 6: ADVANCED ENGAGEMENT ANALYSIS USING CLUSTERING

This is an additional ML component that will make the project more substantial.

Instead of using machine learning only for predicting interactions, the website can also identify groups of posts with similar engagement patterns.

## 6.1 Clustering Model

Use:

**K-Means Clustering**

Dataset:

**Facebook Live Sellers in Thailand**

### Input Features

```python
clustering_features = [
    "num_reactions",
    "num_comments",
    "num_shares"
]
```

### How It Works

1. Select engagement features.
2. Apply a suitable transformation to reduce skew.
3. Standardize the features.
4. Train K-Means for several candidate cluster counts.
5. Compare silhouette scores and cluster sizes.
6. Select an interpretable clustering solution.
7. Generate meaningful descriptions of the resulting clusters.

### Possible Cluster Interpretation

| Cluster   | Interpretation                       |
| --------- | ------------------------------------ |
| Cluster A | Low overall interaction              |
| Cluster B | Reaction-dominated engagement        |
| Cluster C | High commenting and sharing activity |

These are example descriptions. Actual cluster characteristics will be determined after examining the trained model.

This provides a second meaningful use of machine learning.

**Regression predicts future engagement. Clustering analyzes patterns in posts that have already been published.**

### Phase 6 Deliverable

A trained clustering model and an engagement segmentation feature for the website.

---

# PHASE 7: BACKEND DEVELOPMENT

**Objective:** Connect the machine learning models, analytics calculations, database, and website.

### Backend Framework

Python Flask

The backend will be responsible for:

* Loading datasets
* Reading processed analytics data
* Calculating engagement metrics
* Serving dashboard results
* Receiving prediction inputs
* Loading the trained regression model
* Generating predictions
* Reading clustering results
* Saving prediction history
* Handling dataset uploads

## 7.1 Proposed API Endpoints

| Endpoint                   | Method | Purpose                   |
| -------------------------- | ------ | ------------------------- |
| `/api/overview`            | GET    | Dashboard summary         |
| `/api/engagement-trends`   | GET    | Historical engagement     |
| `/api/content-performance` | GET    | Content-type comparison   |
| `/api/posting-times`       | GET    | Time-based analysis       |
| `/api/reactions`           | GET    | Reaction distribution     |
| `/api/segments`            | GET    | Engagement clusters       |
| `/api/predict`             | POST   | ML engagement prediction  |
| `/api/model-info`          | GET    | Model details and metrics |
| `/api/upload`              | POST   | CSV dataset upload        |
| `/api/history`             | GET    | Previous predictions      |
| `/api/export`              | GET    | Download analytical data  |

### Example Prediction Request

```json
{
  "page_likes": 50000,
  "post_type": "Photo",
  "category": 2,
  "month": 10,
  "weekday": 5,
  "hour": 18,
  "paid": 0
}
```

The backend will validate the values, preprocess the input, and submit it to the trained model.

### Example Prediction Response

```json
{
  "predicted_interactions": 185,
  "engagement_category": "Medium",
  "model": "Selected Trained Regressor",
  "prediction_type": "Historical-data estimate"
}
```

The numbers above illustrate the response format, not an actual prediction.

The final API may also provide a historical comparison and an uncertainty indicator where supported by model evaluation.

---

# PHASE 8: DATABASE IMPLEMENTATION

**Objective:** Store user-generated information locally.

### Database

SQLite

A complex database server is unnecessary for this college project.

## 8.1 Database Tables

### Table 1: Dataset Records

Stores metadata about imported datasets.

| Column      | Purpose            |
| ----------- | ------------------ |
| ID          | Dataset identifier |
| Name        | Dataset name       |
| Source      | Dataset origin     |
| Records     | Number of records  |
| Uploaded At | Import timestamp   |
| Status      | Validation status  |

### Table 2: Prediction History

Stores previous predictions.

| Column        | Purpose                |
| ------------- | ---------------------- |
| ID            | Prediction identifier  |
| Post Type     | Selected content type  |
| Posting Time  | Selected posting time  |
| Page Likes    | Input                  |
| Paid          | Promotion status       |
| Prediction    | Estimated interactions |
| Model Version | Model used             |
| Created At    | Prediction timestamp   |

### Table 3: Model Training History

Stores model metadata.

| Column        | Purpose            |
| ------------- | ------------------ |
| Model ID      | Unique identifier  |
| Algorithm     | Algorithm name     |
| Dataset       | Training dataset   |
| MAE           | Evaluation metric  |
| RMSE          | Evaluation metric  |
| R²            | Evaluation metric  |
| Training Date | Training timestamp |

### Phase 8 Deliverable

A functioning SQLite database integrated with the application.

---

# PHASE 9: FRONTEND WEBSITE DEVELOPMENT

This is where the project becomes a complete user-facing application.

The website should look like a modern analytics platform rather than a basic HTML form.

### Proposed Design

**Application Name:** EngageIQ

**Design Style:** Modern SaaS Analytics Dashboard

Recommended appearance:

* Dark navy background
* Purple and blue gradient highlights
* Rounded dashboard cards
* Interactive charts
* Smooth transitions
* Collapsible sidebar
* Responsive layouts
* Professional typography

## Page 1: Overview Dashboard

The landing page will display the main social media engagement statistics.

### Suggested Layout

```text
┌────────────────────────────────────────────────────────┐
│  EngageIQ                           Analytics Dashboard │
├─────────────┬──────────────────────────────────────────┤
│             │                                          │
│  Dashboard  │   Social Media Engagement Overview       │
│             │                                          │
│  Analytics  │  ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│             │  │  Total   │ │  Total   │ │ Average  │  │
│  Prediction │  │  Posts   │ │ Reactions│ │Interaction│  │
│             │  └──────────┘ └──────────┘ └──────────┘  │
│  Content    │                                          │
│             │  ┌────────────────────────────────────┐  │
│  Segments   │  │       Engagement Over Time         │  │
│             │  │                                    │  │
│  Dataset    │  │          LINE CHART                │  │
│             │  │                                    │  │
│  Model Info │  └────────────────────────────────────┘  │
│             │                                          │
│  History    │  ┌─────────────────┐ ┌────────────────┐ │
│             │  │ Content Type    │ │ Reaction       │ │
│             │  │ Comparison      │ │ Distribution   │ │
│             │  └─────────────────┘ └────────────────┘ │
└─────────────┴──────────────────────────────────────────┘
```

## Page 2: Engagement Analytics

This page will contain detailed charts.

### Charts to Implement

| Chart                           | Visualization     |
| ------------------------------- | ----------------- |
| Total interactions over time    | Line chart        |
| Reactions vs comments vs shares | Bar chart         |
| Post-type comparison            | Grouped bar chart |
| Average engagement by hour      | Bar chart         |
| Day and hour performance        | Heatmap           |
| Reaction breakdown              | Doughnut chart    |
| Engagement distribution         | Histogram         |
| Top-performing posts            | Sortable table    |

### Filters

Users should be able to filter the data by:

* Content type
* Date range
* Engagement category
* Reaction type

Charts and summary cards will update according to the selected filters.

---

## Page 3: Machine Learning Prediction

This is one of the most important pages.

### Prediction Interface

**Heading:**

Predict Your Post Engagement

**Description:**

Enter the characteristics of your planned Facebook post to estimate its potential engagement using a model trained on historical data.

### Input Form

| Input            | UI Element      |
| ---------------- | --------------- |
| Page likes       | Number input    |
| Post type        | Dropdown        |
| Content category | Dropdown        |
| Month            | Dropdown        |
| Day              | Dropdown        |
| Posting hour     | Time selector   |
| Paid promotion   | Yes/No selector |

### Prediction Button

**Predict Engagement**

When clicked, the website will send the information to Flask.

Flask will load the trained model and return a prediction.

### Result Display

```text
┌─────────────────────────────────────────────┐
│                                             │
│         ENGAGEMENT PREDICTION               │
│                                             │
│            185 Interactions                 │
│                                             │
│       Estimated Engagement: MEDIUM          │
│                                             │
│    Based on historical Facebook data        │
│                                             │
│    ┌─────────────────────────────────┐      │
│    │  Historical Comparison          │      │
│    │                                 │      │
│    │  Your predicted engagement      │      │
│    │  compared with similar posts    │      │
│    └─────────────────────────────────┘      │
│                                             │
└─────────────────────────────────────────────┘
```

*Illustrative layout only.*

### Additional Feature: Compare Two Posting Scenarios

Allow users to compare two planned posts.

For example:

**Scenario A:** Photo posted at 12 PM.

**Scenario B:** Photo posted at 6 PM.

The model predicts each scenario independently.

The website then displays the difference between the predictions.

This helps demonstrate how the model can support posting decisions, although the difference is not proof that changing the time will cause higher engagement.

---

## Page 4: Content Performance

This page will focus on identifying historically successful post formats.

Users can explore:

* Average interactions per content type
* Comments per content type
* Shares per content type
* Most common reaction categories
* Performance distributions
* Most and least interactive post formats

### Example Output

```text
CONTENT PERFORMANCE ANALYSIS

Videos      ██████████████
Photos      ███████████
Statuses    ████████
Links       █████
```

The displayed values will be calculated from the actual dataset.

---

## Page 5: Engagement Segmentation

This page will display the clustering results.

Features:

* Cluster distribution
* Average interactions by cluster
* Cluster characteristics
* Representative posts
* Scatter plot of engagement metrics

This adds a useful exploratory ML feature to the application.

---

## Page 6: Dataset Upload & Management

Users will be able to upload CSV files containing their own social media statistics.

### Expected Dataset Fields

A supported custom dataset might contain:

```text
post_id
post_type
post_date
likes
comments
shares
reactions
reach
impressions
```

The upload page will:

1. Accept a CSV file.
2. Validate its extension and size.
3. Identify supported columns.
4. Check missing values.
5. Display a data preview.
6. Create relevant charts.
7. Allow users to explore their own data.

### Important Distinction

Uploading a dataset will enable **descriptive analytics** when suitable engagement columns are available.

It will not automatically retrain the predictive model.

Retraining requires the uploaded dataset to include compatible predictor columns, a valid target, and sufficient high-quality records.

---

## Page 7: Model Performance

This page will demonstrate the technical ML implementation to project evaluators.

### Display

* Dataset used for training
* Training and testing record counts
* Algorithms compared
* Selected regression model
* MAE
* RMSE
* R² score
* Model version
* Training date
* Actual vs predicted chart
* Feature importance, where supported
* Model limitations

A model comparison table will also be shown.

This is valuable for a college project presentation because it demonstrates that the prediction functionality comes from an actual training process.

---

# PHASE 10: FRONTEND-BACKEND INTEGRATION

**Objective:** Ensure every part of the application works together.

### Integration Flow

1. User opens the website.
2. Flask serves the HTML interface.
3. JavaScript requests analytics from the backend.
4. The backend reads processed data.
5. Analytics are calculated and returned.
6. Chart.js renders the results.
7. The user enters prediction inputs.
8. The trained model generates an estimate.
9. The result appears on the website.
10. The prediction is saved to SQLite.

### Additional Requirements

The website should:

* Work on desktop and mobile.
* Display loading indicators.
* Validate form inputs.
* Handle API errors.
* Allow users to reset filters.
* Display informative messages if data is unavailable.
* Avoid hardcoded model predictions.
* Use the bundled local datasets without requiring live Facebook API access.

### Phase 10 Deliverable

A fully connected frontend, backend, analytics module, prediction system, and database.

---

# PHASE 11: TESTING, VALIDATION & DOCUMENTATION

This phase is essential before considering the website complete.

## 11.1 Functional Testing

| Test                   | Expected Result                  |
| ---------------------- | -------------------------------- |
| Open dashboard         | Displays dataset statistics      |
| Apply filters          | Charts update correctly          |
| Submit prediction      | Returns model-generated result   |
| Enter invalid values   | Displays validation error        |
| Upload valid CSV       | Processes dataset                |
| Upload invalid CSV     | Shows a meaningful error         |
| View model performance | Displays real evaluation results |
| Open history           | Displays stored predictions      |
| Export report          | Generates downloadable file      |
| Restart server         | Saved history remains available  |

## 11.2 Machine Learning Validation

Verify that:

* The real dataset is used.
* Training and testing data are separated correctly.
* Preprocessing is fitted only on training data.
* Prediction features do not contain future information.
* The selected model outperforms or is accurately compared against the baseline.
* The model artifact loads successfully.
* Repeated predictions return consistent results.
* The Flask API uses the same preprocessing pipeline as model training.

## 11.3 Security & Reliability

Include:

* CSV size limits
* File validation
* Input sanitization
* SQL parameterization
* Graceful API error handling
* Safe model loading from trusted local files
* Upload storage outside public static directories
* Dataset attribution and license information

## 11.4 Documentation

The final project will contain:

**README.md**

Installation and running instructions.

**IMPLEMENTATION_PLAN.md**

Complete phase-wise development documentation.

**DATASET_SOURCES.md**

Dataset descriptions, sources, and attribution.

**MODEL_REPORT.md**

Machine learning algorithms, evaluation methodology, results, and limitations.

**API_DOCUMENTATION.md**

Backend endpoint descriptions.

### Phase 11 Deliverable

A tested and documented college project that can run locally.

---

# 5. COMPLETE MODEL TRAINING STRATEGY

To summarize the exact approach for the two ML components:

| Component           | Engagement Prediction                | Engagement Clustering                 |
| ------------------- | ------------------------------------ | ------------------------------------- |
| Dataset             | Facebook Metrics                     | Facebook Live Sellers                 |
| Records             | 500                                  | Approximately 7,050                   |
| ML Type             | Supervised learning                  | Unsupervised learning                 |
| Algorithm           | Best validated regression model      | K-Means                               |
| Input               | Pre-publication post characteristics | Historical engagement metrics         |
| Output              | Predicted interactions               | Engagement clusters                   |
| Evaluation          | MAE, RMSE, R²                        | Silhouette score and cluster analysis |
| Saved Model         | `engagement_model.joblib`            | `clustering_model.joblib`             |
| Website Integration | Prediction page                      | Segmentation page                     |

### How the Training Will Be Executed

The project will include scripts similar to:

```bash
python scripts/download_datasets.py
```

This downloads the official datasets.

```bash
python scripts/prepare_datasets.py
```

This validates and prepares the data.

```bash
python scripts/train_models.py
```

This trains the regression algorithms, evaluates them, selects the final model, and trains the clustering model.

### Model Persistence

We will save the entire fitted prediction pipeline using Joblib.

This is important because the website needs both the trained model and the fitted preprocessing transformations.

During normal operation, Flask will load the saved model rather than train a new model for every prediction.

**The completed ZIP should already contain the trained model**, so these commands will primarily be needed for reproducibility or retraining.

---

# 6. WHAT THE FINAL PROJECT WILL DELIVER

The intended final ZIP package will contain the following components.

| Component                           | Included in Planned Build                          |
| ----------------------------------- | -------------------------------------------------- |
| Complete responsive website         | Yes                                                |
| Python Flask backend                | Yes                                                |
| SQLite database                     | Yes                                                |
| Real historical engagement datasets | Yes                                                |
| Dataset preprocessing scripts       | Yes                                                |
| Engagement analytics dashboard      | Yes                                                |
| Interactive visualizations          | Yes                                                |
| Content-type analysis               | Yes                                                |
| Posting-time analysis               | Yes                                                |
| Reaction analysis                   | Yes                                                |
| Trained regression model            | Yes, subject to successful training and validation |
| Trained clustering model            | Yes, subject to successful training and validation |
| ML prediction interface             | Yes                                                |
| Scenario comparison                 | Yes                                                |
| Model evaluation dashboard          | Yes                                                |
| CSV upload                          | Yes                                                |
| Prediction history                  | Yes                                                |
| Data export                         | Yes                                                |
| Installation instructions           | Yes                                                |
| Automated tests                     | Yes                                                |
| Project documentation               | Yes                                                |
| Cloud deployment                    | No                                                 |

---

# 7. IMPORTANT TECHNICAL DECISIONS

There are a few decisions I would make to keep the project academically credible.

### 1. Do Not Use Synthetic Engagement Data

The dashboard will be populated from real datasets.

No randomly generated likes or fabricated engagement statistics will be presented as authentic observations.

### 2. Do Not Hardcode ML Predictions

The prediction page will call an actual trained Scikit-learn model.

Predictions must be produced through model inference.

### 3. Avoid Misleading Accuracy Claims

Social media engagement is affected by many factors that these datasets do not capture, including post quality, audience interests, algorithms, and current trends.

The prediction model might achieve only modest accuracy.

If so, that limitation should be documented instead of artificially inflating performance.

### 4. Make Dataset Age Clear

The datasets contain historical Facebook posts from 2012–2018 and 2014, respectively.

They are suitable for a college demonstration and for studying machine learning methods, but they do not represent current social media performance.

### 5. Keep Analytics Separate From Prediction

The larger dataset will drive engagement analysis.

The smaller dataset will drive pre-publication prediction.

This is preferable to improperly combining records from different sources simply to increase the training size.

---

# 8. RECOMMENDED DEVELOPMENT ORDER

This is the implementation sequence I recommend following.

| Phase | Development Area                    | Priority |
| ----- | ----------------------------------- | -------- |
| 1     | Project setup and architecture      | Critical |
| 2     | Download and clean real datasets    | Critical |
| 3     | Exploratory data analysis           | Critical |
| 4     | Build regression training pipeline  | Critical |
| 5     | Train, evaluate, and save the model | Critical |
| 6     | Implement K-Means clustering        | High     |
| 7     | Flask backend and REST APIs         | Critical |
| 8     | Database integration                | High     |
| 9     | Website frontend and dashboards     | Critical |
| 10    | End-to-end integration              | Critical |
| 11    | Testing and documentation           | Critical |
| 12    | Deployment                          | Excluded |

The main milestone is completing Phases 1–5 before integrating the prediction page. That way, the website can be developed around a real trained model rather than a placeholder.

---

# 9. FINAL WEBSITE FUNCTIONALITY

Once implemented, the application should support three primary demonstrations for a college project.

### Demonstration 1: Engagement Analytics

A user opens EngageIQ and immediately sees historical Facebook engagement data.

They explore interactive charts, filter content types, compare posting times, and examine engagement trends.

All statistics are calculated using actual dataset records.

### Demonstration 2: Machine Learning Prediction

A user enters the characteristics of a planned post.

The frontend submits the information to Flask.

The backend processes the data and passes it to the saved model.

The model predicts the expected total interactions.

The result is displayed and stored in the database.

### Demonstration 3: Engagement Pattern Discovery

A user opens the clustering page.

The website displays the engagement groups discovered from historical posts.

Users can examine how reactions, comments, and shares differ between those groups.

This demonstrates unsupervised machine learning alongside the supervised prediction model.

---

## 10. FINAL RECOMMENDATION

For this project, I recommend building **EngageIQ as a complete local analytics platform with two machine learning capabilities**.

The strongest feature combination would be:

| Feature                 | Implementation                                |
| ----------------------- | --------------------------------------------- |
| Historical analytics    | Real 7,050-post dataset                       |
| Predictive analysis     | Trained regression model using 500 real posts |
| Engagement segmentation | K-Means clustering                            |
| Visualization           | Chart.js                                      |
| Web application         | Flask + HTML/CSS/JavaScript                   |
| Data management         | SQLite + CSV upload                           |
| Evaluation              | Real ML performance metrics                   |

This is more substantial than a simple website with an ML prediction form because it demonstrates **data collection, preprocessing, feature engineering, exploratory analysis, supervised learning, unsupervised learning, model evaluation, full-stack integration, and database management**.

### Current Development Status

**The implementation plan and dataset selection are complete.** The datasets are publicly available and their sources have been verified.

The next development step is to download the actual CSV files, train and evaluate the models, and build the working application around them.

The final ZIP should contain the website source code, real datasets, saved trained models, database initialization, automated tests, documentation, and Windows/VS Code running instructions.

**Deployment will remain excluded, as requested.**

[1]: https://www.sciencedirect.com/science/article/pii/S2352340920305552?utm_source=chatgpt.com "Dataset on usage and engagement patterns for Facebook Live sellers in Thailand - ScienceDirect"
[2]: https://www.archive.ics.uci.edu/dataset/368/facebook%2Bmetrics?utm_source=chatgpt.com "UCI Machine Learning Repository"
