# Retailrocket Recommendation System

This project is a step-by-step implementation of a real-world
e-commerce recommendation system using the RetailRocket dataset.

## Phase 1 – Baseline Model

In Phase 1, the goal is to build a clean and reproducible
machine learning pipeline.

### What is implemented
- Data exploration using Jupyter Notebook
- Binary purchase label creation (transaction vs non-transaction)
- Basic time-based features (hour, day of week)
- Logistic Regression baseline model
- Train/test split and evaluation

### Dataset
RetailRocket implicit feedback dataset:
- events.csv (used in Phase 1)



## Phase 2 – Feature Engineering

- Merged item category metadata
- Added category-based features
- Handled class imbalance
- Saved trained model as .pkl


## Phase 3 – Recommendation Logic (Retrieval + Ranking)

In Phase 3, the trained machine learning model is converted into a
working recommendation system.

### What is implemented
- Candidate generation using popular items per category
- Reuse of trained model for purchase probability scoring
- Ranking of candidate items by predicted probability
- Top-N recommendation output

### Recommendation Flow
1. Identify user’s most recent interacted category
2. Retrieve candidate items from the same category
3. Score candidates using the trained ML model
4. Rank items and return Top-N recommendations

## Phase 4 – FastAPI Recommendation Service

In Phase 4, the recommendation logic was exposed as a REST API using FastAPI.

### What is implemented
- FastAPI-based service for real-time recommendations
- Endpoint: `/recommend/{user_id}`
- Query parameter `n` to control number of recommendations
- Model and data loaded once at application startup
- JSON response with recommended item IDs and scores

## Phase 5 – Docker Containerization

In Phase 5, the FastAPI-based recommendation service was containerized using Docker
to ensure reproducibility and environment consistency.

### What is implemented
- Dockerfile to build a lightweight Python image for the FastAPI service
- docker-compose configuration to manage service runtime
- Volume mounting for external data and trained models
- Port exposure for local access to the API

### Why Docker is used
- Eliminates "works on my machine" issues
- Ensures consistent runtime across environments
- Prepares the service for cloud deployment

### How to run locally

Build and run using Docker Compose:
docker-compose up