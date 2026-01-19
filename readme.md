# E-Commerce Recommendation System

This project is a step-by-step implementation of a real-world
e-commerce recommendation system using the RetailRocket dataset.

## Current Status: Phase 1 – Baseline Model

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

### How to run
```bash
python src/evaluate.py