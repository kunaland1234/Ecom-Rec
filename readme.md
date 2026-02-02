# Retailrocket Recommendation System


A production-ready hybrid recommendation system built from scratch, deployed on AWS with FastAPI. This project evolved through multiple phases, from a simple logistic regression baseline to a sophisticated hybrid model combining LightGBM and Matrix Factorization, with iterative improvements along the way.

## Project Overview

This recommendation system predicts which products users are likely to purchase based on their browsing and interaction history. The system handles both warm-start users (with historical data) and cold-start users (new to the platform) through hybrid approach.

**Key Features:**
- Hybrid recommendation model (LightGBM + Matrix Factorization)
- Handles cold-start problems for both users and items
- Real-time inference API with FastAPI
- Deployed on AWS ECS with automated CI/CD
- Comprehensive feature engineering pipeline

## Dataset

**Source:** [Retailrocket recommender system dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

The dataset contains user interaction events from an e-commerce platform, including:
- User views, add-to-carts, and purchases
- Item metadata and categories
- Temporal information (timestamps)
- ~600K total events across 308K users and 114K items

**Cold Start Challenge:**
- 92% of users are cold-start (new users)
- 19.4% of items are cold-start (new products)

## Project Journey

This project was built incrementally through multiple development phases:

### Phase 1: Foundation & Baseline (Logistic Regression)
- Selected dataset from Kaggle
- Built data preprocessing pipeline
- Implemented basic feature engineering
- Trained logistic regression baseline model
- **Deployed FastAPI service on AWS ECS**
- Set up ECR for container registry
- Established CI/CD with GitHub Actions


### Phase 2: Enhanced Features
- Added temporal features (hour, day of week, etc.)
- Created user behavior aggregations
- Engineered item popularity metrics
- Implemented category affinity features
- Performance improved but still room for growth


### Phase 3: Model Upgrade (LightGBM)
- Switched to gradient boosting for better performance
- Hyperparameter tuning with grid search
- Cross-validation for robust evaluation
- Comprehensive metrics tracking
- **Significant performance jump in ROC-AUC**


### Phase 4: Hybrid System (LightGBM + Matrix Factorization)
- Trained collaborative filtering model (CMF)
- Implemented hybrid scoring strategy
- Handled cold-start gracefully
- Manual MF scoring with factor matrices
- Production-ready recommendation engine


### Phase 5: Hybrid Model Optimization
- Experimented with different mixing weights (α = 0.2, 0.3, 0.4)
- Optimized candidate generation strategy
- Improved feature normalization techniques
- Fine-tuned threshold selection for better precision-recall balance
- Enhanced filtering logic for cold-start scenarios


### Phase 6: Production Deployment
- Dockerized application
- Deployed to AWS ECS with load balancer
- S3 integration for model and file storage
- Automated CI/CD pipeline

## Model Evolution & Performance

### Phase 1: Baseline Logistic Regression
**Objective:** Establish baseline and validate data pipeline

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.682 |
| Average Precision | 0.041 |
| Recall@10 | 0.024 |

**Achievements:**
- End-to-end feature pipeline
- Candidate generation workflow
- FastAPI inference service
- AWS ECS deployment with CI/CD

---

### Phase 2: Feature-Enriched Logistic Regression
**Objective:** Improve performance through better feature engineering

**New Features:**
- Temporal signals (hour, day, weekend)
- User activity aggregates
- Item popularity & conversion rates
- User-item interaction history

| Metric | Value | Improvement |
|--------|-------|-------------|
| ROC-AUC | 0.724 | +6.2% |
| Average Precision | 0.058 | +41.5% |
| Recall@10 | 0.031 | +29.2% |

---

### Phase 3: LightGBM Ranking Model
**Objective:** Capture non-linear interactions with gradient boosting

| Metric | Value | Improvement |
|--------|-------|-------------|
| ROC-AUC | 0.761 | +5.1% |
| Average Precision | 0.089 | +53.4% |
| Precision@10 | 0.006 | - |
| Recall@10 | 0.048 | +54.8% |

**Key Insight:** ~10× improvement over random baseline

---

### Phase 4: Hybrid Model (LightGBM + Matrix Factorization)
**Objective:** Leverage collaborative filtering for latent preferences

**Scoring Formula:**
```python
final_score = (1 - α) * lgbm_score + α * mf_score
# where α = 0.3
```

| Metric | Value | Improvement |
|--------|-------|-------------|
| ROC-AUC | 0.768 | +0.9% |
| Average Precision | 0.095 | +6.7% |
| Precision@10 | 0.008 | +33.3% |
| Recall@10 | 0.062 | +29.2% |

**Coverage Strategy:**
- Warm users/items → Hybrid scoring
- Cold users/items → LightGBM fallback

---

### Phase 5: Optimized Hybrid Model
**Objective:** Fine-tune parameters and candidate generation

**Optimizations:**
- Tuned hybrid weight (α = 0.35)
- Enhanced candidate generation

| Metric | Value | Phase Improvement | Overall Improvement |
|--------|-------|-------------------|---------------------|
| ROC-AUC | 0.779 | +1.4% | **+14.2%** |
| Average Precision | 0.102 | +7.4% | **+148.8%** |
| Precision@10 | 0.010 | +25.0% | **+233.3%** |
| Recall@10 | 0.078 | +25.8% | **+225.0%** |


## Development Notebooks

The project development is documented through Jupyter notebooks in the `notebooks/` directory:

1. **`01_data_exploration.ipynb`**
   - Initial data exploration and understanding
   - Distribution analysis of events, users, and items
   - Identifying data quality issues

2. **`02_item_properties_exploration.ipynb`**
   - Item metadata analysis
   - Category distribution
   - Item popularity patterns

3. **`03_Enhancing_ML_Model.ipynb`**
   - Feature engineering experiments
   - Model comparison (Logistic Regression vs LightGBM)
   - Hyperparameter tuning
   - Performance evaluation

4. **`04_Adding_More_Features.ipynb`**
   - Advanced feature engineering
   - Temporal feature analysis
   - User behavior aggregations
   - Matrix Factorization integration

5. **`05_Hybrid_Optimization.ipynb`**
   - Mixing weight experimentation
   - Warm vs cold user analysis
   - Threshold optimization
   - Final model evaluation

These notebooks serve as:
- Experimental playground for new ideas
- Documentation of decision-making process
- Reproducible analysis for model iterations

### API Endpoints

**Base URL:** `http://ecom-rec-alb-711173304.eu-north-1.elb.amazonaws.com`

#### 1. Health Check
```bash
GET /health
```
Returns system status, model version, and dataset statistics.

**Response Example:**
```json
{
  "status": "healthy",
  "service": "E-Commerce Recommendation API",
  "version": "1.0.1",
  "mode": "Hybrid (LGB + MF with manual scoring)",
  "model_version": "v3",
  "stats": {
    "total_events": 2965031,
    "unique_users": 1407580,
    "unique_items": 235061,
    "date_range": {
      "start": "2015-05-03 03:00:04.384000",
      "end": "2015-09-18 02:59:47.788000"
    }
  },
  "model": {
    "version": "v3",
    "features": 16,
    "mf_users": 1123767,
    "mf_items": 212916,
    "hybrid_alpha": 0.3
  }
}
```

#### 2. Get Recommendations
```bash
GET /recommend/{user_id}?n=5
```
Returns top-N personalized recommendations with score breakdown.

**Parameters:**
- `user_id` (path, required): User ID to generate recommendations for
- `n` (query, optional): Number of recommendations (default: 5, max: 50)

**Response Example:**
```json
{
  "user_id": 3,
  "user_type": "warm",
  "recommendations": [
    {
      "itemid": 48030,
      "score": 0.00017298114424949405,
      "lgb_score": 0.000004239249767015611,
      "mf_score": 0.000566712231375277
    },
    {
      "itemid": 461686,
      "score": 0.00015897191572425336,
      "lgb_score": 0.0000035529387509903823,
      "mf_score": 0.0005216161953285336
    },
    {
      "itemid": 161623,
      "score": 0.00011364403458035286,
      "lgb_score": 0.000004012112508431031,
      "mf_score": 0.0003694518527481705
    },
    {
      "itemid": 190070,
      "score": 0.00010910153524588861,
      "lgb_score": 0.0000023458272167092115,
      "mf_score": 0.0003581981873139739
    },
    {
      "itemid": 367664,
      "score": 0.0001088286014816258,
      "lgb_score": 0.000006475251944003882,
      "mf_score": 0.00034765308373607695
    }
  ],
  "total_candidates_scored": 200
}
```

**Note:** The `user_type` indicates whether the user is "warm" (in MF training) or "cold" (new user).

#### 3. Get User Statistics
```bash
GET /user/{user_id}/stats
```
Returns user behavior statistics and warm/cold status.

**Response Example:**
```json
{
  "user_id": 3,
  "user_type": "warm",
  "total_events": 1,
  "purchases": 0,
  "conversion_rate": 0,
  "unique_items_viewed": 1,
  "unique_categories_viewed": 1,
  "first_seen": "2015-08-01 07:10:35.296000",
  "last_seen": "2015-08-01 07:10:35.296000"
}
```

**Error Responses:**
- `404`: User not found
- `400`: Invalid parameters
- `503`: Service not ready
- `500`: Internal server error

## Deployment

### AWS Infrastructure

**Docker Image:**
- Built with FastAPI and all dependencies
- Stored in AWS ECR (Elastic Container Registry)

**ECS Service:**
- Fargate launch type (serverless containers)
- Auto-scaling based on CPU/memory
- Load balancer for high availability

**S3 Storage:**
- Model artifacts (LightGBM, MF)
- Raw data files
- Versioned model storage

### CI/CD Pipeline

The project uses **GitHub Actions** for automated deployment to AWS.

**Workflow (`.github/workflows`):**
1. **Trigger:** Push to main branch
2. **Build:** Create Docker image with all dependencies
3. **Test:** Run unit tests and validation
4. **Push:** Upload image to AWS ECR
5. **Deploy:** Update ECS service with new image
6. **Verify:** Health check on deployed service

**Deployment Process:**
```
Code Push → GitHub Actions → Docker Build → ECR Push → ECS Update → Live Service
```

## Quick Start

### Test the Live API (No Installation Required!)

The recommendation system is already deployed and running. You can test it immediately:

```python
import requests

# Try the live API
BASE_URL = "http://ecom-rec-alb-711173304.eu-north-1.elb.amazonaws.com"

# Clone repository
git clone https://github.com/yourusername/Ecom-Rec.git
cd Ecom-Rec

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export S3_BUCKET=your-bucket-name
export MODEL_VERSION=v3.0
export MF_ALPHA=0.35

# Run locally
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build image
docker build -t recommendation-api .

# Run container
docker run -p 8000:8000 \
  -e S3_BUCKET=your-bucket \
  -e MODEL_VERSION=v3.0 \
  recommendation-api
```

Or use docker-compose:
```bash
docker-compose up
```


## Repository Structure

```
Ecom-Rec/
├── .github/workflows/     # CI/CD pipeline
├── notebooks/             # Jupyter notebooks for development
│   ├── 01_data_exploration.ipynb
│   ├── 02_item_properties_exploration.ipynb
│   ├── 03_Enhancing_ML_Model.ipynb
│   ├── 04_Adding_More_Features.ipynb
│   └── 05_Hybrid_Optimization.ipynb
├── evaluation_results/    # Model evaluation metrics and plots
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── precision_recall_curve.png
│   ├── precision_recall_at_k.png
│   └── roc_curve.png
├── src/                   # Source code
│   ├── __init__.py
│   ├── build_feature.py   # Feature engineering
│   ├── candidate.py       # Candidate generation
│   ├── evaluate_mf.py     # MF model evaluation
│   ├── evaluation.py      # Model evaluation utilities
│   ├── main.py           # FastAPI application (deployment)
│   ├── preprocess.py     # Data preprocessing
│   ├── recommender.py    # Recommendation engine
│   ├── run_preprocessing.py
│   ├── test_recommender.py
│   ├── train.py          # Model training
│   └── train_mf.py       # Matrix Factorization training
├── Dockerfile            # Container definition
├── docker-compose.yml    # Docker compose for local testing
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## Links

- **Live API:** http://ecom-rec-alb-711173304.eu-north-1.elb.amazonaws.com
- **GitHub Repository:** https://github.com/kunaland1234/Ecom-Rec
