# DOTA 2 Big Data Analytics: AWS vs GCP Comparison

Team 18
- Anh Pham (U01836723)
- Jinzhe Bai (U41047897)

Course: MET CS777 - Big Data Analytics
Boston University Metropolitan College

---

## Project Overview

This project compares AWS EMR and GCP Dataproc performance using Dota 2 Pro League match data (2020-2024). We analyze 27.86 GB of professional esports data across 123,691 matches using Apache Spark with both DataFrame and RDD APIs.

Research Questions:
- Which cloud platform delivers better performance for big data analytics?
- How do Spark's DataFrame and RDD APIs compare for complex queries and ML pipelines?
- What are the cost-performance trade-offs between AWS EMR and GCP Dataproc?

Key Results:
- DataFrame API is 6.2x faster than RDD for analytical queries
- GCP Dataproc is 15% faster and 22% cheaper than AWS EMR
- Best ML models: Random Forest (72.7% AUC for classification), Gradient Boosting (R²=0.857 for regression)

---

## Repository Structure
```
cs777-termpaper-team18/
├── README.md                                           
├── code/
│   ├── METCS777-term-paper-code-sample-1.1-Team18.py  
│   ├── METCS777-term-paper-code-sample-1.2-Team18.py  
│   ├── METCS777-term-paper-code-sample-2.1-Team18.py  
│   └── METCS777-term-paper-code-sample-2.2-Team18.py  
├── sample-data.zip/                                    
│   ├── main_metadata.csv                              
│   ├── picks_bans.csv                                 
│   ├── players.csv                                    
│   ├── objectives.csv                                 
│   ├── radiant_gold_adv.csv                           
│   ├── radiant_exp_adv.csv                            
│   ├── teamfights.csv                                 
│   ├── draft_timings.csv                              
│   ├── teams.csv                                      
│   ├── chat.csv                                       
│   ├── cosmetics.csv                                  
│   └── all_word_counts.csv                            
├── results/
│   ├── task1_df_gcp_output.txt                        
│   ├── task1_rdd_gcp_output.txt                       
│   ├── task2_model1_gcp_output.txt                    
│   ├── task2_model2_gcp_output.txt                    
│   ├── output_task1_visualizations_02_hero_trends.png
│   ├── output_task1_visualizations_05_duration_winrate.png
│   └── output_task1_visualizations_12_performance_dashboard.png
└── docs/
    └── METCS777-term-paper-code-sample-doc-Team18.pdf 
```

---

## Dataset

Source: Dota 2 Pro League Matches (2020-2024)
Size: 27.86 GB (full dataset), sample provided in repository
Matches: 123,691 professional games
Files: 12 interconnected CSV files

### Key Files

| File | Description | Key Columns |
|------|-------------|-------------|
| main_metadata.csv | Match outcomes and duration | match_id, duration, radiant_win, radiant_score, dire_score |
| picks_bans.csv | Hero selections | match_id, hero_id, team, is_pick |
| players.csv | Player performance | match_id, kills, deaths, assists, gold_per_min, xp_per_min |
| objectives.csv | Game events | match_id, type, time, team |
| radiant_gold_adv.csv | Gold advantages | match_id, minute, gold |
| teamfights.csv | Teamfight statistics | match_id, start, deaths, gold_delta |

Full Dataset: https://www.kaggle.com/datasets/devinanzelmo/dota-2-matches

---

## Quick Start

### Prerequisites

- Apache Spark 3.x with PySpark
- Python 3.7+
- AWS account or GCP account
- Libraries: matplotlib, pandas, numpy, psutil

### Setup
```bash
# Clone repository
git clone https://github.com/mewwmewie/cs777-termpaper-team18.git
cd cs777-termpaper-team18

# Extract sample data
unzip sample-data.zip

# Install Python dependencies on cluster master node
pip install matplotlib pandas numpy psutil --break-system-packages
```

### AWS EMR Setup
```bash
# Create S3 buckets
aws s3 mb s3://cs777-termpaper-input
aws s3 mb s3://cs777-termpaper-output

# Upload data to S3
aws s3 cp sample-data/ s3://cs777-termpaper-input/ --recursive

# Upload code files
aws s3 cp code/ s3://cs777-termpaper-code/ --recursive

# Create EMR cluster
aws emr create-cluster \
  --name "DOTA2-Analytics-Cluster" \
  --release-label emr-6.10.0 \
  --applications Name=Spark \
  --instance-type m5.xlarge \
  --instance-count 3 \
  --use-default-roles \
  --log-uri s3://cs777-termpaper-logs/
```

### GCP Dataproc Setup
```bash
# Create Cloud Storage buckets
gsutil mb gs://cs777-termpaper-input
gsutil mb gs://cs777-termpaper-output

# Upload data to Cloud Storage
gsutil cp -r sample-data/* gs://cs777-termpaper-input/

# Upload code files
gsutil cp code/* gs://cs777-termpaper-code/

# Create Dataproc cluster
gcloud dataproc clusters create dota2-analytics-cluster \
  --region us-east1 \
  --master-machine-type n1-standard-4 \
  --master-boot-disk-size 50 \
  --num-workers 2 \
  --worker-machine-type n1-standard-4 \
  --worker-boot-disk-size 50 \
  --image-version 2.0-debian10
```

---

## Running the Code

### Task 1.1: Analytics Queries (DataFrame)

**On AWS EMR:**
```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --num-executors 2 \
  --executor-cores 4 \
  --executor-memory 12G \
  --driver-memory 4G \
  s3://cs777-termpaper-code/METCS777-term-paper-code-sample-1.1-Team18.py \
  s3://cs777-termpaper-input/ \
  s3://cs777-termpaper-output/task1-df/
```

**On GCP Dataproc:**
```bash
gcloud dataproc jobs submit pyspark \
  gs://cs777-termpaper-code/METCS777-term-paper-code-sample-1.1-Team18.py \
  --cluster=dota2-analytics-cluster \
  --region=us-east1 \
  -- gs://cs777-termpaper-input/ gs://cs777-termpaper-output/task1-df/
```

**Expected Outputs:**
- `hero_trends.txt` - 612 rows of hero pick/ban statistics by year
- `hero_by_duration.txt` - 372 rows of hero performance by game phase
- `duration_trends.txt` - 49 rows of match duration statistics by month
- `duration_correlation.txt` - 5 rows of win rates by duration bucket
- `visualizations/02_hero_trends.png` - Hero trends chart
- `visualizations/05_duration_winrate.png` - Duration vs win rate chart
- `visualizations/12_performance_dashboard.png` - Performance metrics dashboard
- `performance_metrics.json` - Execution time metrics

### Task 1.2: Analytics Queries (RDD)

**On AWS EMR:**
```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --num-executors 2 \
  --executor-cores 4 \
  --executor-memory 12G \
  --driver-memory 4G \
  s3://cs777-termpaper-code/METCS777-term-paper-code-sample-1.2-Team18.py \
  s3://cs777-termpaper-input/ \
  s3://cs777-termpaper-output/task1-rdd/
```

**On GCP Dataproc:**
```bash
gcloud dataproc jobs submit pyspark \
  gs://cs777-termpaper-code/METCS777-term-paper-code-sample-1.2-Team18.py \
  --cluster=dota2-analytics-cluster \
  --region=us-east1 \
  -- gs://cs777-termpaper-input/ gs://cs777-termpaper-output/task1-rdd/
```

**Expected Outputs:**
Same structure as Task 1.1 but with RDD API performance metrics

### Task 2.1: Match Outcome Prediction

**On AWS EMR:**
```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --num-executors 2 \
  --executor-cores 4 \
  --executor-memory 12G \
  --driver-memory 4G \
  s3://cs777-termpaper-code/METCS777-term-paper-code-sample-2.1-Team18.py \
  s3://cs777-termpaper-input/ \
  s3://cs777-termpaper-output/model1/
```

**On GCP Dataproc:**
```bash
gcloud dataproc jobs submit pyspark \
  gs://cs777-termpaper-code/METCS777-term-paper-code-sample-2.1-Team18.py \
  --cluster=dota2-analytics-cluster \
  --region=us-east1 \
  -- gs://cs777-termpaper-input/ gs://cs777-termpaper-output/model1/
```

**Expected Outputs:**
- `model1_results.json` - Complete metrics in JSON format
- `model1_results_report/part-00000` - Human-readable report with:
  - Classification metrics (AUC, Accuracy) for 3 algorithms
  - Training times for DataFrame and RDD implementations
  - Feature importance rankings
  - Memory usage and inference speed
  - Platform comparison (AWS vs GCP)

### Task 2.2: Match Duration Prediction

**On AWS EMR:**
```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --num-executors 2 \
  --executor-cores 4 \
  --executor-memory 12G \
  --driver-memory 4G \
  s3://cs777-termpaper-code/METCS777-term-paper-code-sample-2.2-Team18.py \
  s3://cs777-termpaper-input/ \
  s3://cs777-termpaper-output/model2/
```

**On GCP Dataproc:**
```bash
gcloud dataproc jobs submit pyspark \
  gs://cs777-termpaper-code/METCS777-term-paper-code-sample-2.2-Team18.py \
  --cluster=dota2-analytics-cluster \
  --region=us-east1 \
  -- gs://cs777-termpaper-input/ gs://cs777-termpaper-output/model2/
```

**Expected Outputs:**
- `model2_results.json` - Complete metrics in JSON format
- `model2_results_report/part-00000` - Human-readable report with:
  - Regression metrics (RMSE, R²) for 3 algorithms
  - Training times for DataFrame and RDD implementations
  - Feature importance rankings
  - Prediction accuracy analysis
  - Platform comparison (AWS vs GCP)

---

## Results Summary

### Task 1: Analytics Query Performance

| Query | GCP DataFrame | GCP RDD | AWS DataFrame | AWS RDD |
|-------|---------------|---------|---------------|---------|
| Hero Trends Over Time | 21.2s | 106.2s | 25.0s | 124.0s |
| Hero Performance by Duration | 4.4s | 24.6s | 5.2s | 28.8s |
| Duration Trends | 1.6s | 23.3s | 1.8s | 27.3s |
| Duration-WinRate Correlation | 0.9s | 19.4s | 1.1s | 22.7s |
| TOTAL | 28.0s | 173.5s | 33.1s | 203.0s |

Key Findings:
- DataFrame API: 6.2x faster than RDD
- GCP: 15% faster than AWS
- Throughput: GCP DF (310 MB/s) > AWS DF (280 MB/s) > GCP RDD (180 MB/s) > AWS RDD (160 MB/s)

### Task 2 Model 1: Match Outcome Prediction

Dataset: 119,801 matches | Features: 25 early-game metrics

| Algorithm | API | Training (GCP) | Accuracy | AUC |
|-----------|-----|----------------|----------|-----|
| Logistic Regression | DataFrame | 17.6s | 64.9% | 0.707 |
| Random Forest | DataFrame | 46.7s | 64.8% | 0.727 |
| Gradient Boosting | DataFrame | 141.8s | 63.9% | 0.717 |
| Random Forest | RDD | 13.4s | 65.4% | 0.655 |
| Gradient Boosting | RDD | 48.7s | 65.3% | 0.655 |

Best Model: Random Forest DataFrame (AUC: 0.727, Training: 46.7s)

Top 5 Features:
1. avg_gold_adv_5_10 (0.1842)
2. total_kills (0.1534)
3. max_hero_damage (0.1286)
4. avg_gpm_all (0.1127)
5. kill_death_ratio (0.0945)

### Task 2 Model 2: Match Duration Prediction

Dataset: 123,491 matches | Features: 26 match characteristics

| Algorithm | API | Training (GCP) | RMSE | R² |
|-----------|-----|----------------|------|-----|
| Linear Regression | DataFrame | 2.5s | 265.9 | 0.798 |
| Linear Regression | RDD | 23.0s | Failed | Failed |
| Random Forest | DataFrame | 59.3s | 226.1 | 0.854 |
| Random Forest | RDD | 15.8s | 241.6 | 0.834 |
| Gradient Boosting | DataFrame | 145.4s | 223.2 | 0.857 |
| Gradient Boosting | RDD | 55.1s | 236.3 | 0.842 |

Best Model: Gradient Boosting DataFrame (RMSE: 223.2s, R²: 0.857)

Top 5 Features:
1. total_score (0.2156)
2. total_objectives (0.1893)
3. gold_swing_std (0.1467)
4. avg_hero_damage (0.1234)
5. total_teamfights (0.0987)

### Cost Comparison

| Task | AWS EMR | GCP Dataproc | Savings |
|------|---------|--------------|---------|
| Startup Time | 8 min | 3 min | 62% faster |
| Hourly Rate | $0.219 | $0.170 | 22% cheaper |
| Task 1 | $0.98 | $0.76 | 22% |
| Model 1 (Best) | $1.86 | $1.72 | 8% |
| Model 2 (Best) | $2.35 | $2.21 | 6% |
| Total Project | $5.19 | $4.69 | 10% |

---

## Key Findings

### DataFrame vs RDD API

DataFrame Advantages:
- 6.2x faster for complex queries
- 2-3% more accurate ML models
- 100x faster inference (0.02ms vs 2ms)
- More reliable (RDD Linear Regression failed with NaN)
- Automatic optimization via Catalyst

RDD Advantages:
- 40% faster feature engineering
- 60% higher data throughput
- More control for advanced optimization

### GCP Dataproc vs AWS EMR

GCP Advantages:
- 15% faster execution
- 62% faster startup (3 min vs 8 min)
- 22% lower hourly cost
- 10% total project savings

AWS Advantages:
- Better ecosystem integration (Redshift, Lambda, Glue)
- More instance options
- Stronger enterprise support

### Machine Learning Insights

Match Outcome Prediction:
- 72.7% AUC using only first 10 minutes of data
- Gold advantage at 5-10 minutes is strongest predictor
- Early game determines 72% of match outcomes

Match Duration Prediction:
- R²=0.857 (explains 85.7% of variance)
- Average prediction error: ±3.7 minutes
- Total score (combined kills) is most predictive feature

---

## Environment Configuration

### AWS EMR Cluster
```bash
Cluster Configuration:
- Release: emr-6.10.0
- Applications: Spark 3.3.0
- Master: 1 x m5.xlarge (4 vCPU, 16 GB RAM)
- Core: 2 x m5.xlarge (4 vCPU, 16 GB RAM)
- Storage: Amazon S3
- Total Resources: 12 vCPU, 48 GB RAM
- Cost: $0.219/hour
```

### GCP Dataproc Cluster
```bash
Cluster Configuration:
- Image: 2.0-debian10
- Spark: 3.1.3
- Master: 1 x n1-standard-4 (4 vCPU, 15 GB RAM)
- Workers: 2 x n1-standard-4 (4 vCPU, 15 GB RAM)
- Storage: Google Cloud Storage
- Total Resources: 12 vCPU, 45 GB RAM
- Cost: $0.170/hour
```

### Spark Configuration

Both platforms use identical Spark settings:
```python
spark.sql.adaptive.enabled = true
spark.sql.shuffle.partitions = 200
spark.default.parallelism = 200
spark.memory.fraction = 0.8
spark.memory.storageFraction = 0.3
spark.executor.memory = 12G
spark.driver.memory = 4G
```

---

## Code Description

### Task 1.1: Analytics Queries (DataFrame)

File: `METCS777-term-paper-code-sample-1.1-Team18.py`

Performs 4 analytical queries using DataFrame API:
1. Hero Trends Over Time - Aggregates picks/bans by year and hero_id, calculates pick_rate and ban_rate
2. Hero Performance by Duration - Joins picks_bans with metadata, categorizes by game phase, calculates win rates
3. Duration Trends - Aggregates match duration by year/month with statistics (avg, min, max, stddev)
4. Duration-WinRate Correlation - Groups matches by duration buckets, calculates radiant win percentages

Output: Text files with query results, JSON performance metrics, PNG visualizations

### Task 1.2: Analytics Queries (RDD)

File: `METCS777-term-paper-code-sample-1.2-Team18.py`

Implements identical queries using RDD API for performance comparison. Uses map, reduceByKey, and join operations with manual optimization.

Output: Same structure as Task 1.1 with RDD performance metrics

### Task 2.1: Match Outcome Prediction

File: `METCS777-term-paper-code-sample-2.1-Team18.py`

Classification pipeline with 25 early-game features:
- Early objectives: first_blood_count, early_tower_kills, early_courier_kills
- Gold/XP advantages: avg_gold_adv_5_10, max_gold_adv_5_10, gold_volatility
- Team composition: radiant_pick_count, dire_pick_count
- Player stats: avg_gpm_all, avg_xpm_all, total_kills, kill_death_ratio

Algorithms: Logistic Regression, Random Forest, Gradient Boosting
Evaluation: AUC, Accuracy, Training Time, Inference Speed
Comparison: DataFrame vs RDD implementations

### Task 2.2: Match Duration Prediction

File: `METCS777-term-paper-code-sample-2.2-Team18.py`

Regression pipeline with 26 match features:
- Match info: leagueid, radiant_score, dire_score, score_difference
- Objectives: total_objectives, total_towers, total_barracks
- Teamfights: total_teamfights, avg_deaths_per_teamfight
- Economy: max_gold_lead, min_gold_lead, gold_swing_std
- Player performance: avg_gpm, avg_xpm, avg_hero_damage

Algorithms: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor
Evaluation: RMSE, R², Training Time
Comparison: DataFrame vs RDD implementations

---

## Sample Insights

Match Duration Trends:
- 2021: Average 2,100 seconds (35 minutes)
- 2024: Average 1,950 seconds (32.5 minutes)
- Game became 7% faster over 4 years

Win Rate Patterns:
- 0-20 min: 58.46% Radiant (one-sided matches)
- 30-40 min: 49.60% Radiant (balanced gameplay)
- 50+ min: 50.59% Radiant (close games)

---

## Technologies

- Big Data: Apache Spark 3.x (PySpark)
- Cloud: AWS EMR, GCP Dataproc
- Storage: Amazon S3, Google Cloud Storage
- ML: Spark MLlib (DataFrame + RDD APIs)
- Visualization: Matplotlib
- Languages: Python 3.7+

---

## Troubleshooting

### Out of Memory Errors

Increase executor memory:
```bash
--executor-memory 16G --driver-memory 8G
```

### RDD Linear Regression NaN Error

This is a known issue with RDD implementation. Use DataFrame API instead.

### Slow Performance

Verify correct number of executors and enable adaptive query execution:
```bash
--conf spark.sql.adaptive.enabled=true
```

### S3/GCS Access Denied

Ensure IAM roles have proper permissions:
- AWS: `AmazonS3FullAccess` and `AmazonEMRFullAccessPolicy_v2`
- GCP: `Storage Admin` and `Dataproc Worker`

---

## Cleanup

### AWS EMR
```bash
# Terminate cluster
aws emr terminate-clusters --cluster-ids j-XXXXXXXXXXXXX

# Delete S3 buckets
aws s3 rb s3://cs777-termpaper-input --force
aws s3 rb s3://cs777-termpaper-output --force
aws s3 rb s3://cs777-termpaper-code --force
```

### GCP Dataproc
```bash
# Delete cluster
gcloud dataproc clusters delete dota2-analytics-cluster --region=us-east1

# Delete Cloud Storage buckets
gsutil -m rm -r gs://cs777-termpaper-input
gsutil -m rm -r gs://cs777-termpaper-output
gsutil -m rm -r gs://cs777-termpaper-code
```

---

## Documentation

Complete documentation in `docs/METCS777-term-paper-code-sample-doc-Team18.pdf`:
- Detailed environment setup
- Step-by-step execution guide
- Comprehensive results analysis
- Dataset and feature explanations

---

## Team Contributions

Jinzhe Bai (U41047897):
- AWS EMR infrastructure setup
- DataFrame API implementation
- Model 1 development
- Cost analysis

Anh Pham (U01836723):
- GCP Dataproc infrastructure setup
- RDD API implementation
- Model 2 development
- Visualization creation

---

## Citations

- Dataset: https://www.kaggle.com/datasets/devinanzelmo/dota-2-matches
- Apache Spark: https://spark.apache.org/docs/latest/
- AWS EMR: https://docs.aws.amazon.com/emr/
- GCP Dataproc: https://cloud.google.com/dataproc/docs


Last Updated: October 31, 2024
