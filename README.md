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

## Quick Start

### Prerequisites

- Apache Spark 3.x with PySpark
- Python 3.7+
- Cloud account (AWS or GCP)
- Libraries: matplotlib, pandas, numpy, psutil

### Setup
```bash
# Clone repository
git clone https://github.com/mewwmewie/cs777-termpaper-team18.git
cd cs777-termpaper-team18

# Extract sample data
unzip sample-data.zip

# Install Python dependencies (on cluster)
pip install matplotlib pandas numpy psutil --break-system-packages
```

### Run on GCP Dataproc
```bash
# Upload data to Cloud Storage
gsutil cp -r sample-data/* gs://your-bucket/input/

# Submit Task 1 (DataFrame)
gcloud dataproc jobs submit pyspark \
  code/METCS777-term-paper-code-sample-1.1-Team18.py \
  --cluster=your-cluster --region=us-east1 \
  -- gs://your-bucket/input/ gs://your-bucket/output/

# Submit Task 1 (RDD)
gcloud dataproc jobs submit pyspark \
  code/METCS777-term-paper-code-sample-1.2-Team18.py \
  --cluster=your-cluster --region=us-east1 \
  -- gs://your-bucket/input/ gs://your-bucket/output/

# Submit Model 1
gcloud dataproc jobs submit pyspark \
  code/METCS777-term-paper-code-sample-2.1-Team18.py \
  --cluster=your-cluster --region=us-east1 \
  -- gs://your-bucket/input/ gs://your-bucket/output/

# Submit Model 2
gcloud dataproc jobs submit pyspark \
  code/METCS777-term-paper-code-sample-2.2-Team18.py \
  --cluster=your-cluster --region=us-east1 \
  -- gs://your-bucket/input/ gs://your-bucket/output/
```

### Run on AWS EMR
```bash
# Upload data to S3
aws s3 cp sample-data/ s3://your-bucket/input/ --recursive

# Submit jobs (replace with appropriate paths)
spark-submit --master yarn --deploy-mode cluster \
  --num-executors 2 --executor-cores 4 \
  --executor-memory 12G --driver-memory 4G \
  s3://your-bucket/code/METCS777-term-paper-code-sample-1.1-Team18.py \
  s3://your-bucket/input/ s3://your-bucket/output/
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

### Task 2 Model 1: Match Outcome Prediction (Classification)

Dataset: 119,801 matches | Features: 25 early-game metrics

| Algorithm | API | Training (GCP) | Accuracy | AUC |
|-----------|-----|----------------|----------|-----|
| Logistic Regression | DataFrame | 17.6s | 64.9% | 0.707 |
| Random Forest | DataFrame | 46.7s | 64.8% | 0.727 (BEST) |
| Gradient Boosting | DataFrame | 141.8s | 63.9% | 0.717 |
| Random Forest | RDD | 13.4s | 65.4% | 0.655 |
| Gradient Boosting | RDD | 48.7s | 65.3% | 0.655 |

Best Model: Random Forest DataFrame (AUC: 0.727)

Top Features:
1. avg_gold_adv_5_10 (0.1842) - Gold advantage at 5-10 minutes
2. total_kills (0.1534)
3. max_hero_damage (0.1286)
4. avg_gpm_all (0.1127)
5. kill_death_ratio (0.0945)

### Task 2 Model 2: Match Duration Prediction (Regression)

Dataset: 123,491 matches | Features: 26 match characteristics

| Algorithm | API | Training (GCP) | RMSE | R² |
|-----------|-----|----------------|------|-----|
| Linear Regression | DataFrame | 2.5s | 265.9 | 0.798 |
| Linear Regression | RDD | 23.0s | Failed (NaN) | Failed |
| Random Forest | DataFrame | 59.3s | 226.1 | 0.854 |
| Random Forest | RDD | 15.8s | 241.6 | 0.834 |
| Gradient Boosting | DataFrame | 145.4s | 223.2 | 0.857 (BEST) |
| Gradient Boosting | RDD | 55.1s | 236.3 | 0.842 |

Best Model: Gradient Boosting DataFrame (RMSE: 223.2s, R²: 0.857)

Top Features:
1. total_score (0.2156) - Combined kills
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

### 1. DataFrame vs RDD API

DataFrame Advantages:
- 6.2x faster for complex queries
- 2-3% more accurate ML models
- 100x faster inference (0.02ms vs 2ms)
- More reliable (RDD Linear Regression failed)
- Automatic optimization (Catalyst)

RDD Advantages:
- 40% faster feature engineering
- 60% higher data throughput
- More control for advanced users
- Prone to failures (NaN errors)

### 2. GCP Dataproc vs AWS EMR

GCP Advantages:
- 15% faster execution
- 62% faster startup (3 min vs 8 min)
- 22% lower hourly cost
- 10% total project savings

AWS Advantages:
- Better ecosystem integration
- More instance options
- Stronger enterprise support

### 3. Machine Learning Insights

Match Outcome Prediction:
- 72.7% AUC using only first 10 minutes
- Gold advantage is strongest predictor
- Early game determines 72% of outcomes

Match Duration Prediction:
- R²=0.857 (explains 85.7% variance)
- Average error: ±3.7 minutes
- Total score most predictive feature

---

## Visualizations

Sample Outputs:

Hero Trends Chart (02_hero_trends.png)
Line charts showing pick/ban rates for top 5 trending heroes over time

Duration vs Win Rate (05_duration_winrate.png)
Bar chart of Radiant win rate by match duration + distribution pie chart

Performance Dashboard (12_performance_dashboard.png)
Execution time comparison, statistics, and category analysis

---

## Environment Setup

### Cloud Configuration

AWS EMR:
- Cluster: 1 master + 2 core nodes
- Instance: m5.xlarge (4 vCPU, 16 GB RAM)
- Storage: Amazon S3
- Cost: $0.219/hour

GCP Dataproc:
- Cluster: 1 master + 2 workers
- Instance: n1-standard-4 (4 vCPU, 15 GB RAM)
- Storage: Google Cloud Storage
- Cost: $0.170/hour

### Software Requirements
```bash
# Core requirements
Apache Spark 3.x
Python 3.7+
PySpark

# Python libraries
pip install matplotlib pandas numpy psutil --break-system-packages
```

---

## Code Description

### Task 1.1: Analytics Queries (DataFrame)

File: METCS777-term-paper-code-sample-1.1-Team18.py

Performs 4 analytical queries using DataFrame API:
1. Hero Trends Over Time - Pick/ban rates by year and hero
2. Hero Performance by Duration - Win rates by game phase
3. Duration Trends - Average match duration by month
4. Duration-WinRate Correlation - Win rates by duration bucket

Output: Text files, JSON metrics, PNG visualizations

### Task 1.2: Analytics Queries (RDD)

File: METCS777-term-paper-code-sample-1.2-Team18.py

Same queries as 1.1 but using RDD API for performance comparison.

### Task 2.1: Match Outcome Prediction

File: METCS777-term-paper-code-sample-2.1-Team18.py

Classification pipeline comparing DataFrame and RDD:
- Features: 25 early-game metrics (first 10 minutes)
- Algorithms: Logistic Regression, Random Forest, Gradient Boosting
- Evaluation: AUC, Accuracy, Training Time, Inference Speed

Output: JSON results, detailed report with feature importance

### Task 2.2: Match Duration Prediction

File: METCS777-term-paper-code-sample-2.2-Team18.py

Regression pipeline comparing DataFrame and RDD:
- Features: 26 match characteristics
- Algorithms: Linear Regression, Random Forest, Gradient Boosting
- Evaluation: RMSE, R², Training Time

Output: JSON results, detailed report with feature importance

---

## Sample Insights

Match Duration Trends:
- 2021: Average 2,100 seconds (35 min)
- 2024: Average 1,950 seconds (32.5 min)
- Game became 7% faster over 4 years

Win Rate Patterns:
- 0-20 min: 58.46% Radiant (stomps)
- 30-40 min: 49.60% Radiant (balanced)
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

## Documentation

Complete documentation available in docs/METCS777-term-paper-code-sample-doc-Team18.pdf:
- Detailed environment setup instructions
- Step-by-step code execution guide
- Comprehensive results analysis
- Dataset description and feature explanations

---

## Team Contributions

Anh Pham (U01836723):
- AWS EMR setup and configuration
- DataFrame API implementation
- Model 1 development
- Cost analysis

Jinzhe Bai (U41047897):
- GCP Dataproc setup and configuration
- RDD API implementation
- Model 2 development
- Visualization creation

---

## Recommendations

Use GCP Dataproc when:
- Cost optimization is priority
- Fast iteration needed
- Performance critical

Use AWS EMR when:
- Deep AWS ecosystem integration needed
- Enterprise support required

Use DataFrame API when:
- Complex queries (default choice)
- Production ML models
- Code maintainability important

Use RDD API when:
- High-throughput ETL pipelines
- Fine-grained control needed

---

## Citations

- Dataset: https://www.kaggle.com/datasets/devinanzelmo/dota-2-matches
- Apache Spark: https://spark.apache.org/docs/latest/
- AWS EMR: https://docs.aws.amazon.com/emr/
- GCP Dataproc: https://cloud.google.com/dataproc/docs

---

## License

Academic project for MET CS777 - Big Data Analytics at Boston University.

---

## Contact

- Anh Pham: U01836723@bu.edu
- Jinzhe Bai: U41047897@bu.edu

---

Course: MET CS777 - Big Data Analytics
Semester: Fall 2024
Institution: Boston University Metropolitan College

---

Last Updated: October 31, 2024
