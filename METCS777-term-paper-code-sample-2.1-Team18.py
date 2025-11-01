#Task 2 Model 1: Match Outcome Prediction - RDD vs DataFrame Comparison

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, avg, stddev, countDistinct, abs as spark_abs
from pyspark.sql.functions import sum as spark_sum, min as spark_min, max as spark_max
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.tree import RandomForest as RDD_RandomForest, GradientBoostedTrees as RDD_GBT
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.linalg import Vectors as MLLibVectors
import time
import json
import sys
import psutil
import os
import builtins
import gc

# Check arguments
if len(sys.argv) != 3:
    print("Usage: script.py <input_path> <output_path>", file=sys.stderr)
    print("Example: spark-submit script.py gs://bucket/merged gs://bucket/output", file=sys.stderr)
    sys.exit(-1)

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

print(f"Input path: {INPUT_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print("="*80)

# Initialize Spark with optimized settings for memory management
spark = SparkSession.builder \
    .appName("Dota2-Model1-RDD-vs-DataFrame-Optimized") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .config("spark.rdd.compress", "true") \
    .config("spark.shuffle.compress", "true") \
    .config("spark.shuffle.spill.compress", "true") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.executor.memoryOverhead", "1024m") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.storage.memoryFraction", "0.3") \
    .config("spark.shuffle.memoryFraction", "0.5") \
    .getOrCreate()

sc = spark.sparkContext
sc.setCheckpointDir("gs://dataproc-temp-us-east1-907542747727-8phk5jld/checkpoints")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def round_float(value, decimals=2):
    """Round a float value to specified decimals"""
    if value is None:
        return 0.0
    return float(f"{value:.{decimals}f}")

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_cpu_percent():
    """Get current CPU utilization percentage"""
    return psutil.cpu_percent(interval=1)

def calculate_throughput(rows_processed, time_seconds):
    """Calculate data throughput in rows per second"""
    if time_seconds > 0:
        return rows_processed / time_seconds
    return 0

def update_peak_memory(current_peak, new_value):
    """Update peak memory using Python's built-in max"""
    return builtins.max(current_peak, new_value)

def calculate_cost_estimate(execution_time_sec, memory_mb, cpu_percent):
    """Estimate cost based on GCP pricing (approximate)"""
    hourly_rate = 0.19
    hours = execution_time_sec / 3600
    base_cost = hours * hourly_rate
    memory_gb = memory_mb / 1024
    memory_cost = memory_gb * 0.01 * hours
    cpu_factor = cpu_percent / 100
    adjusted_cost = (base_cost + memory_cost) * (0.5 + 0.5 * cpu_factor)
    return round_float(adjusted_cost, 4)

def force_cleanup():
    """Force garbage collection and unpersist unused RDDs"""
    gc.collect()
    time.sleep(2)

# ==============================================================================
# INITIALIZE METRICS TRACKING
# ==============================================================================

pipeline_start_time = time.time()
cpu_readings = []
initial_memory = get_memory_usage()

results = {
    "system_metrics": {
        "start_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "initial_memory_mb": round_float(initial_memory, 2),
        "peak_memory_mb": round_float(initial_memory, 2),
        "avg_cpu_percent": 0,
        "total_execution_time_sec": 0,
        "total_cost_estimate_usd": 0
    },
    "etl_metrics": {
        "dataframe": {
            "data_loading_time_sec": 0,
            "feature_engineering_time_sec": 0,
            "total_rows_processed": 0,
            "data_throughput_rows_per_sec": 0,
            "query_execution_time_sec": 0
        },
        "rdd": {
            "data_loading_time_sec": 0,
            "feature_engineering_time_sec": 0,
            "total_rows_processed": 0,
            "data_throughput_rows_per_sec": 0,
            "query_execution_time_sec": 0
        }
    },
    "model1_match_outcome": {
        "dataframe": {},
        "rdd": {}
    },
    "comparison_summary": {}
}

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

load_start = time.time()
memory_before_load = get_memory_usage()
cpu_before_load = get_cpu_percent()

print("Loading datasets...")
players = spark.read.csv(f"{INPUT_PATH}/players.csv", header=True, inferSchema=True)
objectives = spark.read.csv(f"{INPUT_PATH}/objectives.csv", header=True, inferSchema=True)
picks_bans = spark.read.csv(f"{INPUT_PATH}/picks_bans.csv", header=True, inferSchema=True)
draft_timings = spark.read.csv(f"{INPUT_PATH}/draft_timings.csv", header=True, inferSchema=True)
radiant_gold_adv = spark.read.csv(f"{INPUT_PATH}/radiant_gold_adv.csv", header=True, inferSchema=True)
radiant_exp_adv = spark.read.csv(f"{INPUT_PATH}/radiant_exp_adv.csv", header=True, inferSchema=True)
teamfights = spark.read.csv(f"{INPUT_PATH}/teamfights.csv", header=True, inferSchema=True)

# Cache frequently accessed datasets
players.persist()
objectives.persist()

player_count = players.count()
objectives_count = objectives.count()
picks_bans_count = picks_bans.count()

print(f"Players: {player_count} rows")
print(f"Objectives: {objectives_count} rows")
print(f"Picks/Bans: {picks_bans_count} rows")

load_time = time.time() - load_start
memory_after_load = get_memory_usage()
cpu_after_load = get_cpu_percent()

total_rows = player_count + objectives_count + picks_bans_count
data_throughput = calculate_throughput(total_rows, load_time)

print(f"\nData loading completed in {round_float(load_time, 2)} seconds")
print(f"Memory used: {round_float(memory_after_load - memory_before_load, 2)} MB")
print(f"Throughput: {round_float(data_throughput, 2)} rows/sec")

results["system_metrics"]["peak_memory_mb"] = update_peak_memory(
    results["system_metrics"]["peak_memory_mb"], memory_after_load)
cpu_readings.extend([cpu_before_load, cpu_after_load])

# ==============================================================================
# FEATURE ENGINEERING - MODEL 1 (MATCH OUTCOME) - DATAFRAME
# ==============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING - MODEL 1 (MATCH OUTCOME) - DATAFRAME")
print("="*80)

fe1_df_start = time.time()
memory_before_fe1_df = get_memory_usage()
cpu_fe1_df_start = get_cpu_percent()

query_start = time.time()

# Get match outcomes
match_outcomes = players.select('match_id', 'radiant_win').dropDuplicates(['match_id'])
match_outcomes = match_outcomes.withColumn('radiant_win', 
    when(col('radiant_win').cast('string').isin(['True', '1', 'true']), 1).otherwise(0))

# 1. Early game objectives
early_objectives = objectives.filter(col('time') <= 600)
early_obj_stats = early_objectives.groupBy('match_id').agg(
    count(when(col('type') == 'CHAT_MESSAGE_FIRSTBLOOD', 1)).alias('first_blood_count'),
    count(when(col('type').like('%tower%'), 1)).alias('early_tower_kills'),
    count(when(col('type').like('%COURIER%'), 1)).alias('early_courier_kills'),
    count('type').alias('early_objective_count')
)

# 2. Gold and XP advantage metrics
early_gold = radiant_gold_adv.filter(col('minute').between(5, 10))
early_gold_stats = early_gold.groupBy('match_id').agg(
    avg('gold').alias('avg_gold_adv_5_10'),
    spark_max('gold').alias('max_gold_adv_5_10'),
    spark_min('gold').alias('min_gold_adv_5_10'),
    stddev('gold').alias('gold_volatility')
)

early_xp = radiant_exp_adv.filter(col('minute').between(5, 10))
early_xp_stats = early_xp.groupBy('match_id').agg(
    avg('exp').alias('avg_xp_adv_5_10'),
    spark_max('exp').alias('max_xp_adv_5_10')
)

# 3. Team composition features
hero_picks = picks_bans.filter(col('is_pick') == True)

radiant_picks = hero_picks.filter(col('team') == 0).groupBy('match_id').agg(
    count('hero_id').alias('radiant_pick_count'),
    countDistinct('hero_id').alias('radiant_unique_heroes')
)

dire_picks = hero_picks.filter(col('team') == 1).groupBy('match_id').agg(
    count('hero_id').alias('dire_pick_count'),
    countDistinct('hero_id').alias('dire_unique_heroes')
)

# 4. Player performance in early game
early_player_stats = players.groupBy('match_id').agg(
    spark_sum(when(col('firstblood_claimed') > 0, 1).otherwise(0)).alias('firstblood_total'),
    avg('gold_per_min').alias('avg_gpm_all'),
    avg('xp_per_min').alias('avg_xpm_all'),
    spark_max('hero_damage').alias('max_hero_damage'),
    spark_sum('kills').alias('total_kills'),
    spark_sum('deaths').alias('total_deaths'),
    spark_sum('assists').alias('total_assists')
)

# 5. Draft timing features
draft_stats = draft_timings.groupBy('match_id').agg(
    avg('total_time_taken').alias('avg_draft_time'),
    spark_max('total_time_taken').alias('max_draft_time'),
    spark_sum('total_time_taken').alias('total_draft_time')
)

# 6. Early teamfight features
early_teamfights = teamfights.filter(col('start') <= 600)
teamfight_stats = early_teamfights.groupBy('match_id').agg(
    count('*').alias('early_teamfight_count'),
    avg('deaths').alias('avg_teamfight_deaths')
)

query_time = time.time() - query_start

# Merge all features
dataset1_df = match_outcomes \
    .join(early_obj_stats, on='match_id', how='left') \
    .join(early_gold_stats, on='match_id', how='left') \
    .join(early_xp_stats, on='match_id', how='left') \
    .join(radiant_picks, on='match_id', how='left') \
    .join(dire_picks, on='match_id', how='left') \
    .join(early_player_stats, on='match_id', how='left') \
    .join(draft_stats, on='match_id', how='left') \
    .join(teamfight_stats, on='match_id', how='left')

# Fill nulls
dataset1_df = dataset1_df.fillna({
    'first_blood_count': 0, 'early_tower_kills': 0, 'early_courier_kills': 0,
    'early_objective_count': 0, 'avg_gold_adv_5_10': 0, 'max_gold_adv_5_10': 0,
    'min_gold_adv_5_10': 0, 'gold_volatility': 0, 'avg_xp_adv_5_10': 0,
    'max_xp_adv_5_10': 0, 'radiant_pick_count': 5, 'radiant_unique_heroes': 5,
    'dire_pick_count': 5, 'dire_unique_heroes': 5, 'firstblood_total': 0,
    'avg_gpm_all': 400, 'avg_xpm_all': 500, 'max_hero_damage': 0,
    'total_kills': 0, 'total_deaths': 0, 'total_assists': 0,
    'avg_draft_time': 30, 'max_draft_time': 30, 'total_draft_time': 300,
    'early_teamfight_count': 0, 'avg_teamfight_deaths': 0
})

# Create derived features
dataset1_df = dataset1_df.withColumn('kill_death_ratio', 
    when(col('total_deaths') > 0, col('total_kills') / col('total_deaths')).otherwise(col('total_kills')))

dataset1_df = dataset1_df.dropna(subset=['radiant_win'])
dataset1_df = dataset1_df.repartition(50)
dataset1_df.persist()

fe1_df_time = time.time() - fe1_df_start
memory_after_fe1_df = get_memory_usage()
cpu_fe1_df_end = get_cpu_percent()

dataset1_df_count = dataset1_df.count()
print(f"DataFrame - Dataset 1 rows: {dataset1_df_count}")
print(f"Feature engineering time: {round_float(fe1_df_time, 2)} seconds")
print(f"Query execution time: {round_float(query_time, 2)} seconds")
print(f"Total features: {len(dataset1_df.columns) - 2}")
print(f"Memory used: {round_float(memory_after_fe1_df - memory_before_fe1_df, 2)} MB")

results["etl_metrics"]["dataframe"]["feature_engineering_time_sec"] = round_float(fe1_df_time, 2)
results["etl_metrics"]["dataframe"]["query_execution_time_sec"] = round_float(query_time, 2)
results["etl_metrics"]["dataframe"]["total_rows_processed"] = dataset1_df_count
results["etl_metrics"]["dataframe"]["data_throughput_rows_per_sec"] = round_float(
    calculate_throughput(dataset1_df_count, fe1_df_time), 2)

results["system_metrics"]["peak_memory_mb"] = update_peak_memory(
    results["system_metrics"]["peak_memory_mb"], memory_after_fe1_df)
cpu_readings.extend([cpu_fe1_df_start, cpu_fe1_df_end])

# ==============================================================================
# FEATURE ENGINEERING - MODEL 1 (MATCH OUTCOME) - RDD
# ==============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING - MODEL 1 (MATCH OUTCOME) - RDD")
print("="*80)

fe1_rdd_start = time.time()
memory_before_fe1_rdd = get_memory_usage()
cpu_fe1_rdd_start = get_cpu_percent()

query_rdd_start = time.time()

# Convert DataFrames to RDDs for processing
players_rdd = players.rdd
objectives_rdd = objectives.rdd
picks_bans_rdd = picks_bans.rdd
draft_timings_rdd = draft_timings.rdd
radiant_gold_adv_rdd = radiant_gold_adv.rdd
radiant_exp_adv_rdd = radiant_exp_adv.rdd
teamfights_rdd = teamfights.rdd

# 1. Match outcomes
def extract_match_outcome(row):
    match_id = row['match_id']
    radiant_win = 1 if str(row['radiant_win']).lower() in ['true', '1'] else 0
    return (match_id, radiant_win)

match_outcomes_rdd = players_rdd.map(extract_match_outcome).distinct()

# 2. Early objectives
def process_early_objectives(row):
    time_val = row['time']
    if time_val is not None and time_val <= 600:
        match_id = row['match_id']
        obj_type = str(row['type']) if row['type'] is not None else ''
        first_blood = 1 if obj_type == 'CHAT_MESSAGE_FIRSTBLOOD' else 0
        tower = 1 if 'tower' in obj_type.lower() else 0
        courier = 1 if 'COURIER' in obj_type else 0
        return (match_id, (first_blood, tower, courier, 1))
    return None

early_obj_rdd = objectives_rdd.map(process_early_objectives) \
    .filter(lambda x: x is not None) \
    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]))

# 3. Gold stats
def process_gold_adv(row):
    minute = row['minute']
    gold = row['gold']
    if minute is not None and gold is not None and 5 <= minute <= 10:
        return (row['match_id'], gold)
    return None

gold_rdd = radiant_gold_adv_rdd.map(process_gold_adv).filter(lambda x: x is not None)
gold_stats_rdd = gold_rdd.groupByKey().mapValues(lambda vals: {
    'avg': sum(vals) / len(vals) if len(vals) > 0 else 0,
    'max': max(vals) if len(vals) > 0 else 0,
    'min': min(vals) if len(vals) > 0 else 0
})

# 4. XP stats
def process_xp_adv(row):
    minute = row['minute']
    exp = row['exp']
    if minute is not None and exp is not None and 5 <= minute <= 10:
        return (row['match_id'], exp)
    return None

xp_rdd = radiant_exp_adv_rdd.map(process_xp_adv).filter(lambda x: x is not None)
xp_stats_rdd = xp_rdd.groupByKey().mapValues(lambda vals: {
    'avg': sum(vals) / len(vals) if len(vals) > 0 else 0,
    'max': max(vals) if len(vals) > 0 else 0
})

# 5. Hero picks
def process_picks(row):
    if row['is_pick']:
        team = row['team']
        return ((row['match_id'], team), 1)
    return None

picks_rdd = picks_bans_rdd.map(process_picks).filter(lambda x: x is not None)
picks_count_rdd = picks_rdd.reduceByKey(lambda a, b: a + b)

# 6. Player stats
def process_player_stats(row):
    match_id = row['match_id']
    firstblood = 1 if (row['firstblood_claimed'] is not None and row['firstblood_claimed'] > 0) else 0
    return (match_id, (
        firstblood,
        row['gold_per_min'] if row['gold_per_min'] is not None else 0,
        row['xp_per_min'] if row['xp_per_min'] is not None else 0,
        row['hero_damage'] if row['hero_damage'] is not None else 0,
        row['kills'] if row['kills'] is not None else 0,
        row['deaths'] if row['deaths'] is not None else 0,
        row['assists'] if row['assists'] is not None else 0,
        1
    ))

player_stats_rdd = players_rdd.map(process_player_stats) \
    .reduceByKey(lambda a, b: tuple(a[i] + b[i] for i in range(8)))

query_rdd_time = time.time() - query_rdd_start

# Combine all features
def combine_features(match_id, outcome, obj_stats, gold_stats, xp_stats, player_stats):
    features = [
        obj_stats[0] if obj_stats else 0,
        obj_stats[1] if obj_stats else 0,
        obj_stats[2] if obj_stats else 0,
        obj_stats[3] if obj_stats else 0,
        gold_stats['avg'] if gold_stats else 0,
        gold_stats['max'] if gold_stats else 0,
        gold_stats['min'] if gold_stats else 0,
        xp_stats['avg'] if xp_stats else 0,
        xp_stats['max'] if xp_stats else 0,
        player_stats[0] if player_stats else 0,
        player_stats[1] / player_stats[7] if player_stats and player_stats[7] > 0 else 400,
        player_stats[2] / player_stats[7] if player_stats and player_stats[7] > 0 else 500,
        player_stats[3] if player_stats else 0,
        player_stats[4] if player_stats else 0,
        player_stats[5] if player_stats else 0,
        player_stats[6] if player_stats else 0,
    ]
    
    kdr = features[13] / features[14] if features[14] > 0 else features[13]
    features.append(kdr)
    
    return LabeledPoint(outcome, MLLibVectors.dense(features))

# Join all RDDs
combined_rdd = match_outcomes_rdd \
    .leftOuterJoin(early_obj_rdd) \
    .leftOuterJoin(gold_stats_rdd) \
    .leftOuterJoin(xp_stats_rdd) \
    .leftOuterJoin(player_stats_rdd) \
    .map(lambda x: combine_features(x[0], x[1][0][0][0][0], x[1][0][0][0][1], 
                                     x[1][0][0][1], x[1][0][1], x[1][1]))

dataset1_rdd = combined_rdd.repartition(50)
dataset1_rdd.persist()

fe1_rdd_time = time.time() - fe1_rdd_start
memory_after_fe1_rdd = get_memory_usage()
cpu_fe1_rdd_end = get_cpu_percent()

dataset1_rdd_count = dataset1_rdd.count()
print(f"RDD - Dataset 1 rows: {dataset1_rdd_count}")
print(f"Feature engineering time: {round_float(fe1_rdd_time, 2)} seconds")
print(f"Query execution time: {round_float(query_rdd_time, 2)} seconds")
print(f"Memory used: {round_float(memory_after_fe1_rdd - memory_before_fe1_rdd, 2)} MB")

results["etl_metrics"]["rdd"]["feature_engineering_time_sec"] = round_float(fe1_rdd_time, 2)
results["etl_metrics"]["rdd"]["query_execution_time_sec"] = round_float(query_rdd_time, 2)
results["etl_metrics"]["rdd"]["total_rows_processed"] = dataset1_rdd_count
results["etl_metrics"]["rdd"]["data_throughput_rows_per_sec"] = round_float(
    calculate_throughput(dataset1_rdd_count, fe1_rdd_time), 2)

results["system_metrics"]["peak_memory_mb"] = update_peak_memory(
    results["system_metrics"]["peak_memory_mb"], memory_after_fe1_rdd)
cpu_readings.extend([cpu_fe1_rdd_start, cpu_fe1_rdd_end])

# Unpersist source DataFrames to free memory
players.unpersist()
objectives.unpersist()

force_cleanup()

# ==============================================================================
# MODEL 1: MATCH OUTCOME PREDICTION - DATAFRAME
# ==============================================================================
print("\n" + "="*80)
print("MODEL 1: MATCH OUTCOME PREDICTION - DATAFRAME")
print("="*80)

model1_df_start = time.time()
memory_before_model1_df = get_memory_usage()

feature_cols_1 = [
    'first_blood_count', 'early_tower_kills', 'early_courier_kills', 'early_objective_count',
    'avg_gold_adv_5_10', 'max_gold_adv_5_10', 'min_gold_adv_5_10', 'gold_volatility',
    'avg_xp_adv_5_10', 'max_xp_adv_5_10',
    'radiant_pick_count', 'dire_pick_count',
    'firstblood_total', 'avg_gpm_all', 'avg_xpm_all', 'max_hero_damage',
    'total_kills', 'total_deaths', 'total_assists', 'kill_death_ratio',
    'avg_draft_time', 'max_draft_time', 'total_draft_time',
    'early_teamfight_count', 'avg_teamfight_deaths'
]

assembler1 = VectorAssembler(inputCols=feature_cols_1, outputCol="features_raw")
scaler1 = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

temp_df = assembler1.transform(dataset1_df)
scaler_model = scaler1.fit(temp_df)
dataset1_prepared = scaler_model.transform(temp_df)
dataset1_prepared = dataset1_prepared.select('features', 'radiant_win')

train1_df, test1_df = dataset1_prepared.randomSplit([0.8, 0.2], seed=42)
train1_df.persist()
test1_df.persist()

train_count_1_df = train1_df.count()
test_count_1_df = test1_df.count()

print(f"DataFrame - Train: {train_count_1_df}, Test: {test_count_1_df}")
print(f"Number of features: {len(feature_cols_1)}")

results["model1_match_outcome"]["dataframe"]["feature_count"] = len(feature_cols_1)
results["model1_match_outcome"]["dataframe"]["training_rows"] = train_count_1_df
results["model1_match_outcome"]["dataframe"]["test_rows"] = test_count_1_df

# Logistic Regression - DataFrame
print("\n--- Logistic Regression (DataFrame) ---")
lr_df_memory_before = get_memory_usage()
lr_df_train_start = time.time()
lr_df = LogisticRegression(featuresCol="features", labelCol="radiant_win", maxIter=100, regParam=0.01)
lr_df_model = lr_df.fit(train1_df)
lr_df_train_time = time.time() - lr_df_train_start
lr_df_memory_after = get_memory_usage()

lr_df_pred_start = time.time()
lr_df_predictions = lr_df_model.transform(test1_df)
lr_df_pred_count = lr_df_predictions.count()
lr_df_inference_time = (time.time() - lr_df_pred_start) / lr_df_pred_count if lr_df_pred_count > 0 else 0

evaluator_auc = BinaryClassificationEvaluator(labelCol="radiant_win", metricName="areaUnderROC")
evaluator_acc = MulticlassClassificationEvaluator(labelCol="radiant_win", predictionCol="prediction", metricName="accuracy")

lr_df_auc = evaluator_auc.evaluate(lr_df_predictions)
lr_df_accuracy = evaluator_acc.evaluate(lr_df_predictions)
lr_df_memory_footprint = lr_df_memory_after - lr_df_memory_before

print(f"Training: {round_float(lr_df_train_time, 2)}s")
print(f"AUC: {round_float(lr_df_auc, 4)}, Accuracy: {round_float(lr_df_accuracy, 4)}")
print(f"Inference: {round_float(lr_df_inference_time*1000, 4)}ms per row")
print(f"Memory: {round_float(lr_df_memory_footprint, 2)} MB")

results["model1_match_outcome"]["dataframe"]["logistic_regression"] = {
    "training_time_sec": round_float(lr_df_train_time, 2),
    "auc": round_float(lr_df_auc, 4),
    "accuracy": round_float(lr_df_accuracy, 4),
    "inference_time_ms": round_float(lr_df_inference_time * 1000, 4),
    "memory_footprint_mb": round_float(lr_df_memory_footprint, 2)
}

cpu_readings.append(get_cpu_percent())

# Random Forest - DataFrame
print("\n--- Random Forest (DataFrame) ---")
rf_df_memory_before = get_memory_usage()
rf_df_train_start = time.time()
rf_df = RandomForestClassifier(featuresCol="features", labelCol="radiant_win", 
                               numTrees=50, maxDepth=10, seed=42)
rf_df_model = rf_df.fit(train1_df)
rf_df_train_time = time.time() - rf_df_train_start
rf_df_memory_after = get_memory_usage()

rf_df_pred_start = time.time()
rf_df_predictions = rf_df_model.transform(test1_df)
rf_df_pred_count = rf_df_predictions.count()
rf_df_inference_time = (time.time() - rf_df_pred_start) / rf_df_pred_count if rf_df_pred_count > 0 else 0

rf_df_auc = evaluator_auc.evaluate(rf_df_predictions)
rf_df_accuracy = evaluator_acc.evaluate(rf_df_predictions)
rf_df_memory_footprint = rf_df_memory_after - rf_df_memory_before

feature_importance_df = [(feature_cols_1[i], float(rf_df_model.featureImportances[i])) 
                         for i in range(len(feature_cols_1))]
feature_importance_df.sort(key=lambda x: x[1], reverse=True)

print(f"Training: {round_float(rf_df_train_time, 2)}s")
print(f"AUC: {round_float(rf_df_auc, 4)}, Accuracy: {round_float(rf_df_accuracy, 4)}")
print(f"Inference: {round_float(rf_df_inference_time*1000, 4)}ms per row")
print(f"Memory: {round_float(rf_df_memory_footprint, 2)} MB")
print("Top 5 Feature Importance:")
for feat, imp in feature_importance_df[:5]:
    print(f"  {feat}: {round_float(imp, 4)}")

results["model1_match_outcome"]["dataframe"]["random_forest"] = {
    "training_time_sec": round_float(rf_df_train_time, 2),
    "auc": round_float(rf_df_auc, 4),
    "accuracy": round_float(rf_df_accuracy, 4),
    "inference_time_ms": round_float(rf_df_inference_time * 1000, 4),
    "memory_footprint_mb": round_float(rf_df_memory_footprint, 2),
    "feature_importance": {feat: round_float(imp, 4) for feat, imp in feature_importance_df[:10]}
}

cpu_readings.append(get_cpu_percent())

# Gradient Boosting - DataFrame
print("\n--- Gradient Boosting (DataFrame) ---")
gbt_df_memory_before = get_memory_usage()
gbt_df_train_start = time.time()
gbt_df = GBTClassifier(featuresCol="features", labelCol="radiant_win", maxIter=30, maxDepth=8, seed=42)
gbt_df_model = gbt_df.fit(train1_df)
gbt_df_train_time = time.time() - gbt_df_train_start
gbt_df_memory_after = get_memory_usage()

gbt_df_pred_start = time.time()
gbt_df_predictions = gbt_df_model.transform(test1_df)
gbt_df_pred_count = gbt_df_predictions.count()
gbt_df_inference_time = (time.time() - gbt_df_pred_start) / gbt_df_pred_count if gbt_df_pred_count > 0 else 0

gbt_df_auc = evaluator_auc.evaluate(gbt_df_predictions)
gbt_df_accuracy = evaluator_acc.evaluate(gbt_df_predictions)
gbt_df_memory_footprint = gbt_df_memory_after - gbt_df_memory_before

gbt_feature_importance_df = [(feature_cols_1[i], float(gbt_df_model.featureImportances[i])) 
                             for i in range(len(feature_cols_1))]
gbt_feature_importance_df.sort(key=lambda x: x[1], reverse=True)

print(f"Training: {round_float(gbt_df_train_time, 2)}s")
print(f"AUC: {round_float(gbt_df_auc, 4)}, Accuracy: {round_float(gbt_df_accuracy, 4)}")
print(f"Inference: {round_float(gbt_df_inference_time*1000, 4)}ms per row")
print(f"Memory: {round_float(gbt_df_memory_footprint, 2)} MB")

results["model1_match_outcome"]["dataframe"]["gradient_boosting"] = {
    "training_time_sec": round_float(gbt_df_train_time, 2),
    "auc": round_float(gbt_df_auc, 4),
    "accuracy": round_float(gbt_df_accuracy, 4),
    "inference_time_ms": round_float(gbt_df_inference_time * 1000, 4),
    "memory_footprint_mb": round_float(gbt_df_memory_footprint, 2),
    "feature_importance": {feat: round_float(imp, 4) for feat, imp in gbt_feature_importance_df[:10]}
}

model1_df_total_time = time.time() - model1_df_start
memory_after_model1_df = get_memory_usage()

results["model1_match_outcome"]["dataframe"]["ml_training_time_sec"] = round_float(model1_df_total_time, 2)
results["model1_match_outcome"]["dataframe"]["total_memory_used_mb"] = round_float(
    memory_after_model1_df - memory_before_model1_df, 2)

cpu_readings.append(get_cpu_percent())

# Clean up DataFrame models
train1_df.unpersist()
test1_df.unpersist()
dataset1_df.unpersist()

force_cleanup()

# ==============================================================================
# MODEL 1: MATCH OUTCOME PREDICTION - RDD
# ==============================================================================
print("\n" + "="*80)
print("MODEL 1: MATCH OUTCOME PREDICTION - RDD")
print("="*80)

model1_rdd_start = time.time()
memory_before_model1_rdd = get_memory_usage()

# Split data
train1_rdd, test1_rdd = dataset1_rdd.randomSplit([0.8, 0.2], seed=42)
train1_rdd.persist()
test1_rdd.persist()

train_count_1_rdd = train1_rdd.count()
test_count_1_rdd = test1_rdd.count()

print(f"RDD - Train: {train_count_1_rdd}, Test: {test_count_1_rdd}")

results["model1_match_outcome"]["rdd"]["feature_count"] = 17
results["model1_match_outcome"]["rdd"]["training_rows"] = train_count_1_rdd
results["model1_match_outcome"]["rdd"]["test_rows"] = test_count_1_rdd

# Logistic Regression - RDD
print("\n--- Logistic Regression (RDD) ---")
lr_rdd_memory_before = get_memory_usage()
lr_rdd_train_start = time.time()
lr_rdd_model = LogisticRegressionWithSGD.train(train1_rdd, iterations=100, step=0.01)
lr_rdd_train_time = time.time() - lr_rdd_train_start
lr_rdd_memory_after = get_memory_usage()

lr_rdd_pred_start = time.time()
lr_rdd_pred_tuples = test1_rdd.map(lambda lp: (float(lr_rdd_model.predict(lp.features)), lp.label)).collect()
lr_rdd_pred_count = len(lr_rdd_pred_tuples)
lr_rdd_inference_time = (time.time() - lr_rdd_pred_start) / lr_rdd_pred_count if lr_rdd_pred_count > 0 else 0

lr_rdd_predictions = sc.parallelize(lr_rdd_pred_tuples)
lr_rdd_metrics = BinaryClassificationMetrics(lr_rdd_predictions)
lr_rdd_auc = lr_rdd_metrics.areaUnderROC
lr_rdd_accuracy = sum(1 for pred, label in lr_rdd_pred_tuples if (pred > 0.5 and label == 1) or (pred <= 0.5 and label == 0)) / lr_rdd_pred_count if lr_rdd_pred_count > 0 else 0
lr_rdd_memory_footprint = lr_rdd_memory_after - lr_rdd_memory_before

print(f"Training: {round_float(lr_rdd_train_time, 2)}s")
print(f"AUC: {round_float(lr_rdd_auc, 4)}, Accuracy: {round_float(lr_rdd_accuracy, 4)}")
print(f"Inference: {round_float(lr_rdd_inference_time*1000, 4)}ms per row")
print(f"Memory: {round_float(lr_rdd_memory_footprint, 2)} MB")

results["model1_match_outcome"]["rdd"]["logistic_regression"] = {
    "training_time_sec": round_float(lr_rdd_train_time, 2),
    "auc": round_float(lr_rdd_auc, 4),
    "accuracy": round_float(lr_rdd_accuracy, 4),
    "inference_time_ms": round_float(lr_rdd_inference_time * 1000, 4),
    "memory_footprint_mb": round_float(lr_rdd_memory_footprint, 2)
}

cpu_readings.append(get_cpu_percent())
force_cleanup()

# Random Forest - RDD
print("\n--- Random Forest (RDD) ---")
rf_rdd_memory_before = get_memory_usage()
rf_rdd_train_start = time.time()
rf_rdd_model = RDD_RandomForest.trainClassifier(train1_rdd, numClasses=2, categoricalFeaturesInfo={},
                                                numTrees=30, featureSubsetStrategy="auto",
                                                impurity='gini', maxDepth=8, maxBins=32, seed=42)
rf_rdd_train_time = time.time() - rf_rdd_train_start
rf_rdd_memory_after = get_memory_usage()

rf_rdd_pred_start = time.time()
test_data_collected = test1_rdd.collect()
rf_rdd_pred_tuples = [(float(rf_rdd_model.predict(lp.features)), lp.label) for lp in test_data_collected]
rf_rdd_pred_count = len(rf_rdd_pred_tuples)
rf_rdd_inference_time = (time.time() - rf_rdd_pred_start) / rf_rdd_pred_count if rf_rdd_pred_count > 0 else 0

rf_rdd_predictions = sc.parallelize(rf_rdd_pred_tuples)
rf_rdd_metrics = BinaryClassificationMetrics(rf_rdd_predictions)
rf_rdd_auc = rf_rdd_metrics.areaUnderROC
rf_rdd_accuracy = sum(1 for pred, label in rf_rdd_pred_tuples if int(pred) == int(label)) / rf_rdd_pred_count if rf_rdd_pred_count > 0 else 0
rf_rdd_memory_footprint = rf_rdd_memory_after - rf_rdd_memory_before

print(f"Training: {round_float(rf_rdd_train_time, 2)}s")
print(f"AUC: {round_float(rf_rdd_auc, 4)}, Accuracy: {round_float(rf_rdd_accuracy, 4)}")
print(f"Inference: {round_float(rf_rdd_inference_time*1000, 4)}ms per row")
print(f"Memory: {round_float(rf_rdd_memory_footprint, 2)} MB")

results["model1_match_outcome"]["rdd"]["random_forest"] = {
    "training_time_sec": round_float(rf_rdd_train_time, 2),
    "auc": round_float(rf_rdd_auc, 4),
    "accuracy": round_float(rf_rdd_accuracy, 4),
    "inference_time_ms": round_float(rf_rdd_inference_time * 1000, 4),
    "memory_footprint_mb": round_float(rf_rdd_memory_footprint, 2)
}

cpu_readings.append(get_cpu_percent())
force_cleanup()

# Gradient Boosting - RDD
print("\n--- Gradient Boosting (RDD) ---")
gbt_rdd_memory_before = get_memory_usage()
gbt_rdd_train_start = time.time()
gbt_rdd_model = RDD_GBT.trainClassifier(train1_rdd, categoricalFeaturesInfo={},
                                        numIterations=20, maxDepth=6, learningRate=0.1)
gbt_rdd_train_time = time.time() - gbt_rdd_train_start
gbt_rdd_memory_after = get_memory_usage()

gbt_rdd_pred_start = time.time()
gbt_rdd_pred_tuples = [(float(gbt_rdd_model.predict(lp.features)), lp.label) for lp in test_data_collected]
gbt_rdd_pred_count = len(gbt_rdd_pred_tuples)
gbt_rdd_inference_time = (time.time() - gbt_rdd_pred_start) / gbt_rdd_pred_count if gbt_rdd_pred_count > 0 else 0

gbt_rdd_predictions = sc.parallelize(gbt_rdd_pred_tuples)
gbt_rdd_metrics = BinaryClassificationMetrics(gbt_rdd_predictions)
gbt_rdd_auc = gbt_rdd_metrics.areaUnderROC
gbt_rdd_accuracy = sum(1 for pred, label in gbt_rdd_pred_tuples if int(pred) == int(label)) / gbt_rdd_pred_count if gbt_rdd_pred_count > 0 else 0
gbt_rdd_memory_footprint = gbt_rdd_memory_after - gbt_rdd_memory_before

print(f"Training: {round_float(gbt_rdd_train_time, 2)}s")
print(f"AUC: {round_float(gbt_rdd_auc, 4)}, Accuracy: {round_float(gbt_rdd_accuracy, 4)}")
print(f"Inference: {round_float(gbt_rdd_inference_time*1000, 4)}ms per row")
print(f"Memory: {round_float(gbt_rdd_memory_footprint, 2)} MB")

results["model1_match_outcome"]["rdd"]["gradient_boosting"] = {
    "training_time_sec": round_float(gbt_rdd_train_time, 2),
    "auc": round_float(gbt_rdd_auc, 4),
    "accuracy": round_float(gbt_rdd_accuracy, 4),
    "inference_time_ms": round_float(gbt_rdd_inference_time * 1000, 4),
    "memory_footprint_mb": round_float(gbt_rdd_memory_footprint, 2)
}

model1_rdd_total_time = time.time() - model1_rdd_start
memory_after_model1_rdd = get_memory_usage()

results["model1_match_outcome"]["rdd"]["ml_training_time_sec"] = round_float(model1_rdd_total_time, 2)
results["model1_match_outcome"]["rdd"]["total_memory_used_mb"] = round_float(
    memory_after_model1_rdd - memory_before_model1_rdd, 2)

cpu_readings.append(get_cpu_percent())

# Clean up Model 1 RDDs
train1_rdd.unpersist()
test1_rdd.unpersist()
dataset1_rdd.unpersist()

force_cleanup()

# ==============================================================================
# FINALIZE SYSTEM METRICS & COMPARISON
# ==============================================================================

pipeline_end_time = time.time()
total_execution_time = pipeline_end_time - pipeline_start_time
avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0
peak_memory = results["system_metrics"]["peak_memory_mb"]

results["system_metrics"]["total_execution_time_sec"] = round_float(total_execution_time, 2)
results["system_metrics"]["avg_cpu_percent"] = round_float(avg_cpu, 2)
results["system_metrics"]["total_cost_estimate_usd"] = calculate_cost_estimate(
    total_execution_time, peak_memory, avg_cpu)

# Calculate comparison metrics
results["comparison_summary"] = {
    "etl_processing": {
        "dataframe_vs_rdd": {
            "feature_engineering_speedup": round_float(
                results["etl_metrics"]["rdd"]["feature_engineering_time_sec"] / 
                results["etl_metrics"]["dataframe"]["feature_engineering_time_sec"], 2),
            "query_execution_speedup": round_float(
                results["etl_metrics"]["rdd"]["query_execution_time_sec"] / 
                results["etl_metrics"]["dataframe"]["query_execution_time_sec"], 2),
            "throughput_comparison": {
                "dataframe_rows_per_sec": results["etl_metrics"]["dataframe"]["data_throughput_rows_per_sec"],
                "rdd_rows_per_sec": results["etl_metrics"]["rdd"]["data_throughput_rows_per_sec"]
            }
        }
    },
    "model1_comparison": {
        "logistic_regression": {
            "training_time_ratio_df_to_rdd": round_float(
                results["model1_match_outcome"]["dataframe"]["logistic_regression"]["training_time_sec"] /
                results["model1_match_outcome"]["rdd"]["logistic_regression"]["training_time_sec"], 2),
            "accuracy_difference": round_float(
                results["model1_match_outcome"]["dataframe"]["logistic_regression"]["accuracy"] -
                results["model1_match_outcome"]["rdd"]["logistic_regression"]["accuracy"], 4),
            "memory_difference_mb": round_float(
                results["model1_match_outcome"]["dataframe"]["logistic_regression"]["memory_footprint_mb"] -
                results["model1_match_outcome"]["rdd"]["logistic_regression"]["memory_footprint_mb"], 2)
        },
        "random_forest": {
            "training_time_ratio_df_to_rdd": round_float(
                results["model1_match_outcome"]["dataframe"]["random_forest"]["training_time_sec"] /
                results["model1_match_outcome"]["rdd"]["random_forest"]["training_time_sec"], 2),
            "accuracy_difference": round_float(
                results["model1_match_outcome"]["dataframe"]["random_forest"]["accuracy"] -
                results["model1_match_outcome"]["rdd"]["random_forest"]["accuracy"], 4),
            "memory_difference_mb": round_float(
                results["model1_match_outcome"]["dataframe"]["random_forest"]["memory_footprint_mb"] -
                results["model1_match_outcome"]["rdd"]["random_forest"]["memory_footprint_mb"], 2)
        },
        "gradient_boosting": {
            "training_time_ratio_df_to_rdd": round_float(
                results["model1_match_outcome"]["dataframe"]["gradient_boosting"]["training_time_sec"] /
                results["model1_match_outcome"]["rdd"]["gradient_boosting"]["training_time_sec"], 2),
            "accuracy_difference": round_float(
                results["model1_match_outcome"]["dataframe"]["gradient_boosting"]["accuracy"] -
                results["model1_match_outcome"]["rdd"]["gradient_boosting"]["accuracy"], 4),
            "memory_difference_mb": round_float(
                results["model1_match_outcome"]["dataframe"]["gradient_boosting"]["memory_footprint_mb"] -
                results["model1_match_outcome"]["rdd"]["gradient_boosting"]["memory_footprint_mb"], 2)
        }
    },
    "scalability_metrics": {
        "total_rows_processed": results["etl_metrics"]["dataframe"]["total_rows_processed"],
        "peak_memory_mb": peak_memory,
        "avg_cpu_utilization": avg_cpu,
        "total_execution_time_sec": total_execution_time,
        "estimated_cost_usd": results["system_metrics"]["total_cost_estimate_usd"]
    }
}

# ==============================================================================
# SAVE COMPREHENSIVE REPORT
# ==============================================================================
print("\n" + "="*80)
print("GENERATING COMPREHENSIVE REPORT")
print("="*80)

# Determine best models
best_model1_df = builtins.max(
    [("Logistic Regression", results["model1_match_outcome"]["dataframe"]["logistic_regression"]["auc"]),
     ("Random Forest", results["model1_match_outcome"]["dataframe"]["random_forest"]["auc"]),
     ("Gradient Boosting", results["model1_match_outcome"]["dataframe"]["gradient_boosting"]["auc"])],
    key=lambda x: x[1]
)

best_model1_rdd = builtins.max(
    [("Logistic Regression", results["model1_match_outcome"]["rdd"]["logistic_regression"]["auc"]),
     ("Random Forest", results["model1_match_outcome"]["rdd"]["random_forest"]["auc"]),
     ("Gradient Boosting", results["model1_match_outcome"]["rdd"]["gradient_boosting"]["auc"])],
    key=lambda x: x[1]
)

# Create comprehensive report
report = []
report.append("="*80)
report.append("DOTA 2 MODEL 1: MATCH OUTCOME PREDICTION")
report.append("RDD vs DATAFRAME COMPARISON")
report.append("="*80)
report.append(f"\nExecution Date: {results['system_metrics']['start_time']}")
report.append(f"Total Execution Time: {results['system_metrics']['total_execution_time_sec']} seconds")
report.append(f"Estimated Cost: ${results['system_metrics']['total_cost_estimate_usd']} USD")

# System Metrics
report.append("\n" + "="*80)
report.append("SYSTEM RESOURCE METRICS")
report.append("="*80)
report.append(f"Initial Memory: {results['system_metrics']['initial_memory_mb']} MB")
report.append(f"Peak Memory Usage: {results['system_metrics']['peak_memory_mb']} MB")
report.append(f"Average CPU Utilization: {results['system_metrics']['avg_cpu_percent']}%")

# ETL Comparison
report.append("\n" + "="*80)
report.append("ETL PROCESSING METRICS COMPARISON")
report.append("="*80)
report.append("\nDataFrame Approach:")
report.append(f"  Feature Engineering Time: {results['etl_metrics']['dataframe']['feature_engineering_time_sec']} sec")
report.append(f"  Query Execution Time: {results['etl_metrics']['dataframe']['query_execution_time_sec']} sec")
report.append(f"  Data Throughput: {results['etl_metrics']['dataframe']['data_throughput_rows_per_sec']:,.2f} rows/sec")

report.append("\nRDD Approach:")
report.append(f"  Feature Engineering Time: {results['etl_metrics']['rdd']['feature_engineering_time_sec']} sec")
report.append(f"  Query Execution Time: {results['etl_metrics']['rdd']['query_execution_time_sec']} sec")
report.append(f"  Data Throughput: {results['etl_metrics']['rdd']['data_throughput_rows_per_sec']:,.2f} rows/sec")

report.append("\nSpeedup Metrics:")
report.append(f"  Feature Engineering Speedup (RDD/DF): {results['comparison_summary']['etl_processing']['dataframe_vs_rdd']['feature_engineering_speedup']}x")
report.append(f"  Query Execution Speedup (RDD/DF): {results['comparison_summary']['etl_processing']['dataframe_vs_rdd']['query_execution_speedup']}x")

# Model 1 Comparison
report.append("\n" + "="*80)
report.append("MODEL 1: MATCH OUTCOME PREDICTION COMPARISON")
report.append("="*80)

for model_name in ["logistic_regression", "random_forest", "gradient_boosting"]:
    report.append(f"\n{model_name.replace('_', ' ').title()}:")
    report.append("  DataFrame:")
    df_metrics = results["model1_match_outcome"]["dataframe"][model_name]
    for key, val in df_metrics.items():
        if key != "feature_importance":
            report.append(f"    {key}: {val}")
    
    if model_name in results["model1_match_outcome"]["rdd"]:
        report.append("  RDD:")
        rdd_metrics = results["model1_match_outcome"]["rdd"][model_name]
        for key, val in rdd_metrics.items():
            report.append(f"    {key}: {val}")
        
        if model_name in results["comparison_summary"]["model1_comparison"]:
            comp = results["comparison_summary"]["model1_comparison"][model_name]
            report.append(f"  Comparison:")
            report.append(f"    Training Time Ratio (DF/RDD): {comp['training_time_ratio_df_to_rdd']}x")
            report.append(f"    Accuracy Difference (DF-RDD): {comp['accuracy_difference']}")
            report.append(f"    Memory Difference (DF-RDD): {comp['memory_difference_mb']} MB")

# Scalability Summary
report.append("\n" + "="*80)
report.append("SCALABILITY & PERFORMANCE SUMMARY")
report.append("="*80)
report.append(f"Total Rows Processed: {results['comparison_summary']['scalability_metrics']['total_rows_processed']:,}")
report.append(f"Peak Memory: {results['comparison_summary']['scalability_metrics']['peak_memory_mb']} MB")
report.append(f"Average CPU: {results['comparison_summary']['scalability_metrics']['avg_cpu_utilization']}%")
report.append(f"Total Time: {results['comparison_summary']['scalability_metrics']['total_execution_time_sec']} sec")
report.append(f"Estimated Cost: ${results['comparison_summary']['scalability_metrics']['estimated_cost_usd']}")

# Key Findings
report.append("\n" + "="*80)
report.append("KEY FINDINGS")
report.append("="*80)
report.append(f"\nBest DataFrame Model: {best_model1_df[0]} (AUC: {round_float(best_model1_df[1], 4)})")
report.append(f"Best RDD Model: {best_model1_rdd[0]} (AUC: {round_float(best_model1_rdd[1], 4)})")

# Convert to single string
report_text = "\n".join(report)

# Print to console
print("\n" + report_text)

# Save as JSON
json_output = json.dumps(results, indent=2)
json_rdd = sc.parallelize([json_output])
json_rdd.coalesce(1).saveAsTextFile(f"{OUTPUT_PATH}/model1_results.json")

# Save as text report
report_rdd = sc.parallelize([report_text])
report_rdd.coalesce(1).saveAsTextFile(f"{OUTPUT_PATH}/model1_results_report")

print(f"\nResults saved to:")
print(f"  - {OUTPUT_PATH}/model1_results.json")
print(f"  - {OUTPUT_PATH}/model1_results_report")

spark.stop()