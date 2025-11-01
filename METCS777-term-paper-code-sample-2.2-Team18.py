# Task 2 Model 2: Match Duration Prediction - RDD vs DataFrame Comparison

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, avg, stddev, countDistinct, abs as spark_abs
from pyspark.sql.functions import sum as spark_sum, min as spark_min, max as spark_max
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.tree import RandomForest as RDD_RandomForest, GradientBoostedTrees as RDD_GBT
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.linalg import Vectors as MLLibVectors
import time
import json
import sys
import psutil
import os
import builtins
import gc
import math

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
    .appName("Dota2-Model2-RDD-vs-DataFrame-Optimized") \
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
    if value is None or math.isnan(value) or math.isinf(value):
        return None
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

def safe_round(value, decimals=2):
    """Safely round a value, handling NaN and infinity"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return round_float(value, decimals)
    return value

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
    "model2_match_duration": {
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
teamfights = spark.read.csv(f"{INPUT_PATH}/teamfights.csv", header=True, inferSchema=True)
main_metadata = spark.read.csv(f"{INPUT_PATH}/main_metadata.csv", header=True, inferSchema=True)

players.persist()
objectives.persist()

player_count = players.count()
objectives_count = objectives.count()
metadata_count = main_metadata.count()

print(f"Players: {player_count} rows")
print(f"Objectives: {objectives_count} rows")
print(f"Metadata: {metadata_count} rows")

load_time = time.time() - load_start
memory_after_load = get_memory_usage()
cpu_after_load = get_cpu_percent()

total_rows = player_count + objectives_count + metadata_count
data_throughput = calculate_throughput(total_rows, load_time)

print(f"\nData loading completed in {round_float(load_time, 2)} seconds")
print(f"Memory used: {round_float(memory_after_load - memory_before_load, 2)} MB")
print(f"Throughput: {round_float(data_throughput, 2)} rows/sec")

results["system_metrics"]["peak_memory_mb"] = update_peak_memory(
    results["system_metrics"]["peak_memory_mb"], memory_after_load)
cpu_readings.extend([cpu_before_load, cpu_after_load])

# ==============================================================================
# FEATURE ENGINEERING - MODEL 2 (MATCH DURATION) - DATAFRAME
# ==============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING - MODEL 2 (MATCH DURATION) - DATAFRAME")
print("="*80)

fe2_df_start = time.time()
memory_before_fe2_df = get_memory_usage()
cpu_fe2_df_start = get_cpu_percent()

query2_start = time.time()

match_duration = main_metadata.select('match_id', 'duration', 'leagueid', 'radiant_score', 
                                      'dire_score', 'first_blood_time', 'radiant_win')
match_duration = match_duration.withColumn('leagueid', col('leagueid').cast('int'))
match_duration = match_duration.withColumn('first_blood_time', col('first_blood_time').cast('int'))

total_objectives = objectives.groupBy('match_id').agg(
    count('*').alias('total_objectives'),
    count(when(col('type').like('%tower%'), 1)).alias('total_towers'),
    count(when(col('type').like('%barracks%'), 1)).alias('total_barracks')
)

total_teamfights = teamfights.groupBy('match_id').agg(
    count('*').alias('total_teamfights'),
    avg('deaths').alias('avg_deaths_per_teamfight'),
    spark_sum('deaths').alias('total_teamfight_deaths')
)

gold_swing = radiant_gold_adv.groupBy('match_id').agg(
    spark_max('gold').alias('max_gold_lead'),
    spark_min('gold').alias('min_gold_lead'),
    stddev('gold').alias('gold_swing_std')
)

hero_diversity = picks_bans.filter(col('is_pick') == True).groupBy('match_id').agg(
    countDistinct('hero_id').alias('unique_heroes_picked')
)

player_aggregates = players.groupBy('match_id').agg(
    avg('kills').alias('avg_kills_per_player'),
    avg('deaths').alias('avg_deaths_per_player'),
    avg('assists').alias('avg_assists_per_player'),
    avg('gold_per_min').alias('avg_gold_per_min'),
    avg('xp_per_min').alias('avg_xp_per_min'),
    avg('hero_damage').alias('avg_hero_damage'),
    spark_sum('tower_damage').alias('total_tower_damage')
)

hero_picks = picks_bans.filter(col('is_pick') == True)

radiant_picks = hero_picks.filter(col('team') == 0).groupBy('match_id').agg(
    count('hero_id').alias('radiant_pick_count')
)

dire_picks = hero_picks.filter(col('team') == 1).groupBy('match_id').agg(
    count('hero_id').alias('dire_pick_count')
)

draft_stats = draft_timings.groupBy('match_id').agg(
    avg('total_time_taken').alias('avg_draft_time')
)

query2_time = time.time() - query2_start

dataset2_df = match_duration \
    .join(radiant_picks, on='match_id', how='left') \
    .join(dire_picks, on='match_id', how='left') \
    .join(draft_stats, on='match_id', how='left') \
    .join(total_objectives, on='match_id', how='left') \
    .join(total_teamfights, on='match_id', how='left') \
    .join(gold_swing, on='match_id', how='left') \
    .join(hero_diversity, on='match_id', how='left') \
    .join(player_aggregates, on='match_id', how='left')

dataset2_df = dataset2_df.fillna({
    'leagueid': 0, 'radiant_score': 0, 'dire_score': 0, 'first_blood_time': 0,
    'radiant_pick_count': 5, 'dire_pick_count': 5, 'avg_draft_time': 30,
    'total_objectives': 0, 'total_towers': 0, 'total_barracks': 0,
    'total_teamfights': 0, 'avg_deaths_per_teamfight': 0, 'total_teamfight_deaths': 0,
    'max_gold_lead': 0, 'min_gold_lead': 0, 'gold_swing_std': 0,
    'unique_heroes_picked': 10, 'avg_kills_per_player': 0, 'avg_deaths_per_player': 0,
    'avg_assists_per_player': 0, 'avg_gold_per_min': 400, 'avg_xp_per_min': 500,
    'avg_hero_damage': 0, 'total_tower_damage': 0
})

dataset2_df = dataset2_df.withColumn('score_difference', spark_abs(col('radiant_score') - col('dire_score')))
dataset2_df = dataset2_df.withColumn('total_score', col('radiant_score') + col('dire_score'))

dataset2_df = dataset2_df.filter((col('duration') >= 600) & (col('duration') <= 7200))
dataset2_df = dataset2_df.dropna(subset=['duration'])
dataset2_df = dataset2_df.repartition(50)
dataset2_df.persist()

fe2_df_time = time.time() - fe2_df_start
memory_after_fe2_df = get_memory_usage()
cpu_fe2_df_end = get_cpu_percent()

dataset2_df_count = dataset2_df.count()
print(f"DataFrame - Dataset 2 rows: {dataset2_df_count}")
print(f"Feature engineering time: {round_float(fe2_df_time, 2)} seconds")
print(f"Query execution time: {round_float(query2_time, 2)} seconds")
print(f"Total features: {len(dataset2_df.columns) - 2}")
print(f"Memory used: {round_float(memory_after_fe2_df - memory_before_fe2_df, 2)} MB")

results["etl_metrics"]["dataframe"]["feature_engineering_time_sec"] = round_float(fe2_df_time, 2)
results["etl_metrics"]["dataframe"]["query_execution_time_sec"] = round_float(query2_time, 2)
results["etl_metrics"]["dataframe"]["total_rows_processed"] = dataset2_df_count
results["etl_metrics"]["dataframe"]["data_throughput_rows_per_sec"] = round_float(
    calculate_throughput(dataset2_df_count, fe2_df_time), 2)

results["system_metrics"]["peak_memory_mb"] = update_peak_memory(
    results["system_metrics"]["peak_memory_mb"], memory_after_fe2_df)
cpu_readings.extend([cpu_fe2_df_start, cpu_fe2_df_end])

# ==============================================================================
# FEATURE ENGINEERING - MODEL 2 (MATCH DURATION) - RDD
# ==============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING - MODEL 2 (MATCH DURATION) - RDD")
print("="*80)

fe2_rdd_start = time.time()
memory_before_fe2_rdd = get_memory_usage()
cpu_fe2_rdd_start = get_cpu_percent()

query2_rdd_start = time.time()

main_metadata_rdd = main_metadata.rdd
objectives_rdd = objectives.rdd
teamfights_rdd = teamfights.rdd
radiant_gold_adv_rdd = radiant_gold_adv.rdd
players_rdd = players.rdd

def extract_duration_features(row):
    duration = row['duration']
    if duration and 600 <= duration <= 7200:
        match_id = row['match_id']
        return (match_id, {
            'duration': duration,
            'leagueid': row['leagueid'] if row['leagueid'] is not None else 0,
            'radiant_score': row['radiant_score'] if row['radiant_score'] is not None else 0,
            'dire_score': row['dire_score'] if row['dire_score'] is not None else 0,
            'first_blood_time': row['first_blood_time'] if row['first_blood_time'] is not None else 0
        })
    return None

duration_features_rdd = main_metadata_rdd.map(extract_duration_features).filter(lambda x: x is not None)

def count_objectives(row):
    match_id = row['match_id']
    obj_type = str(row['type']) if row['type'] is not None else ''
    tower = 1 if 'tower' in obj_type.lower() else 0
    barracks = 1 if 'barracks' in obj_type.lower() else 0
    return (match_id, (1, tower, barracks))

objectives_stats_rdd = objectives_rdd.map(count_objectives) \
    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2]))

def process_teamfights(row):
    deaths = row['deaths'] if row['deaths'] is not None else 0
    return (row['match_id'], (1, deaths))

teamfight_stats_rdd = teamfights_rdd.map(process_teamfights) \
    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))

def process_gold(row):
    gold = row['gold']
    if gold is not None:
        return (row['match_id'], gold)
    return None

gold_all_rdd = radiant_gold_adv_rdd.map(process_gold).filter(lambda x: x is not None)
gold_swing_rdd = gold_all_rdd.groupByKey().mapValues(lambda vals: {
    'max': max(vals) if len(vals) > 0 else 0,
    'min': min(vals) if len(vals) > 0 else 0
})

def aggregate_player_stats(row):
    return (row['match_id'], (
        row['kills'] if row['kills'] is not None else 0,
        row['deaths'] if row['deaths'] is not None else 0,
        row['assists'] if row['assists'] is not None else 0,
        row['gold_per_min'] if row['gold_per_min'] is not None else 0,
        row['xp_per_min'] if row['xp_per_min'] is not None else 0,
        row['hero_damage'] if row['hero_damage'] is not None else 0,
        row['tower_damage'] if row['tower_damage'] is not None else 0,
        1
    ))

player_agg_rdd = players_rdd.map(aggregate_player_stats) \
    .reduceByKey(lambda a, b: tuple(a[i] + b[i] for i in range(8)))

query2_rdd_time = time.time() - query2_rdd_start

def combine_duration_features(match_id, base, obj_stats, tf_stats, gold, player_stats):
    features = [
        base['leagueid'],
        base['radiant_score'],
        base['dire_score'],
        abs(base['radiant_score'] - base['dire_score']),
        base['radiant_score'] + base['dire_score'],
        base['first_blood_time'],
        obj_stats[0] if obj_stats else 0,
        obj_stats[1] if obj_stats else 0,
        obj_stats[2] if obj_stats else 0,
        tf_stats[0] if tf_stats else 0,
        tf_stats[1] / tf_stats[0] if tf_stats and tf_stats[0] > 0 else 0,
        tf_stats[1] if tf_stats else 0,
        gold['max'] if gold else 0,
        gold['min'] if gold else 0,
    ]
    
    if player_stats:
        count = player_stats[7]
        features.extend([
            player_stats[0] / count if count > 0 else 0,
            player_stats[1] / count if count > 0 else 0,
            player_stats[2] / count if count > 0 else 0,
            player_stats[3] / count if count > 0 else 400,
            player_stats[4] / count if count > 0 else 500,
            player_stats[5] / count if count > 0 else 0,
            player_stats[6],
        ])
    else:
        features.extend([0, 0, 0, 400, 500, 0, 0])
    
    return LabeledPoint(base['duration'], MLLibVectors.dense(features))

combined2_rdd = duration_features_rdd \
    .leftOuterJoin(objectives_stats_rdd) \
    .leftOuterJoin(teamfight_stats_rdd) \
    .leftOuterJoin(gold_swing_rdd) \
    .leftOuterJoin(player_agg_rdd) \
    .map(lambda x: combine_duration_features(x[0], x[1][0][0][0][0], x[1][0][0][0][1], 
                                             x[1][0][0][1], x[1][0][1], x[1][1]))

dataset2_rdd = combined2_rdd.repartition(50)
dataset2_rdd.persist()

fe2_rdd_time = time.time() - fe2_rdd_start
memory_after_fe2_rdd = get_memory_usage()
cpu_fe2_rdd_end = get_cpu_percent()

dataset2_rdd_count = dataset2_rdd.count()
print(f"RDD - Dataset 2 rows: {dataset2_rdd_count}")
print(f"Feature engineering time: {round_float(fe2_rdd_time, 2)} seconds")
print(f"Query execution time: {round_float(query2_rdd_time, 2)} seconds")
print(f"Memory used: {round_float(memory_after_fe2_rdd - memory_before_fe2_rdd, 2)} MB")

results["etl_metrics"]["rdd"]["feature_engineering_time_sec"] = round_float(fe2_rdd_time, 2)
results["etl_metrics"]["rdd"]["query_execution_time_sec"] = round_float(query2_rdd_time, 2)
results["etl_metrics"]["rdd"]["total_rows_processed"] = dataset2_rdd_count
results["etl_metrics"]["rdd"]["data_throughput_rows_per_sec"] = round_float(
    calculate_throughput(dataset2_rdd_count, fe2_rdd_time), 2)

results["system_metrics"]["peak_memory_mb"] = update_peak_memory(
    results["system_metrics"]["peak_memory_mb"], memory_after_fe2_rdd)
cpu_readings.extend([cpu_fe2_rdd_start, cpu_fe2_rdd_end])

players.unpersist()
objectives.unpersist()

force_cleanup()

# ==============================================================================
# MODEL 2: MATCH DURATION PREDICTION - DATAFRAME
# ==============================================================================
print("\n" + "="*80)
print("MODEL 2: MATCH DURATION PREDICTION - DATAFRAME")
print("="*80)

model2_df_start = time.time()
memory_before_model2_df = get_memory_usage()

feature_cols_2 = [
    'leagueid', 'radiant_score', 'dire_score', 'score_difference', 'total_score', 'first_blood_time',
    'radiant_pick_count', 'dire_pick_count', 'avg_draft_time',
    'total_objectives', 'total_towers', 'total_barracks',
    'total_teamfights', 'avg_deaths_per_teamfight', 'total_teamfight_deaths',
    'max_gold_lead', 'min_gold_lead', 'gold_swing_std',
    'unique_heroes_picked', 'avg_kills_per_player', 'avg_deaths_per_player',
    'avg_assists_per_player', 'avg_gold_per_min', 'avg_xp_per_min',
    'avg_hero_damage', 'total_tower_damage'
]

assembler2 = VectorAssembler(inputCols=feature_cols_2, outputCol="features_raw")
scaler2 = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

temp_df2 = assembler2.transform(dataset2_df)
scaler_model2 = scaler2.fit(temp_df2)
dataset2_prepared = scaler_model2.transform(temp_df2)
dataset2_prepared = dataset2_prepared.select('features', 'duration')

train2_df, test2_df = dataset2_prepared.randomSplit([0.8, 0.2], seed=42)
train2_df.persist()
test2_df.persist()

train_count_2_df = train2_df.count()
test_count_2_df = test2_df.count()

print(f"DataFrame - Train: {train_count_2_df}, Test: {test_count_2_df}")
print(f"Number of features: {len(feature_cols_2)}")

results["model2_match_duration"]["dataframe"]["feature_count"] = len(feature_cols_2)
results["model2_match_duration"]["dataframe"]["training_rows"] = train_count_2_df
results["model2_match_duration"]["dataframe"]["test_rows"] = test_count_2_df

# Linear Regression - DataFrame
print("\n--- Linear Regression (DataFrame) ---")
linreg_df_memory_before = get_memory_usage()
linreg_df_train_start = time.time()
linreg_df = LinearRegression(featuresCol="features", labelCol="duration", maxIter=100, regParam=0.01)
linreg_df_model = linreg_df.fit(train2_df)
linreg_df_train_time = time.time() - linreg_df_train_start
linreg_df_memory_after = get_memory_usage()

linreg_df_pred_start = time.time()
linreg_df_predictions = linreg_df_model.transform(test2_df)
linreg_df_pred_count = linreg_df_predictions.count()
linreg_df_inference_time = (time.time() - linreg_df_pred_start) / linreg_df_pred_count if linreg_df_pred_count > 0 else 0

evaluator_rmse = RegressionEvaluator(labelCol="duration", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="duration", predictionCol="prediction", metricName="r2")

linreg_df_rmse = evaluator_rmse.evaluate(linreg_df_predictions)
linreg_df_r2 = evaluator_r2.evaluate(linreg_df_predictions)
linreg_df_memory_footprint = linreg_df_memory_after - linreg_df_memory_before

print(f"Training: {round_float(linreg_df_train_time, 2)}s")
print(f"RMSE: {round_float(linreg_df_rmse, 2)}, R²: {round_float(linreg_df_r2, 4)}")
print(f"Inference: {round_float(linreg_df_inference_time*1000, 4)}ms per row")
print(f"Memory: {round_float(linreg_df_memory_footprint, 2)} MB")

results["model2_match_duration"]["dataframe"]["linear_regression"] = {
    "training_time_sec": round_float(linreg_df_train_time, 2),
    "rmse": safe_round(linreg_df_rmse, 2),
    "r2": safe_round(linreg_df_r2, 4),
    "inference_time_ms": round_float(linreg_df_inference_time * 1000, 4),
    "memory_footprint_mb": round_float(linreg_df_memory_footprint, 2)
}

cpu_readings.append(get_cpu_percent())

# Random Forest Regressor - DataFrame
print("\n--- Random Forest Regressor (DataFrame) ---")
rfr_df_memory_before = get_memory_usage()
rfr_df_train_start = time.time()
rfr_df = RandomForestRegressor(featuresCol="features", labelCol="duration", 
                               numTrees=50, maxDepth=10, seed=42)
rfr_df_model = rfr_df.fit(train2_df)
rfr_df_train_time = time.time() - rfr_df_train_start
rfr_df_memory_after = get_memory_usage()

rfr_df_pred_start = time.time()
rfr_df_predictions = rfr_df_model.transform(test2_df)
rfr_df_pred_count = rfr_df_predictions.count()
rfr_df_inference_time = (time.time() - rfr_df_pred_start) / rfr_df_pred_count if rfr_df_pred_count > 0 else 0

rfr_df_rmse = evaluator_rmse.evaluate(rfr_df_predictions)
rfr_df_r2 = evaluator_r2.evaluate(rfr_df_predictions)
rfr_df_memory_footprint = rfr_df_memory_after - rfr_df_memory_before

rfr_feature_importance = [(feature_cols_2[i], float(rfr_df_model.featureImportances[i])) 
                          for i in range(len(feature_cols_2))]
rfr_feature_importance.sort(key=lambda x: x[1], reverse=True)

print(f"Training: {round_float(rfr_df_train_time, 2)}s")
print(f"RMSE: {round_float(rfr_df_rmse, 2)}, R²: {round_float(rfr_df_r2, 4)}")
print(f"Inference: {round_float(rfr_df_inference_time*1000, 4)}ms per row")
print(f"Memory: {round_float(rfr_df_memory_footprint, 2)} MB")
print("Top 5 Feature Importance:")
for feat, imp in rfr_feature_importance[:5]:
    print(f"  {feat}: {round_float(imp, 4)}")

results["model2_match_duration"]["dataframe"]["random_forest"] = {
    "training_time_sec": round_float(rfr_df_train_time, 2),
    "rmse": safe_round(rfr_df_rmse, 2),
    "r2": safe_round(rfr_df_r2, 4),
    "inference_time_ms": round_float(rfr_df_inference_time * 1000, 4),
    "memory_footprint_mb": round_float(rfr_df_memory_footprint, 2),
    "feature_importance": {feat: round_float(imp, 4) for feat, imp in rfr_feature_importance[:10]}
}

cpu_readings.append(get_cpu_percent())

# Gradient Boosting Regressor - DataFrame
print("\n--- Gradient Boosting Regressor (DataFrame) ---")
gbt2_df_memory_before = get_memory_usage()
gbt2_df_train_start = time.time()
gbt2_df = GBTRegressor(featuresCol="features", labelCol="duration", maxIter=30, maxDepth=8, seed=42)
gbt2_df_model = gbt2_df.fit(train2_df)
gbt2_df_train_time = time.time() - gbt2_df_train_start
gbt2_df_memory_after = get_memory_usage()

gbt2_df_pred_start = time.time()
gbt2_df_predictions = gbt2_df_model.transform(test2_df)
gbt2_df_pred_count = gbt2_df_predictions.count()
gbt2_df_inference_time = (time.time() - gbt2_df_pred_start) / gbt2_df_pred_count if gbt2_df_pred_count > 0 else 0

gbt2_df_rmse = evaluator_rmse.evaluate(gbt2_df_predictions)
gbt2_df_r2 = evaluator_r2.evaluate(gbt2_df_predictions)
gbt2_df_memory_footprint = gbt2_df_memory_after - gbt2_df_memory_before

gbt2_feature_importance = [(feature_cols_2[i], float(gbt2_df_model.featureImportances[i])) 
                           for i in range(len(feature_cols_2))]
gbt2_feature_importance.sort(key=lambda x: x[1], reverse=True)

print(f"Training: {round_float(gbt2_df_train_time, 2)}s")
print(f"RMSE: {round_float(gbt2_df_rmse, 2)}, R²: {round_float(gbt2_df_r2, 4)}")
print(f"Inference: {round_float(gbt2_df_inference_time*1000, 4)}ms per row")
print(f"Memory: {round_float(gbt2_df_memory_footprint, 2)} MB")

results["model2_match_duration"]["dataframe"]["gradient_boosting"] = {
    "training_time_sec": round_float(gbt2_df_train_time, 2),
    "rmse": safe_round(gbt2_df_rmse, 2),
    "r2": safe_round(gbt2_df_r2, 4),
    "inference_time_ms": round_float(gbt2_df_inference_time * 1000, 4),
    "memory_footprint_mb": round_float(gbt2_df_memory_footprint, 2),
    "feature_importance": {feat: round_float(imp, 4) for feat, imp in gbt2_feature_importance[:10]}
}

model2_df_total_time = time.time() - model2_df_start
memory_after_model2_df = get_memory_usage()

results["model2_match_duration"]["dataframe"]["ml_training_time_sec"] = round_float(model2_df_total_time, 2)
results["model2_match_duration"]["dataframe"]["total_memory_used_mb"] = round_float(
    memory_after_model2_df - memory_before_model2_df, 2)

cpu_readings.append(get_cpu_percent())

train2_df.unpersist()
test2_df.unpersist()
dataset2_df.unpersist()

force_cleanup()

# ==============================================================================
# MODEL 2: MATCH DURATION PREDICTION - RDD
# ==============================================================================
print("\n" + "="*80)
print("MODEL 2: MATCH DURATION PREDICTION - RDD")
print("="*80)

model2_rdd_start = time.time()
memory_before_model2_rdd = get_memory_usage()

train2_rdd, test2_rdd = dataset2_rdd.randomSplit([0.8, 0.2], seed=42)
train2_rdd.persist()
test2_rdd.persist()

train_count_2_rdd = train2_rdd.count()
test_count_2_rdd = test2_rdd.count()

print(f"RDD - Train: {train_count_2_rdd}, Test: {test_count_2_rdd}")

results["model2_match_duration"]["rdd"]["feature_count"] = 21
results["model2_match_duration"]["rdd"]["training_rows"] = train_count_2_rdd
results["model2_match_duration"]["rdd"]["test_rows"] = test_count_2_rdd

# Linear Regression - RDD
print("\n--- Linear Regression (RDD) ---")
linreg_rdd_memory_before = get_memory_usage()
linreg_rdd_train_start = time.time()
linreg_rdd_model = LinearRegressionWithSGD.train(train2_rdd, iterations=100, step=0.01)
linreg_rdd_train_time = time.time() - linreg_rdd_train_start
linreg_rdd_memory_after = get_memory_usage()

linreg_rdd_pred_start = time.time()
linreg_rdd_pred_tuples = test2_rdd.map(lambda lp: (float(linreg_rdd_model.predict(lp.features)), lp.label)).collect()
linreg_rdd_pred_count = len(linreg_rdd_pred_tuples)
linreg_rdd_inference_time = (time.time() - linreg_rdd_pred_start) / linreg_rdd_pred_count if linreg_rdd_pred_count > 0 else 0

linreg_rdd_predictions = sc.parallelize(linreg_rdd_pred_tuples)
linreg_rdd_metrics = RegressionMetrics(linreg_rdd_predictions)
linreg_rdd_rmse = linreg_rdd_metrics.rootMeanSquaredError
linreg_rdd_r2 = linreg_rdd_metrics.r2
linreg_rdd_memory_footprint = linreg_rdd_memory_after - linreg_rdd_memory_before

print(f"Training: {round_float(linreg_rdd_train_time, 2)}s")
print(f"RMSE: {safe_round(linreg_rdd_rmse, 2)}, R²: {safe_round(linreg_rdd_r2, 4)}")
print(f"Inference: {round_float(linreg_rdd_inference_time*1000, 4)}ms per row")
print(f"Memory: {round_float(linreg_rdd_memory_footprint, 2)} MB")

results["model2_match_duration"]["rdd"]["linear_regression"] = {
    "training_time_sec": round_float(linreg_rdd_train_time, 2),
    "rmse": safe_round(linreg_rdd_rmse, 2),
    "r2": safe_round(linreg_rdd_r2, 4),
    "inference_time_ms": round_float(linreg_rdd_inference_time * 1000, 4),
    "memory_footprint_mb": round_float(linreg_rdd_memory_footprint, 2)
}

cpu_readings.append(get_cpu_percent())
force_cleanup()

# Random Forest Regressor - RDD
print("\n--- Random Forest Regressor (RDD) ---")
rfr_rdd_memory_before = get_memory_usage()
rfr_rdd_train_start = time.time()
rfr_rdd_model = RDD_RandomForest.trainRegressor(train2_rdd, categoricalFeaturesInfo={},
                                                numTrees=30, featureSubsetStrategy="auto",
                                                impurity='variance', maxDepth=8, maxBins=32, seed=42)
rfr_rdd_train_time = time.time() - rfr_rdd_train_start
rfr_rdd_memory_after = get_memory_usage()

rfr_rdd_pred_start = time.time()
test_data_collected = test2_rdd.collect()
rfr_rdd_pred_tuples = [(float(rfr_rdd_model.predict(lp.features)), lp.label) for lp in test_data_collected]
rfr_rdd_pred_count = len(rfr_rdd_pred_tuples)
rfr_rdd_inference_time = (time.time() - rfr_rdd_pred_start) / rfr_rdd_pred_count if rfr_rdd_pred_count > 0 else 0

rfr_rdd_predictions = sc.parallelize(rfr_rdd_pred_tuples)
rfr_rdd_metrics = RegressionMetrics(rfr_rdd_predictions)
rfr_rdd_rmse = rfr_rdd_metrics.rootMeanSquaredError
rfr_rdd_r2 = rfr_rdd_metrics.r2
rfr_rdd_memory_footprint = rfr_rdd_memory_after - rfr_rdd_memory_before

print(f"Training: {round_float(rfr_rdd_train_time, 2)}s")
print(f"RMSE: {safe_round(rfr_rdd_rmse, 2)}, R²: {safe_round(rfr_rdd_r2, 4)}")
print(f"Inference: {round_float(rfr_rdd_inference_time*1000, 4)}ms per row")
print(f"Memory: {round_float(rfr_rdd_memory_footprint, 2)} MB")

results["model2_match_duration"]["rdd"]["random_forest"] = {
    "training_time_sec": round_float(rfr_rdd_train_time, 2),
    "rmse": safe_round(rfr_rdd_rmse, 2),
    "r2": safe_round(rfr_rdd_r2, 4),
    "inference_time_ms": round_float(rfr_rdd_inference_time * 1000, 4),
    "memory_footprint_mb": round_float(rfr_rdd_memory_footprint, 2)
}

cpu_readings.append(get_cpu_percent())
force_cleanup()

# Gradient Boosting Regressor - RDD
print("\n--- Gradient Boosting Regressor (RDD) ---")
gbt2_rdd_memory_before = get_memory_usage()
gbt2_rdd_train_start = time.time()
gbt2_rdd_model = RDD_GBT.trainRegressor(train2_rdd, categoricalFeaturesInfo={},
                                        numIterations=20, maxDepth=6, learningRate=0.1)
gbt2_rdd_train_time = time.time() - gbt2_rdd_train_start
gbt2_rdd_memory_after = get_memory_usage()

gbt2_rdd_pred_start = time.time()
gbt2_rdd_pred_tuples = [(float(gbt2_rdd_model.predict(lp.features)), lp.label) for lp in test_data_collected]
gbt2_rdd_pred_count = len(gbt2_rdd_pred_tuples)
gbt2_rdd_inference_time = (time.time() - gbt2_rdd_pred_start) / gbt2_rdd_pred_count if gbt2_rdd_pred_count > 0 else 0

gbt2_rdd_predictions = sc.parallelize(gbt2_rdd_pred_tuples)
gbt2_rdd_metrics = RegressionMetrics(gbt2_rdd_predictions)
gbt2_rdd_rmse = gbt2_rdd_metrics.rootMeanSquaredError
gbt2_rdd_r2 = gbt2_rdd_metrics.r2
gbt2_rdd_memory_footprint = gbt2_rdd_memory_after - gbt2_rdd_memory_before

print(f"Training: {round_float(gbt2_rdd_train_time, 2)}s")
print(f"RMSE: {safe_round(gbt2_rdd_rmse, 2)}, R²: {safe_round(gbt2_rdd_r2, 4)}")
print(f"Inference: {round_float(gbt2_rdd_inference_time*1000, 4)}ms per row")
print(f"Memory: {round_float(gbt2_rdd_memory_footprint, 2)} MB")

results["model2_match_duration"]["rdd"]["gradient_boosting"] = {
    "training_time_sec": round_float(gbt2_rdd_train_time, 2),
    "rmse": safe_round(gbt2_rdd_rmse, 2),
    "r2": safe_round(gbt2_rdd_r2, 4),
    "inference_time_ms": round_float(gbt2_rdd_inference_time * 1000, 4),
    "memory_footprint_mb": round_float(gbt2_rdd_memory_footprint, 2)
}

model2_rdd_total_time = time.time() - model2_rdd_start
memory_after_model2_rdd = get_memory_usage()

results["model2_match_duration"]["rdd"]["ml_training_time_sec"] = round_float(model2_rdd_total_time, 2)
results["model2_match_duration"]["rdd"]["total_memory_used_mb"] = round_float(
    memory_after_model2_rdd - memory_before_model2_rdd, 2)

cpu_readings.append(get_cpu_percent())

train2_rdd.unpersist()
test2_rdd.unpersist()
dataset2_rdd.unpersist()

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

def safe_divide(a, b):
    """Safely divide two values, handling None and zero"""
    if a is None or b is None or b == 0:
        return None
    return round_float(a / b, 2)

def safe_subtract(a, b):
    """Safely subtract two values, handling None"""
    if a is None or b is None:
        return None
    return round_float(a - b, 2)

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
    "model2_comparison": {
        "linear_regression": {
            "training_time_ratio_df_to_rdd": safe_divide(
                results["model2_match_duration"]["dataframe"]["linear_regression"]["training_time_sec"],
                results["model2_match_duration"]["rdd"]["linear_regression"]["training_time_sec"]),
            "rmse_difference": safe_subtract(
                results["model2_match_duration"]["dataframe"]["linear_regression"]["rmse"],
                results["model2_match_duration"]["rdd"]["linear_regression"]["rmse"]),
            "memory_difference_mb": safe_subtract(
                results["model2_match_duration"]["dataframe"]["linear_regression"]["memory_footprint_mb"],
                results["model2_match_duration"]["rdd"]["linear_regression"]["memory_footprint_mb"])
        },
        "random_forest": {
            "training_time_ratio_df_to_rdd": safe_divide(
                results["model2_match_duration"]["dataframe"]["random_forest"]["training_time_sec"],
                results["model2_match_duration"]["rdd"]["random_forest"]["training_time_sec"]),
            "rmse_difference": safe_subtract(
                results["model2_match_duration"]["dataframe"]["random_forest"]["rmse"],
                results["model2_match_duration"]["rdd"]["random_forest"]["rmse"]),
            "memory_difference_mb": safe_subtract(
                results["model2_match_duration"]["dataframe"]["random_forest"]["memory_footprint_mb"],
                results["model2_match_duration"]["rdd"]["random_forest"]["memory_footprint_mb"])
        },
        "gradient_boosting": {
            "training_time_ratio_df_to_rdd": safe_divide(
                results["model2_match_duration"]["dataframe"]["gradient_boosting"]["training_time_sec"],
                results["model2_match_duration"]["rdd"]["gradient_boosting"]["training_time_sec"]),
            "rmse_difference": safe_subtract(
                results["model2_match_duration"]["dataframe"]["gradient_boosting"]["rmse"],
                results["model2_match_duration"]["rdd"]["gradient_boosting"]["rmse"]),
            "memory_difference_mb": safe_subtract(
                results["model2_match_duration"]["dataframe"]["gradient_boosting"]["memory_footprint_mb"],
                results["model2_match_duration"]["rdd"]["gradient_boosting"]["memory_footprint_mb"])
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

# Determine best models - filter out None values
valid_df_models = [
    (name, metric["rmse"]) 
    for name, metric in [
        ("Linear Regression", results["model2_match_duration"]["dataframe"]["linear_regression"]),
        ("Random Forest", results["model2_match_duration"]["dataframe"]["random_forest"]),
        ("Gradient Boosting", results["model2_match_duration"]["dataframe"]["gradient_boosting"])
    ]
    if metric["rmse"] is not None
]

valid_rdd_models = [
    (name, metric["rmse"]) 
    for name, metric in [
        ("Linear Regression", results["model2_match_duration"]["rdd"]["linear_regression"]),
        ("Random Forest", results["model2_match_duration"]["rdd"]["random_forest"]),
        ("Gradient Boosting", results["model2_match_duration"]["rdd"]["gradient_boosting"])
    ]
    if metric["rmse"] is not None
]

best_model2_df = builtins.min(valid_df_models, key=lambda x: x[1]) if valid_df_models else ("None", None)
best_model2_rdd = builtins.min(valid_rdd_models, key=lambda x: x[1]) if valid_rdd_models else ("None", None)

report = []
report.append("="*80)
report.append("DOTA 2 MODEL 2: MATCH DURATION PREDICTION")
report.append("RDD vs DATAFRAME COMPARISON")
report.append("="*80)
report.append(f"\nExecution Date: {results['system_metrics']['start_time']}")
report.append(f"Total Execution Time: {results['system_metrics']['total_execution_time_sec']} seconds")
report.append(f"Estimated Cost: ${results['system_metrics']['total_cost_estimate_usd']} USD")

report.append("\n" + "="*80)
report.append("SYSTEM RESOURCE METRICS")
report.append("="*80)
report.append(f"Initial Memory: {results['system_metrics']['initial_memory_mb']} MB")
report.append(f"Peak Memory Usage: {results['system_metrics']['peak_memory_mb']} MB")
report.append(f"Average CPU Utilization: {results['system_metrics']['avg_cpu_percent']}%")

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

report.append("\n" + "="*80)
report.append("MODEL 2: MATCH DURATION PREDICTION COMPARISON")
report.append("="*80)

for model_name in ["linear_regression", "random_forest", "gradient_boosting"]:
    report.append(f"\n{model_name.replace('_', ' ').title()}:")
    report.append("  DataFrame:")
    df_metrics = results["model2_match_duration"]["dataframe"][model_name]
    for key, val in df_metrics.items():
        if key != "feature_importance":
            report.append(f"    {key}: {val if val is not None else 'N/A'}")
    
    if model_name in results["model2_match_duration"]["rdd"]:
        report.append("  RDD:")
        rdd_metrics = results["model2_match_duration"]["rdd"][model_name]
        for key, val in rdd_metrics.items():
            report.append(f"    {key}: {val if val is not None else 'N/A'}")
        
        if model_name in results["comparison_summary"]["model2_comparison"]:
            comp = results["comparison_summary"]["model2_comparison"][model_name]
            report.append(f"  Comparison:")
            report.append(f"    Training Time Ratio (DF/RDD): {comp['training_time_ratio_df_to_rdd'] if comp['training_time_ratio_df_to_rdd'] is not None else 'N/A'}x")
            report.append(f"    RMSE Difference (DF-RDD): {comp['rmse_difference'] if comp['rmse_difference'] is not None else 'N/A'}")
            report.append(f"    Memory Difference (DF-RDD): {comp['memory_difference_mb'] if comp['memory_difference_mb'] is not None else 'N/A'} MB")

report.append("\n" + "="*80)
report.append("SCALABILITY & PERFORMANCE SUMMARY")
report.append("="*80)
report.append(f"Total Rows Processed: {results['comparison_summary']['scalability_metrics']['total_rows_processed']:,}")
report.append(f"Peak Memory: {results['comparison_summary']['scalability_metrics']['peak_memory_mb']} MB")
report.append(f"Average CPU: {results['comparison_summary']['scalability_metrics']['avg_cpu_utilization']}%")
report.append(f"Total Time: {results['comparison_summary']['scalability_metrics']['total_execution_time_sec']} sec")
report.append(f"Estimated Cost: ${results['comparison_summary']['scalability_metrics']['estimated_cost_usd']}")

report.append("\n" + "="*80)
report.append("KEY FINDINGS")
report.append("="*80)
if best_model2_df[1] is not None:
    report.append(f"\nBest DataFrame Model: {best_model2_df[0]} (RMSE: {round_float(best_model2_df[1], 2)})")
else:
    report.append(f"\nBest DataFrame Model: {best_model2_df[0]}")

if best_model2_rdd[1] is not None:
    report.append(f"Best RDD Model: {best_model2_rdd[0]} (RMSE: {round_float(best_model2_rdd[1], 2)})")
else:
    report.append(f"Best RDD Model: {best_model2_rdd[0]}")

report_text = "\n".join(report)

print("\n" + report_text)

json_output = json.dumps(results, indent=2)
json_rdd = sc.parallelize([json_output])
json_rdd.coalesce(1).saveAsTextFile(f"{OUTPUT_PATH}/model2_results.json")

report_rdd = sc.parallelize([report_text])
report_rdd.coalesce(1).saveAsTextFile(f"{OUTPUT_PATH}/model2_results_report")

print(f"\nResults saved to:")
print(f"  - {OUTPUT_PATH}/model2_results.json")
print(f"  - {OUTPUT_PATH}/model2_results_report")

spark.stop()