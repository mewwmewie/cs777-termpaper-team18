"""
DOTA2 Scalable Analytics with PySpark
Task 1: Complex Analytical Queries with Performance Benchmarks (RDD)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, sum as spark_sum, min, max, stddev, desc,
    when, lit, round as spark_round,  year, month
)
from pyspark.sql.window import Window
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os


class DOTA2Analytics:
    """
    DOTA2 Data Analytics Engine
    Performs Task 1 & 2 on DOTA2 match data with performance tracking
    """

    def __init__(self, data_path, output_path="/tmp/dota2_results"):
        """
        Initialize Spark session and data paths

        Args:
            data_path: Root directory path for data (folder containing 3 CSVs)
            output_path: Path for saving results
        """
        self.data_path = data_path.rstrip("/")
        self.output_path = output_path.rstrip("/")
        self.performance_metrics = {}
        self.viz_data = {}

        self.spark = (
            SparkSession.builder
            .appName("DOTA2_Scalable_Analytics_T12_CSV")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.default.parallelism", "200")
            .config("spark.sql.files.maxPartitionBytes", "128MB")
            .config("spark.memory.fraction", "0.8")
            .config("spark.memory.storageFraction", "0.3")
            .config("spark.sql.parquet.compression.codec", "snappy")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("WARN")

        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

        print("=" * 80)
        print("DOTA2 Scalable Analytics (Task 1 & 2, CSV mode) Initialized")
        print("=" * 80)

    def load_data(self):
        """Load CSV files: main_metadata.csv, picks_bans.csv, players.csv"""
        print("\n[1/3] Loading merged CSVs from folder ...")

        def _csv(file_name):
            path = f"{self.data_path}/{file_name}"
            return (
                self.spark.read
                .option("header", "true")
                .option("multiLine", "false")
                .csv(path)
            )

        try:
            meta = _csv("main_metadata.csv")

            time_cols = [c for c in ["start_date_time", "start_time", "start_time_utc", "match_start_time"] if c in meta.columns]
            if time_cols:
                from pyspark.sql.functions import to_timestamp
                tcol = time_cols[0]
                meta = meta.withColumn(tcol, to_timestamp(col(tcol)))
                meta = meta.withColumn("year", year(col(tcol))).withColumn("month", month(col(tcol)))
            else:
                meta = meta.withColumn("year", lit(None).cast("int")).withColumn("month", lit(None).cast("int"))

            meta = (
                meta
                .withColumn("match_id", col("match_id").cast("long"))
                .withColumn("duration", col("duration").cast("int"))
                .withColumn(
                    "radiant_win",
                    when(col("radiant_win").isin("1", 1, "true", "True"), lit(True))
                    .when(col("radiant_win").isin("0", 0, "false", "False"), lit(False))
                    .otherwise(None).cast("boolean")
                )
                .withColumn("radiant_score", col("radiant_score").cast("int"))
                .withColumn("dire_score", col("dire_score").cast("int"))
                .withColumn("leagueid", col("leagueid").cast("int"))
            )
            self.main_metadata = meta
            print(f"Loaded main_metadata.csv: {self.main_metadata.count():,} records")

            pb = _csv("picks_bans.csv")
            pb = (
                pb
                .withColumn("match_id", col("match_id").cast("long"))
                .withColumn("hero_id", col("hero_id").cast("int"))
                .withColumn("team", col("team").cast("int"))
                .withColumn("is_pick_int", when(col("is_pick").isin("1", 1, "true", "True", "TRUE"), lit(1)).otherwise(lit(0)))
            )
            
            pb = pb.withColumn("is_ban", lit(1) - col("is_pick_int"))
            pb = pb.drop("is_pick").withColumnRenamed("is_pick_int", "is_pick")
            
            self.picks_bans = pb.join(
                self.main_metadata.select(
                    col("match_id").alias("match_id_meta"), 
                    "year", 
                    "month", 
                    col("radiant_win").alias("match_radiant_win")
                ), 
                pb["match_id"] == col("match_id_meta"), 
                "left"
            ).drop("match_id_meta")
            
            print(f"Loaded picks_bans.csv: {self.picks_bans.count():,} records")

            pl = _csv("players.csv")
            self.players = pl.withColumn("match_id", col("match_id").cast("long"))
            print(f"Loaded players.csv: {self.players.count():,} records")

            self.main_metadata.cache()
            self.picks_bans.cache()
            self.players.cache()
            print("\nAll CSVs loaded & lightly typed.\n")

        except Exception as e:
            print(f"\nError loading CSV data: {str(e)}")
            print("Expected folder structure under data_path/:")
            print("  main_metadata.csv")
            print("  picks_bans.csv")
            print("  players.csv")
            raise

    def time_query(self, query_name, func):
        """
        Execute query and track performance metrics
        """
        print(f"\n{'=' * 80}\nExecuting: {query_name}\n{'=' * 80}")
        start_time = time.time()
        result = func()
        execution_time = time.time() - start_time
        self.performance_metrics[query_name] = {
            "execution_time": round(execution_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        print(f"\nCompleted in {execution_time:.2f} seconds")
        return result

    def hero_meta_analysis(self):
        """Task 1: hero trends & win-rates by duration buckets"""

        def hero_trends_over_time():
            print("\n[1.1] Hero Pick/Ban Trends Over Time")
            trends = self.picks_bans.groupBy("year", "hero_id").agg(
                spark_sum(when(col("is_pick") == 1, 1).otherwise(0)).alias("picks"),
                spark_sum(when(col("is_ban") == 1, 1).otherwise(0)).alias("bans"),
                count("*").alias("total_appearances"),
            ).withColumn("pick_rate", spark_round((col("picks") / col("total_appearances")) * 100, 2)
            ).withColumn("ban_rate", spark_round((col("bans") / col("total_appearances")) * 100, 2))

            top_trending = trends.groupBy("hero_id").agg(
                avg("pick_rate").alias("avg_pick_rate"),
                stddev("pick_rate").alias("pick_rate_volatility"),
                avg("ban_rate").alias("avg_ban_rate"),
            ).orderBy(desc("pick_rate_volatility")).limit(10)

            self.viz_data["hero_trends"] = trends.toPandas()
            self.viz_data["top_trending_heroes"] = top_trending.toPandas()
            return trends

        def hero_by_duration():
            print("\n[1.2] Hero Performance by Match Duration")
            picks_with_duration = (
                self.picks_bans.filter(col("is_pick") == 1)
                .join(
                    self.main_metadata.select(
                        col("match_id").alias("match_id_dur"), 
                        col("duration").alias("match_duration"), 
                        col("radiant_win").alias("match_result")
                    ), 
                    self.picks_bans["match_id"] == col("match_id_dur"), 
                    "inner"
                )
                .drop("match_id_dur")
                .withColumn(
                    "game_phase",
                    when(col("match_duration") < 1800, "Early (<30min)")
                    .when(col("match_duration") < 2700, "Mid (30-45min)")
                    .otherwise("Late (>45min)")
                )
                .withColumn(
                    "win",
                    when(
                        (col("team") == 0) & (col("match_result") == True) |
                        (col("team") == 1) & (col("match_result") == False),
                        1
                    ).otherwise(0)
                )
            )
            hero_phase_stats = (
                picks_with_duration.groupBy("hero_id", "game_phase")
                .agg(count("*").alias("games"), spark_sum("win").alias("wins"))
                .withColumn("win_rate", spark_round((col("wins") / col("games")) * 100, 2))
                .filter(col("games") >= 50)
            )
            self.viz_data["hero_phase_stats"] = hero_phase_stats.toPandas()
            return hero_phase_stats

        results = {}
        results['trends'] = self.time_query("Hero Trends Over Time", hero_trends_over_time)
        results['by_duration'] = self.time_query("Hero Performance by Duration", hero_by_duration)
        return results

    def match_duration_analysis(self):
        """Task 2: duration trends & correlation"""

        def duration_trends():
            print("\n[2.1] Average Match Duration Trends")
            trends = (
                self.main_metadata.groupBy("year", "month")
                .agg(
                    avg("duration").alias("avg_duration_sec"),
                    count("*").alias("total_matches"),
                    min("duration").alias("shortest"),
                    max("duration").alias("longest"),
                    stddev("duration").alias("std_dev"),
                )
                .withColumn("avg_duration_min", spark_round(col("avg_duration_sec") / 60, 2))
                .orderBy("year", "month")
            )
            self.viz_data["duration_trends"] = trends.toPandas()
            return trends

        def duration_winrate_correlation():
            print("\n[2.2] Match Duration vs Win Rate Correlation")
            duration_buckets = (
                self.main_metadata.withColumn(
                    "duration_bucket",
                    when(col("duration") < 1200, "0-20min")
                    .when(col("duration") < 1800, "20-30min")
                    .when(col("duration") < 2400, "30-40min")
                    .when(col("duration") < 3000, "40-50min")
                    .otherwise("50min+")
                )
                .groupBy("duration_bucket")
                .agg(
                    count("*").alias("total_matches"),
                    spark_sum(when(col("radiant_win") == True, 1).otherwise(0)).alias("radiant_wins"),
                )
                .withColumn("radiant_win_rate", spark_round((col("radiant_wins") / col("total_matches")) * 100, 2))
            )
            self.viz_data["duration_correlation"] = duration_buckets.toPandas()
            return duration_buckets

        results = {}
        results['trends'] = self.time_query("Duration Trends", duration_trends)
        results['correlation'] = self.time_query("Duration-WinRate Correlation", duration_winrate_correlation)
        return results

    def create_visualizations(self):
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS (02, 05, 12)")
        print("=" * 80)

        if self.output_path.startswith("gs://"):
            viz_path = "/tmp/visualizations"
            os.makedirs(viz_path, exist_ok=True)
        else:
            viz_path = f"{self.output_path}/visualizations"
            os.makedirs(viz_path, exist_ok=True)

        if "hero_trends" in self.viz_data and "top_trending_heroes" in self.viz_data:
            self._plot_hero_trends(viz_path)

        if "duration_correlation" in self.viz_data:
            self._plot_duration_winrate(viz_path)

        self._create_performance_dashboard(viz_path)

        if self.output_path.startswith("gs://"):
            import subprocess
            subprocess.run(["gsutil", "-m", "cp", "-r", f"{viz_path}/*", f"{self.output_path}/visualizations/"])
            print(f"\nVisualizations uploaded to: {self.output_path}/visualizations/")
        else:
            print(f"\nSelected visualizations saved to: {viz_path}")

    def _plot_hero_trends(self, viz_path):
        df = self.viz_data["hero_trends"]
        top_heroes = self.viz_data["top_trending_heroes"]["hero_id"].head(5).tolist()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for hero_id in top_heroes:
            hero_data = df[df["hero_id"] == hero_id].sort_values("year")
            ax1.plot(hero_data["year"], hero_data["pick_rate"], marker="o", label=f"Hero {hero_id}", linewidth=2)

        ax1.set_xlabel("Year"); ax1.set_ylabel("Pick Rate (%)"); ax1.set_title("Hero Pick Rate Trends Over Time")
        ax1.legend(loc="best"); ax1.grid(True, alpha=0.3)

        for hero_id in top_heroes:
            hero_data = df[df["hero_id"] == hero_id].sort_values("year")
            ax2.plot(hero_data["year"], hero_data["ban_rate"], marker="s", label=f"Hero {hero_id}", linewidth=2)

        ax2.set_xlabel("Year"); ax2.set_ylabel("Ban Rate (%)"); ax2.set_title("Hero Ban Rate Trends Over Time")
        ax2.legend(loc="best"); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{viz_path}/02_hero_trends.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Created: 02_hero_trends.png")

    def _plot_duration_winrate(self, viz_path):
        df = self.viz_data["duration_correlation"]
        order = ["0-20min", "20-30min", "30-40min", "40-50min", "50min+"]
        df["duration_bucket"] = pd.Categorical(df["duration_bucket"], categories=order, ordered=True)
        df = df.sort_values("duration_bucket")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        bars = ax1.bar(df["duration_bucket"], df["radiant_win_rate"], alpha=0.9)
        ax1.axhline(y=50, color="red", linestyle="--", label="50% (Balanced)")
        ax1.set_xlabel("Match Duration"); ax1.set_ylabel("Radiant Win Rate (%)")
        ax1.set_title("Radiant Win Rate by Match Duration"); ax1.legend(); ax1.grid(axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

        labels = df["duration_bucket"].astype(str).tolist()
        ax2.pie(df["total_matches"], labels=labels, autopct="%1.1f%%", startangle=90)
        ax2.set_title("Match Distribution by Duration")

        plt.tight_layout()
        plt.savefig(f"{viz_path}/05_duration_winrate.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Created: 05_duration_winrate.png")

    def _create_performance_dashboard(self, viz_path):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :])
        queries = list(self.performance_metrics.keys())
        times = [m["execution_time"] for m in self.performance_metrics.values()]
        bars = ax1.barh(queries, times)
        ax1.set_xlabel("Execution Time (seconds)")
        ax1.set_title("Query Execution Time Performance"); ax1.grid(axis="x", alpha=0.3)
        for b in bars:
            w = b.get_width()
            ax1.text(w, b.get_y() + b.get_height()/2., f"{w:.2f}s", ha="left", va="center", fontsize=9, fontweight="bold")

        ax2 = fig.add_subplot(gs[1, 0])
        total = np.sum(times) if times else 0.0
        avg_t = np.mean(times) if times else 0.0
        stats_text = f"""
        PERFORMANCE SUMMARY
        {'=' * 40}
        Total Execution Time:    {total:.2f}s
        Average Query Time:      {avg_t:.2f}s
        Total Queries:           {len(queries)}
        """
        ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes,
                 fontsize=11, verticalalignment="center", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[1, 1])
        if times:
            ax3.pie(times, labels=[q.split(':')[0] if ':' in q else q for q in queries],
                    autopct="%1.1f%%", startangle=90)
        ax3.set_title("Time Distribution by Query")

        ax4 = fig.add_subplot(gs[2, :])
        hero_time = np.sum([t for q, t in zip(queries, times) if "Hero" in q])
        duration_time = np.sum([t for q, t in zip(queries, times) if "Duration" in q])
        cats = ["Hero\nAnalysis", "Duration\nAnalysis"]
        vals = [hero_time, duration_time]
        bars2 = ax4.bar(cats, vals)
        ax4.set_ylabel("Total Execution Time (s)"); ax4.set_title("Execution Time by Analysis Category"); ax4.grid(axis="y", alpha=0.3)
        for b in bars2:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width()/2., h, f"{h:.2f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")

        plt.suptitle("DOTA2 Analytics Performance Dashboard", fontsize=16, fontweight="bold", y=0.98)
        plt.savefig(f"{viz_path}/12_performance_dashboard.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Created: 12_performance_dashboard.png")

    def collect_performance_metrics(self):
        """Collect and output performance metrics"""
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARKS")
        print("=" * 80)

        sc = self.spark.sparkContext
        print(f"\nSpark Configuration:")
        print(f"  App Name:              {sc.appName}")
        print(f"  Master:                {sc.master}")
        print(f"  Default Parallelism:   {sc.defaultParallelism}")

        print(f"\n{'Query Name':<50} {'Execution Time (s)':<20}")
        print("-" * 80)
        total_time = 0
        for query_name, metrics in self.performance_metrics.items():
            exec_time = metrics['execution_time']
            total_time += exec_time
            print(f"{query_name:<50} {exec_time:<20.2f}")
        print("-" * 80)
        print(f"{'TOTAL EXECUTION TIME':<50} {total_time:<20.2f}")

        if self.output_path.startswith("gs://"):
            metrics_file = "/tmp/performance_metrics.json"
        else:
            os.makedirs(self.output_path, exist_ok=True)
            metrics_file = f"{self.output_path}/performance_metrics.json"

        with open(metrics_file, 'w') as f:
            json.dump({
                'spark_config': {
                    'app_name': sc.appName,
                    'master': sc.master,
                    'default_parallelism': sc.defaultParallelism
                },
                'queries': self.performance_metrics,
                'summary': {
                    'total_execution_time': total_time,
                    'average_query_time': total_time / len(self.performance_metrics) if self.performance_metrics else 0.0,
                    'total_queries': len(self.performance_metrics)
                }
            }, f, indent=2)

        if self.output_path.startswith("gs://"):
            import subprocess
            subprocess.run(["gsutil", "cp", metrics_file, f"{self.output_path}/performance_metrics.json"])
            print(f"\nPerformance metrics uploaded to: {self.output_path}/performance_metrics.json")
        else:
            print(f"\nPerformance metrics saved to: {metrics_file}")

    def run_all_analytics(self):
        """Run Task 1 & 2, create visualizations, collect metrics"""
        print("\n" + "=" * 80)
        print("STARTING DOTA2 SCALABLE ANALYTICS (Task 1 & 2)")
        print("=" * 80)

        self.load_data()

        print("\n" + "=" * 80)
        print("TASK 1: HERO META ANALYSIS")
        print("=" * 80)
        hero_results = self.hero_meta_analysis()

        print("\n" + "=" * 80)
        print("TASK 2: MATCH DURATION ANALYSIS")
        print("=" * 80)
        duration_results = self.match_duration_analysis()

        self.create_visualizations()
        self.collect_performance_metrics()

        print("\n" + "=" * 80)
        print("ALL ANALYTICS (T1&T2) COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        return {'hero': hero_results, 'duration': duration_results}

    def save_results(self, results):
        """Save analysis results to text files"""
        print(f"\nSaving results to: {self.output_path}")
        
        if self.output_path.startswith("gs://"):
            local_output = "/tmp/results"
            os.makedirs(local_output, exist_ok=True)
        else:
            local_output = self.output_path
            os.makedirs(local_output, exist_ok=True)

        for category, result_dict in results.items():
            for name, df in result_dict.items():
                output_file = f"{local_output}/{category}_{name}.txt"
                df_pandas = df.toPandas()
                with open(output_file, 'w') as f:
                    f.write(f"={'=' * 80}\n")
                    f.write(f"{category.upper()} - {name.upper()}\n")
                    f.write(f"={'=' * 80}\n\n")
                    f.write(df_pandas.to_string(index=False))
                    f.write(f"\n\n")
                    f.write(f"Total Records: {len(df_pandas)}\n")
                print(f"  Saved: {category}_{name}.txt")

        if self.output_path.startswith("gs://"):
            import subprocess
            subprocess.run(["gsutil", "-m", "cp", f"{local_output}/*.txt", f"{self.output_path}/"])
            print(f"\nResults uploaded to: {self.output_path}")
        
        print("\nAll results saved successfully!")

    def stop(self):
        """Stop Spark session"""
        self.spark.stop()
        print("\nSpark session stopped")


def main():
    """
    Usage:
        spark-submit dota2_analytics_t12_csv.py s3://<bucket>/dota_merged/ s3://<bucket>/dota_results/
    Or local:
        python dota2_analytics_t12_csv.py /path/to/folder /tmp/dota2_results

    data_path folder must contain:
        main_metadata.csv, picks_bans.csv, players.csv
    """
    DATA_PATH = "/task2_clean_out"
    OUTPUT_PATH = "/dota2_analysis_results"

    import sys
    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_PATH = sys.argv[2]

    print(f"\nData Path:   {DATA_PATH}")
    print(f"Output Path: {OUTPUT_PATH}")

    try:
        analyzer = DOTA2Analytics(DATA_PATH, OUTPUT_PATH)
        results = analyzer.run_all_analytics()
        analyzer.save_results(results)
        analyzer.stop()

        print("\n" + "=" * 80)
        print("ANALYTICS PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nResults available at:        {OUTPUT_PATH}")
        print(f"Visualizations:              {OUTPUT_PATH}/visualizations/")
        print(f"Performance metrics:         {OUTPUT_PATH}/performance_metrics.json")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())