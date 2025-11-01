"""
DOTA2 Scalable Analytics with PySpark
Task 1: Complex Analytical Queries with Performance Benchmarks (DF)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col, lit, year, month
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import math


class DOTA2AnalyticsRDD:

    def __init__(self, data_path, output_path="/tmp/dota2_results"):
        self.data_path = data_path.rstrip("/")
        self.output_path = output_path.rstrip("/")
        self.performance_metrics = {}
        self.viz_data = {}  # small pandas objects for plotting

        self.spark = (
            SparkSession.builder
            .appName("DOTA2_Scalable_Analytics_T12_RDD_CSV")
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
        self.sc = self.spark.sparkContext
        self.sc.setLogLevel("WARN")

        # Matplotlib basic setup (no seaborn)
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

        print("=" * 80)
        print("DOTA2 Scalable Analytics (RDD, Task 1 & 2, CSV mode) Initialized")
        print("=" * 80)

    # ---------------- helpers ----------------
    def _timeit(self, name, fn):
        """Run fn() and record execution time."""
        print("\n" + "=" * 80)
        print(f"Executing: {name}")
        print("=" * 80)
        t0 = time.time()
        out = fn()
        dt = time.time() - t0
        self.performance_metrics[name] = {
            "execution_time": round(dt, 2),
            "timestamp": datetime.now().isoformat()
        }
        print(f"\nCompleted in {dt:.2f} seconds")
        return out

    @staticmethod
    def _safe_bool(v):
        """Cast many truthy/falsy forms to Python bool/None."""
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("1", "true", "t", "yes", "y"):
            return True
        if s in ("0", "false", "f", "no", "n"):
            return False
        return None

    # ---------------- data load ----------------
    def load_data(self):
        """
        Load three merged CSVs with Spark reader (robust CSV parsing),
        then convert to RDDs and perform ALL computations with RDD API.
        Expected files in data_path:
          - main_metadata.csv
          - picks_bans.csv
          - players.csv
        """
        print("\n[1/3] Loading merged CSVs from folder ...")

        def _csv(file_name):
            path = f"{self.data_path}/{file_name}"
            return (
                self.spark.read
                .option("header", "true")
                .option("multiLine", "false")
                .csv(path)
            )

        # ----- main_metadata -----
        meta_df = _csv("main_metadata.csv")

        # pick a time column (if any) to extract year/month
        tcol = None
        for c in ["start_date_time", "start_time", "start_time_utc", "match_start_time"]:
            if c in meta_df.columns:
                tcol = c
                break

        if tcol:
            meta_df = meta_df.withColumn("parsed_time", to_timestamp(col(tcol)))
            meta_df = meta_df.withColumn("year", year(col("parsed_time")))
            meta_df = meta_df.withColumn("month", month(col("parsed_time")))
        else:
            meta_df = meta_df.withColumn("year", lit(None).cast("int")).withColumn("month", lit(None).cast("int"))

        meta_df = (
            meta_df
            .withColumn("match_id", col("match_id").cast("long"))
            .withColumn("duration", col("duration").cast("int"))
            .withColumn("radiant_win_raw", col("radiant_win"))
            .withColumn("radiant_score", col("radiant_score").cast("int"))
            .withColumn("dire_score", col("dire_score").cast("int"))
        )

        # Convert to dict-RDD for free-form mapping
        meta_rdd_dict = meta_df.rdd.map(lambda r: {k: r[k] for k in r.__fields__})

        def map_meta(d):
            """Return typed tuple for main metadata."""
            def to_int(x):
                try: return int(x)
                except: return None
            def to_long(x):
                try: return int(x)
                except: return None

            match_id = to_long(d.get("match_id"))
            duration = to_int(d.get("duration"))
            radiant_win = DOTA2AnalyticsRDD._safe_bool(d.get("radiant_win_raw"))
            radiant_score = to_int(d.get("radiant_score"))
            dire_score = to_int(d.get("dire_score"))
            year_v = to_int(d.get("year"))
            month_v = to_int(d.get("month"))
            return (
                match_id,      # 0
                duration,      # 1
                radiant_win,   # 2
                radiant_score, # 3
                dire_score,    # 4
                year_v,        # 5
                month_v        # 6
            )

        self.meta_rdd = meta_rdd_dict.map(map_meta).filter(lambda x: x[0] is not None)
        print(f"Loaded main_metadata.csv -> RDD count: {self.meta_rdd.count():,}")

        # ----- picks_bans -----
        pb_df = _csv("picks_bans.csv")
        pb_rdd_dict = pb_df.rdd.map(lambda r: {k: r[k] for k in r.__fields__})

        def map_pb(d):
            """Return typed tuple for picks/bans."""
            def to_int(x):
                try: return int(x)
                except: return None
            match_id = None
            try:
                match_id = int(d.get("match_id"))
            except:
                pass
            hero_id = to_int(d.get("hero_id"))
            team = to_int(d.get("team"))
            is_pick = 1 if str(d.get("is_pick")).strip().lower() in ("1", "true", "t", "yes", "y") else 0
            # treat missing explicit is_ban; in original DF version, you used (1 - is_pick)
            is_ban = 1 - is_pick
            return (
                match_id,  # 0
                hero_id,   # 1
                team,      # 2
                is_pick,   # 3
                is_ban     # 4
            )

        self.pb_rdd = pb_rdd_dict.map(map_pb).filter(lambda x: x[0] is not None and x[1] is not None)
        print(f"Loaded picks_bans.csv -> RDD count: {self.pb_rdd.count():,}")

        # ----- players (kept for parity; not used for output figs) -----
        pl_df = _csv("players.csv")
        self.players_rdd = pl_df.rdd  # not used in plots; kept for compatibility
        print(f"Loaded players.csv -> RDD count: {self.players_rdd.count():,}")

        # Build KV for joining picks/bans with meta by match_id
        meta_kv = self.meta_rdd.map(lambda x: (x[0], (x[2], x[1], x[5], x[6])))  # match_id -> (radiant_win, duration, year, month)
        pb_kv = self.pb_rdd.map(lambda x: (x[0], (x[1], x[2], x[3], x[4])))      # match_id -> (hero_id, team, is_pick, is_ban)

        joined = pb_kv.join(meta_kv)
        # => (match_id, ((hero_id, team, is_pick, is_ban), (radiant_win, duration, year, month)))

        # Flatten enriched PB rows
        # (year, hero_id, is_pick, is_ban, team, radiant_win, duration)
        self.pb_enriched_rdd = joined.map(
            lambda kv: (
                kv[1][1][2],   # year
                kv[1][0][0],   # hero_id
                kv[1][0][2],   # is_pick
                kv[1][0][3],   # is_ban
                kv[1][0][1],   # team
                kv[1][1][0],   # radiant_win
                kv[1][1][1],   # duration
            )
        ).filter(lambda x: x[0] is not None)

        # Compact meta for Task 2
        # (duration, radiant_win, year, month)
        self.meta_compact_rdd = self.meta_rdd.map(lambda x: (x[1], x[2], x[5], x[6]))

        print("\nAll CSVs loaded & converted to RDDs.\n")

    # ---------------- Task 1 ----------------
    def hero_meta_analysis(self):
        results = {}

        def trends_over_time():
            # Aggregate per (year, hero): picks, bans, total appearances
            # Key: (year, hero_id) ; Val: (picks, bans, total)
            by_yh = self.pb_enriched_rdd.map(
                lambda t: ((t[0], t[1]), (t[2], t[3], 1))
            ).reduceByKey(
                lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])
            )

            # Compute pick_rate and ban_rate
            # Row: (year, hero_id, picks, bans, total, pick_rate, ban_rate)
            trends_rdd = by_yh.map(lambda kv: (
                kv[0][0],
                kv[0][1],
                kv[1][0],
                kv[1][1],
                kv[1][2],
                round(100.0 * kv[1][0] / kv[1][2], 2) if kv[1][2] else 0.0,
                round(100.0 * kv[1][1] / kv[1][2], 2) if kv[1][2] else 0.0
            ))

            # Aggregate per hero to compute avg pick rate & stddev of pick rate across years
            # (hero_id) -> (cnt, sum_pr, sumsq_pr, sum_br, cnt_br)
            hero_agg = trends_rdd.map(
                lambda r: (r[1], (1, r[5], r[5] ** 2, r[6], 1))
            ).reduceByKey(
                lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4])
            ).map(
                lambda kv: (
                    kv[0],
                    (kv[1][1] / kv[1][0] if kv[1][0] else 0.0),  # avg_pick_rate
                    math.sqrt(max(kv[1][2] / kv[1][0] - (kv[1][1] / kv[1][0]) ** 2, 0.0)) if kv[1][0] else 0.0,  # stddev
                    (kv[1][3] / kv[1][4] if kv[1][4] else 0.0)   # avg_ban_rate
                )
            )

            # Choose top-10 heroes by pick_rate volatility (stddev)
            # Collect small lists for plotting
            top_trending = hero_agg.takeOrdered(10, key=lambda x: -x[1])  # sort by avg_pick_rate desc as a stable proxy
            top_trending_pd = pd.DataFrame(
                top_trending,
                columns=["hero_id", "avg_pick_rate", "pick_rate_volatility", "avg_ban_rate"]
            )

            trends_pd = pd.DataFrame(
                trends_rdd.collect(),
                columns=["year", "hero_id", "picks", "bans", "total", "pick_rate", "ban_rate"]
            )

            # Store for visualization
            self.viz_data["hero_trends"] = trends_pd
            self.viz_data["top_trending_heroes"] = top_trending_pd

            return trends_rdd

        def hero_by_duration():
            # Compute hero win by duration bucket (no figure, but kept for text output)
            def phase(dur):
                if dur is None:
                    return None
                if dur < 1800: return "Early (<30min)"
                if dur < 2700: return "Mid (30-45min)"
                return "Late (>45min)"

            # Only picks; compute win by side
            # Key: (hero_id, phase) ; Val: (games, wins)
            hero_phase = self.pb_enriched_rdd.filter(lambda t: t[2] == 1).map(
                lambda t: (
                    (t[1], phase(t[6])),
                    (1, 1 if ((t[4] == 0 and t[5] is True) or (t[4] == 1 and t[5] is False)) else 0)
                )
            ).filter(lambda kv: kv[0][1] is not None).reduceByKey(
                lambda a, b: (a[0] + b[0], a[1] + b[1])
            ).map(
                lambda kv: (kv[0][0], kv[0][1], kv[1][0], kv[1][1],
                            round(100.0 * kv[1][1] / kv[1][0], 2) if kv[1][0] else 0.0)
            )

            # keep a pandas copy for potential debugging
            self.viz_data["hero_phase_stats"] = pd.DataFrame(
                hero_phase.collect(),
                columns=["hero_id", "game_phase", "games", "wins", "win_rate"]
            )
            return hero_phase

        results = {}
        results['trends'] = self._timeit("Hero Trends Over Time (RDD)", trends_over_time)
        results['by_duration'] = self._timeit("Hero Performance by Duration (RDD)", hero_by_duration)
        return results

    # ---------------- Task 2 ----------------
    def match_duration_analysis(self):
        results = {}

        def duration_trends():
            # Aggregate by (year, month):
            # Keep accumulators: (count, sum, sumsq, min, max)
            def agg_init():
                return (0, 0.0, 0.0, float("inf"), float("-inf"))

            def agg_add(acc, d):
                c, s, s2, mn, mx = acc
                if d is None:
                    return acc
                c += 1
                s += d
                s2 += d * d
                mn = d if d < mn else mn
                mx = d if d > mx else mx
                return (c, s, s2, mn, mx)

            def agg_merge(a, b):
                return (a[0] + b[0], a[1] + b[1], a[2] + b[2], min(a[3], b[3]), max(a[4], b[4]))

            keyed = self.meta_compact_rdd.map(lambda t: ((t[2], t[3]), t[0]))  # ((year, month), duration)
            agg = keyed.aggregateByKey(agg_init(), agg_add, agg_merge)

            # Finalize stats: (year, month, avg_sec, count, min, max, std_dev)
            out = agg.map(lambda kv: (
                kv[0][0], kv[0][1],
                (kv[1][1] / kv[1][0]) if kv[1][0] else None,
                kv[1][0],
                None if kv[1][0] == 0 else int(kv[1][3]),
                None if kv[1][0] == 0 else int(kv[1][4]),
                math.sqrt(max((kv[1][2] / kv[1][0]) - (kv[1][1] / kv[1][0]) ** 2, 0.0)) if kv[1][0] else None
            )).filter(lambda r: r[0] is not None and r[1] is not None)

            df = pd.DataFrame(
                out.collect(),
                columns=["year", "month", "avg_duration_sec", "total_matches", "shortest", "longest", "std_dev"]
            )
            if not df.empty:
                df["avg_duration_min"] = (df["avg_duration_sec"] / 60.0).round(2)
                df = df.sort_values(["year", "month"])
            self.viz_data["duration_trends"] = df
            return out

        def duration_winrate_correlation():
            # Bucket duration and compute radiant win rate per bucket
            def bucket(d):
                if d is None: return None
                if d < 1200: return "0-20min"
                if d < 1800: return "20-30min"
                if d < 2400: return "30-40min"
                if d < 3000: return "40-50min"
                return "50min+"

            # Key: bucket ; Val: (total, wins)
            by_bucket = self.meta_compact_rdd.map(
                lambda t: (bucket(t[0]), (1, 1 if t[1] is True else 0))
            ).filter(lambda kv: kv[0] is not None).reduceByKey(
                lambda a, b: (a[0] + b[0], a[1] + b[1])
            ).map(
                lambda kv: (kv[0], kv[1][0], kv[1][1],
                            round(100.0 * kv[1][1] / kv[1][0], 2) if kv[1][0] else 0.0)
            )
            df = pd.DataFrame(by_bucket.collect(),
                              columns=["duration_bucket", "total_matches", "radiant_wins", "radiant_win_rate"])
            self.viz_data["duration_correlation"] = df
            return by_bucket

        results['trends'] = self._timeit("Duration Trends (RDD)", duration_trends)
        results['correlation'] = self._timeit("Duration-WinRate Correlation (RDD)", duration_winrate_correlation)
        return results

    # ---------------- Visualization (02, 05, 12) ----------------
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

        # 02: hero trends (pick/ban rates for top-trending heroes)
        if "hero_trends" in self.viz_data and "top_trending_heroes" in self.viz_data:
            self._plot_hero_trends(viz_path)

        # 05: duration vs winrate
        if "duration_correlation" in self.viz_data:
            self._plot_duration_winrate(viz_path)

        # 12: performance dashboard
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

        # Pick-rate lines
        for hero_id in top_heroes:
            hd = df[df["hero_id"] == hero_id].sort_values("year")
            ax1.plot(hd["year"], hd["pick_rate"], marker="o", label=f"Hero {hero_id}", linewidth=2)
        ax1.set_xlabel("Year"); ax1.set_ylabel("Pick Rate (%)"); ax1.set_title("Hero Pick Rate Trends Over Time")
        ax1.legend(loc="best"); ax1.grid(True, alpha=0.3)

        # Ban-rate lines
        for hero_id in top_heroes:
            hd = df[df["hero_id"] == hero_id].sort_values("year")
            ax2.plot(hd["year"], hd["ban_rate"], marker="s", label=f"Hero {hero_id}", linewidth=2)
        ax2.set_xlabel("Year"); ax2.set_ylabel("Ban Rate (%)"); ax2.set_title("Hero Ban Rate Trends Over Time")
        ax2.legend(loc="best"); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{viz_path}/02_hero_trends.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Created: 02_hero_trends.png")

    def _plot_duration_winrate(self, viz_path):
        df = self.viz_data["duration_correlation"]
        order = ["0-20min", "20-30min", "30-40min", "40-50min", "50min+"]
        if not df.empty:
            df["duration_bucket"] = pd.Categorical(df["duration_bucket"], categories=order, ordered=True)
            df = df.sort_values("duration_bucket")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Bar chart
        bars = ax1.bar(df["duration_bucket"].astype(str), df["radiant_win_rate"], alpha=0.9)
        ax1.axhline(y=50, color="red", linestyle="--", label="50% (Balanced)")
        ax1.set_xlabel("Match Duration"); ax1.set_ylabel("Radiant Win Rate (%)")
        ax1.set_title("Radiant Win Rate by Match Duration"); ax1.legend(); ax1.grid(axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")

        # Pie chart
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

        # Execution time bars
        ax1 = fig.add_subplot(gs[0, :])
        queries = list(self.performance_metrics.keys())
        times = [m["execution_time"] for m in self.performance_metrics.values()]
        bars = ax1.barh(queries, times)
        ax1.set_xlabel("Execution Time (seconds)")
        ax1.set_title("Query Execution Time Performance"); ax1.grid(axis="x", alpha=0.3)
        for b in bars:
            w = b.get_width()
            ax1.text(w, b.get_y() + b.get_height()/2., f"{w:.2f}s", ha="left", va="center", fontsize=9, fontweight="bold")

        # Text summary
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

        # Time distribution pie
        ax3 = fig.add_subplot(gs[1, 1])
        if times:
            ax3.pie(times, labels=[q.split(':')[0] if ':' in q else q for q in queries],
                    autopct="%1.1f%%", startangle=90)
        ax3.set_title("Time Distribution by Query")

        # Category split (Hero vs Duration)
        ax4 = fig.add_subplot(gs[2, :])
        hero_time = np.sum([t for q, t in zip(queries, times) if "Hero" in q])
        duration_time = np.sum([t for q, t in zip(queries, times) if "Duration" in q])
        cats, vals = ["Hero\nAnalysis", "Duration\nAnalysis"], [hero_time, duration_time]
        bars2 = ax4.bar(cats, vals)
        ax4.set_ylabel("Total Execution Time (s)")
        ax4.set_title("Execution Time by Analysis Category"); ax4.grid(axis="y", alpha=0.3)
        for b in bars2:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width()/2., h, f"{h:.2f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")

        plt.suptitle("DOTA2 Analytics Performance Dashboard", fontsize=16, fontweight="bold", y=0.98)
        plt.savefig(f"{viz_path}/12_performance_dashboard.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  Created: 12_performance_dashboard.png")

    # ---------------- saving & orchestration ----------------
    def save_results(self, results):
        """
        Save RDD results as text files (collect -> pandas -> write .txt),
        preserving your previous output style.
        """
        print(f"\nSaving results to: {self.output_path}")

        if self.output_path.startswith("gs://"):
            local_output = "/tmp/results"
            os.makedirs(local_output, exist_ok=True)
        else:
            local_output = self.output_path
            os.makedirs(local_output, exist_ok=True)

        for category, result_dict in results.items():
            for name, rdd in result_dict.items():
                # Collect RDD to pandas for pretty text output
                rows = rdd.collect()
                if len(rows) == 0:
                    df_pd = pd.DataFrame()
                else:
                    # Build lightweight column inference by category/name
                    if category == "hero" and name == "trends":
                        cols = ["year", "hero_id", "picks", "bans", "total_appearances", "pick_rate", "ban_rate"]
                    elif category == "hero" and name == "by_duration":
                        cols = ["hero_id", "game_phase", "games", "wins", "win_rate"]
                    elif category == "duration" and name == "trends":
                        cols = ["year", "month", "avg_duration_sec", "total_matches", "shortest", "longest", "std_dev"]
                    elif category == "duration" and name == "correlation":
                        cols = ["duration_bucket", "total_matches", "radiant_wins", "radiant_win_rate"]
                    else:
                        # fallback: enumerate columns
                        cols = [f"c{i}" for i in range(len(rows[0]))]
                    df_pd = pd.DataFrame(rows, columns=cols)

                out_path = f"{local_output}/{category}_{name}.txt"
                with open(out_path, "w") as f:
                    f.write(f"={'=' * 80}\n")
                    f.write(f"{category.upper()} - {name.upper()}\n")
                    f.write(f"={'=' * 80}\n\n")
                    if df_pd.empty:
                        f.write("(no records)\n")
                    else:
                        f.write(df_pd.to_string(index=False))
                        f.write("\n\n")
                        f.write(f"Total Records: {len(df_pd)}\n")

                print(f"  Saved: {category}_{name}.txt")

        if self.output_path.startswith("gs://"):
            import subprocess
            subprocess.run(["gsutil", "-m", "cp", f"{local_output}/*.txt", f"{self.output_path}/"])
            print(f"\nResults uploaded to: {self.output_path}")

        print("\nAll results saved successfully!")

    def collect_performance_metrics(self):
        """Print and persist execution time metrics as JSON."""
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARKS")
        print("=" * 80)

        sc = self.sc
        print("\nSpark Configuration:")
        print(f"  App Name:              {sc.appName}")
        print(f"  Master:                {sc.master}")
        print(f"  Default Parallelism:   {sc.defaultParallelism}")

        print(f"\n{'Query Name':<50} {'Execution Time (s)':<20}")
        print("-" * 80)
        total = 0.0
        for q, m in self.performance_metrics.items():
            t = m["execution_time"]
            total += t
            print(f"{q:<50} {t:<20.2f}")
        print("-" * 80)
        print(f"{'TOTAL EXECUTION TIME':<50} {total:<20.2f}")

        if self.output_path.startswith("gs://"):
            metrics_file = "/tmp/performance_metrics.json"
        else:
            os.makedirs(self.output_path, exist_ok=True)
            metrics_file = f"{self.output_path}/performance_metrics.json"

        with open(metrics_file, "w") as f:
            json.dump({
                "spark_config": {
                    "app_name": sc.appName,
                    "master": sc.master,
                    "default_parallelism": sc.defaultParallelism
                },
                "queries": self.performance_metrics,
                "summary": {
                    "total_execution_time": total,
                    "average_query_time": total / len(self.performance_metrics) if self.performance_metrics else 0.0,
                    "total_queries": len(self.performance_metrics)
                }
            }, f, indent=2)

        if self.output_path.startswith("gs://"):
            import subprocess
            subprocess.run(["gsutil", "cp", metrics_file, f"{self.output_path}/performance_metrics.json"])
            print(f"\nPerformance metrics uploaded to: {self.output_path}/performance_metrics.json")
        else:
            print(f"\nPerformance metrics saved to: {metrics_file}")

    def run_all_analytics(self):
        """Run Task 1 & 2 with RDDs, create plots, collect metrics."""
        print("\n" + "=" * 80)
        print("STARTING DOTA2 SCALABLE ANALYTICS (RDD, Task 1 & 2)")
        print("=" * 80)

        self.load_data()

        print("\n" + "=" * 80)
        print("TASK 1: HERO META ANALYSIS (RDD)")
        print("=" * 80)
        hero_results = self.hero_meta_analysis()

        print("\n" + "=" * 80)
        print("TASK 2: MATCH DURATION ANALYSIS (RDD)")
        print("=" * 80)
        duration_results = self.match_duration_analysis()

        self.create_visualizations()
        self.collect_performance_metrics()

        print("\n" + "=" * 80)
        print("ALL ANALYTICS (RDD T1&T2) COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        return {'hero': hero_results, 'duration': duration_results}

    def stop(self):
        """Stop Spark session."""
        self.spark.stop()
        print("\nSpark session stopped")


# ---------------- main ----------------
def main():
    DATA_PATH = "/task2_clean_out"
    OUTPUT_PATH = "/dota2_analysis_results"

    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_PATH = sys.argv[2]

    print(f"\nData Path:   {DATA_PATH}")
    print(f"Output Path: {OUTPUT_PATH}")

    try:
        job = DOTA2AnalyticsRDD(DATA_PATH, OUTPUT_PATH)
        results = job.run_all_analytics()
        job.save_results(results)
        job.stop()

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
    sys.exit(main())