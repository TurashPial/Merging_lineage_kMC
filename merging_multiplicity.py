#!/usr/bin/env python3
"""
Plot: log(V) = log(V0) + M*log(X) and see how close it to pure equal volume addition of multiplicative factor=2
For both max and min branch size.
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
from scipy.odr import ODR, Model, RealData
from scipy import stats



## Helper functions
R0=9
def volume_from_radius(r):
    return (4.0 / 3.0) * np.pi * (r ** 3)


def get_max_min_branch_depth_sql(conn, final_pid):
    """
    Compute maximum and minimum branch depth for one final particle using recursive SQL.
    """

    query = """
    WITH RECURSIVE ancestors(pid, depth) AS (
        SELECT parent_pid, 1
        FROM particle_relations
        WHERE child_pid = ?

        UNION ALL

        SELECT pr.parent_pid, ancestors.depth + 1
        FROM particle_relations pr
        JOIN ancestors ON pr.child_pid = ancestors.pid
    )
    SELECT
        MIN(depth) AS min_edge_depth,
        MAX(depth) AS max_edge_depth
    FROM ancestors
    JOIN particles p ON p.pid = ancestors.pid
    WHERE p.is_initial = 1;
    """

    df = pd.read_sql_query(query, conn, params=(int(final_pid),))

    if df.empty:
        return 1, 1

    min_edge_depth = df["min_edge_depth"].iloc[0]
    max_edge_depth = df["max_edge_depth"].iloc[0]

    # If final particle has no parents, it is itself initial
    if pd.isna(min_edge_depth) or pd.isna(max_edge_depth):
        return 1, 1

    # Convert edge depth to node-count branch depth
    min_branch_size = int(min_edge_depth) + 1
    max_branch_size = int(max_edge_depth) + 1

    return max_branch_size, min_branch_size


def extract_final_particle_records_sql(db_file):
    """
    Extract final particle records from one database.
    """

    records = []

    conn = sqlite3.connect(db_file)

    final_particles = pd.read_sql_query("""
        SELECT pid, final_r
        FROM final_particles
    """, conn)

    for _, row in final_particles.iterrows():
        final_pid = int(row["pid"])
        r_float = float(row["final_r"])

        V = volume_from_radius(r_float)

        max_branch_size, min_branch_size = get_max_min_branch_depth_sql(
            conn,
            final_pid
        )

        records.append({
            "file": os.path.basename(db_file),
            "pid": final_pid,
            "max_branch_size": max_branch_size,
            "min_branch_size": min_branch_size,
            "volume": V,
            "radius": r_float
        })

    conn.close()

    return records


def load_all_sql_lineage_files(pattern):
    """
    Load all lineage DB files matching the glob pattern.
    """

    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    all_records = []

    for file in files:
        try:
            print(f"Loading {os.path.basename(file)}...")
            recs = extract_final_particle_records_sql(file)
            all_records.extend(recs)

        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")

    return all_records, files



def odr_linear_model(beta, x):
    """
    Linear model for ODR:
        y = a + b*x
    beta = [a, b]
    """
    return beta[0] + beta[1] * x


def estimate_uncertainty_from_residuals(x, y):
    
    # Fit OLS for initial guess
    slope_ols, intercept_ols, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Compute residuals
    y_pred_ols = intercept_ols + slope_ols * x
    residuals = y - y_pred_ols
    
    # Estimate noise std from residuals using Median Absolute Deviation (MAD)
    median_residual = np.median(residuals)
    mad = np.median(np.abs(residuals - median_residual))
    
    # Convert MAD to standard deviation (for normal distribution)
    sigma_residual = 1.4826 * mad
    sx = np.abs(x) * 0.05
    
    # Uncertainty in y: use estimated residual noise
    sy = np.ones_like(y) * sigma_residual
    
    sx = np.maximum(sx, 1e-12)
    sy = np.maximum(sy, 1e-12)
    
    beta0 = [intercept_ols, slope_ols]
    
    return sx, sy, sigma_residual, beta0, r_value**2


def fit_odr_residual_uncertainty(x, y, beta0_override=None):
    
    # Estimate uncertainties from residuals
    sx, sy, sigma_residual, beta0_ols, r_squared_ols = estimate_uncertainty_from_residuals(x, y)
    
    # Use provided beta0 if given, otherwise use OLS estimate
    beta0 = beta0_override if beta0_override is not None else beta0_ols
    
    # Create ODR model and fit
    model = Model(odr_linear_model)
    data = RealData(x, y, sx=sx, sy=sy)
    odr = ODR(data, model, beta0=beta0)
    output = odr.run()
    
    return output, sigma_residual, r_squared_ols


def plot_log_volume(records):

    max_branch = np.array(
        [r["max_branch_size"] for r in records],
        dtype=float
    )

    min_branch = np.array(
        [r["min_branch_size"] for r in records],
        dtype=float
    )

    V = np.array(
        [r["volume"] for r in records],
        dtype=float
    )

    mask = (
        np.isfinite(max_branch)
        & np.isfinite(min_branch)
        & np.isfinite(V)
        & (V > 0)
        & (max_branch > 0)
        & (min_branch > 0)
    )

    max_branch = max_branch[mask]
    min_branch = min_branch[mask]
    V = V[mask]

    V0 = volume_from_radius(R0)
    logV = np.log(V)
    logV0 = np.log(V0)

    output_max, sigma_max, r2_ols_max = fit_odr_residual_uncertainty(
        max_branch, 
        logV
    )

    a_max, b_max = output_max.beta
    X_max = np.exp(b_max)
    sd_a_max, sd_b_max = output_max.sd_beta

    print(f"  log(V0) = {a_max:.6f} ± {sd_a_max:.6f}")
    print(f"  log(X)  = {b_max:.6f} ± {sd_b_max:.6f}")
    print(f"  X       = {X_max:.6f}")
    print(f"  Growth factor per step: {X_max:.4f}")

    output_min, sigma_min, r2_ols_min = fit_odr_residual_uncertainty(
        min_branch, 
        logV
    )

    a_min, b_min = output_min.beta
    X_min = np.exp(b_min)
    sd_a_min, sd_b_min = output_min.sd_beta


    print(f"  log(V0) = {a_min:.6f} ± {sd_a_min:.6f}")
    print(f"  log(X)  = {b_min:.6f} ± {sd_b_min:.6f}")
    print(f"  X       = {X_min:.6f}")
    print(f"  Growth factor per step: {X_min:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]

    ax.scatter(
        max_branch,
        logV,
        alpha=0.6,
        s=30,
        color="steelblue",
        label="Data"
    )

    x_max = np.linspace(max_branch.min(), max_branch.max(), 200)
    y_max = a_max + b_max * x_max

    ax.plot(
        x_max,
        y_max,
        "r-",
        lw=2.5,
        label=(
            f"ODR fit (residual uncertainty):\n"
            f"log(V) = {a_max:.3f} + {b_max:.4f}M\n"
            f"X = {X_max:.4f} (σ={sigma_max:.4f})"
        )
    )

    ax.set_xlabel("Maximum Branch Size M", fontsize=12)
    ax.set_ylabel("log(V)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    ax = axes[1]

    ax.scatter(
        min_branch,
        logV,
        alpha=0.6,
        s=30,
        color="darkorange",
        label="Data"
    )

    x_min = np.linspace(min_branch.min(), min_branch.max(), 200)
    y_min = a_min + b_min * x_min

    ax.plot(
        x_min,
        y_min,
        "r-",
        lw=2.5,
        label=(
            f"ODR fit (residual uncertainty):\n"
            f"log(V) = {a_min:.3f} + {b_min:.4f}M\n"
            f"X = {X_min:.4f} (σ={sigma_min:.4f})"
        )
    )

    ax.set_xlabel("Minimum Branch Size M", fontsize=12)
    ax.set_ylabel("log(V)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

def save_records_csv(records, output_file="sql_branch_volume_records.csv"):

    if len(records) == 0:
        print("No records to save.")
        return

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)

#main

def main():
    pattern = "full_sim_*_lineage_R09_N4000.db"
    try:
        records, files = load_all_sql_lineage_files(pattern)

        save_records_csv(records)

        plot_log_volume(records)


    except FileNotFoundError as e:
        print(f" {e}")

    except Exception as e:
        print(f" Error: {e}")
        raise


if __name__ == "__main__":
    main()