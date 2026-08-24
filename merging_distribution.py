#!/usr/bin/env python3

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from collections import Counter
from scipy import stats
import pandas as pd


def get_initial_particles_for_final(conn, final_pid):
    """
    Extract original initial particles in the ancestor tree of one final particle
    and compute their merge distance to the final particle.
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
        ancestors.pid AS pid,
        MAX(ancestors.depth) AS branch_size
    FROM ancestors
    JOIN particles p ON p.pid = ancestors.pid
    WHERE p.is_initial = 1
    GROUP BY ancestors.pid
    ORDER BY branch_size;
    """

    df = pd.read_sql_query(query, conn, params=(int(final_pid),))

    particles = []

    for _, row in df.iterrows():
        particles.append({
            "pid": int(row["pid"]),
            "branch_size": int(row["branch_size"])
        })

    return particles


def load_all_sql_lineage_files(pattern):
    """
    Load all lineage DB files matching the glob pattern.
    """

    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    all_branch_sizes = []
    all_particle_info = []

    for file in files:

        try:
            conn = sqlite3.connect(file)

            final_particles = pd.read_sql_query("""
                SELECT pid
                FROM final_particles
            """, conn)

            for _, fp in final_particles.iterrows():
                final_pid = int(fp["pid"])

                particles_in_tree = get_initial_particles_for_final(conn, final_pid)

                for particle in particles_in_tree:
                    all_branch_sizes.append(particle["branch_size"])

                    all_particle_info.append({
                        "file": os.path.basename(file),
                        "final_pid": final_pid,
                        "pid": particle["pid"],
                        "branch_size": particle["branch_size"]
                    })

            conn.close()

        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")

    return np.array(all_branch_sizes), all_particle_info, files



def fit_normal_and_lognormal(data):
    """
    Fit normal and log-normal distributions
    """

    data = np.array(data)
    data = data[data > 0]  # log-normal requires positive values

    if len(data) == 0:
        print("No positive data for fitting.")
        return None

    # Fit normal
    mu_norm, std_norm = stats.norm.fit(data)

    # Fit log-normal
    shape_ln, loc_ln, scale_ln = stats.lognorm.fit(data, floc=0)

    print(f"Normal fit:     mu = {mu_norm:.4f}, std = {std_norm:.4f}")
    print(f"Log-normal fit: shape = {shape_ln:.4f}, loc = {loc_ln:.4f}, scale = {scale_ln:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.arange(np.min(data) - 0.5, np.max(data) + 1.5, 1)

    ax.hist(
        data,
        bins=bins,
        density=True,
        alpha=0.6,
        color="steelblue",
        edgecolor="black",
        label="Data"
    )

    x = np.linspace(np.min(data), np.max(data), 500)

    normal_pdf = stats.norm.pdf(x, mu_norm, std_norm)
    ax.plot(x, normal_pdf, "r-", lw=2.5, label="Normal fit")

    lognorm_pdf = stats.lognorm.pdf(x, shape_ln, loc_ln, scale_ln)
    ax.plot(x, lognorm_pdf, "g-", lw=2.5, label="Log-normal fit")

    ax.set_xlabel("Merge Distance to Final Particle", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "normal": (mu_norm, std_norm),
        "lognormal": (shape_ln, loc_ln, scale_ln)
    }


def save_branch_info_csv(particle_info, output_file):                                                       

    df = pd.DataFrame(particle_info)
    df.to_csv(output_file, index=False)


##main

def main():
    pattern = "full_sim_*_lineage_R09_N4000.db"

    try:
        branch_sizes, particle_info, files = load_all_sql_lineage_files(pattern)

        for item in particle_info[:30]:
            print(
                f"  File: {item['file']}, "
                f"Final PID: {item['final_pid']}, "
                f"Initial PID: {item['pid']}, "
                f"Merge Distance: {item['branch_size']}"
            )

        # save extracted data
        save_branch_info_csv(
            particle_info,
            output_file="sql_initial_particle_branch_sizes.csv"
        )

        fit_normal_and_lognormal(branch_sizes)

    except FileNotFoundError as e:
        print(f"{e}")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()