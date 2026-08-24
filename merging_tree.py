#!/usr/bin/env python3

import sqlite3
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load kmc lineage data

def load_all_sql_data(db_pattern):
    """
    Load and combine data from all SQLite DB files matching db_pattern.
    """

    db_files = sorted(glob.glob(db_pattern))

    if len(db_files) == 0:
        raise FileNotFoundError(f"No database files found matching pattern: {db_pattern}")

    final_particles_list = []
    particles_list = []
    relations_list = []
    simulations_list = []

    for db_file in db_files:
        print(f"Loading: {db_file}")

        conn = sqlite3.connect(db_file)

        final_particles = pd.read_sql_query("""
            SELECT *
            FROM final_particles
        """, conn)

        particles = pd.read_sql_query("""
            SELECT *
            FROM particles
        """, conn)

        relations = pd.read_sql_query("""
            SELECT *
            FROM particle_relations
        """, conn)

        simulations = pd.read_sql_query("""
            SELECT *
            FROM simulations
        """, conn)

        conn.close()

        base = os.path.basename(db_file)

        final_particles["db_file"] = base
        particles["db_file"] = base
        relations["db_file"] = base
        simulations["db_file"] = base

        final_particles_list.append(final_particles)
        particles_list.append(particles)
        relations_list.append(relations)
        simulations_list.append(simulations)

    sql_data = {
        "db_files": db_files,
        "final_particles": pd.concat(final_particles_list, ignore_index=True),
        "particles": pd.concat(particles_list, ignore_index=True),
        "relations": pd.concat(relations_list, ignore_index=True),
        "simulations": pd.concat(simulations_list, ignore_index=True)
    }

    return sql_data


##analyze and plot

def plot_size_vs_mergers_all(sql_data, ax=None):
    """
    Final particle radius vs number of mergers across all DB files.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    final_particles = sql_data["final_particles"]

    if final_particles.empty:
        ax.text(
            0.5, 0.5,
            "No final particle data",
            ha="center",
            va="center",
            transform=ax.transAxes
        )
        return ax

    merge_counts = final_particles["merge_count"].values
    final_radii = final_particles["final_r"].values

    ax.scatter(
        merge_counts,
        final_radii,
        s=60,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        c="steelblue"
    )

    ax.set_xlabel("Number of Mergers", fontsize=12)
    ax.set_ylabel("Final Radius (nm)", fontsize=12)
    ax.set_title(
        f"Final Particle Size vs Merger Count\n"
        f"{len(final_particles)} final particles from {len(sql_data['db_files'])} DB files",
        fontsize=14
    )
    ax.grid(True, alpha=0.3)

    return ax



def build_lineage_from_sql(target_pid, particles_df, relations_df):
    """
    Build nodes and edges for the full ancestor tree of one target particle.
    """

    particle_lookup = {}

    for _, row in particles_df.iterrows():
        pid = int(row["pid"])
        particle_lookup[pid] = {
            "r": float(row["r"]),
            "fiPEG": float(row["fiPEG"]),
            "fiRNA": float(row["fiRNA"]),
            "birth_time": float(row["birth_time"]),
            "death_time": None if pd.isna(row["death_time"]) else float(row["death_time"]),
            "merge_count": int(row["merge_count"]),
            "is_initial": bool(row["is_initial"])
        }

    parent_map = {}

    for _, row in relations_df.iterrows():
        child_pid = int(row["child_pid"])
        parent_pid = int(row["parent_pid"])

        if child_pid not in parent_map:
            parent_map[child_pid] = []

        parent_map[child_pid].append(parent_pid)

    nodes = {}
    edges = []

    def recurse(pid):
        if pid not in particle_lookup:
            return

        if pid in nodes:
            return

        nodes[pid] = particle_lookup[pid]

        parents = parent_map.get(pid, [])

        for parent_pid in parents:
            edges.append((parent_pid, pid))
            recurse(parent_pid)

    recurse(int(target_pid))

    return nodes, edges



def plot_merger_tree_sql(target_pid, particles_df, relations_df, axes=None, use_log_time=True):
    """
    Plot merger tree for one final particle.
    - longest single initial-parent-to-final path
    - shortest single initial-parent-to-final path
    """

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 8))

    ax_radius, ax_time = axes

    nodes, edges = build_lineage_from_sql(target_pid, particles_df, relations_df)

    if not nodes:
        for ax in axes:
            ax.text(
                0.5, 0.5,
                "No lineage data",
                ha="center",
                va="center",
                transform=ax.transAxes
            )
        return axes

    final_pid = int(target_pid)

    def get_parents(pid):
        return [parent for parent, child in edges if child == pid]

    all_paths = []

    def collect_paths_backward(pid, current_path):
        """
        Walk backward from final particle to initial particles.
        """
        parents = get_parents(pid)

        # No parents means original initial particle
        if len(parents) == 0:
            path_initial_to_final = list(reversed(current_path))
            all_paths.append(path_initial_to_final)
            return

        for parent_pid in parents:
            collect_paths_backward(parent_pid, current_path + [parent_pid])

    collect_paths_backward(final_pid, [final_pid])

    if len(all_paths) == 0:
        print("Warning: no paths found.")
        return axes

    # Longest and shortest single paths

    longest_path = max(all_paths, key=len)
    shortest_path = min(all_paths, key=len)

    longest_path_set = set(longest_path)
    shortest_path_set = set(shortest_path)

    longest_edges = set(
        (longest_path[i], longest_path[i + 1])
        for i in range(len(longest_path) - 1)
    )

    shortest_edges = set(
        (shortest_path[i], shortest_path[i + 1])
        for i in range(len(shortest_path) - 1)
    )

    print(f"  Longest path:         {longest_path}")
    print(f"  Shortest path:        {shortest_path}")

    def get_node_color(pid):
        if pid == final_pid:
            return "red"
        elif pid in longest_path_set:
            return "gold"
        elif pid in shortest_path_set:
            return "limegreen"
        else:
            return "steelblue"

    def get_edge_color(parent_pid, child_pid):
        if (parent_pid, child_pid) in longest_edges:
            return "gold"
        elif (parent_pid, child_pid) in shortest_edges:
            return "limegreen"
        else:
            return "black"

    def get_edge_width(parent_pid, child_pid):
        if (parent_pid, child_pid) in longest_edges:
            return 2.0
        elif (parent_pid, child_pid) in shortest_edges:
            return 2.0
        else:
            return 1.0

    times = [node["birth_time"] for node in nodes.values()]
    positive_times = [t for t in times if t > 0]

    min_time = min(positive_times) if positive_times else 1e-10

    def time_to_y(t):
        if use_log_time:
            if t <= 0:
                t = min_time * 0.1
            return np.log10(t)
        else:
            return t

    x_positions = {}

    def assign_x_positions(pid, x_left, x_right):
        x_positions[pid] = (x_left + x_right) / 2

        parents = get_parents(pid)

        if not parents:
            return

        width = x_right - x_left
        n_parents = len(parents)

        for i, parent_pid in enumerate(parents):
            p_left = x_left + i * width / n_parents
            p_right = x_left + (i + 1) * width / n_parents
            assign_x_positions(parent_pid, p_left, p_right)

    assign_x_positions(final_pid, 0, 1)


    # Plot radius and time trees
    for ax, y_type in [(ax_radius, "radius"), (ax_time, "time")]:

        for parent_pid, child_pid in edges:
            if parent_pid not in x_positions or child_pid not in x_positions:
                continue

            x1 = x_positions[parent_pid]
            x2 = x_positions[child_pid]

            if y_type == "radius":
                y1 = nodes[parent_pid]["r"]
                y2 = nodes[child_pid]["r"]
            else:
                y1 = time_to_y(nodes[parent_pid]["birth_time"])
                y2 = time_to_y(nodes[child_pid]["birth_time"])

            edge_color = get_edge_color(parent_pid, child_pid)
            edge_width = get_edge_width(parent_pid, child_pid)
            edge_alpha = 1

            # vertical part
            ax.plot(
                [x1, x1],
                [y1, y2],
                color=edge_color,
                linewidth=edge_width,
                solid_capstyle="round",
                zorder=5,
                alpha=edge_alpha
            )

            # horizontal part
            ax.plot(
                [x1, x2],
                [y2, y2],
                color=edge_color,
                linewidth=edge_width,
                solid_capstyle="round",
                zorder=5,
                alpha=edge_alpha
            )

        # Plot nodes
        for pid, node in nodes.items():
            if pid not in x_positions:
                continue

            x = x_positions[pid]

            if y_type == "radius":
                y = node["r"]
            else:
                y = time_to_y(node["birth_time"])

            size = np.clip(node["r"] ** 2 / 20, 10, 150)

            color = get_node_color(pid)

            if pid == final_pid:
                marker = "*"
                zorder = 30
                size = size * 2.8
                edge_width = 1.2
            elif pid == longest_path[0]:
                marker = "D"
                zorder = 25
                size = size * 1.8
                edge_width = 1.0
            elif pid == shortest_path[0]:
                marker = "s"
                zorder = 25
                size = size * 1.8
                edge_width = 1.0
            elif pid in longest_path_set or pid in shortest_path_set:
                marker = "o"
                zorder = 20
                size = size * 1.8
                edge_width = 0.8
            else:
                marker = "o"
                zorder = 20
                size = size * 1.5
                edge_width = 0.8

            ax.scatter(
                x,
                y,
                s=size,
                c=color,
                marker=marker,
                edgecolor="black",
                linewidth=edge_width,
                zorder=zorder,
                alpha=1
            )
        if y_type == "radius":
            ax.set_ylabel("Radius (nm)", fontsize=13)
            ax.set_title("Merger Tree by Radius", fontsize=14)
        else:
            if use_log_time:
                ax.set_ylabel("Time (log$_{10}$ s)", fontsize=13)
            else:
                ax.set_ylabel("Time (s)", fontsize=13)
            ax.set_title("Merger Tree by Time", fontsize=14)

        ax.set_xlabel("Branch", fontsize=13)
        ax.set_xticks([])
        ax.grid(True, alpha=0.2, axis="y")

        if y_type == "radius":
            ax.scatter(
                [], [], c="red", s=200, marker="*",
                label="Selected final particle",
                edgecolor="black"
            )

            ax.scatter(
                [], [], c="gold", s=120, marker="D",
                label=f"Longest path",
                edgecolor="black"
            )

            ax.scatter(
                [], [], c="limegreen", s=120, marker="s",
                label=f"Shortest path",
                edgecolor="black"
            )

            ax.scatter(
                [], [], c="steelblue", s=80, marker="o",
                label="Other lineage nodes",
                edgecolor="black"
            )

            ax.legend(loc="upper left", fontsize=9)

    return axes

#main

if __name__ == "__main__":

    db_pattern = "full_sim_*_lineage_R09_N4000.db"

    sql_data = load_all_sql_data(db_pattern)

    final_particles = sql_data["final_particles"]
    particles = sql_data["particles"]
    relations = sql_data["relations"]
    simulations = sql_data["simulations"]


    print(f"Total particles:      {len(particles)}")

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    plot_size_vs_mergers_all(sql_data, ax=ax1)
    plt.tight_layout()
    plt.show()


    top_particle = final_particles.sort_values("final_r", ascending=False).iloc[2]

    target_pid = int(top_particle["pid"])
    target_db_file = top_particle["db_file"]
    merge_count = int(top_particle["merge_count"])
    num_ancestors = int(top_particle["num_ancestors"])

    particles_one = particles[particles["db_file"] == target_db_file].copy()
    relations_one = relations[relations["db_file"] == target_db_file].copy()

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 8))

    plot_merger_tree_sql(
        target_pid=target_pid,
        particles_df=particles_one,
        relations_df=relations_one,
        axes=axes2,
        use_log_time=True
    )

    fig2.suptitle(
        f"Particle {target_pid} - {merge_count} mergers, {num_ancestors} ancestors\n"
        f"{target_db_file}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()

    print("Done")