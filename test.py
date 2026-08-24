#!/usr/bin/env python3

import sqlite3
import pandas as pd
import networkx as nx
import glob
import os

def list_tables(conn):
    query = """
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    ORDER BY name;
    """
    df = pd.read_sql_query(query, conn)
    return df["name"].tolist()


def get_table_schema(conn, table_name):
    query = f"PRAGMA table_info({table_name});"
    return pd.read_sql_query(query, conn)


def get_table_count(conn, table_name):
    query = f"SELECT COUNT(*) AS n_rows FROM {table_name};"
    df = pd.read_sql_query(query, conn)
    return int(df["n_rows"].iloc[0])


def preview_table(conn, table_name, n=5):
    query = f"SELECT * FROM {table_name} LIMIT {n};"
    return pd.read_sql_query(query, conn)



def inspect_database(db_file, preview_rows=5):

    conn = sqlite3.connect(db_file)

    tables = list_tables(conn)

    print("\nTABLES FOUND:")
    for table in tables:
        print(f"  - {table}")

    row_counts = {}

    for table in tables:
        count = get_table_count(conn, table)
        row_counts[table] = count
        print(f"{table:25s} {count:10d} rows")

    for table in tables:
        print(f"\nSchema for table: {table}")
        schema = get_table_schema(conn, table)
        print(schema[["cid", "name", "type", "notnull", "pk"]].to_string(index=False))


    print(f"FIRST {preview_rows} ROWS FROM EACH TABLE")

    for table in tables:
        print(f"\nPreview: {table}")
        df = preview_table(conn, table, n=preview_rows)
        print(df.to_string(index=False))

    conn.close()

    return row_counts

def simulation_summary(conn):

    try:
        simulations = pd.read_sql_query("SELECT * FROM simulations;", conn)
        print(simulations.to_string(index=False))
    except Exception as e:
        print(f"Could not read simulations table: {e}")

    try:
        final_particles = pd.read_sql_query("SELECT * FROM final_particles;", conn)

        print(f"  Number of final particles: {len(final_particles)}")

        if len(final_particles) > 0:
            print(f"  Mean final radius: {final_particles['final_r'].mean():.4f}")

            print(f"  Mean merge_count:  {final_particles['merge_count'].mean():.4f}")
            print(f"  Max merge_count:   {final_particles['merge_count'].max()}")
            print(f"  Min merge_count:   {final_particles['merge_count'].min()}")

            print("\nTop 10 final particles by merge_count:")
            print(
                final_particles
                .sort_values("merge_count", ascending=False)
                .head(10)
                .to_string(index=False)
            )

    except Exception as e:
        print(f"Could not read final_particles table: {e}")

    try:
        particles = pd.read_sql_query("SELECT * FROM particles;", conn)
        print(f"  Total particles ever created: {len(particles)}")
        print(f"  Initial particles: {(particles['is_initial'] == 1).sum()}")

    except Exception as e:
        print(f"Could not read particles table: {e}")

    try:
        relations = pd.read_sql_query("SELECT * FROM particle_relations;", conn)
        print(f"  Parent-child relations: {len(relations)}")

        if len(relations) > 0:
            print("\nFirst 20 parent-child relations:")
            print(relations.head(20).to_string(index=False))

    except Exception as e:
        print(f"Could not read particle_relations table: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    db_pattern = "full_sim_*_lineage_R09_N4000.db"
    # For N0=5000 in kMC, use:
    # db_pattern = "full_sim_*_lineage_R09_N5000.db"

    files = sorted(glob.glob(db_pattern))
    
    for f in files:
        print(f"  {f}")

    # Inspect first DB file in detail
    db_file = files[0]

    row_counts = inspect_database(db_file, preview_rows=5)

    conn = sqlite3.connect(db_file)

    simulation_summary(conn)

    # Pick final particle with largest merge_count
    final_particles = pd.read_sql_query("""
        SELECT *
        FROM final_particles
        ORDER BY merge_count DESC
        LIMIT 1;
    """, conn)

    conn.close()


if __name__ == "__main__":
    main()