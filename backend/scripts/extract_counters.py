import os
import polars as pl

def main():
    data_dir = r"c:\Dev\group-projects\WIF3009-project\backend\data"
    parquet_path = os.path.join(data_dir, "Parquets", "final_draft.parquet")
    output_path = os.path.join(data_dir, "Parquets", "champ_counters.parquet")
    
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} does not exist.")
        return
        
    df = pl.read_parquet(parquet_path)
    print("Loaded final_draft.parquet. Shape:", df.shape)
    
    roles = ["top", "jng", "mid", "bot", "sup"]
    matchup_dfs = []
    
    for role in roles:
        blue_col = f"blue_{role}"
        red_col = f"red_{role}"
        
        # Symmetrical aggregates:
        # 1. Blue side perspective
        blue_side = df.select([
            pl.col(blue_col).alias("champion"),
            pl.col(red_col).alias("enemy_champion"),
            pl.lit(role.upper()).alias("role"),
            pl.col("blue_win").cast(pl.Int32).alias("wins"),
            pl.lit(1).alias("matches")
        ]).filter(pl.col("champion").is_not_null() & (pl.col("champion") != "") & 
                  pl.col("enemy_champion").is_not_null() & (pl.col("enemy_champion") != ""))
        
        # 2. Red side perspective (wins = 1 - blue_win)
        red_side = df.select([
            pl.col(red_col).alias("champion"),
            pl.col(blue_col).alias("enemy_champion"),
            pl.lit(role.upper()).alias("role"),
            (1 - pl.col("blue_win").cast(pl.Int32)).alias("wins"),
            pl.lit(1).alias("matches")
        ]).filter(pl.col("champion").is_not_null() & (pl.col("champion") != "") & 
                  pl.col("enemy_champion").is_not_null() & (pl.col("enemy_champion") != ""))
        
        matchup_dfs.extend([blue_side, red_side])
        
    # Concatenate all
    all_matchups = pl.concat(matchup_dfs)
    
    # Group by champion, enemy_champion, and role
    grouped = all_matchups.group_by(["champion", "enemy_champion", "role"]).agg([
        pl.col("wins").sum().alias("wins"),
        pl.col("matches").sum().alias("matches")
    ])
    
    # Calculate head-to-head win rate
    grouped = grouped.with_columns(
        (pl.col("wins") / pl.col("matches")).alias("head_to_head_winrate")
    )
    
    print(f"Aggregated {grouped.height} unique champion matchup pairs.")
    
    # Filter matches >= 2 to reduce statistical noise
    min_games = 2
    filtered = grouped.filter(pl.col("matches") >= min_games)
    print(f"Filtered matchups with >= {min_games} games: {filtered.height} matchups remaining.")
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered.write_parquet(output_path)
    print(f"Successfully saved polars-based counter metrics to: {output_path}")

if __name__ == "__main__":
    main()
