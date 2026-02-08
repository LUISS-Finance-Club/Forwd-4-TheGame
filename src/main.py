from src.data.fetch_data import NFLDataFetcher
from src.features.engineer import FeatureEngineer
from src.models.train import SuperBowlPredictor
import json

def main():
    # Configuration
    ROLLING_WINDOW = 5  # Can be made interactive
    TEAM_1 = "SEA"  # Seahawks
    TEAM_2 = "NE"   # Patriots
    
    print("=" * 60)
    print("FORWD Super Bowl LX Prediction Model")
    print("=" * 60)
    
    # Step 1: Fetch data
    print("\n[1/5] Fetching NFL data...")
    fetcher = NFLDataFetcher(2025)
    pbp, schedules, weekly = fetcher.fetch_season_data()
    
    # Step 2: Calculate team statistics
    print("\n[2/5] Calculating team statistics...")
    sea_stats = fetcher.get_team_season_stats(TEAM_1, schedules, pbp)
    ne_stats = fetcher.get_team_season_stats(TEAM_2, schedules, pbp)
    
    # Step 3: Feature engineering
    print(f"\n[3/5] Engineering features (rolling window={ROLLING_WINDOW})...")
    engineer = FeatureEngineer(window_size=ROLLING_WINDOW)
    
    sea_stats = engineer.create_rolling_features(sea_stats)
    ne_stats = engineer.create_rolling_features(ne_stats)
    
    # Create Super Bowl matchup features
    sb_features = engineer.create_matchup_features(sea_stats, ne_stats)
    
    # Step 4: Train models
    print("\n[4/5] Training models...")
    
    # Prepare training data from regular season
    all_teams = schedules['home_team'].unique()
    all_team_stats = {}
    
    for team in all_teams:
        stats = fetcher.get_team_season_stats(team, schedules, pbp)
        stats = engineer.create_rolling_features(stats)
        all_team_stats[team] = stats
    
    training_df = engineer.prepare_training_data(all_team_stats, schedules)
    
    feature_cols = [col for col in training_df.columns 
                   if col not in ['home_win', 'point_diff']]
    
    X = training_df[feature_cols]
    y_win = training_df['home_win']
    y_diff = training_df['point_diff']
    
    predictor = SuperBowlPredictor()
    predictor.feature_cols = feature_cols
    
    xgb_params, xgb_acc, xgb_loss = predictor.train_xgboost(X, y_win, tune=False)
    pd_alpha, pd_rmse = predictor.train_point_diff(X, y_diff)
    
    # Step 5: Make Super Bowl prediction
    print("\n[5/5] Predicting Super Bowl LX...")
    print("=" * 60)
    
    X_sb = sb_features[feature_cols]
    
    # XGBoost prediction
    xgb_proba = predictor.xgb_model.predict_proba(X_sb)[0][1]  # Prob of team 1 win
    
    # Point differential prediction
    point_diff_pred = predictor.point_diff_model.predict(X_sb)[0]
    
    # Blended prediction
    blended_proba, pd_proba = predictor.blend_predictions(xgb_proba, point_diff_pred)
    
    # Display results
    print(f"\n{TEAM_1} (15-4, 2-0 playoffs)")
    print(f"  OFF: {sea_stats.iloc[-1][f'points_rolling_{ROLLING_WINDOW}']:.1f} ppg | "
          f"{sea_stats.iloc[-1][f'epa_rolling_{ROLLING_WINDOW}']:.3f} EPA | "
          f"{sea_stats.iloc[-1][f'success_rolling_{ROLLING_WINDOW}']:.0%} success")
    print(f"  DEF: {sea_stats.iloc[-1][f'pts_allowed_rolling_{ROLLING_WINDOW}']:.1f} ppg allowed | "
          f"{sea_stats.iloc[-1][f'sacks_rolling_{ROLLING_WINDOW}']:.1f} sacks/g")
    
    print(f"\n{TEAM_2} (16-4, 3-0 playoffs)")
    print(f"  OFF: {ne_stats.iloc[-1][f'points_rolling_{ROLLING_WINDOW}']:.1f} ppg | "
          f"{ne_stats.iloc[-1][f'epa_rolling_{ROLLING_WINDOW}']:.3f} EPA | "
          f"{ne_stats.iloc[-1][f'success_rolling_{ROLLING_WINDOW}']:.0%} success")
    print(f"  DEF: {ne_stats.iloc[-1][f'pts_allowed_rolling_{ROLLING_WINDOW}']:.1f} ppg allowed | "
          f"{ne_stats.iloc[-1][f'sacks_rolling_{ROLLING_WINDOW}']:.1f} sacks/g")
    
    print("\n" + "=" * 60)
    print("Model breakdown:")
    print(f"  XGBoost:        {TEAM_1} {xgb_proba:.1%} | {TEAM_2} {1-xgb_proba:.1%}")
    print(f"  Point diff:     {TEAM_1} {pd_proba:.1%} | {TEAM_2} {1-pd_proba:.1%}")
    
    print(f"\nFinal (40/60 blend):")
    print(f"  {TEAM_1}: {blended_proba:.1%}")
    print(f"  {TEAM_2}: {1-blended_proba:.1%}")
    print(f">> {TEAM_1 if blended_proba > 0.5 else TEAM_2} WIN ({max(blended_proba, 1-blended_proba):.1%})")
    print("=" * 60)
    
    # Save models and predictions
    predictor.save_models('models/')
    
    # Save prediction results for webapp
    results = {
        'team1': TEAM_1,
        'team2': TEAM_2,
        'team1_record': '15-4',
        'team2_record': '16-4',
        'xgb_proba': float(xgb_proba),
        'point_diff_proba': float(pd_proba),
        'blended_proba': float(blended_proba),
        'point_diff_pred': float(point_diff_pred),
        'xgb_accuracy': float(xgb_acc),
        'xgb_params': xgb_params,
        'point_diff_rmse': float(pd_rmse),
        'point_diff_alpha': float(pd_alpha),
        'rolling_window': ROLLING_WINDOW,
        'feature_importance': dict(zip(
            feature_cols,
            predictor.xgb_model.feature_importances_.tolist()
        ))
    }
    
    with open('webapp/public/prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved for webapp")

if __name__ == "__main__":
    main()
