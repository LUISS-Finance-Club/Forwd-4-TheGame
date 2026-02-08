import pandas as pd
import numpy as np

class FeatureEngineer:
    """Create rolling average features and matchup differentials"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
    
    def create_rolling_features(self, team_stats):
        """Calculate rolling averages for recent performance"""
        
        # Sort by week
        team_stats = team_stats.sort_values('week')
        
        # Rolling statistics
        rolling_cols = ['points', 'epa', 'success', 'def_epa', 
                       'pts_allowed', 'turnovers', 'sacks']
        
        for col in rolling_cols:
            team_stats[f'{col}_rolling_{self.window_size}'] = (
                team_stats[col]
                .rolling(window=self.window_size, min_periods=1)
                .mean()
            )
        
        return team_stats
    
    def create_matchup_features(self, team1_stats, team2_stats):
        """Create differential features between two teams"""
        
        # Get most recent stats (last row)
        t1 = team1_stats.iloc[-1]
        t2 = team2_stats.iloc[-1]
        
        window = self.window_size
        
        features = {
            'pts_diff': t1[f'points_rolling_{window}'] - t2[f'points_rolling_{window}'],
            'epa_diff': t1[f'epa_rolling_{window}'] - t2[f'epa_rolling_{window}'],
            'success_diff': t1[f'success_rolling_{window}'] - t2[f'success_rolling_{window}'],
            'turnover_diff': t1[f'turnovers_rolling_{window}'] - t2[f'turnovers_rolling_{window}'],
            't1_epa': t1[f'epa_rolling_{window}'],
            't2_epa': t2[f'epa_rolling_{window}'],
            't1_success': t1[f'success_rolling_{window}'],
            't2_success': t2[f'success_rolling_{window}'],
            'def_epa_diff': (t1[f'def_epa_rolling_{window}'] - t2[f'def_epa_rolling_{window}']) * 2.0,
            'pts_allowed_diff': (t1[f'pts_allowed_rolling_{window}'] - t2[f'pts_allowed_rolling_{window}']) * 2.0,
            'sacks_diff': (t1[f'sacks_rolling_{window}'] - t2[f'sacks_rolling_{window}']) * 2.0,
            'net_pts_diff': (t1[f'points_rolling_{window}'] - t1[f'pts_allowed_rolling_{window}']) - 
                           (t2[f'points_rolling_{window}'] - t2[f'pts_allowed_rolling_{window}'])
        }
        
        return pd.DataFrame([features])
    
    def prepare_training_data(self, all_team_stats, schedules):
        """Create training dataset from historical matchups"""
        
        training_data = []
        
        # Filter completed games
        completed = schedules[schedules['game_type'] == 'REG'].copy()
        
        for idx, game in completed.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            week = game['week']
            
            # Get team stats up to this game
            home_stats = all_team_stats[home_team][
                all_team_stats[home_team]['week'] < week
            ]
            away_stats = all_team_stats[away_team][
                all_team_stats[away_team]['week'] < week
            ]
            
            if len(home_stats) < 3 or len(away_stats) < 3:
                continue
            
            # Create features
            features = self.create_matchup_features(home_stats, away_stats)
            features['home_win'] = 1 if game['home_score'] > game['away_score'] else 0
            features['point_diff'] = game['home_score'] - game['away_score']
            
            training_data.append(features)
        
        return pd.concat(training_data, ignore_index=True)
