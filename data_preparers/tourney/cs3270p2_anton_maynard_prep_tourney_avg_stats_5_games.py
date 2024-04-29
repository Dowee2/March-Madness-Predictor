
#pylint: disable=C0301,C0114,W0718,C0103
import pandas as pd

def prepare_matchup_data(games_df, stats):
    """
    Merges game data with team stats to prepare matchup data.

    Parameters:
    - games_df (DataFrame): The DataFrame containing game results.
    - stats (DataFrame): The DataFrame containing average stats per team.

    Returns:
    - DataFrame: Matchup data with team stats and game outcome.
    """
    stats.set_index('TeamID', inplace=True)
    processed_data = []

    for _, row in games_df.iterrows():
        team_1, team_2 = sorted((row['WTeamID'], row['LTeamID']))
        team_1_won = 1 if team_1 == row['WTeamID'] else 0
        team_1_stats = stats.loc[team_1].add_prefix('team_1_').to_dict()
        team_2_stats = stats.loc[team_2].add_prefix('team_2_').to_dict()

        matchup_data = {
            'Season': row['Season'],
            'DayNum': row['DayNum'],
            'team_1': team_1,
            'team_2': team_2,
            'team_1_won': team_1_won
        }
        matchup_data.update(team_1_stats)
        matchup_data.update(team_2_stats)

        processed_data.append(matchup_data)

    return pd.DataFrame(processed_data)


def main():
    """
    The main method that starts the program.
    """
    data_location = 'data/Mens/Season/'
    for season in range(2010, 2024):
        try:
            teams_df = pd.read_csv(f'{data_location}/{season}/MRegularSeasonDetailedResults_{season}_avg_5_games.csv').groupby('TeamID').last().reset_index()
            games_df = pd.read_csv(f'{data_location}/{season}/MNCAATourneyCompactResults_{season}.csv')
            matchup_data = prepare_matchup_data(games_df, teams_df)
            matchup_data.to_csv(f'{data_location}/{season}/MNCAATourneyDetailedResults_{season}_tourney_avg_5_games.csv', index=False)
        except FileNotFoundError as error:
            print(f"Data for season {season} not found.")
            print(error)


if __name__ == "__main__":
    main()