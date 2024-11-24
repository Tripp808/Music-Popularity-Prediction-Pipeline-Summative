import pandas as pd

def preprocess_data(file_path):
    """Preprocesses the dataset for training or inference."""
    # Load dataset
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    # Cleaning numeric columns stored as strings
    numeric_cols = ['Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach',
                    'YouTube Views', 'YouTube Likes', 'TikTok Posts', 'TikTok Likes', 'TikTok Views',
                    'YouTube Playlist Reach', 'AirPlay Spins', 'SiriusXM Spins',
                    'Deezer Playlist Reach', 'Pandora Streams', 'Pandora Track Stations',
                    'Soundcloud Streams', 'Shazam Counts']

    for col in numeric_cols:
        data[col] = data[col].str.replace(',', '').astype(float)

    # Drop empty column
    data.drop(columns=['TIDAL Popularity'], inplace=True, errors='ignore')

    # Fill missing values
    data.fillna(data.mean(numeric_only=True), inplace=True)
    data['Artist'].fillna('Unknown', inplace=True)

    # Create target column
    track_score_median = data['Track Score'].median()
    data['High_Potential'] = (data['Track Score'] > track_score_median).astype(int)

    # Drop unnecessary columns
    X = data.drop(columns=['High_Potential', 'Track', 'Album Name', 'Artist', 
                           'Release Date', 'ISRC', 'All Time Rank', 'Track Score'])
    y = data['High_Potential']

    # Encode categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    return X, y
