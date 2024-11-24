import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data

def train_model(file_path, model_path='music_popularity_model.pkl'):
    """Trains and saves the model."""
    # Preprocess the dataset
    X, y = preprocess_data(file_path)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    return model, X_test, y_test
