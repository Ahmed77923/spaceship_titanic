from sklearn.model_selection import train_test_split
from config.config import Config
def split_data(df):
    """
    split the data into train and test sets
    Parameters:
        df (pd.DataFrame): the data frame that we want to split
    Returns:
        X_train (pd.DataFrame): the training features
        X_test (pd.DataFrame): the testing features
        y_train (pd.Series): the training target
        y_test (pd.Series): the testing target
    """
    X = df.drop(columns=[Config.TARGET])
    y = df[Config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)

    print('='* 50)
    print('\nData splitting completed successfully!\n')
    print('='* 50)
    print(f'Training set size: {X_train.shape[0]} samples')
    print(f'Testing set size: {X_test.shape[0]} samples')
    print('='* 50)
    
    return X_train, X_test, y_train, y_test
    