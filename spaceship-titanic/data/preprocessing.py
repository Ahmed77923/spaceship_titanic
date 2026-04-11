# step 1
#1) we have a passengerld column we can know the person's group or not split the 
#2) column into two and take the number of group and deleted 
# we have cabin column we can split it into three columns (Deck,Num,Side) and we can do to_dumies for the deck and side and we can drop the num column because it is not important

#3) we have (Home planet,Destination) it need to to_dumies
#4) we have (cryo sleep,VIP,Transported) it is boolean we can change it to 0 and 1
# step 2
# we can fill the missing value in column Age by meadin()
# we can fill the missing value in column Cabin by fillna('Unknown/0/U')
# we can fill the missing value in column Home planet by fillna('Unknown')
# we can fill the missing value in column VIP by fillna(False)
# we can fill the missing value in column Destination by fillna('Unknown')
# we can fill the missing value in column Cryo sleep by fillna(False)
# we can fill the missing value in column spending by fillna(0)
# the column name drop 
import pandas as pd

    
def preprocess_data(df):
    """
    the perprocessing steps for the data
    Parameters:
        df (pd.DataFrame): the data frame that we want to preprocess
    Returns:
        pd.DataFrame: the preprocessed data frame
    """
    df = df.copy()

    # PassengerId-derived features
    if 'PassengerId' in df.columns:
        split_cols = df['PassengerId'].astype(str).str.split('_', expand=True)
        if split_cols.shape[1] >= 2:
            df['Group'] = pd.to_numeric(split_cols[0], errors='coerce')
            df['Group_Num'] = pd.to_numeric(split_cols[1], errors='coerce')
            df['Group_size'] = df.groupby('Group')['Group'].transform('count')
            df['is_alone'] = (df['Group_size'] == 1).astype(int)

    df.drop(columns=['PassengerId', 'Name'], inplace=True, errors='ignore')

    # Cabin features
    if 'Cabin' in df.columns:
        df['Cabin'] = df['Cabin'].fillna('Unknown/0/U')
        cabin_split = df['Cabin'].str.split('/', expand=True)
        if cabin_split.shape[1] >= 3:
            df['Deck'] = cabin_split[0]
            df['Num'] = pd.to_numeric(cabin_split[1], errors='coerce')
            df['Side'] = cabin_split[2]
        df.drop(columns=['Cabin'], inplace=True)

    if 'Num' in df.columns:
        if df['Num'].nunique(dropna=True) >= 2:
            df['Num_bin'] = pd.qcut(df['Num'], q=4, duplicates='drop')
        else:
            df['Num_bin'] = 'Unknown'

    # Categorical missing
    if 'HomePlanet' in df.columns:
        df['HomePlanet'] = df['HomePlanet'].fillna('Unknown')
    if 'Destination' in df.columns:
        df['Destination'] = df['Destination'].fillna('Unknown')

    # Boolean columns that may or may not exist (test has no Transported)
    for col in ['CryoSleep', 'VIP']:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)
    if 'Transported' in df.columns:
        df['Transported'] = df['Transported'].fillna(False).astype(int)

    # Age
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())

    # Spending
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    existing_spending_cols = [col for col in spending_cols if col in df.columns]
    if existing_spending_cols:
        df[existing_spending_cols] = df[existing_spending_cols].fillna(0)
        df['Total_spending'] = df[existing_spending_cols].sum(axis=1)
    else:
        df['Total_spending'] = 0
    df['No_spending'] = (df['Total_spending'] == 0).astype(int)

    return df

if __name__ == "__main__":
    df = pd.read_csv('data/train.csv')
    df = preprocess_data(df)