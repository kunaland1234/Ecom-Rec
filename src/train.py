from preprocess import loadandcleanevent
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

path='data/events.csv'

def train():
    df=loadandcleanevent(path)
    y=df['label']
    X=df.drop('label',axis=1)
    #Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,stratify=y)

    #Model Training
    model=LogisticRegression()
    model.fit(X_train,y_train)

    return model,X_test,y_test


