from train import train
from sklearn.metrics import classification_report


def evaluation():
    print("Evaluation started")
    model,X_test,y_test=train()
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

if __name__ == "__main__":
    evaluation()