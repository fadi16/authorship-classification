import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    path = "./checkpoints_and_eval/cross-encoder-10/test_e2e_classification_authors10.csv"
    df = pd.read_csv(path)

    predicted = df["predicted_labels"].tolist()
    actuals = df["actual_labels"].tolist()

    #  i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
    confusion_matrix = confusion_matrix(y_true=actuals, y_pred=predicted)

    labels = sorted(set(predicted))

    ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix - 10 Authors\n\n')
    ax.set_xlabel('\nPredicted')
    ax.set_ylabel('Actual')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()

    print(confusion_matrix)