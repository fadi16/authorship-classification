import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, matthews_corrcoef, \
    cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt


def evaluation_stats(results_csv_path, print_stats=False):
    df = pd.read_csv(results_csv_path)

    predicted = df["predicted_labels"].tolist()
    actuals = df["actual_labels"].tolist()

    ##### confusion matrix
    #  i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
    matrix = confusion_matrix(y_true=actuals, y_pred=predicted)
    labels = sorted(set(predicted))
    ax = sns.heatmap(matrix, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix - 10 Authors\n\n')
    ax.set_xlabel('\nPredicted')
    ax.set_ylabel('Actual')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    plt.savefig(results_csv_path.replace(".csv", "_confusion_matrix.png"))

    if print_stats:
        ##### Accuracy
        accuracy = accuracy_score(actuals, predicted)

        ##### Precision, Recall, F1
        # overall
        overall_precision_recall_f1 = precision_recall_fscore_support(actuals, predicted)
        overall_precision = overall_precision_recall_f1[0]
        overall_recall = overall_precision_recall_f1[1]
        overall_f1 = overall_precision_recall_f1[2]

        # per class
        per_class_precision_recall_f1 = precision_recall_fscore_support(actuals, predicted, labels=labels)
        per_class_precision = per_class_precision_recall_f1[0]
        per_class_recall = per_class_precision_recall_f1[1]
        per_class_f1 = per_class_precision_recall_f1[2]

        ###### Matthews Correlation Coefficient
        mcc = matthews_corrcoef(actuals, predicted)

        ##### Cohen's Kappa
        kappa = cohen_kappa_score(actuals, predicted)

        stats = {
            "accuracy": accuracy,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_recall,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "per_class_f1": per_class_f1,
            "MCC": mcc,
            "K": kappa
        }

        print(stats)


if __name__ == "__main__":
    path = "./checkpoints_and_eval/cross-encoder-10/test_e2e_classification_authors10.csv"
