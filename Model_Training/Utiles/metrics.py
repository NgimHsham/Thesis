from sklearn.metrics import confusion_matrix, classification_report

def get_classification_metrics(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)
    return cm, report
