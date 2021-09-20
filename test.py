import numpy as np
from classifier import BertClassifier
from bert_processing import Y_val
from model import bert_predict, val_dataloader
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split


probs = bert_predict(BertClassifier, val_dataloader)

# Evaluate the Bert classifier

def evaluate_roc(probs, y_true):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
  
if __name__ == '__main__':
    evaluate_roc(probs, Y_val.astype(float))
