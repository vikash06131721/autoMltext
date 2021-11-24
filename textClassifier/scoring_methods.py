from sklearn.metrics import f1_score, \
precision_score, recall_score, roc_auc_score, \
accuracy_score
import warnings
warnings.filterwarnings("ignore") 

def scorer_f1(y_true,y_pred):
    return f1_score(y_true,y_pred)

def scorer_precision(y_true,y_pred):
    return precision_score(y_true,y_pred)

def scorer_recall(y_true,y_pred):
    return recall_score(y_true,y_pred)

def scorer_accuracy(y_true,y_pred):
    return accuracy_score(y_true,y_pred)

def scorer_roc(y_true,y_pred):
    return roc_auc_score(y_true,y_pred)