# test_model.py

from sklearn.metrics import precision_score, recall_score, f1_score
import fasttext

def evaluate_model(model_file, test_data):
    model = fasttext.load_model(model_file)
    y_true, y_pred = [], []
    
    with open(test_data, 'r') as f:
        for line in f:
            # Ignore empty lines or lines without a label and text
            if not line.strip() or ' ' not in line:
                continue
            
            label, text = line.strip().split(' ', 1)
            true_label = label.replace("__label__", "")
            y_true.append(true_label)
            
            pred_label = model.predict(text)[0][0].replace("__label__", "")
            y_pred.append(pred_label)
            
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    return precision, recall, f1
