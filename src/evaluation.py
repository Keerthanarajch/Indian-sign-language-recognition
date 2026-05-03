import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

def evaluate_model(model_path, X, y_true, class_list, title):
    model = tf.keras.models.load_model(model_path)

    y_pred = np.argmax(model.predict(X), axis=1)

    print(f"\n{title} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_list))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_list, yticklabels=class_list)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()