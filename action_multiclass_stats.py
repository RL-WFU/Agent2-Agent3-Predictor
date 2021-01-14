#LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

history = np.load("action_multiclass_history.npy", allow_pickle=True).item()

model = load_model("actionMultiClassNetwork.keras")
model.summary()

#PLOTS
plt.plot(history['agent2classifier_loss'], label='Training Agent2 loss')
plt.plot(history['agent3classifier_loss'], label='Training Agent3 loss')
plt.plot(history['val_agent2classifier_loss'], label='Validation Agent2 loss')
plt.plot(history['val_agent3classifier_loss'], label='Validation Agent2 loss')
plt.title("Agent Predictor Network Loss")
plt.legend()
plt.show()

plt.clf()   # clear figure

plt.plot(history['agent2classifier_acc'], label='Training Agent2 acc')
plt.plot(history['agent3classifier_acc'], label='Training Agent3 acc')
plt.plot(history['val_agent2classifier_acc'], label='Validation Agent2 acc')
plt.plot(history['val_agent3classifier_acc'], label='Validation Agent2 acc')
plt.title("Agent Predictor Network Accuracy")
plt.legend()
plt.show()

print(history['val_agent3classifier_acc'][-1])
print(history['val_agent2classifier_acc'][-1])

