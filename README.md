# Digits-classification-Confusion-matrix
This is my Pytorch practice implementation of Confusion matrix of a digits classification problem.

I use torchmetrics to compute the Confusion matrix and use seaborn to visualize it.

With the visualized result, we can have a clearer observation of the model performance on the data.
Therefore, we can commit and decide how to improve the model next.


# General Information
## Model
CNN model, built and trained by me. The architecture is COMPLETELY RANDOM.

## Evaluation steps
I use CNN model to predict the input data.

Then, from the model prediction and data true labels, I use torchmetrics to compute the Confusion matrix.

Finally, I use seaborn to plot the Confusion Matrix.


# Reference
torchmetrics:

[torchmetrics.ConfusionMatrix](https://torchmetrics.readthedocs.io/en/stable/classification/confusion_matrix.html)

seaborn:

[seaborn.heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap)
