Onboarding Project:

This project synthetically generates a map of mountains and uses a GNN to classify which nodes are mountain peaks and which ones are not.

Hyperparameter tuning:

We tested the following hyperparameter configurations:

dims    learning rate     training_acc      testing_acc

[8,2]      0.01               0.998            0.998
[16,2]     0.01               0.998            0.998
[32, 2]    0.01               0.998            0.998
[16,16,2]  0.01               0.998            0.998
[16, 2]    0.005              0.998            0.998


All models, no matter the hyperparameter configuration, achieved very similar accuracies on both testing/training. This means that our models is very generliazable. Since the accuracy did not change despite making architecture deeper, we can see that even a shallower GNN can learn patterns in this task. Similariy, since decreasing learning rate did not improve performance, we can confirm the simplicity of the task. 