# Poisoning Attacks with Back-gradient optimization on Physics-informed Turbulent Flow prediction

This is an implemetation of the paper "Towards Poisoning of Deep Learning Algorithms with Back-gradient Optimization" (https://arxiv.org/abs/1708.08689) on the model "Towards Physics-informed Deep Learning for Turbulent Flow Prediction" (https://arxiv.org/abs/1911.08655).

To generate the poison points, run the code:
```sh
  python poison_turb1.2.py --gen_poison
```

To test the poison points by mixing them with the training data and evaluating on Validation set:
```sh
  python poison_turb1.2.py --poison_train
```
The code was run on ACESipu, where the data is located.
