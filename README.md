# Poisoning Attacks with Back-gradient optimization on Physics-informed Turbulent Flow prediction

To generate the poison points, run the code:
```sh
  python poison_turb1.2.py --gen_poison
```

To test the poison points by mixing them with the training data and evaluating on Validation set:
```sh
  python poison_turb1.2.py --poison_train
```
The code was run on ACESipu, where the data is located.
