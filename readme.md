# Before reading, you can find training logs and inference images in the `MAIN NOTEBOOK.ipynb`

# Skip Connections Experimentations

## Training setup

- 3 levels UNet
- lr: 0.001
- epochs: 120
- batch_size: 16
- reduce lr on plateau:
  - lr_reduction_factor: 0.7
  - reduce_on_plateau_patience: 10

UNets (by design) lose spatial information with downsampling.

skip connections were introduced to give the upsampler access to spatial information.

this effictively results in better segmentation masks.

It's intuitive and totally understandable that the more skip connections there are AND the closer they are to the upper levels [the ones with more spatial info], the better the masks get.

This is demonstrated through this experiment where we try different networks with either full skip connections or a single skip connection at a single level per experiment and observing the mDice and mIOU scores.

The following table shows the results of each experiment:

| Configuration       | Best Test mDice | Best Test mIoU |
| ------------------- | --------------- | -------------- |
| **All Connections** | **0.8559**      | **0.7917**     |
| Level 1             | 0.8068          | 0.7296         |
| Level 2             | 0.7912          | 0.7328         |
| Level 3             | 0.7582          | 0.6822         |

as expected, the network with the full connections got full access to spatial information at different levels, which resulted in better informed masks and consequently in higher mIOU and mDice scores.

level 1, 2, and 3 got only partial information about spatial features, therefore their metrics weren't as high as the one with the full connections.

level 1 got access to more spatial features, level 2 got access to a less amount and level 3 had access to an even lesser amount.

the reason i chose BCEDiceLoss is because it handles Class Imbalance well, because this task has many classes with very rare presence in the dataset.

a normal BCELoss would just give the same weight for all class, and it'd result in lower classes having very bad scores.

here's a table showing the Dice score per class through all experiments.

## `Dice Score` per Class

| Class | All Connections |    Level 1 |    Level 2 | Level 3 |
| ----- | --------------: | ---------: | ---------: | ------: |
| 0     |          0.5000 |     0.5000 |     0.5000 |  0.5000 |
| 1     |      **0.9359** |     0.8978 |     0.9242 |  0.8898 |
| 2     |          0.9720 |     0.9624 | **0.9727** |  0.9619 |
| 3     |      **0.9903** |     0.9877 |     0.9881 |  0.9847 |
| 4     |      **0.7368** |     0.5595 |     0.6663 |  0.6239 |
| 5     |      **0.9609** |     0.9405 |     0.9433 |  0.8990 |
| 6     |      **0.9016** |     0.9124 |     0.8937 |  0.8480 |
| 7     |          0.8292 | **0.8654** |     0.8592 |  0.8097 |
| 8     |          0.8615 |     0.8330 | **0.8745** |  0.8378 |
| 9     |      **0.9398** |     0.9299 |     0.9331 |  0.9045 |
| 10    |      **0.7868** |     0.4858 |     0.1481 |  0.0809 |

and this table shows IoU score per class across all experiments.

## `IoU Score` per Class

| Class | All Connections |    Level 1 |    Level 2 | Level 3 |
| ----- | --------------: | ---------: | ---------: | ------: |
| 0     |          0.5000 |     0.5000 |     0.5000 |  0.5000 |
| 1     |      **0.8825** |     0.8145 |     0.8762 |  0.8107 |
| 2     |          0.9457 |     0.9276 | **0.9494** |  0.9274 |
| 3     |      **0.9808** |     0.9757 |     0.9792 |  0.9698 |
| 4     |      **0.5973** |     0.3992 |     0.5412 |  0.4717 |
| 5     |      **0.9247** |     0.8876 |     0.9170 |  0.8168 |
| 6     |          0.8208 | **0.8395** |     0.8230 |  0.7363 |
| 7     |          0.7112 |     0.7650 | **0.8061** |  0.6802 |
| 8     |      **0.7577** |     0.7138 |     0.7323 |  0.7212 |
| 9     |      **0.8865** |     0.8690 |     0.8712 |  0.8257 |
| 10    |      **0.7010** |     0.3332 |     0.0653 |  0.0440 |

so, as can be seen, the network with full connections got the best scores per class than the rest of the networks.

the network with level 3 connections got the worst scores.

## Masks Quality Observations

one more observations is that the masks of the network with full connections look more coherent, connected, and well detailed around the object, and it assigned correct classes per objec.

whereas the masks of networks with lower connections have chunky and flawed masks as well as some wrong classes due to the loss of information.
