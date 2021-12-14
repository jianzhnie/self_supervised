<!--
 * @Author: jianzhnie
 * @Date: 2021-12-13 12:19:08
 * @LastEditTime: 2021-12-14 14:47:11
 * @LastEditors: jianzhnie
 * @Description:
 *
-->
# Self-Supervised Learning

### Features

[self_supervised](./self_supervised) offers features like

- modular framework
- support for multi-gpu training using PyTorch Lightning
- easy to use and written in a PyTorch like style
- supports custom backbone models for self-supervised pre-training

#### Supported Models

- [MoCo, 2019](https://arxiv.org/abs/1911.05722)
- [SimCLR, 2020](https://arxiv.org/abs/2002.05709)
- [SimSiam, 2021](https://arxiv.org/abs/2011.10566)
- [Barlow Twins, 2021](https://arxiv.org/abs/2103.03230)
- [BYOL, 2020](https://arxiv.org/abs/2006.07733)
- [NNCLR, 2021](https://arxiv.org/abs/2104.14548)
- [SwaV, 2020](https://arxiv.org/abs/2006.09882)
- [MocoV2, 2020]()
- [MocoV3, 2021]()


#### Supported Loss Function

- [NegativeCosineSimilarity]()
- [SwaVLoss]()

### Benchmarks
Currently implemented models and their accuracy on cifar10 and imagenette.
#### ImageNette

| Model       | Epochs | Batch Size | Test Accuracy |
|-------------|--------|------------|---------------|
| MoCo        |  800   | 256        | 0.827         |
| SimCLR      |  800   | 256        | 0.847         |
| SimSiam     |  800   | 256        | 0.827          |
| BarlowTwins |  800   | 256        | 0.801         |
| BYOL        |  800   | 256        | 0.851         |


#### Cifar10

| Model       | Epochs | Batch Size | Test Accuracy |
|-------------|--------|------------|---------------|
| MoCo        |  200   | 128        | 0.83          |
| SimCLR      |  200   | 128        | 0.78          |
| SimSiam     |  200   | 128        | 0.73          |
| BarlowTwins |  200   | 128        | 0.84          |
| BYOL        |  200   | 128        | 0.85          |
| MoCo        |  200   | 512        | 0.85          |
| SimCLR      |  200   | 512        | 0.83          |
| SimSiam     |  200   | 512        | 0.81          |
| BarlowTwins |  200   | 512        | 0.78          |
| BYOL        |  200   | 512        | 0.84          |
| MoCo        |  800   | 128        | 0.89          |
| SimCLR      |  800   | 128        | 0.87          |
| SimSiam     |  800   | 128        | 0.80          |
| MoCo        |  800   | 512        | 0.90          |
| SimCLR      |  800   | 512        | 0.89          |
| SimSiam     |  800   | 512        | 0.91          |

### Tutorials

Want to jump to the tutorials and see lightly in action?

- [Train MoCo on CIFAR-10]()
- [Train SimCLR on clothing data]()
- [Train SimSiam on satellite images]()
- [Use lightly with custom augmentations]()

## Further Reading

**Self-supervised Learning:**
- [A Simple Framework for Contrastive Learning of Visual Representations (2020)](https://arxiv.org/abs/2002.05709)
- [Momentum Contrast for Unsupervised Visual Representation Learning (2020)](https://arxiv.org/abs/1911.05722)
- [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (2020)](https://arxiv.org/abs/2006.09882)
- [What Should Not Be Contrastive in Contrastive Learning (2020)](https://arxiv.org/abs/2008.05659)
