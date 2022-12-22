# DeepCTR-Torch
e
[![Python Versions](https://img.shields.io/pypi/pyversions/deepctr-torch.svg)](https://pypi.org/project/deepctr-torch)
[![Downloads](https://pepy.tech/badge/deepctr-torch)](https://pepy.tech/project/deepctr-torch)
[![PyPI Version](https://img.shields.io/pypi/v/deepctr-torch.svg)](https://pypi.org/project/deepctr-torch)
[![GitHub Issues](https://img.shields.io/github/issues/shenweichen/deepctr-torch.svg
)](https://github.com/shenweichen/deepctr-torch/issues)

[![Documentation Status](https://readthedocs.org/projects/deepctr-torch/badge/?version=latest)](https://deepctr-torch.readthedocs.io/)
![CI status](https://github.com/shenweichen/deepctr-torch/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/shenweichen/DeepCTR-Torch/branch/master/graph/badge.svg?token=m6v89eYOjp)](https://codecov.io/gh/shenweichen/DeepCTR-Torch)
[![Disscussion](https://img.shields.io/badge/chat-wechat-brightgreen?style=flat)](./README.md#disscussiongroup)
[![License](https://img.shields.io/github/license/shenweichen/deepctr-torch.svg)](https://github.com/shenweichen/deepctr-torch/blob/master/LICENSE)

PyTorch version of [DeepCTR](https://github.com/shenweichen/DeepCTR).

DeepCTR is a **Easy-to-use**,**Modular** and **Extendible** package of deep-learning based CTR models along with lots of core components layers which can be used to build your own custom model
easily.You can use any complex model with `model.fit()`and `model.predict()` .Install through `pip install -U deepctr-torch`.

Let's [**Get Started!**](https://deepctr-torch.readthedocs.io/en/latest/Quick-Start.html)([Chinese Introduction](https://zhuanlan.zhihu.com/p/53231955))

## Models List

|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  Convolutional Click Prediction Model  | [CIKM 2015][A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)             |
| Factorization-supported Neural Network | [ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)                    |
|      Product-based Neural Network      | [ICDM 2016][Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                   |
|              Wide & Deep               | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                 |
|                 DeepFM                 | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                           |
|        Piece-wise Linear Model         | [arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)                                 |
|          Deep & Cross Network          | [ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                   |
|   Attentional Factorization Machine    | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |
|      Neural Factorization Machine      | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                               |
|                xDeepFM                 | [KDD 2018][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                         |
|         Deep Interest Network          | [KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)                                                       |
|    Deep Interest Evolution Network     | [AAAI 2019][Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)                                            |
|                AutoInt                 | [CIKM 2019][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                              |
|                  ONN                   | [arxiv 2019][Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                                |
|                FiBiNET                 | [RecSys 2019][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)   |
|                  IFM                   | [IJCAI 2019][An Input-aware Factorization Machine for Sparse Prediction](https://www.ijcai.org/Proceedings/2019/0203.pdf)   |
|                  DCN V2                | [arxiv 2020][DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535)   |
|                  DIFM                  | [IJCAI 2020][A Dual Input-aware Factorization Machine for CTR Prediction](https://www.ijcai.org/Proceedings/2020/0434.pdf)   |
|                  AFN                   | [AAAI 2020][Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://arxiv.org/pdf/1909.03276)   |
|               SharedBottom             | [arxiv 2017][An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/pdf/1706.05098.pdf)  |
|                  ESMM                  | [SIGIR 2018][Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://dl.acm.org/doi/10.1145/3209978.3210104)                       |
|                  MMOE                  | [KDD 2018][Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007)                   |
|                  PLE                   | [RecSys 2020][Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236)                   |

## DisscussionGroup & Related Projects

- [Github Discussions](https://github.com/shenweichen/DeepCTR/discussions)
- Wechat Discussions


- Related Projects

    - [AlgoNotes](https://github.com/shenweichen/AlgoNotes)
    - [DeepCTR](https://github.com/shenweichen/DeepCTR)
    - [DeepMatch](https://github.com/shenweichen/DeepMatch)
    - [GraphEmbedding](https://github.com/shenweichen/GraphEmbedding)

## Main Contributors([welcome to join us!](./CONTRIBUTING.md))
