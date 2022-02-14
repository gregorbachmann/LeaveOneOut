# **Generalization Through the Lens of Leave-One-Out Error**
This repository contains the code for the ICLR paper [Generalization Through the Lens of Leave-One-Out Error](https://openreview.net/pdf?id=7grkzyj89A_) by Gregor Bachmann, Thomas Hofmann and Aurelien Lucchi.
## Dependencies
You will need to install the following dependencies:
1. numpy        1.21.5
2. jax          0.2.26
3. jaxlib       0.1.75
4. torchvision  0.11.2
5. neural-tangents 0.3.9
## Experiments
In the file

> leave_one_out.ipynb 

we provide a Jupyter notebook that illustrates the usage of the leave-one-out error as a proxy for the test error in varying regimes. It is accompanied with the corresponding 
mathematical statements and their implementation. If you prefer to use standard non-notebook files, you can run the file

> example.py

to produce test and the corresponding leave-one-out statistics. 

## Reference
f you use this code, please cite the following paper:
``` bibtex
@inproceedings{
bachmann2022generalization,
title={Generalization Through the Lens of Leave-One-Out Error},
author={Gregor Bachmann and Thomas Hofmann and Aurelien Lucchi},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=7grkzyj89A_}
}
