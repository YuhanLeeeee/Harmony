# Harmony
Official implementation for paper "Harmony: Hierarchical and Balanced Multi-Modal Fusion for Reaction Yield Prediction".

Our code is improved based on the code of [UAM](https://github.com/The-Real-JerryChen/reaction_yield_prediction). We would like to express our gratitude to the author of UAM for making it open-source!

---
## Requirements

Please run the code on a device that supports the `bfloat16` data type. You can check the compatibility using the following code:
`print(torch.cuda.get_device_capability(device)[0] >= 8)`

Required packages and recommended version:

```
python (>=3.9)
pytorch (>= 2.1.0+cu121)
transformers (>=4.50.0)
rdkit (>=2024.9.6)
mamba-ssm (>=2.2.2)
causal-conv1d (>=1.4.0)
numpy (>=1.26.0)
torch-geometric
torch-scatter
...
```

### Quick Runtime Environment Setup 
We have provided `requirements.txt`, `mamba_ssm-2.2.2+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl`, and `causal_conv1d-1.4.0+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl` in the `requirements` directory of this project. You can quickly set up the runtime environment by executing the `env.sh` script.

```
. env.sh
```

---
## Datasets

We have provided three main benchmark datasets used in the experimental process in the `data` folder
of this project. `BH.pkl` and `SM.pkl` correspond to the original 
Buchwald-Hartwig and Suzuki-Miyaura datasets respectively.
`BH_split.pkl` and `SM_split.pkl` are the datasets divided according 
to the ratio of training data to test data of 7:3. 
`ACR_split622.pkl` indicates that the Amide coupling reaction dataset
is divided into a training set, a validation set and a test set according to
the ratio of 6:2:2. The complete ACR dataset can be accessed [here](https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc03902a).

---
## Preparations Before Running
1. Please download [seyonec/ChemBERTa-zinc-base-v1](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1) 
from HuggingFace and place it in the `ChemBERTa` directory of this project.
2. The pre-trained weights (`checkpoints`) and partial required dependencies (`requirements`) 
can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1HDeS0IK6V1gwGixyFW02o6ueTPrHIiID?usp=sharing).
 After downloading, please place them in the corresponding folders.


---
## Training & Evaluation

We have provided the model weights and configuration files trained on each benchmark dataset
in the `checkpoint` directory of the project. You can modify the hyperparameters
by editing the `config.yaml` file in the project.

You can use the `train.sh` script to complete the training and testing 
of Harmony. When the `epoch` is set to 0, only the testing of the model
will be carried out. 

During the testing, it is necessary to ensure that the `exp_name` parameter
is the same as the name of the folder where the model weights are saved.
The dataset can be switched through the `dataset_path` parameter.

```
. train.sh
```

We will provide a more detailed training and evaluation process after the paper is accepted.

[//]: # (## References)

[//]: # (If you find this repository useful in your research, please cite the following paper:)

[//]: # (```)

[//]: # (@inproceedings{chen2024uncertainty,)

[//]: # (  title={Uncertainty-Aware Yield Prediction with Multimodal Molecular Features},)

[//]: # (  author={Chen, Jiayuan and Guo, Kehan and Liu, Zhen and Isayev, Olexandr and Zhang, Xiangliang},)

[//]: # (  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},)

[//]: # (  year={2024})

[//]: # (})

[//]: # (```)
