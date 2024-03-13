# AHL

Official implementation of ["Anomaly Heterogeneity Learning for Open-set Supervised Anomaly Detection"](https://arxiv.org/pdf/2310.12790.pdf).(accepted by CVPR 2024)

The code will be released soon.

## Setup
numpy >= 1.22.3
python >= 3.10.4
pytorch >= 1.12.1
torch >= 1.12.1
torchvision >= 0.13.1
tqdm >= 4.64.0
scipy >= 1.10.1
scikit-image >= 0.19.2
einops >= 0.6.0

## Run
#### Step 1. Setup the Anomaly Detection dataset

Download the Anomaly Detection Dataset and convert it to MVTec AD format. (For datasets we used in the paper, we provided the convert script.) 
The dataset folder structure should look like:

```
DATA_PATH/
    subset_1/
        train/
            good/
        test/
            good/
            defect_class_1/
            defect_class_2/
            defect_class_3/
            ...
    ...
```
#### Step 2. Running the base model([DRA](https://github.com/Choubo/DRA/tree/main), [DevNet](https://github.com/Choubo/deviation-network-image)) and save model weights.

#### Step 3. Save augmentation features and multi-scale features extracted from base model's feature extractor.

#### Step 4. Running AHL

## Citation

```bibtex
@inproceedings{zhu2023anomaly,
      title={Anomaly Heterogeneity Learning for Open-set Supervised Anomaly Detection}, 
      author={Jiawen Zhu and Choubo Ding and Yu Tian and Guansong Pang},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2024},
}
```
