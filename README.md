# **BoNet**

This repository provides a PyTorch implementation of BoNet, presented in the paper [Hand Pose Estimation for Pediatric Bone Age Assessment](https://doi.org/10.1007/978-3-030-32226-7_60). Oral presentation at [MICCAI](https://www.miccai2019.org/),2019. 
<br/>

## Paper
[Hand Pose Estimation for Pediatric Bone Age Assessment](https://doi.org/10.1007/978-3-030-32226-7_60) <br/>
[María Escobar](https://mc-escobar11.github.io/)<sup> 1* </sup>, [Cristina González](https://cigonzalez.github.io/)<sup> 1* </sup>, [Felipe Torres](https://ftorres11.github.io/) <sup>1</sup>,[Laura Daza](https://sites.google.com/view/ldaza/en)<sup>1</sup>, [Gustavo Triana](http://radiologiafsfb.org/site/index.php?option=com_content&view=category&id=176&Itemid=332)<sup>2</sup>, [Pablo Arbeláez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=en)<sup>1</sup> <br/>
<sup>*</sup>Equal contribution.
<sup>1 </sup>Biomedical Computer Vision ([BCV](https://biomedicalcomputervision.uniandes.edu.co/)) Lab, Universidad de Los Andes. <br/>
<sup>2 </sup>Radiology department, Fundación Santa Fe de Bogotá. <br/>
<br/>

<p align="center"><img src="Figures/overview.png" /></p>

## Citation
```
@inproceedings{escobar2019hand,
  title={Hand Pose Estimation for Pediatric Bone Age Assessment},
  author={Escobar, Mar{\'\i}a and Gonz{\'a}lez, Cristina and Torres, Felipe and Daza, Laura and Triana, Gustavo and Arbel{\'a}ez, Pablo},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={1--9},
  year={2019},
  organization={Springer}
}

```
<br/>

## Dependencies
* [Python](https://www.continuum.io/downloads) (2.7, 3.5+)
* [PyTorch](http://pytorch.org/) (0.3, 0.4, 1.0)
<br/>

## Usage

### Cloning the repository
```bash
$ git clone https://github.com/BCV-Uniandes/SMIT.git
$ cd SMIT
```

### Downloading the dataset
To download the CelebA dataset:
```bash
$ bash generate_data/download.sh
```

### Train command:
```bash
./main.py --GPU=$gpu_id --dataset_fake=CelebA
```
Each dataset must has `datasets/<dataset>.py` and `datasets/<dataset>.yaml` files. All models and figures will be stored at `snapshot/models/$dataset_fake/<epoch>_<iter>.pth` and `snapshot/samples/$dataset_fake/<epoch>_<iter>.jpg`, respectivelly.

### Test command:
```bash
./main.py --GPU=$gpu_id --dataset_fake=CelebA --mode=test
```
SMIT will expect the `.pth` weights are stored at `snapshot/models/$dataset_fake/` (or --pretrained_model=location/model.pth should be provided). If there are several models, it will take the last alphabetical one. 

### Demo:
```bash
./main.py --GPU=$gpu_id --dataset_fake=CelebA --mode=test --DEMO_PATH=location/image_jpg/or/location/dir
```
DEMO performs transformation per attribute, that is swapping attributes with respect to the original input as in the images below. Therefore, *--DEMO_LABEL* is provided for the real attribute if *DEMO_PATH* is an image (If it is not provided, the discriminator acts as classifier for the real attributes).

### [Pretrained models](http://marr.uniandes.edu.co/weights/SMIT)
Models trained using Pytorch 1.0.

### Multi-GPU
For multiple GPUs we use [Horovod](https://github.com/horovod/horovod). Example for training with 4 GPUs:
```bash
mpirun -n 4 ./main.py --dataset_fake=CelebA
```
<br/>

