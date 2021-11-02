# Context-Sensitive-Temporal-Feature-Learning-for-Gait-Recognition
ICCV 2021 PAPER is available at <a href="https://openaccess.thecvf.com/content/ICCV2021/html/Huang_Context-Sensitive_Temporal_Feature_Learning_for_Gait_Recognition_ICCV_2021_paper.html" title="CSTL">Context-Sensitive-Temporal-Feature-Learning-for-Gait-Recognition</a>.
## Note
Due to the 64 x 44 silhouettes are in a low resolution, the experiment results will oscillate from time to time. Therefore, we highly recommend you guys to conduct experiments with 128 x 88 resolution, which produces relatively stable results.

## Usage
The dataset path can be set in config.py.

Train command: 
```
python train.py
```

Inference command:
```
python test.py
```
## Citation
Citation Format:
```
@inproceedings{huang2021context, 
  title={Context-Sensitive Temporal Feature Learning for Gait Recognition}, 
  author={Huang, Xiaohu and Zhu, Duowang and Wang, Hao and Wang, Xinggang and Yang, Bo and He, Botao and Liu, Wenyu and Feng, Bin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12909--12918},
  year={2021}
}
```
