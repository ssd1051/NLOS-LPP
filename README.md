# Generalizable-Non-Line-of-Sight-Imaging-with-Learnable-Physical-Priors

This is the official codebase of the paper Generalizable-Non-Line-of-Sight-Imaging-with-Learnable-Physical-Priors (ICCV2025).

### Dataset

#### Synthetic Dataset

We utilized the synthetic data (~3000 motorbike dataset) provided by [LFE](https://github.com/princeton-computational-imaging/NLOSFeatureEmbeddings). \
Please download in [Here](https://drive.google.com/file/d/183VAD_wuVtwkyvfaBoguUHZgHu065BNW/view).

#### Real-world Data

We utilized the real-world data provided by [FK](https://github.com/computational-imaging/nlos-fk) and [NLOST](https://github.com/Depth2World/NLOST). \

Besides, the self-captured data will be released soon.

### Training

1. Download the synthetic dataset.
2. Modify the data path and result root in `DL_inference/inference/config.py`.
3. Run `train256.py`.

### Acknowledgements

We thank the authors who shared the code of their works. Particularly [LFE](https://github.com/princeton-computational-imaging/NLOSFeatureEmbeddings).

### Citation

```
@inproceedings{sun2025generalizable,
  title={Generalizable Non-Line-of-Sight Imaging with Learnable Physical Priors},
  author={Sun, Shida and Li, Yue and Zhang, Yueyi and Xiong, Zhiwei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```
