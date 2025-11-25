# Project Development Log & Analysis

## Real-Time Sign Language Recognition System with Alphabet Fallback Mechanism for Out-Of-Vocabulary words

- - -

## Development Summary

### How to install and run the model

This repository implenmets realtime American Sign Language recognition with deep Learning and MediaPipe, supporting both Word-level WLASL and Alphabet static models.

## Installation

### Clone the repository: https://github.com/ayaaiden/20251125_Myrzatay_Ayaulym_92003661_bachelor_thesis 

## **A. Create Virtual Environment**

#### macOS

```
python3 -m venv .thesis_venv 
source .thesis_venv/bin/activate
```

## **B. Install Dependencies**

```
pip install --upgrade pip 
pip install tensorflow == 2.13.0
pip install mediapipe == 0.10.21
pip install opencv-python==4.9
pip install numpy==1.24 
pip install scikit-learn==1.2
pip install matplotlib seaborn
pip install yt-dlp
```

## **C. Download Datasets**

This project uses two datasets that must be downloaded separatly due to licensing restrictions.

### WLASL Dataset

**Download**

1. Visit: https://github.com/dxli94/WLASL
2. Download `WLASL_v0.3.json` and video files
3. Extract to `data/raw_videos`

### ASL Dataset

**Download**

1. Visit: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. Extract to `data/asl_alphabet/`

## **D. Preprocessing**

After downloading datasets:

### Extract keypoints from WLASL videos

```
python3 scripts/5_preprocess_video_keypoints.py
```

### Extract keypoints from ASL Alphabet images

```
python3 scripts/2_preprocess_keypoints.py
```

### Augmented data

### WLASL

```
16_augment_wlasl_video.py
```

### ASL Alphabet

```
python3 scripts/16_augment_image.py
```
## **E. Train the models**
### WLASL 
```
python3 scripts/16_train_wlasl_video.py 
```
### ASL Alphabet 
```
14_train_hand_only_model.py
```
## **F. Evaluate the models**
### WLASL 
```
python3 scripts/7_evaluate_vide_model.py
```
### ASL Alphabet 
```
python3 scripts/8_evaluate_image_model.py
```
## **Run the realtime recognition builder**
```
python3 scripts/13_realtime_video_builder.py
```
- press 'q' to quit the display 
- press 'space' to freeze 
- press 'enter' when building the words using letters


**Citation**

* Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M., Kudlur, M., Levenberg, J., Monga, R., Moore, S., Murray, D. G., Steiner, B., Tucker, P., Vasudevan, V., Warden, P., ... Zheng, X. (2016). TensorFlow: A system for large-scale machine learning. In *Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation* (pp. 265-283). USENIX Association. [https://www.tensorflow.org/](https://www.tensorflow.org/)
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">Bradski, G. (2000). The OpenCV library. *Dr. Dobb's Journal of Software Tools, 25*(11), 120-125. [https://opencv.org/](https://opencv.org/)</span>
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">Camgöz, N. C., Hadfield, S., Koller, O., Ney, H., & Bowden, R. (2018). Neural sign language translation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 7784-7793). IEEE. [https://doi.org/10.1109/CVPR.2018.00812](https://doi.org/10.1109/CVPR.2018.00812)</span>
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">Chollet, F., & others. (2015). *Keras* [Computer software]. GitHub. [https://keras.io/](https://keras.io/)</span>
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">Fang, B., Co, J., & Zhang, M. (2017). DeepASL: Enabling ubiquitous and non-intrusive word and sentence-level sign language translation. In *Proceedings of the 15th ACM Conference on Embedded Network Sensor Systems* (pp. 1-13). ACM. [https://doi.org/10.1145/3131672.3131693](https://doi.org/10.1145/3131672.3131693)</span>
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">Graves, A., Mohamed, A., & Hinton, G. (2013). Speech recognition with deep recurrent neural networks. In *2013 IEEE International Conference on Acoustics, Speech and Signal Processing* (pp. 6645-6649). IEEE. [https://doi.org/10.1109/ICASSP.2013.6638947](https://doi.org/10.1109/ICASSP.2013.6638947)</span>
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., ... Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357-362. [https://doi.org/10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)</span>
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735-1780. [https://doi.org/10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)</span>
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">Kaggle. (n.d.). *ASL Alphabet dataset*. Kaggle Datasets. [https://www.kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)</span>
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision* (pp. 1459-1469). IEEE. [https://doi.org/10.1109/WACV45572.2020.9093512](https://doi.org/10.1109/WACV45572.2020.9093512)</span>
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M. G., Lee, J., Chang, W.-T., Hua, W., Georg, M., & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. *arXiv preprint arXiv:1906.08172*. [https://doi.org/10.48550/arXiv.1906.08172](https://doi.org/10.48550/arXiv.1906.08172)</span>
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">McKinney, W. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference* (Vol. 445, pp. 51-56). SciPy. [https://doi.org/10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a)</span>
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830. [https://scikit-learn.org/](https://scikit-learn.org/)</span>
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">Python Software Foundation. (n.d.). *PEP 8 – Style guide for Python code*. Python.org. [https://peps.python.org/pep-0008/](https://peps.python.org/pep-0008/)</span>
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data, 6*(1), Article 60. [https://doi.org/10.1186/s40537-019-0197-0](https://doi.org/10.1186/s40537-019-0197-0)</span>
* <span class="colour" style="color:oklch(0.9296 0.007 106.53)">Van Rossum, G., & Drake, F. L. (2009). *Python 3 reference manual*. CreateSpace.</span>

<br>
<br>
# 20251125_Myrzatay_Ayaulym_92003661_bachelor_thesis
