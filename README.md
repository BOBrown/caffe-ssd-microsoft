# caffe-ssd-microsoft

# SSD: Single Shot MultiBox Detector
By [Wei Liu](http://www.cs.unc.edu/~wliu/), [Dragomir Anguelov](http://research.google.com/pubs/DragomirAnguelov.html), [Dumitru Erhan](http://research.google.com/pubs/DumitruErhan.html), [Christian Szegedy](http://research.google.com/pubs/ChristianSzegedy.html), [Scott Reed](http://www-personal.umich.edu/~reedscot/), Cheng-Yang Fu, [Alexander C. Berg](http://acberg.com).

### Citing SSD

Please cite SSD in your publications if it helps your research:

    @article{liu15ssd,
      Title = {{SSD}: Single Shot MultiBox Detector},
      Author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
      Journal = {arXiv preprint arXiv:1512.02325},
      Year = {2015}
    }
	
### Installation
Requriements: CUDA8.0 and cudnn v5.x or  cudnn v6.x (NOTE: inappropriate cudnn version could lead to the failure of this project)

### Usage
  1. Clone or download this resository:<br>
  https://github.com/BOBrown/caffe-ssd-microsoft.git<br>
  2. Settings:<br>
  cd caffe-ssd-microsoft/windows<br>
  cp CommonSettings.props.example CommonSettings.props<br>
  Enter file named CommonSettings.props, change following items<br>
    \<CudaVersion\>8.0\<\/CudaVersion\>  //here is your CUDA version<br>
	\<UseCuDNN\>true\<\/UseCuDNN\>       // true stands for using cudnn lib<br>
	\<PythonSupport\>true\<\/PythonSupport\> // if you choose to support python, you must change <PythonDir> item with your python dir in your Windows<br>
	\<PythonDir\>C:Python27\<\/PythonDir\> <br>
  3. Compiling libcaffe:<br>
  NuGet will be loaded automatically when you compile libcaffe project.<br>
  Some compile errors could take place due to the inappropriate OpenCv version and glog version.<br>
  Trying update the corresponding NuGet package.<br>
  4. ssd_detect.cpp:<br>
  ssd_detect.cpp is a demo of using SSD lib. It will detect a binary images and store the detecton result in a txt file.<br>
  see detect.cpp for more details.<br>

### pretrain model
  Including Reduced VGG-16 model and simimages dataset pretrain(reflection,single-label,multi-label model) https://pan.baidu.com/s/1-_yv_QATJFyEAoY10Kry7w

### Questions

Holobo: 515765944@qq.com  