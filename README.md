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

###Usage
  1. Clone or download this resository:
  https://github.com/BOBrown/caffe-ssd-microsoft.git
  2. Settings:
  cd caffe-ssd-microsoft/windows
  cp CommonSettings.props.example CommonSettings.props
  Enter file named CommonSettings.props, change following items
    <CudaVersion>8.0</CudaVersion>  //here is your CUDA version
	<UseCuDNN>true</UseCuDNN>       // true stands for using cudnn lib
	<PythonSupport>true</PythonSupport> // if you choose to support python, you must change <PythonDir> item with your python dir in your Windows
	<PythonDir>C:\Python27</PythonDir> 
  3. Compiling libcaffe:
  NuGet will be loaded automatically when you compile libcaffe project.
  some compile errors could be due to the inappropriate OpenCv version and glog version.
  trying update the corresponding NuGet package.
  4. ssd_detect.cpp:
  ssd_detect.cpp is a demo of using SSD lib. It will detect a binary images and store the detecton result in a txt file.
  see detect.cpp for more details.

###Questions:
Holobo: 515765944@qq.com  