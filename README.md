# YOLOv3-simple-object-detection

<p align="center">
    <img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/gifs/YOLOv3/traffic.gif" width=470></td>
</p>

This GitHub repository contains Jupyter notebooks that showcase simple object detection using YOLOv3 and Tiny YOLOv3 models. 
The notebooks demonstrate how to apply these models to both images and video files, and provide step-by-step instructions for implementing the object detection algorithm. 
Whether you're new to deep learning or just want to learn more about YOLOv3, this repository provides a great starting point for experimenting with object detection.

## Repo Overview:
* [YOLOv3_img_simple_object_detection.ipynb](https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/YOLOv3_img_simple_object_detection%20.ipynb) : Jupyter notebook for object detection on image files with YOLOv3 and YOLOv3-tiny.
* [YOLOv3_video_simple_object_detection.ipynb](https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/YOLOv3_video_simple_object_detection.ipynb) : Jupyter notebook for object detection on video files with YOLOv3 and YOLOv3-tiny.
* [configs](https://github.com/mohamedamine99/YOLOv3-simple-object-detection/tree/main/configs) : contains cfg files for yolov3 and tiny yolov3.
* [result imgs](https://github.com/mohamedamine99/YOLOv3-simple-object-detection/tree/main/result%20imgs) : contains results of object detection on image files.
* [result vids](https://github.com/mohamedamine99/YOLOv3-simple-object-detection/tree/main/result%20vids) : contains results of object detection on video files in .avi format.
* [gifs](https://github.com/mohamedamine99/YOLOv3-simple-object-detection/tree/main/gifs) : contains results of object detection on video files in GIF format..
* [test imgs](https://github.com/mohamedamine99/YOLOv3-simple-object-detection/tree/main/test%20imgs) : contains images of random scenes.
* [test vids](https://github.com/mohamedamine99/YOLOv3-simple-object-detection/tree/main/test%20vids) : contains videos of random scenes.
* [coco.names](https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/coco.names) : The "coco.names" file is a plain text file that contains the names of the 80 object classes in the Microsoft Common Objects in Context (COCO) dataset.


## Object detection on images : YOLOv3 vs YOLOv3-tiny

<div align="center">  
<table style="margin: 0 auto; border-style: none; width:100%">
  <tr>
    <td>YOLOv3</td>
    <td>YOLOv3-Tiny</td>
  </tr>

  <tr>
    <td><img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/result%20imgs/YOLOv3/2%20cats.PNG" width=280></td>
    <td><img src=https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/result%20imgs/YOLOv3_tiny/2%20cats.PNG width=280></td>
  </tr>
  
  
  <tr>
    <td><img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/result%20imgs/YOLOv3/2%20dogs.PNG" width=280></td>
    <td><img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/result%20imgs/YOLOv3_tiny/2%20dogs.PNG" width=280></td>
  </tr>
  
  <tr>
    <td><img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/result%20imgs/YOLOv3/people%20crossing%20the%20street.jpg" width=280></td>
    <td><img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/result%20imgs/YOLOv3_tiny/people%20crossing%20the%20street.jpg" width=280></td>
  </tr>
 
  <tr>
    <td><img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/result%20imgs/YOLOv3/highway.PNG" width=280></td>
    <td><img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/result%20imgs/YOLOv3_tiny/highway.PNG" width=280></td>
  </tr>
  
  <tr>
    <td><img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/result%20imgs/YOLOv3/nyc%20street.PNG" width=280></td>
    <td><img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/result%20imgs/YOLOv3_tiny/nyc%20street.PNG" width=280></td>
  </tr>

    
   <tr>
    <td><img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/result%20imgs/YOLOv3/street%202.PNG" width=280></td>
    <td><img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/result%20imgs/YOLOv3_tiny/street%202.PNG" width=280></td>
    </tr>
  
      
  
</table>
</div>

## Object detection on video files : YOLOv3 vs YOLOv3-tiny

<div align="center">  
<table style="margin: 0 auto; border-style: none; width:100%">
  <tr>
    <td>YOLOv3</td>
    <td>YOLOv3-Tiny</td>
  </tr>

  <tr>
    <td><img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/gifs/YOLOv3/traffic.gif" width=280></td>
    <td><img src=https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/gifs/YOLOv3_tiny/traffic.gif width=280></td>
  </tr>
  
  
  <tr>
 <td align="center">
<img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/gifs/YOLOv3/dog%20video%202.gif" width=170></td>
<td align="center">
<img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/gifs/YOLOv3_tiny/dog%20video%202.gif" width=170></td>
  </tr>
  
  <tr>
<td align="center">
<img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/gifs/YOLOv3/cat%20video%202.gif" width=170></td>
<td align="center"> <img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/gifs/YOLOv3_tiny/cat%20video%202.gif" width=170></td>
  </tr>
      
  
</table>
</div>

## Interpreting results : YOLOv3 vs YOLOv3-tiny

<p align="center">
    <img src="https://github.com/mohamedamine99/YOLOv3-simple-object-detection/blob/main/readme%20imgs/YOLO%20performance.PNG" width=470></td>
</p>

The **mAP** (mean average precision) is a metric used to evaluate the performance of object detection models. It measures the accuracy of the model in terms of both precision (the fraction of true positives out of all positive predictions) and recall (the fraction of true positives out of all actual positives).

**GFlops** (GigaFLOPS) is a measure of computational power, specifically the number of floating-point operations per second that a computer or device can perform. In the context of deep learning models, GFlops are often used as a measure of the computational complexity of the model.

* YOLOv3-320 has a mAP of 55.3 for 65.86 GFlops which means the model is able on average to correctly detect and identify 55.3% of the objects present in a given image and require a compuational power of 65.86 billion floating point operations per second. We can conclude that the object detection model is relatively accurate and moderately complex. It may be suitable for some applications but may not be efficient enough for applications with strict real-time performance requirements if we dont have access to high performing GPUs, which is further confirmed by the results in the above section.

* YOLOv3-tiny has a mAP of 33.1 for 5.56 GFlops which means the model is able on average to correctly detect and identify 33.1% of the objects present in a given image and require a compuational power of 5.56 billion floating point operations per second. We can conclude that the object detection model is is moderately accurate and relatively simple compared to the YOLOv3-320.Therefore it is h

In summary, YOLOv3 and YOLOv3 Tiny are both object detection models, but YOLOv3 Tiny is a smaller and faster version of YOLOv3, designed for use in scenarios where real-time object detection is required on lower-end hardware. Although YOLOv3 Tiny sacrifices some accuracy compared to YOLOv3, it can still achieve good performance in many real-world applications while using fewer computational resources.
