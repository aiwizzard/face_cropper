# face_cropper
A python script for cropping faces from images or from videos. Need to crop faces from an image or video? use this.


## How to use


### 1: crop from image

```
git clone https://github.com/Ajmal-K/face_cropper.git
cd face_cropper
python crop_faces.py --type image --image image.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
--output output_path
```

### 1: crop from video

```
git clone https://github.com/Ajmal-K/face_cropper.git
cd face_cropper
python crop_faces.py --type video --video video.mp4 --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
--output output_path
