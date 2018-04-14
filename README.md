# Basic-ALPR
Number Recognition on License Plate using Python, OpenCV, Keras
Data training: MNIST Handwritten digit

## Contributors
1. Agni Wira Buana
2. Azmi (https://github.com/azminajid)
3. M Izzan

## How to Use
```
python find.py foto/[nama file]
```

## Output
Menggunakan 2 metode Template Matching dan CNN LeNet
![alpr](https://raw.githubusercontent.com/awbuana/basic-ALPR/master/ALPR.JPG)

## Packages
```
- keras 2.0.8
- opencv-python 3.3.0.10
- numpy 1.13.3
- scipy 1.0.0
- tensorflow 1.2.1
```

## Method
```
- ubah ke grayscale
- apply bilateralfilter
- Top-hat transform
- thresholding 
- erotion
- crop vertikal lokasi plat nomor dengan Horiozntal Histogram
- crop horizontal lokasi plat nomor dengan Vertical Histogram
- number segmentation menggunakan contour
- crop setiap nomer
- masukan ke program template matching atau CNN(Keras)
```
