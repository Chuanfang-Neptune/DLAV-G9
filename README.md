# DLAV 
Group 9

## Milestone 1
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zyZLkp4B0IAd13fslAqpC1DfcLYskfV4?usp=sharing)

[![Github Milestone-1](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBkPSJNMTIgMGMtNi42MjYgMC0xMiA1LjM3My0xMiAxMiAwIDUuMzAyIDMuNDM4IDkuOCA4LjIwNyAxMS4zODcuNTk5LjExMS43OTMtLjI2MS43OTMtLjU3N3YtMi4yMzRjLTMuMzM4LjcyNi00LjAzMy0xLjQxNi00LjAzMy0xLjQxNi0uNTQ2LTEuMzg3LTEuMzMzLTEuNzU2LTEuMzMzLTEuNzU2LTEuMDg5LS43NDUuMDgzLS43MjkuMDgzLS43MjkgMS4yMDUuMDg0IDEuODM5IDEuMjM3IDEuODM5IDEuMjM3IDEuMDcgMS44MzQgMi44MDcgMS4zMDQgMy40OTIuOTk3LjEwNy0uNzc1LjQxOC0xLjMwNS43NjItMS42MDQtMi42NjUtLjMwNS01LjQ2Ny0xLjMzNC01LjQ2Ny01LjkzMSAwLTEuMzExLjQ2OS0yLjM4MSAxLjIzNi0zLjIyMS0uMTI0LS4zMDMtLjUzNS0xLjUyNC4xMTctMy4xNzYgMCAwIDEuMDA4LS4zMjIgMy4zMDEgMS4yMy45NTctLjI2NiAxLjk4My0uMzk5IDMuMDAzLS40MDQgMS4wMi4wMDUgMi4wNDcuMTM4IDMuMDA2LjQwNCAyLjI5MS0xLjU1MiAzLjI5Ny0xLjIzIDMuMjk3LTEuMjMuNjUzIDEuNjUzLjI0MiAyLjg3NC4xMTggMy4xNzYuNzcuODQgMS4yMzUgMS45MTEgMS4yMzUgMy4yMjEgMCA0LjYwOS0yLjgwNyA1LjYyNC01LjQ3OSA1LjkyMS40My4zNzIuODIzIDEuMTAyLjgyMyAyLjIyMnYzLjI5M2MwIC4zMTkuMTkyLjY5NC44MDEuNTc2IDQuNzY1LTEuNTg5IDguMTk5LTYuMDg2IDguMTk5LTExLjM4NiAwLTYuNjI3LTUuMzczLTEyLTEyLTEyeiIvPjwvc3ZnPg==)](https://github.com/Qingbeixi/darknet)    <img src="https://google.github.io/mediapipe/images/mediapipe_small.png" alt="drawing" style="width:120px;"/>   <img src="https://res.cloudinary.com/practicaldev/image/fetch/s--nV6F2cwB--/c_imagga_scale,f_auto,fl_progressive,h_900,q_auto,w_1600/https://dev-to-uploads.s3.amazonaws.com/i/5482jc4y2k1ksvvz2hzd.png" alt="drawing" style="width:30px;"/> 

**Instructions**
1. Run a JavaScript to obtain the frame from our webcam from the browser.
2. Convert the returned base64 to an OpenCV image format, Numpy ndarray with a shape (h, w, c), RGB
3. Run YOLOv4 to detect objects in the returned frame from browser, here we only keep objects with human label (label 0) 
4. Iterate through all the detected human, and crop the original image only keep the objects inside each bounding boxes.
5. Use mediapipe to inference hand landmarks (key points) in each croped image with human objects.
6. Build a KNN model to implement hand pose recognition, here we build a small data set with `eight_sign`, `five_sign`, `four_sign`, `ok`, `one_sign`, `six_sign`, `spider`, `ten_sign`, `three_sign`, `two_sign`, 10 different hand gestures. 
7. In order to accelerate our reference speed, a embedding method is introdeced. The Dimensions of landmarks are reduced from 21 points to 5 points.
![](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)
8. When the detected person is doing the right hand gesture, we will render an alpha image only with bounding boxes.

This is how we done to detect the Point of Interests (POI).

 
## Milestone 2
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I9wODc3xlLdw8inOF9CIJWf_UpLhsf1j#scrollTo=mrJhOqvuBcJI)

![Deep Sort](https://miro.medium.com/max/1400/0*-S2EkuGhkP9tp9It.JPG)

Based on milestone1, we added the `DeepSort` algorithm for tracking points of interest. 

**Something about the Coordinates**

For a openCV picture, the 0-dim is y-axis, the 1-dim is x-axis, the 2-dim is channel
1. xywh -> xc, yc, width, height
2. xyxy -> left, top, right, bottom
3. xyah -> xc, yc, w/h, height
4. xysr -> xc, yc, square, h/w

**Instructions**

First, run of the cells one by one, and we upload all the required files to Google Drive. Use the `gdown` with a `id` parameter to download given file from Google Drive.
> For example:  `gdown 1dWOhStdDXK_kBefa9t9hDYLZ6kyrBwgP`

By giving our app a hand gesture, you will be the POI (point of interest), then our app will keep tracking you whenever you are in or out of the camera. If you are out of the camera too long, our app will count your leaving time, and if run out of time, the app will re-initialize and try to find a new POI.

Here you can set the gesture you want and tuning the max leaving time, by give the variables `target_pose` and `max_count` some new values (See cell below).
>Supported hand poses are `eight_sign`, `five_sign`, `four_sign`, `ok`, `one_sign`, `six_sign`, `spider`, `ten_sign`, `three_sign`, `two_sign`.

Our milestone 1&2 is quite light-weighted. 🚀 Enjoy 🍻!

## Milestone 3
[![Github Milestone-1](data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBkPSJNMTIgMGMtNi42MjYgMC0xMiA1LjM3My0xMiAxMiAwIDUuMzAyIDMuNDM4IDkuOCA4LjIwNyAxMS4zODcuNTk5LjExMS43OTMtLjI2MS43OTMtLjU3N3YtMi4yMzRjLTMuMzM4LjcyNi00LjAzMy0xLjQxNi00LjAzMy0xLjQxNi0uNTQ2LTEuMzg3LTEuMzMzLTEuNzU2LTEuMzMzLTEuNzU2LTEuMDg5LS43NDUuMDgzLS43MjkuMDgzLS43MjkgMS4yMDUuMDg0IDEuODM5IDEuMjM3IDEuODM5IDEuMjM3IDEuMDcgMS44MzQgMi44MDcgMS4zMDQgMy40OTIuOTk3LjEwNy0uNzc1LjQxOC0xLjMwNS43NjItMS42MDQtMi42NjUtLjMwNS01LjQ2Ny0xLjMzNC01LjQ2Ny01LjkzMSAwLTEuMzExLjQ2OS0yLjM4MSAxLjIzNi0zLjIyMS0uMTI0LS4zMDMtLjUzNS0xLjUyNC4xMTctMy4xNzYgMCAwIDEuMDA4LS4zMjIgMy4zMDEgMS4yMy45NTctLjI2NiAxLjk4My0uMzk5IDMuMDAzLS40MDQgMS4wMi4wMDUgMi4wNDcuMTM4IDMuMDA2LjQwNCAyLjI5MS0xLjU1MiAzLjI5Ny0xLjIzIDMuMjk3LTEuMjMuNjUzIDEuNjUzLjI0MiAyLjg3NC4xMTggMy4xNzYuNzcuODQgMS4yMzUgMS45MTEgMS4yMzUgMy4yMjEgMCA0LjYwOS0yLjgwNyA1LjYyNC01LjQ3OSA1LjkyMS40My4zNzIuODIzIDEuMTAyLjgyMyAyLjIyMnYzLjI5M2MwIC4zMTkuMTkyLjY5NC44MDEuNTc2IDQuNzY1LTEuNTg5IDguMTk5LTYuMDg2IDguMTk5LTExLjM4NiAwLTYuNjI3LTUuMzczLTEyLTEyLTEyeiIvPjwvc3ZnPg==)](https://github.com/Chuanfang-Neptune/DLAV-G9) 

In milestone 1&2, we implement Object Detection, Keypoint Detection, KNN Classification, and Multi Object Tracking (MOT).
<img src="https://store.segway.com/media/catalog/product/cache/1be073bb625205c2a3aab025b4fe3368/l/o/loomo_708x708.png" alt="drawing" style="width:200px;"/>

**Code Structure**

```
milestone3
│───deepsort.py
│───detector.py
│───client.py  
│───requirements.txt
│
│───hand_knn
│   │───embedder.py
│   │───hand_detect.py
│   └───dataset_embedded.npz # KNN embedding data set with 5-dim and 10 classes
│   
│───deep
│   │───fastreid
│   │───checkpoint # Download the checkpoint first
│   │───feature_extractor.py
│   └───.....
│
└───sort
    │───detection.py
    │───track.py
    │───tracker.py
    └───.....
```
1. `client.py` is the main interferece of our application, which receive frames from loomo.
2. `detector.py` is the core part of our application. There is a `forward()` function inside, which will processing the frames from client and return the tracked points(x, y) and the flags.  Note that each of them is a `python <list> type`.
3. `deep/checkpoint` is the directory for fast-ReId checkpoint(weights), **here is the link** [Link to fast-ReID checkpooint](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50.pth) 
4. 	`hand_knn/dataset_embedded.npz` is the embedded dataset only with `5-dims` (`21 landmarks` --> `5 representations`)
5. `requirements.txt` is all the packages used in our project. Please bulid a new python environment with this file to avoid env configuration error.


## Enjoy our 🤖️/🚗


