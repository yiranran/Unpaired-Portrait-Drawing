## Preprocessing steps

During training, face photos and drawings are aligned and have nose,eyes,lips mask detected. 

During test, the alignment step is optional and the masks are not needed.

### 1. Align, resize, crop images to 512x512

All training and testing images in our model are aligned using facial landmarks. And landmarks after alignment are needed in our code.

- First, 5 facial landmark for a face photo need to be detected (we detect using [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)(MTCNNv1)).

- Then, we provide a matlab function in `face_align_512.m` to align, resize and crop face photos (and corresponding drawings) to 512x512. Call this function in MATLAB to align the image to 512x512.
For example, for `ia_selfie_10515.jpg` in `example` dir, 5 detected facial landmark is saved in `example/ia_selfie_10515_facial5point.mat`. Call following in MATLAB:
```bash
load('example/ia_selfie_10515_facial5point.mat');
[trans_img]=face_align_512('example/ia_selfie_10515.jpg',facial5point,'example');
```

This will align the image and output aligned image  in `example` folder.
See `face_align_512.m` for more instructions.


### 2. Prepare nose,eyes,lips masks

In our work, we use the face parsing network in https://github.com/cientgu/Mask_Guided_Portrait_Editing to get nose,eyes,lips regions and then dilate the regions to make them cover these facial features (some examples are shown in `example` folder).

- The background masks need to be copied to `datasets/portrait_drawing/train/A(B)(_eyes)(_lips)`, and has the **same filename** with aligned face photos.  
