# Conditional Gans

In this research project we work with several GANs models and try to make them conditionable.

# Basic Terminology

- Unaligned Model : When the input data for a model does not have to be pair wise the model is unaligned
- Aligned Model : When the input data for a model does not have to be pair wise the model is aligned
- Absolute Distance: In context of our specific dataset of images taken from several cameras from the front of a moving car, the absolute distance refers to the position of cameras, where the camera in middle is at 0. 
- Relative Distance: In context of our specific dataset of images taken from several cameras from the front of a moving car, the relative distance between camera_1 and camera_2 = absolute_distance(camera_2) - absolute_distance(camera_1)

# To set up the project

```
pip install requirements.txt
```

# Folder Structure Explanation

- conditional_models
    - aligned_models
        - stargan_relative_distance
        - stargan_absolute_distance
    - unaligned_models
        - stargan (https://github.com/yunjey/stargan)
        - cyclegans (https://junyanz.github.io/CycleGAN/)
- evaluation
    - image_similarity.py 
- misc
    - camera_demo_app
    - dataset_builder
    - filesrenamer

From above stargan_relative_distance, stargan_absolute_distance & stargan are made to work on the sythetically generated dataset from Carla from cameras of front of a moving car.

While cyclegans folder contains the variation of conditional cycle gan and was tylored to work on a weather dataset conditioned on weather class, we tested this model on both real and synthetic dataset. Cyclegan is for conversrion from and to two classes only.

Inside the evaluation folder, the file image_similarity.py is used to measure similarity scores between sets of two images and store the results in csv.


The folder Misc contains all utilities during this project, most of them are usable in importable and usable as independant packages.

camera_demo_app is a dockerized, flask based application web-app which works which uses models from stargan_relative_distance and provides a demo for how model results might look in action.

filerenamer is a python package to collection of files in a folder. (Look in to the internal readme for more details)

dataset_builder is a python package to sample and restructre images in to a specific datastructure. (use --help for more details)

# How to train
- ## Download Datasets

```
python download_data.py
```
This command will download the dataset for stargan_relative_distance & stargan_absolute_position & stargan and store it in the main directory for this project in folder called data/multicamera

- ## stargan_relative_distance & stargan_absolute_position & stargan
```
python main.py --mode train --rafd_crop_size 128 --n_critic 2 --rafd_image_dir ../../../data/multicamera/train --image_size 128 --c_dim 6 --sample_dir mymodel/samples --model_save_dir mymodel/model
```
|Attribute|Description|Possible values|
|---------|-----------|---------------|
|mode|The mode for running the script| train, test|
|image_size|Resize the image to this size always a square|>0|
|rafd_crop_size|Crop the image to this size always a square|>0 and < image_size|
|c_dim|number of classes|>0|
|n_critic|train generator every Kth iteration, 1 would mean train generator in each iteration|>0|
|rafd_image_dir|path to image directory|../../../data/multicamera/train|
|sample_dir|path to where intermediate samples are saved|mymodel/samples|
|model_save_dir|path to where intermediate models are saved|mymodel/model|



The data folder structure for above models is as follow.

 - Data
    - train
        - classA  - Where the name classA, classB, .. can be any other string
        - classB
        - classC
        - .
    - test
        - classA
        - classB
        - classC
        - .
- ## cyclegans
```
python train.py --dataroot datasets\weather --name test --model cycle_gan  --A_add_channel 2 --A_one_hot_encoding 0 --B_add_channel 2 --B_one_hot_encoding 1 --lambda_identity 0
```
The data folder structure for above models is as follow.

 - Data
    - train
        - trainA - Where the name trainA, trainB must be the exact folder names
        - trainB
    - test
        - testA - Where the name testA, testB must be the exact folder names
        - testB

# How to test
- ## Download Pretrained models

```
python download_models.py
```
This command will download the pretrained models for stargan_relative_distance & stargan_absolute_position and store it in the directory for each stargan_relative_distance and stargan_absolute_position.


- ## Pretrained Models and their descriptions
|Model|Cycle loss|classification loss in discriminator| classification loss in generator|input image size|Target|Remarks|
|----|-----|-----|-----|-----|-----|-----|
|multicamera|Yes|Yes|One way|256|Absolue position of target camera||
|multicamera_rel|Yes|Yes|One way|128|relative position of target camera||
|multicamera_rel_nocycle|No|Yes|One way|128|relative position of target camera||
|multicamera_rel_nocycle_nocal|No|No|One way|128|relative position of target camera||
|multicamera_rel_nocycle_calforboth|No|Yes|Two way|128|relative position of target camera||
|multicamera_rel_nocycle_calforboth_missing_camera|No|Yes|Two way|128|relative position of target camera|This model was trained on all cameras except camera2, and test set for this model shoul dbe only images of camera2|

- ## stargan_relative_distance

Use following to convert to all classes 

```
python main.py --mode test --batch_size 1  --rafd_crop_size 128  --image_size 128  --c_dim 6  --test_iters 60000  --rafd_image_dir ../../../data/multicamera/test   --model_save_dir multicamera_rel_nocycle/models  --result_dir multicamera_rel_nocycle/results
```
 Following are the new attributes used during testing when converting to all.

|Attribute|Description|Possible values|
|---------|-----------|---------------|
|batch_size|Size of batch for testing| >0|
|test_iters|this is the sufffix of the model to be loaded| 10000, 50000, 100000...|
|result_dir|Path where the result images are supposed to be stored |mymodel/result|




Use following to convert to one target class
```
python main.py --mode test --batch_size 1  --rafd_crop_size 128  --image_size 128  --c_dim 6  --test_iters 60000  --rafd_image_dir ../../../data/multicamera/test   --model_save_dir multicamera_rel_nocycle/models  --result_dir multicamera_rel_nocycle/results  --target_camera -0.3

```
Following are the new attributes used during testing when converting to one.

|Attribute|Description|Possible values|
|---------|-----------|---------------|
|target_camera|relative distance from current testing sample| -0.5, 1, 0 (This should give the same image as output)|

- ## stargan_absolute_position & stargan
Note : The pretrained model for absolute distance was trained on image size of 256

```
python main.py --mode test --batch_size 1  --rafd_crop_size 256  --image_size 256  --c_dim 6  --test_iters 70000  --rafd_image_dir ../../../data/multicamera/test   --model_save_dir multicamera/models  --result_dir multicamera/results
```




# How to evaluate
You can use image_similarity.py to compute similarity scores between predcited images and ground truth images.
Just put all the predicted images in one folder and put all ground truths in another, make sure the predicted images have same name as the ground truth images.

And then simply run the following.

```
python image_similarity.py --real_dir path_with_real_pictures --fake_dir path_with_fake_pictures --out scores.csv
```

Above command will generate a csv for you with following scores against each pair of image.

- Vanilla MSE (Mean Squared Error) : https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
- SSIM (Structural similarity score) : https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
- Cosine distance of VGG feature : https://deepai.org/machine-learning-model/image-similarity

This is also an independent importable module.

- ## Our evaluations
Our evaluations and scores for comparision of various models can be found in following two sheets.

|File| Descrition|
|----|-----------|
|performance_comparision_between_models.xlsx| This file contains comparision of all above mentioned pretrained models on vaious metrices|
| analysis_with_missing_camera.xlsx| This file compares the real camera2 images with prediction of camera2 images generated from model multicamera_rel_nocycle_calforboth when camera2 was not inclded in the training data|

You can find both of above files in evaluation directory.