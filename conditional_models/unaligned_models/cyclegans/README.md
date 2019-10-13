Repository for training Conditional GANs which generates multiple weathers from a single generator network

Example command for training.py testing:

`python train.py --dataroot datasets\weather --name test --model cycle_gan --gpu_ids -1 --A_add_channel 2 --A_one_hot_encoding 0 --B_add_channel 2 --B_one_hot_encoding 1 --lambda_identity 0`
