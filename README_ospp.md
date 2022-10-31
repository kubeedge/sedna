# README
------
I modified sedna source code to get my algorithm integrated into lifelong learning. I would display the framework and illustrute what I modified.
## Framework
![](framework_with_gan_selftaughtlearning.png)

## What I Modified

1. Delete redundant annotations.
2. Replce `print` with `logger`.
3. Integrate my algorithm into `sedna.lib.sedna.algorithms.unseen_task_processing.unseen_task_processing.py`.
4. Replace absolute path with relative path.
5. Remove redundant code.
6. Provide link for developers to download trained model.

## What I Refer to
I refer to [FastGAN-pytorch](https://github.com/odegeasslbc/FastGAN-pytorch) to implement my GAN module.

## Model to Download
In `GAN.lpips.weights`, developers may need the pre-trained model.    
Also, the trained GAN model and trained encoder model is also provided.      
Click here [link](https://drive.google.com/drive/folders/1IOQCQ3sntxrbt7RtJIsSlBo0PFrR7Ets?usp=share_link) to download.