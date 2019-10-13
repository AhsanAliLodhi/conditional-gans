
# coding: utf-8

# In[92]:


from robocarsdk.python import image as rbimg
from PIL import Image
import numpy as np
import os, random
import shutil
from tqdm import tqdm
from argparse import ArgumentParser

# In[50]:


def sample(count,source_dir, save_dir=None,seed=None,show=False, resize = 256):
    if seed is not None:
        random.seed(seed)
    for i in tqdm(range(count),desc="Samples"):
        rfname = np.random.choice(os.listdir(source_dir),replace=False)
        img = rbimg.load_image(source_dir+rfname)
        img = Image.fromarray(np.uint8(img))
        if resize is not None:
            img = img.resize((int(resize),int(resize)),Image.BICUBIC)
        if save_dir is not None:
            img.save(save_dir+rfname)
        if show:
            img.show()


# In[95]:


def generate_dataset(count,source_dir,save_dir,test_split=0.2,seed=None,source_dir_suffix ="stereo/centre/",resize = 256):
    if seed is not None:
        random.seed(seed)
    for classname in tqdm(os.listdir(source_dir),desc="Total"):
        if not os.path.isdir(source_dir+classname):
            continue
        try:
            shutil.rmtree(save_dir+'train/'+classname)
            shutil.rmtree(save_dir+'test/'+classname)
        except Exception as e:
            pass
        os.makedirs(save_dir+'train/'+classname)
        os.makedirs(save_dir+'test/'+classname)
        rsamples = np.random.choice(os.listdir(source_dir+classname+'/'+source_dir_suffix),size=count,replace=False)
        for index,sample in tqdm(enumerate(rsamples),desc=classname,total=len(rsamples)):
            try:
                img = rbimg.load_image(source_dir+classname+'/'+source_dir_suffix+sample)
                img = Image.fromarray(np.uint8(img))
            except Exception as e:
                img = Image.open(source_dir+classname+'/'+source_dir_suffix+sample)
            if resize is not None:
                img = img.resize((int(resize),int(resize)),Image.BICUBIC)
            if (index*1.0/count) < (1 - test_split):
                img.save(save_dir+'train/'+classname+'/'+sample)
            else:
                img.save(save_dir+'test/'+classname+'/'+sample)


# In[96]:
if __name__ == "__main__" :
    parser = ArgumentParser()

    parser.add_argument("-c", "--count", dest="count",
                        help="count of samples", default = 100)
    parser.add_argument("-r", "--resize", dest="resize",
                        help="value in pixels to resize the images to a square", default = None)
    parser.add_argument("-s", "--source_dir", dest="source_dir",
                        help="In case you want to generate a dataset this will be the"
                        "folder where all classes live other wise this will be folder with all images", default = "./")
    parser.add_argument("-d", "--dest_dir", dest="dest_dir",
                        help="In case you want to generate a dataset this will be the"
                        "folder where the train and test folders will be made other wise this will be output location with all images", default = None)
    parser.add_argument("-t", "--test_split", dest="test_split",
                        help="You must provide this argument in order to generate a dataset, should be a float between 0-1", default = None)
    parser.add_argument("-e", "--random_seed", dest="seed",
                        help="seed for suedorandom generator", default = None)
    parser.add_argument("-k", "--source_dir_suffix", dest="source_dir_suffix",
                        help="Used in case images in the source_dir are nested inside subfolders", default = "stereo/centre/")                   
    parser.add_argument("-i", "--show", dest="show",
                        help="can be used to show the image in an image viewer (wont work if you provide test_split)", default = False)

    args = parser.parse_args()

    if args.test_split is None:
        sample(int(args.count),str(args.source_dir), save_dir=args.dest_dir,seed=args.seed,show=bool(args.show),resize = args.resize)
    else:
        if args.source_dir is None:
            args.source_dir = "output/"
        generate_dataset(int(args.count),str(args.source_dir),str(args.dest_dir),
            test_split=float(args.test_split),seed=args.seed,source_dir_suffix =str(args.source_dir_suffix),resize = args.resize)

    """
    Example use:
    
    Source folder structure
    my_source_folder/
        /class1/
            subdir/
        /class2/
            subdir/
    
    Dest folder structure
    output/
        class1/
        class2/

    Generate dataset for 100 samples with 0.2 test split
    python dataset_builder.py -c 100 -s my_source_folder/ -d output/ -t 0.2 -k subdir/

    My usage
    python dataset_builder.py -c 1000 -s robocardata/ -d real/ -t 0.2 -r 256

    python dataset_builder.py -c 10 -s robocardata/sunny/stereo/centre/  -r 256 -i True

    python dataset_builder.py -c 1000 -s test/cgan/ -d augmented/ -t 0.2 -r 256  -k ""

    
    """





