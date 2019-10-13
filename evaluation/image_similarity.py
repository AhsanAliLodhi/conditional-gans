import requests
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import minmax_scale

# Structural similarity by sikit-image package
from skimage.measure import compare_ssim as ssim

api_url = "https://api.deepai.org/api/image-similarity"
api_key = '5655d7b5-3af7-4897-b70d-b60648709bbf'


# Scaled cosine distance between features extracted by VGG of images
# Lower the score, similar the images
# Look into for definition of similarity, https://deepai.org/machine-learning-model/image-similarity

parser = argparse.ArgumentParser()

# Model configuration.
parser.add_argument('--real_dir', default='real', help='Directory for real images')
parser.add_argument('--fake_dir', default='fake', help='Directory for fake images')
parser.add_argument('--out', default='out.csv', help='File for outputs')


config = parser.parse_args()

# Vanilla MSE
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def get_similarity_scores(real_dir =  None,fake_dir = None, verbose = False):
    if real_dir is None:
        real_dir = config.real_dir
    if fake_dir is None:
        fake_dir = config.fake_dir
    vgg_scores = {}
    mse_score = {}
    ssim_score = {}
    if verbose:
        images = tqdm(os.listdir(real_dir))
    else:
        images = os.listdir(real_dir)
    for image in images:
        image2 = Image.open(fake_dir+'/'+image)
        image1 = Image.open(real_dir+'/'+image)
        image1.thumbnail(image2.size,Image.ANTIALIAS)
        image1 = np.array(image1)
        image2 = np.array(image2)
        r = requests.post(api_url,
            files={
                'image1': open(real_dir+'/'+image, 'rb'),
                'image2': open(fake_dir+'/'+image, 'rb'),
            },
            headers={'api-key':api_key }
        )
        vgg_scores[image] = r.json()['output']['distance']
        mse_score[image] = mse(image1,image2)
        ssim_score[image] = ssim(image1,image2,multichannel=True)
    return vgg_scores,mse_score,ssim_score


if __name__ == '__main__':
    vgg_scores,mse_score,ssim_score = get_similarity_scores(verbose = False)
    df = pd.DataFrame()
    df['vgg_scores'] = list(vgg_scores.values())
    df['mse_score'] = list(mse_score.values())
    df['mse_score'] = minmax_scale(df['mse_score'])
    df['ssim_score'] = list(ssim_score.values())
    df.index = list(mse_score.keys())
    df.to_csv(config.out)