from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision import datasets
from PIL import Image
import torch
import os
import random
try:
    from camera_info import get_abs_pos
except Exception as e:
    from stargan.camera_info import get_abs_pos

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images





class multi_camera_dataset(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(multi_camera_dataset, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        
        rand_target_class = random.choice(self.classes)
        rand_target_class_idx =self.class_to_idx[rand_target_class]
        if path.rfind("/train\\") > -1:
            target_path = path[:path.rfind("/train\\")+7] + rand_target_class + path[path.rfind("\\"):]
        else:
            target_path = path[:path.rfind("/test\\")+6] + rand_target_class + path[path.rfind("\\"):]
        
        org_pos = get_abs_pos(self.classes[original_tuple[1]])
        target_pos = get_abs_pos(rand_target_class)
        
        rel_dist = target_pos - org_pos

        target_img = self.loader(target_path)
        if self.transform is not None:
            target_img = self.transform(target_img)
        #print(target_img.shape)
        
        # print(path[path.rfind("/train\\")+7:])
        # print(target_path[target_path.rfind("/train\\")+7:])
        # print(org_pos,target_pos,rel_dist)
        # make a new tuple that includes original and the path
        tuple_target_and_path = (original_tuple + (path,org_pos,target_img,rand_target_class_idx,target_path,target_pos,rel_dist))
        return tuple_target_and_path


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    # if mode == 'train':
    #     transform.append(T.RandomHorizontalFlip())
    # Ahsan: flipped resize and crop order
    transform.append(T.Resize(image_size))
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = multi_camera_dataset(image_dir, transform) #Ahsan: using extention to debug filenames while training, ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader