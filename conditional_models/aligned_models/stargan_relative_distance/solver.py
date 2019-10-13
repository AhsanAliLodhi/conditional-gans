try:
    from model import Generator
except Exception as e:
    from stargan.model import Generator
try:
    from model import Discriminator
except Exception as e:
    from stargan.model import Discriminator
try:
    from camera_info import get_abs_pos,get_rel_dists
except Exception as e:
    from stargan.camera_info import get_abs_pos,get_rel_dists

import ntpath
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import os
import time
import datetime

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters
        self.target_camera = config.target_camera

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, 1, self.g_repeat_num) # Ahsan: generator takes one extra dimension instead of self.c_dim which distance instead of actual classes
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        try:
            from logger import Logger
        except Exception as e:
            from stargan.logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org, x_real_paths,org_pos,x_target,label_target,x_target_paths,target_pos,rel_dist  = next(data_iter) #Ahsan: path is not returned from regular image folder class

        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
        rel_dist_list = []
        for camera in range(self.c_dim):
            t = torch.zeros(c_org.size(0), 1)
            for j in range(t.size(0)):
                t[j][0]=  get_abs_pos(data_loader.dataset.classes[camera]) - org_pos[j]
                t = t.to(self.device)
            rel_dist_list.append(t)

        #print(x_fixed.shape,c_fixed_list, x_real_paths)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org, x_real_paths,org_pos,x_target,label_target,x_target_paths,target_pos,rel_dist = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org, x_real_paths,org_pos,x_target,label_target,x_target_paths,target_pos,rel_dist = next(data_iter)

            # print(label_org)
            # print(label_target)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]
            label_trg = label_target #Ahsan: overwriting random target labels with aligned target labels
            #print(label_org.size(0),rand_idx,label_org,label_trg)

            #print("Read : ",x_real.shape,label_org.numpy(),rand_idx.numpy(),label_trg.numpy(), x_real_paths)

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)
            t = torch.zeros(label_org.size(0), 1)
            for j in range(t.size(0)):
                t[j][0]=rel_dist[j]
            rel_dist = t
            #print(c_trg,rel_dist)
            x_real = x_real.to(self.device)           # Input source images.
            x_target = x_target.to(self.device)       # Input target images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.
            rel_dist = rel_dist.to(self.device)       # target relative distance
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            #print("Train Disc")
            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)
            #print("Disc returns with real ",out_src.shape, out_cls,d_loss_real,d_loss_cls)

            # Compute loss with fake images.
            #print(x_real.type(),x_real.size(),rel_dist.type(),rel_dist.size())
            x_fake = self.G(x_real, rel_dist) #Ahsan: instead of target classes in one hot encoded c_trg, we use relative distances #rel_dist
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake_cls = self.classification_loss(out_cls, label_trg, self.dataset) #Ahsan: added classification loss for fake image to enforce correct classification of generated image
            d_loss_fake = torch.mean(out_src)
            #print("Disc returns with fake ",x_fake.shape,out_src.shape, out_cls,d_loss_fake)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)
            #print("Disc returns with x_hat ", out_src.shape,d_loss_gp)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake  + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp #Ahsan: no longer this loss + d_loss_fake_cls
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_fake_cls'] = d_loss_fake_cls.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            if (i+1) % self.n_critic == 0:
                #print("Train Generator")
                # Original-to-target domain.
                x_fake = self.G(x_real, rel_dist) # Ahsan: using relative distance instead of class targets c_trg #rel_dist
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src) # real or not
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset) # correct class or not
                g_loss_rec2 = torch.mean(torch.abs(x_fake - x_target)) # Ahsan: recon loss for generated image
                
                # Target-to-original domain.
                # x_reconst = self.G(x_fake, c_org) # Ahsan: from target camera, go in the opposite direction but same distance #-1 * rel_dist
                # g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Replaced regular cycle loss with true loss of generator in reversed direction
                x_reconst = self.G(x_target, -1 * rel_dist) # Ahsan: from target camera, go in the opposite direction but same distance #-1 * rel_dist
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # if both ways are same then may be both ways should be treated same
                out_src2, out_cls2 = self.D(x_reconst)
                g_loss_fake2 = - torch.mean(out_src2) # real or not
                g_loss_cls2 = self.classification_loss(out_cls2, label_org, self.dataset) # correct class or not

                # Backward and optimize.
                #g_loss = g_loss_fake + self.lambda_rec * (g_loss_rec)  + g_loss_rec2 + self.lambda_cls * g_loss_cls # Ahsan: use recon loss for target image
                # Since g_loss_rec is not cycle loss any more we treet both reconstructions as normal loss
                g_loss = g_loss_fake + g_loss_fake2 + g_loss_rec  + g_loss_rec2 + self.lambda_cls * g_loss_cls + self.lambda_cls * g_loss_cls2 # Ahsan: use recon loss for target image
                
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_rec2'] = g_loss_rec2.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in rel_dist_list: #: Ahsan: testing now uses relative target labels instead of target class c_fixed_list #rel_dist_list
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_multi(self):
        """Train StarGAN with multiple datasets."""        
        # Data iterators.
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)

        # Fetch fixed inputs for debugging.
        x_fixed, c_org = next(celeba_iter)
        x_fixed = x_fixed.to(self.device)
        c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
        c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
        zero_celeba = torch.zeros(x_fixed.size(0), self.c_dim).to(self.device)           # Zero vector for CelebA.
        zero_rafd = torch.zeros(x_fixed.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
        mask_celeba = self.label2onehot(torch.zeros(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
        mask_rafd = self.label2onehot(torch.ones(x_fixed.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for dataset in ['CelebA', 'RaFD']:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                
                # Fetch real images and labels.
                data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter
                
                try:
                    x_real, label_org = next(data_iter)
                except:
                    if dataset == 'CelebA':
                        celeba_iter = iter(self.celeba_loader)
                        x_real, label_org = next(celeba_iter)
                    elif dataset == 'RaFD':
                        rafd_iter = iter(self.rafd_loader)
                        x_real, label_org = next(rafd_iter)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'CelebA':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                    zero = torch.zeros(x_real.size(0), self.c2_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                    c_org = torch.cat([c_org, zero, mask], dim=1)
                    c_trg = torch.cat([c_trg, zero, mask], dim=1)
                elif dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)
                    zero = torch.zeros(x_real.size(0), self.c_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                    c_org = torch.cat([zero, c_org, mask], dim=1)
                    c_trg = torch.cat([zero, c_trg, mask], dim=1)

                x_real = x_real.to(self.device)             # Input images.
                c_org = c_org.to(self.device)               # Original domain labels.
                c_trg = c_trg.to(self.device)               # Target domain labels.
                label_org = label_org.to(self.device)       # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)       # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org, dataset)

                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()
            
                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i+1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg, dataset)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training info.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i+1, self.num_iters, dataset)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_celeba_list:
                        c_trg = torch.cat([c_fixed, zero_rafd, mask_celeba], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    for c_fixed in c_rafd_list:
                        c_trg = torch.cat([zero_celeba, c_fixed, mask_rafd], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
    
    def pass_from_generator(self,image,dist):
        image = self.rafd_loader.dataset.transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        result = self.G(image, torch.tensor([[dist]]).to(self.device))
        trans = T.ToPILImage()
        result = self.denorm(result.data.cpu())
        result = trans(result.squeeze())
        return result

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        
        with torch.no_grad():
            for i, (x_real, c_org, x_real_paths,org_pos,x_target,label_target,x_target_paths,target_pos,rel_dist) in enumerate(data_loader): #x_real, c_org
                #print(x_real_paths)
                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                x_target = x_target.to(self.device)
                #c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
                # print(org_pos)
                # if self.target_camera is not None:
                #     org_pos = [pos for pos in org_pos if org_pos == self.target_camera]
                #     print(org_pos)
                # if self.target_camera is None:
                rel_dists = get_rel_dists(org_pos,target_pos = self.target_camera)#abs_pos = [0,-0.4,-0.8,-1.2,-1.6,-2.0,-2.2,-2.4,-2.8,-3.0]
                
                # else:
                #     rel_dists = [self.target_camera - org_pos[0]]
                #print(rel_dists)
                # t = torch.zeros(x_real.size(0), 1)
                # for j in range(t.size(0)):
                #     print(t.size(),rel_dists[j])
                #     t[j][0]=rel_dists[j][1]
                # rel_dists = t
                rel_dists = [dist[1][:,np.newaxis].float().to(self.device) for dist in rel_dists]
                
                # Translate images.
                if self.target_camera is None:
                    x_fake_list = [x_real]
                    t_f_name = None
                else:
                    x_fake_list = []
                    t_f_name = ntpath.basename(x_real_paths[0])
                for dist in rel_dists:
                    #print(x_real.type(),x_real.size(),dist.type(),dist.size())
                    r = self.G(x_real, dist)
                    #print(r.size(),x_target.size(),torch.cat((r, x_target), 0).size())

                    x_fake_list.append(r)

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                if t_f_name is None:
                    result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                else:
                    result_path = os.path.join(self.result_dir, t_f_name)
                    
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
    


    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.celeba_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
                c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
                zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(self.device)            # Zero vector for CelebA.
                zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))