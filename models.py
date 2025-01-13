import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from fft_utils import *
from misc_utils import *
import copy
from pathlib import Path
from optimizer import IIPG
"""Kernel size 11: Receptive Field를 위해 어쩔 수 없음"""
"""features_out 24에서 32으로 바꾸었음"""
DEFAULT_OPTS = {'kernel_size':11,
                'features_in':1,
                'features_out':16,
                'do_prox_map':True,
                'pad':11,
                'vmin':-1.0,'vmax':1.0,
                'lamb_init':1.0,
                'num_act_weights':31,
                'init_type':'linear',
                'init_scale':0.04,
                'sampling_pattern':'cartesian',
                'num_stages':10,
                'seed':1,
                'optimizer':'adam','lr':1e-4,
                'activation':'rbf',
                'loss_type':'complex', 
                'momentum':0.,
                'error_scale':10,
                'loss_weight':1}

class SirenActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w): 
        """Forward pass for Siren Activation
        Weight normalization 구현 
        
        Paramters:
        -------------
        ctx:
        input: torch tensor (NxCxHxW) Batch size, Channel, Height, Width 
        w: Torch tensor (1 x C x 1 x 1 x # of RBF kernels): Dimension for broadcasting 
            weights for the SIREN activation 
        
        Returns:
        --------
        torch tensor: linear weight combination of SIREN activation 
        """
        num_act_weights = w.shape[-1] # siren activation의 weight 개수를 의미 
        output = input.new_zeros(input.shape) # output을 정의할 건데 output하고 input의 차원이 일치해야 함
        siren_grad_input = input.new_zeros(input.shape) # gradient를 정의
        
        for i in range(num_act_weights):	 
                tmp = w[:,:,:,:,i] * torch.sin(w[:,:,:,:,i]*input)
                output += tmp
                siren_grad_input += w[:,:,:,:,i]*w[:,:,:,:,i]*torch.cos(w[:,:,:,:,i]*input)

        ctx.save_for_backward(input, w, siren_grad_input) # Forward propagation에서 gradient 계산 
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, w, siren_grad_input = ctx.saved_tensors # input, weight, and gradients
        num_act_weights = w.shape[-1]

        grad_input = grad_output * siren_grad_input #upstream gradient * local gradient 
        
        grad_w = w.new_zeros(w.shape)
        
        for i in range(num_act_weights):
            cos_term = input * torch.cos(w[:,:,:,:,i] * input)  # x * cos(w * x)

            # Combine terms and sum over batch, height, and width dimensions # 채널별 activation 합 
            tmp = (grad_output * cos_term).sum((0, 2, 3)) 
            grad_w[:,:,:,:,i] = tmp.view(w.shape[0:-1])  # Reshape to match weight shape
        return grad_input, grad_w
        
class SirenActivation(nn.Module):
    """SIREN activation function with trainable weight"""
    def __init__(self, **kwargs):
        super().__init__()
        self.options = kwargs

        """Weight initializations from original SIREN paper"""
        if kwargs.get('is_first_layer', True):
            bound = 1/kwargs['num_act_weights']
            w_0 = np.random.uniform(-bound, bound, kwargs['num_act_weights'])
        else:
            bound = np.sqrt(6.0 / kwargs['num_act_weights'])
            w_0 = np.random.uniform(
                -bound/30,
                bound/30,
                kwargs['num_act_weights']
            ).astype(np.float32)

        w_0 = np.reshape(w_0, (1,1,1,1, kwargs['num_act_weights'])) # n수에 맞게 복제 
        w_0 = np.repeat(w_0, kwargs['features_out'], axis=1) # channel 수에 맞게 복제

        w_0 = 30.0 * w_0

        self.w = torch.nn.Parameter(torch.from_numpy(w_0))
        self.siren_act = SirenActivationFunction.apply # apply : 사용자 정의 autograd 함수를 호출하는 함수 
    def forward(self, x):
        output = self.siren_act(x, self.w)
        return output

"""추가 fitting network by Spatial Encoding by NeRF"""

class NeRFFittingNetwork(pl.LightningModule):
    def __init__(self, sidelen=320, hidden_features=256, hidden_layers=3, lr=1e-4):
        super().__init__()
        self.lr = lr
        
        # Network layers
        self.layers = nn.ModuleList()
        
        # First layer (2D coordinates to hidden features)
        first_layer_options = {
            'features_in': 2,
            'features_out': hidden_features,
            'num_act_weights': hidden_features,
            'is_first_layer': True
        }
        self.layers.append(nn.Linear(2, hidden_features))
        self.layers.append(SirenActivation(**first_layer_options))
        
        # Hidden layers
        for _ in range(hidden_layers):
            hidden_layer_options = {
                'features_in': hidden_features,
                'features_out': hidden_features,
                'num_act_weights': hidden_features,
                'is_first_layer': False
            }
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.layers.append(SirenActivation(**hidden_layer_options))
            
        # Final layer (hidden features to 2 channels for real and imaginary)
        final_layer_options = {
            'features_in': hidden_features,
            'features_out': 2,
            'num_act_weights': 2,
            'is_first_layer': False
        }
        self.layers.append(nn.Linear(hidden_features, 2))
        self.layers.append(SirenActivation(**final_layer_options))
        
    def get_mgrid(self, sidelen, dim=2):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
        sidelen: int
        dim: int'''
        tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, dim)
        print(sidelen)
        return mgrid
        
    def forward(self, coords):
        x = coords
        
        for layer in self.layers:
            x = layer(x)
            
        return x

class RBFActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w, mu, sigma):
        """ Forward pass for RBF activation

        Parameters:
        ----------
        ctx: 
        input: torch tensor (NxCxHxW)
            input tensor
        w: torch tensor (1 x C x 1 x 1 x # of RBF kernels)
            weight of the RBF kernels
        mu: torch tensor (# of RBF kernels)
            center of the RBF
        sigma: torch tensor (1)
            std of the RBF

        Returns:
        ----------
        torch tensor: linear weight combination of RBF of input
        """
        num_act_weights = w.shape[-1]
        output = input.new_zeros(input.shape)
        rbf_grad_input = input.new_zeros(input.shape)
        for i in range(num_act_weights):
            tmp = w[:,:,:,:,i] * torch.exp(-torch.square(input - mu[i])/(2* sigma ** 2))
            output += tmp
            rbf_grad_input += tmp*(-(input-mu[i]))/(sigma**2)
        del tmp
        ctx.save_for_backward(input,w,mu,sigma,rbf_grad_input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w, mu, sigma, rbf_grad_input = ctx.saved_tensors
        num_act_weights = w.shape[-1]

        #if ctx.needs_input_grad[0]:
        grad_input = grad_output * rbf_grad_input #upstream gradient * local gradient 

        #if ctx.need_input_grad[1]:
        grad_w = w.new_zeros(w.shape)
        for i in range(num_act_weights):
            tmp = (grad_output*torch.exp(-torch.square(input-mu[i])/(2*sigma**2))).sum((0,2,3))
            grad_w[:,:,:,:,i] = tmp.view(w.shape[0:-1])
    
        return grad_input, grad_w, None, None


class RBFActivation(nn.Module):
    """ RBF activation function with trainable weights """
    def __init__(self, **kwargs):
        super().__init__()
        self.options = kwargs
        x_0 = np.linspace(kwargs['vmin'],kwargs['vmax'],kwargs['num_act_weights'],dtype=np.float32)
        mu = np.linspace(kwargs['vmin'],kwargs['vmax'],kwargs['num_act_weights'],dtype=np.float32)
        self.sigma = 2*kwargs['vmax']/(kwargs['num_act_weights'] - 1)
        self.sigma = torch.tensor(self.sigma)
        if kwargs['init_type'] == 'linear':
            w_0 = kwargs['init_scale']*x_0
        elif kwargs['init_type'] == 'tv':
            w_0 = kwargs['init_scale'] * np.sign(x_0)
        elif kwargs['init_type'] == 'relu':
            w_0 = kwargs['init_scale'] * np.maximum(x_0, 0)
        elif kwargs['init_type'] == 'student-t':
            alpha = 100
            w_0 = kwargs['init_scale'] * np.sqrt(alpha)*x_0/(1+0.5*alpha*x_0**2)
        else:
            raise ValueError("init_type '%s' not defined!" % kwargs['init_type'])
        w_0 = np.reshape(w_0,(1,1,1,1,kwargs['num_act_weights']))
        w_0 = np.repeat(w_0,kwargs['features_out'],1)
        self.w = torch.nn.Parameter(torch.from_numpy(w_0))
        self.mu = torch.from_numpy(mu)
        self.rbf_act = RBFActivationFunction.apply

    def forward(self,x):
        # x = x.unsqueeze(-1)
        # x = x.repeat((1,1,1,1,self.mu.shape[-1]))
        # if not self.mu.device == x.device:
        #     self.mu = self.mu.to(x.device)
        #     self.std = self.std.to(x.device)
        # gaussian = torch.exp(-torch.square(x - self.mu)/(2*self.std ** 2))
        # weighted_gaussian = self.w_0 * gaussian
        # out = torch.sum(weighted_gaussian,axis=-1,keepdim=False)
        if not self.mu.device == x.device:
        	self.mu = self.mu.to(x.device)
        	self.sigma = self.sigma.to(x.device)

        # out = torch.zeros(x.shape,dtype=torch.float32,device=x.device)
        # for i in range(self.options['num_act_weights']):
        # 	out += self.w_0[:,:,:,:,i] * torch.exp(-torch.square(x - self.mu[:,:,:,:,i])/(2*self.std ** 2))
        output = self.rbf_act(x,self.w,self.mu,self.sigma)
        	
        return output

class VnMriReconCell(nn.Module):
    """ One cell of variational network """
    def __init__(self, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options
        """Convolution Kernel의 He initialization"""
        conv_kernel_init = np.random.randn(options['features_out'],options['features_in'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*1*options['features_in'])
        conv_kernel_1 = np.random.randn(options['features_out'],options['features_out'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*1*options['features_out'])
        conv_kernel_2 = np.random.randn(2*options['features_out'],options['features_out'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*1*options['features_out'])
        conv_kernel_3 = np.random.randn(2*options['features_out'],2*options['features_out'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*2*options['features_out'])
        conv_kernel_4 = np.random.randn(4*options['features_out'],2*options['features_out'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*2*options['features_out'])
        conv_kernel_5 = np.random.randn(4*options['features_out'],4*options['features_out'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*4*options['features_out'])
        ### pooling으로 대체 ㅠㅠ
        concat_kernel_1 = np.random.randn(4*options['features_out'],8*options['features_out'],11,11,2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*8*options['features_out'])
        concat_kernel_2 = np.random.randn(2*options['features_out'],4*options['features_out'],11,11,2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*4*options['features_out'])
        concat_kernel_3 = np.random.randn(1*options['features_out'],2*options['features_out'],11,11,2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*2*options['features_out'])
        
        """Initialization 값 잘 변경할것"""
        deconv_kernel_1 = np.random.randn(4*options['features_out'],4*options['features_out'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*4*options['features_out'])
        deconv_kernel_2 = np.random.randn(4*options['features_out'],4*options['features_out'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*4*options['features_out'])
        deconv_kernel_3 = np.random.randn(4*options['features_out'],2*options['features_out'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*4*options['features_out'])
        deconv_kernel_4 = np.random.randn(2*options['features_out'],2*options['features_out'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*2*options['features_out'])
        deconv_kernel_5 = np.random.randn(2*options['features_out'],1*options['features_out'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*2*options['features_out'])
        deconv_kernel_6 = np.random.randn(1*options['features_out'],1*options['features_out'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*1*options['features_out'])
        deconv_kernel_final = np.random.randn(options['features_out'],1*options['features_in'],options['kernel_size'],options['kernel_size'],2).astype(np.float32)\
                    /np.sqrt(options['kernel_size']**2*2*1*options['features_out'])

            
        """굳이 필요 없는 과정, 한 번 더 중심화"""
        conv_kernel_init -= np.mean(conv_kernel_init, axis=(1,2,3,4),keepdims=True)
        conv_kernel_init = torch.from_numpy(conv_kernel_init) # Array -> Tensor로 변환

        conv_kernel_1 -= np.mean(conv_kernel_1, axis=(1,2,3,4),keepdims=True)
        conv_kernel_1 = torch.from_numpy(conv_kernel_1) # Array -> Tensor로 변환

        conv_kernel_2 -= np.mean(conv_kernel_2, axis=(1,2,3,4),keepdims=True)
        conv_kernel_2 = torch.from_numpy(conv_kernel_2) # Array -> Tensor로 변환

        conv_kernel_3 -= np.mean(conv_kernel_3, axis=(1,2,3,4),keepdims=True)
        conv_kernel_3 = torch.from_numpy(conv_kernel_3) # Array -> Tensor로 변환

        conv_kernel_4 -= np.mean(conv_kernel_4, axis=(1,2,3,4),keepdims=True)
        conv_kernel_4 = torch.from_numpy(conv_kernel_4) # Array -> Tensor로 변환

        conv_kernel_5 -= np.mean(conv_kernel_5, axis=(1,2,3,4),keepdims=True)
        conv_kernel_5 = torch.from_numpy(conv_kernel_5) # Array -> Tensor로 변환

        deconv_kernel_1 -= np.mean(deconv_kernel_1, axis=(1,2,3,4),keepdims=True)
        deconv_kernel_1 = torch.from_numpy(deconv_kernel_1) # Array -> Tensor로 변환

        deconv_kernel_2 -= np.mean(deconv_kernel_2, axis=(1,2,3,4),keepdims=True)
        deconv_kernel_2 = torch.from_numpy(deconv_kernel_2) # Array -> Tensor로 변환

        deconv_kernel_3 -= np.mean(deconv_kernel_3, axis=(1,2,3,4),keepdims=True)
        deconv_kernel_3 = torch.from_numpy(deconv_kernel_3) # Array -> Tensor로 변환

        deconv_kernel_4 -= np.mean(deconv_kernel_4, axis=(1,2,3,4),keepdims=True)
        deconv_kernel_4 = torch.from_numpy(deconv_kernel_4) # Array -> Tensor로 변환

        deconv_kernel_5 -= np.mean(deconv_kernel_5, axis=(1,2,3,4),keepdims=True)
        deconv_kernel_5 = torch.from_numpy(deconv_kernel_5) # Array -> Tensor로 변환

        deconv_kernel_6 -= np.mean(deconv_kernel_6, axis=(1,2,3,4),keepdims=True)
        deconv_kernel_6 = torch.from_numpy(deconv_kernel_6) # Array -> Tensor로 변환

        deconv_kernel_final -= np.mean(deconv_kernel_final, axis=(1,2,3,4),keepdims=True)
        deconv_kernel_final = torch.from_numpy(deconv_kernel_final) # Array -> Tensor로 변환

        
        concat_kernel_1 -= np.mean(concat_kernel_1, axis=(1,2,3,4),keepdims=True)
        concat_kernel_1 = torch.from_numpy(concat_kernel_1) # Array -> Tensor로 변환

        concat_kernel_2 -= np.mean(concat_kernel_2, axis=(1,2,3,4),keepdims=True)
        concat_kernel_2 = torch.from_numpy(concat_kernel_2) # Array -> Tensor로 변환

        concat_kernel_3 -= np.mean(concat_kernel_3, axis=(1,2,3,4),keepdims=True)
        concat_kernel_3 = torch.from_numpy(concat_kernel_3) # Array -> Tensor로 변환


        if options['do_prox_map']:
            conv_kernel_init = zero_mean_norm_ball(conv_kernel_init,axis=(1,2,3,4))
            conv_kernel_1 = zero_mean_norm_ball(conv_kernel_1,axis=(1,2,3,4))
            conv_kernel_2 = zero_mean_norm_ball(conv_kernel_2,axis=(1,2,3,4))
            conv_kernel_3 = zero_mean_norm_ball(conv_kernel_3,axis=(1,2,3,4))
            conv_kernel_4 = zero_mean_norm_ball(conv_kernel_4,axis=(1,2,3,4))
            conv_kernel_5 = zero_mean_norm_ball(conv_kernel_5,axis=(1,2,3,4))
        
            deconv_kernel_1 = zero_mean_norm_ball(deconv_kernel_1,axis=(1,2,3,4))
            deconv_kernel_2 = zero_mean_norm_ball(deconv_kernel_2,axis=(1,2,3,4))
            deconv_kernel_3 = zero_mean_norm_ball(deconv_kernel_3,axis=(1,2,3,4))
            deconv_kernel_4 = zero_mean_norm_ball(deconv_kernel_4,axis=(1,2,3,4))
            deconv_kernel_5 = zero_mean_norm_ball(deconv_kernel_5,axis=(1,2,3,4))
            deconv_kernel_6 = zero_mean_norm_ball(deconv_kernel_6,axis=(1,2,3,4))
            deconv_kernel_final = zero_mean_norm_ball(deconv_kernel_final,axis=(1,2,3,4))
            concat_kernel_1 = zero_mean_norm_ball(concat_kernel_1,axis=(1,2,3,4))
            concat_kernel_2 = zero_mean_norm_ball(concat_kernel_2,axis=(1,2,3,4))
            concat_kernel_3 = zero_mean_norm_ball(concat_kernel_3,axis=(1,2,3,4))
            


        """Learnable Parameter로 지정"""
        self.conv_kernel_init = torch.nn.Parameter(conv_kernel_init)
        self.conv_kernel_1 = torch.nn.Parameter(conv_kernel_1)
        self.conv_kernel_2 = torch.nn.Parameter(conv_kernel_2)
        self.conv_kernel_3 = torch.nn.Parameter(conv_kernel_3)
        self.conv_kernel_4 = torch.nn.Parameter(conv_kernel_4)
        self.conv_kernel_5 = torch.nn.Parameter(conv_kernel_5)

        self.concat_kernel_1 = torch.nn.Parameter(concat_kernel_1)
        self.concat_kernel_2 = torch.nn.Parameter(concat_kernel_2)
        self.concat_kernel_3 = torch.nn.Parameter(concat_kernel_3)
        
        self.deconv_kernel_1 = torch.nn.Parameter(deconv_kernel_1)
        self.deconv_kernel_2 = torch.nn.Parameter(deconv_kernel_2)
        self.deconv_kernel_3 = torch.nn.Parameter(deconv_kernel_3)
        self.deconv_kernel_4 = torch.nn.Parameter(deconv_kernel_4)
        self.deconv_kernel_5 = torch.nn.Parameter(deconv_kernel_5)
        self.deconv_kernel_6 = torch.nn.Parameter(deconv_kernel_6)
        self.deconv_kernel_final = torch.nn.Parameter(deconv_kernel_final)
        if 'stage_idx' in options:
            options['is_first_layer'] = (options['stage_idx'] == 0)

        if self.options['activation'] == 'rbf':
            self.activation = RBFActivation(**options)
            options['is_first_layer'] = (options['stage_idx'] == 0) # 여기 수정해야 함 
            
        elif self.options['activation'] == 'relu':
            self.activation = torch.nn.ReLU()
            options['is_first_layer'] = (options['stage_idx'] == 0) # 여기 수정해야 함 
            
        self.lamb = torch.nn.Parameter(torch.tensor(options['lamb_init'],dtype=torch.float32))
        # self.siren_net = NeRFFittingNetwork(sidelen=320)
        # kwargs.get('image_size', 320)

        """SENet init"""
        # Define SENet layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        ### Pooling으로 strided convolution 대체
        self.fc1 = nn.Linear(4*options['features_out'], 4*options['features_out'] // 4)
        self.fc2 = nn.Linear(4*options['features_out'] // 4, 4*options['features_out'])
 
        self.relu = torch.nn.ReLU()
        self.bn1_re = torch.nn.BatchNorm2d(options['features_out'])
        self.bn1_im = torch.nn.BatchNorm2d(options['features_out'])

        self.bn2_re = torch.nn.BatchNorm2d(options['features_out'])
        self.bn2_im = torch.nn.BatchNorm2d(options['features_out'])  

        self.bn3_re = torch.nn.BatchNorm2d(2*options['features_out'])
        self.bn3_im = torch.nn.BatchNorm2d(2*options['features_out'])

        self.bn4_re = torch.nn.BatchNorm2d(2*options['features_out'])
        self.bn4_im = torch.nn.BatchNorm2d(2*options['features_out'])

        self.bn5_re = torch.nn.BatchNorm2d(4*options['features_out'])
        self.bn5_im = torch.nn.BatchNorm2d(4*options['features_out'])

        self.bn6_re = torch.nn.BatchNorm2d(4*options['features_out'])
        self.bn6_im = torch.nn.BatchNorm2d(4*options['features_out'])

        self.in1_re = torch.nn.InstanceNorm2d(4*options['features_out'])
        self.in1_im = torch.nn.InstanceNorm2d(4*options['features_out'])

        self.in2_re = torch.nn.InstanceNorm2d(4*options['features_out'])
        self.in2_im = torch.nn.InstanceNorm2d(4*options['features_out'])

        self.in3_re = torch.nn.InstanceNorm2d(2*options['features_out'])
        self.in3_im = torch.nn.InstanceNorm2d(2*options['features_out'])

        self.in4_re = torch.nn.InstanceNorm2d(2*options['features_out'])
        self.in4_im = torch.nn.InstanceNorm2d(2*options['features_out'])

        self.in5_re = torch.nn.InstanceNorm2d(options['features_out'])
        self.in5_im = torch.nn.InstanceNorm2d(options['features_out'])

        self.in6_re = torch.nn.InstanceNorm2d(options['features_out'])
        self.in6_im = torch.nn.InstanceNorm2d(options['features_out'])
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    def mri_forward_op(self, u, coil_sens, sampling_mask, os=False):
        """
        Forward pass with kspace
        (2X the size)
        
        Parameters:
        ----------
        u: torch tensor NxHxWx2
            complex input image
        coil_sens: torch tensor NxCxHxWx2
            coil sensitivity map
        sampling_mask: torch tensor NxHxW
            sampling mask to undersample kspace
        os: bool
            whether the data is oversampled in frequency encoding

        Returns:
        -----------
        kspace of u with applied coil sensitivity and sampling mask
        """
        if os:
            pad_u = torch.tensor((sampling_mask.shape[1]*0.25 + 1),dtype=torch.int16)
            pad_l = torch.tensor((sampling_mask.shape[1]*0.25 - 1),dtype=torch.int16)
            u_pad = F.pad(u,[0,0,0,0,pad_u,pad_l])
        else:
            u_pad = u
        u_pad = u_pad.unsqueeze(1)
        coil_imgs = complex_mul(u_pad, coil_sens) # NxCxHxWx2
        
        Fu = fftc2d(coil_imgs) #
        
        mask = sampling_mask.unsqueeze(1) # Nx1xHxW
        mask = mask.unsqueeze(4) # Nx1xHxWx1
        mask = mask.repeat([1,1,1,1,2]) # Nx1xHxWx2

        kspace = mask*Fu # NxCxHxWx2
        return kspace

    def mri_adjoint_op(self, f, coil_sens, sampling_mask, os=False):
        """
        Adjoint operation that convert kspace to coil-combined under-sampled image
        by using coil_sens and sampling mask
        
        Parameters:
        ----------
        f: torch tensor NxCxHxWx2
            multi channel kspace
        coil_sens: torch tensor NxCxHxWx2
            coil sensitivity map
        sampling_mask: torch tensor NxHxW
            sampling mask to undersample kspace
        os: bool
            whether the data is oversampled in frequency encoding
        Returns:
        -----------
        Undersampled, coil-combined image
        """
        
        # Apply mask and perform inverse centered Fourier transform
        mask = sampling_mask.unsqueeze(1) # Nx1xHxW
        mask = mask.unsqueeze(4) # Nx1xHxWx1
        mask = mask.repeat([1,1,1,1,2]) # Nx1xHxWx2

        Finv = ifftc2d(mask*f) # NxCxHxWx2
        # multiply coil images with sensitivities and sum up over channels
        img = torch.sum(complex_mul(Finv,conj(coil_sens)),1)

        if os:
            # Padding to remove FE oversampling
            pad_u = torch.tensor((sampling_mask.shape[1]*0.25 + 1),dtype=torch.int16)
            pad_l = torch.tensor((sampling_mask.shape[1]*0.25 - 1),dtype=torch.int16)
            img = img[:,pad_u:-pad_l,:,:]
            
        return img

    def forward(self, inputs):
        
        u_t_1 = inputs['u_t'] #NxHxWx2
        f = inputs['f']
        c = inputs['coil_sens']
        m = inputs['sampling_mask']

        u_t_1 = u_t_1.unsqueeze(1) #Nx1xHxWx2 to broadcast: 1x1x320x368x2
        
        # pad the image to avoid problems at the border
        pad = self.options['pad']
        u_t_real = u_t_1[:,:,:,:,0] # 1x1x320x368
        u_t_imag = u_t_1[:,:,:,:,1]
        """u_t_ shape before padding: torch.Size([1, 1, 320, 368]) """

        
        """u_t_real/Imag shape After Padding: torch.Size([1, 1, 342, 390]): Input dim=4D Tensor"""

        """Encoding"""
        u_t_real = F.pad(u_t_real,[pad,pad,pad,pad],mode='reflect') #to do: implement symmetric padding
        u_t_imag = F.pad(u_t_imag,[pad,pad,pad,pad],mode='reflect') # 1x1x342x390
        
        # split the image in real and imaginary part and perform convolution 1x1x342x390 -> 1x32x342x390
        u_k_1_real = F.conv2d(u_t_real, self.conv_kernel_init[:,:,:,:,0],stride=1,padding=5) # ([1, 32, 342, 390])
        u_k_1_imag = F.conv2d(u_t_imag, self.conv_kernel_init[:,:,:,:,1],stride=1,padding=5)
        u_k_1_real = self.bn1_re(u_k_1_real)
        u_k_1_imag = self.bn1_im(u_k_1_imag)
        
        
        f_u_k_1_real = self.activation(u_k_1_real)
        f_u_k_1_imag = self.activation(u_k_1_imag)
        
        u_k_2_real = F.conv2d(f_u_k_1_real,self.conv_kernel_1[:,:,:,:,0],stride=1,padding=5) # ([1, 32, 342, 390]) 
        u_k_2_imag = F.conv2d(f_u_k_1_imag,self.conv_kernel_1[:,:,:,:,1],stride=1,padding=5)
        u_k_2_real = self.bn2_re(u_k_2_real)
        u_k_2_real = self.bn2_im(u_k_2_real)
        # u_k_2 = u_k_2_real + u_k_2_imag
        f_u_k_2_real = self.activation(u_k_2_real)
        f_u_k_2_imag = self.activation(u_k_2_imag)
        # f_u_k_2 = self.activation(u_k_2)
        
        """메모리 이슈로 Strided Convolution을 max pooling으로 대체함 ... Original UNet paper처럼"""
        u_k_3_real = self.maxpool(f_u_k_2_real) # 1x32x171x195 by maxpool2D
        u_k_3_imag = self.maxpool(f_u_k_2_imag)


        """층층이 UNet처럼 만들기 ... strided convolution을 pooling으로 대체하면서, 그림을 반드시 다시 그려야함"""
        u_k_4_real = F.conv2d(u_k_3_real,self.conv_kernel_2[:,:,:,:,0],stride=1,padding=5) # 차원을 맞추기 위해
        u_k_4_imag = F.conv2d(u_k_3_imag,self.conv_kernel_2[:,:,:,:,1],stride=1,padding=5) # # ([1, 64, 171, 195])
        u_k_4_real = self.bn3_re(u_k_4_real) 
        u_k_4_imag = self.bn3_im(u_k_4_imag)
        f_u_k_4_real = self.activation(u_k_4_real)
        f_u_k_4_imag = self.activation(u_k_4_imag)
        # print("dimension of f_u_k_4", u_k_4_real.size())
        
        u_k_5_real = F.conv2d(f_u_k_4_real,self.conv_kernel_3[:,:,:,:,0],stride=1,padding=5) # 차원을 맞추기 위해
        u_k_5_imag = F.conv2d(f_u_k_4_imag,self.conv_kernel_3[:,:,:,:,1],stride=1,padding=5) # # ([1, 64, 171, 195])
        u_k_5_real = self.bn4_re(u_k_5_real) 
        u_k_5_imag = self.bn4_im(u_k_5_imag)
        f_u_k_5_real = self.activation(u_k_5_real)
        f_u_k_5_imag = self.activation(u_k_5_imag)
        
        
        u_k_6_real = self.maxpool(f_u_k_5_real) # torch.Size([1, 64, 86, 98])
        u_k_6_imag = self.maxpool(f_u_k_5_imag)

        """1 x 128 x 86 x 98"""
        u_k_7_real = F.conv2d(u_k_6_real,self.conv_kernel_4[:,:,:,:,0],stride=1,padding=5) 
        u_k_7_imag = F.conv2d(u_k_6_imag,self.conv_kernel_4[:,:,:,:,1],stride=1,padding=5) # torch.Size([1, 64, 86, 98])
        u_k_7_real = self.bn5_re(u_k_7_real) 
        u_k_7_imag = self.bn5_im(u_k_7_imag)
        f_u_k_7_real = self.activation(u_k_7_real)
        f_u_k_7_imag = self.activation(u_k_7_imag)

        u_k_8_real = F.conv2d(f_u_k_7_real,self.conv_kernel_5[:,:,:,:,0],stride=1,padding=5) 
        u_k_8_imag = F.conv2d(f_u_k_7_imag,self.conv_kernel_5[:,:,:,:,1],stride=1,padding=5) # torch.Size([1, 64, 86, 98])
        u_k_8_real = self.bn6_re(u_k_8_real) 
        u_k_8_imag = self.bn6_im(u_k_8_imag)
        f_u_k_8_real = self.activation(u_k_8_real)
        f_u_k_8_imag = self.activation(u_k_8_imag)
        """f_u_k_7.shape()= torch.Size([1, 256, 43, 49])"""
        u_k_9_real = self.maxpool(f_u_k_8_real) # Final at encoder
        u_k_9_imag = self.maxpool(f_u_k_8_imag)
        
        
        """Attention layer:: Warning: Out of Memory"""
        """Real Part Attention"""
        batch_real, channels_real, _, _ = u_k_9_real.shape  # (N, C, H, W) 
        se_u_k_real = self.global_pool(u_k_9_real) # 1, 256, 1, 1
        se_u_k_real = se_u_k_real.view(batch_real, channels_real)
        se_u_k_real = self.fc1(se_u_k_real)
        se_u_k_real = self.relu(se_u_k_real)
        se_u_k_real = self.fc2(se_u_k_real)
        att_real = torch.sigmoid(se_u_k_real).view(batch_real, channels_real, 1, 1)
        """Img part attention"""
        batch_imag, channels_imag, _, _ = u_k_9_imag.shape  # (N, C, H, W) 
        se_u_k_imag = self.global_pool(u_k_9_imag)
        se_u_k_imag = se_u_k_imag.view(batch_imag, channels_imag)
        se_u_k_imag = self.fc1(se_u_k_imag)
        se_u_k_imag = self.relu(se_u_k_imag)
        se_u_k_imag = self.fc2(se_u_k_imag)
        att_imag = torch.sigmoid(se_u_k_imag).view(batch_imag, channels_imag, 1, 1)
        """Mul"""
        u_k_att_real = att_real*u_k_9_real
        u_k_att_imag = att_imag*u_k_9_imag
        
        
        ### 1 x 128 x 86 x 98
        u_k_T_real_1 = F.conv_transpose2d(u_k_att_real, self.deconv_kernel_1[:,:,:,:,0], stride=2, padding=5, output_padding=1)
        u_k_T_imag_1 = F.conv_transpose2d(u_k_att_imag, self.deconv_kernel_1[:,:,:,:,1], stride=2, padding=5, output_padding=1)
        """Deconv with residual connection"""
        # u_k_T_real_1 = F.conv_transpose2d(u_k_9_real, self.deconv_kernel_1[:,:,:,:,0], stride=2, padding=5, output_padding=1)
        # u_k_T_imag_1 = F.conv_transpose2d(u_k_9_imag, self.deconv_kernel_1[:,:,:,:,1], stride=2, padding=5, output_padding=1)
        """Residual Connection을 덧셈에서 concat으로 바꿔야 함. """
        concat_u_t_real_0 = torch.cat((u_k_T_real_1, f_u_k_8_real), dim=1)
        concat_u_t_imag_0 = torch.cat((u_k_T_imag_1, f_u_k_8_imag), dim=1)
        
        u_k_T_real_2 = F.conv2d(concat_u_t_real_0, self.concat_kernel_1[:,:,:,:,0], stride=1, padding=5)
        u_k_T_imag_2 = F.conv2d(concat_u_t_imag_0, self.concat_kernel_1[:,:,:,:,1], stride=1, padding=5)
        u_k_T_real_2 = self.in1_re(u_k_T_real_2)
        u_k_T_imag_2 = self.in1_im(u_k_T_imag_2)
        f_u_k_T_real_2 = self.activation(u_k_T_real_2)
        f_u_k_T_imag_2 = self.activation(u_k_T_imag_2)
        """아직 남음"""
        """2025/1/13: 크아아... ㅠㅠ Out Of Memory Issue 발생 ㅠㅠㅠ"""

        u_k_T_real_3 = F.conv2d(u_k_T_real_2,self.deconv_kernel_2[:,:,:,:,0],stride=1,padding=5)
        u_k_T_imag_3 = F.conv2d(u_k_T_imag_2,self.deconv_kernel_2[:,:,:,:,1],stride=1,padding=5)
        u_k_T_real_3 = self.in2_re(u_k_T_real_3)
        u_k_T_real_3 = self.in2_im(u_k_T_imag_3)
        f_u_k_T_real_3 = self.activation(u_k_T_real_3)
        f_u_k_T_real_3 = self.activation(u_k_T_real_3)
        
        # 1 x 64 x 171 x 195 !
        u_k_T_real_4 = F.conv_transpose2d(f_u_k_T_real_3,self.deconv_kernel_3[:,:,:,:,0],stride=2,padding=5)
        u_k_T_imag_4 = F.conv_transpose2d(f_u_k_T_real_3,self.deconv_kernel_3[:,:,:,:,1],stride=2,padding=5)

        # 1 x 128 x 171 x 195 !
        concat_u_t_real_1 = torch.cat((u_k_T_real_4, f_u_k_5_real), dim=1) 
        concat_u_t_imag_1 = torch.cat((u_k_T_imag_4, f_u_k_5_imag), dim=1)


        u_k_T_real_5 = F.conv2d(concat_u_t_real_1,self.concat_kernel_2[:,:,:,:,0],stride=1,padding=5)
        u_k_T_imag_5 = F.conv2d(concat_u_t_imag_1,self.concat_kernel_2[:,:,:,:,1],stride=1,padding=5)
        u_k_T_real_5 = self.in3_re(u_k_T_real_5)
        u_k_T_imag_5 = self.in3_im(u_k_T_imag_5)
        f_u_k_T_real_5 = self.activation(u_k_T_real_5)
        f_u_k_T_imag_5 = self.activation(u_k_T_imag_5)

        u_k_T_real_6 = F.conv2d(f_u_k_T_real_5,self.deconv_kernel_4[:,:,:,:,0],stride=1,padding=5)
        u_k_T_imag_6 = F.conv2d(f_u_k_T_imag_5,self.deconv_kernel_4[:,:,:,:,1],stride=1,padding=5)
        u_k_T_real_6 = self.in4_re(u_k_T_real_6)
        u_k_T_imag_6 = self.in4_im(u_k_T_imag_6)
        f_u_k_T_real_6 = self.activation(u_k_T_real_6)
        f_u_k_T_imag_6 = self.activation(u_k_T_imag_6)

        # 1 x 64 x 342 x 390 !
        u_k_T_real_7 = F.conv_transpose2d(f_u_k_T_real_6,self.deconv_kernel_5[:,:,:,:,0],stride=2,padding=5,output_padding=1)
        u_k_T_imag_7 = F.conv_transpose2d(f_u_k_T_imag_6,self.deconv_kernel_5[:,:,:,:,1],stride=2,padding=5,output_padding=1)
        concat_u_t_real_2 = torch.cat((u_k_T_real_7, f_u_k_2_real), dim=1) 
        concat_u_t_imag_2 = torch.cat((u_k_T_imag_7, f_u_k_2_imag), dim=1)
        u_k_T_real_8 = F.conv2d(concat_u_t_real_2,self.concat_kernel_3[:,:,:,:,0],stride=1,padding=5)
        u_k_T_imag_8 = F.conv2d(concat_u_t_imag_2,self.concat_kernel_3[:,:,:,:,1],stride=1,padding=5)

        u_k_T_real_8 = self.in5_re(u_k_T_real_8)
        u_k_T_imag_8 = self.in5_im(u_k_T_imag_8)
        f_u_k_T_real_8 = self.activation(u_k_T_real_8)
        f_u_k_T_imag_8 = self.activation(u_k_T_imag_8)

        u_k_T_real_9 = F.conv2d(f_u_k_T_real_8,self.deconv_kernel_6[:,:,:,:,0],stride=1,padding=5)
        u_k_T_imag_9 = F.conv2d(f_u_k_T_imag_8,self.deconv_kernel_6[:,:,:,:,1],stride=1,padding=5)
        u_k_T_real_9 = self.in6_re(u_k_T_real_9)
        u_k_T_imag_9 = self.in6_im(u_k_T_imag_9)
        f_u_k_T_real_9 = self.activation(u_k_T_real_9)
        f_u_k_T_imag_9 = self.activation(u_k_T_imag_9)
        """Final deconv"""
        
        u_k_T_real_10 = F.conv_transpose2d(f_u_k_T_real_9,self.deconv_kernel_final[:,:,:,:,0],stride=1,padding=5)
        u_k_T_imag_10 = F.conv_transpose2d(f_u_k_T_imag_9,self.deconv_kernel_final[:,:,:,:,1],stride=1,padding=5)

        # perform transpose convolution for real and imaginary part
        """Rebuild complex image"""
    
        u_k_T_real = u_k_T_real_10.unsqueeze(-1)
        u_k_T_imag = u_k_T_imag_10.unsqueeze(-1)
        u_k_T =  torch.cat((u_k_T_real,u_k_T_imag),dim=-1)

        #Remove padding and normalize by number of filter
        Ru = u_k_T[:,0,pad:-pad,pad:-pad,:] #NxHxWx2
        Ru /= self.options['features_out']

        if self.options['sampling_pattern'] == 'cartesian':
            os = False
        elif not 'sampling_pattern' in self.options or self.options['sampling_pattern'] == 'cartesian_with_os':
            os = True

        # Siren Network
        # u_t_1을 SIREN network에 통과
        """이미지의 차원 = 1, 1, 320, 364?, 2인듯 함
        GPU 받고 재도전"""
        # coords = self.siren_net.get_mgrid(u_t_1.shape[2], dim=2)  # u_t_1의 크기에 맞게 좌표 그리드 생성
        
        # coords = coords.to(u_t_1.device)
        # coords = coords.view(1, 1, u_t_1.shape[2], u_t_1.shape[2], 2)  # (1, 1, 320, 320, 2)
        # coords = coords.permute(0, 3, 1, 2, 4)  # (1, 320, 1, 320, 2)   
        # # coords = coords.unsqueeze(0).repeat(u_t_1.shape[0], 1, 1)
        # N = u_t_1.shape[0]
        
        # coords = coords.expand(N, -1, -1, -1, -1)  # 배치 크기 확장
        # coords = coords.reshape(N, -1, 2)  # (N, 320*320, 2)
        
        # siren_output = self.siren_net(coords)  # Siren 네트워크 통과
        # siren_output = siren_output.view(u_t_1.shape[0], 320, 1, 320)  # (N, 320, 1, 320)

        # Au = self.mri_forward_op(siren_output[:,0,:,:,:],c,m,os)
        Au = self.mri_forward_op(u_t_1[:,0,:,:,:],c,m,os)
        At_Au_f = self.mri_adjoint_op(Au - f, c, m,os)
        Du = At_Au_f * self.lamb
        u_t = u_t_1[:,0,:,:,:] - Ru - Du
        output = {'u_t':u_t,'f':f,'coil_sens':c,'sampling_mask':m}
        return output #NxHxWx2
    # def forward(self, inputs):
    #     u_t_1 = inputs['u_t'] #NxHxWx2
    #     f = inputs['f']
    #     c = inputs['coil_sens']
    #     m = inputs['sampling_mask']

    #     u_t_1 = u_t_1.unsqueeze(1) #Nx1xHxWx2
    #     # pad the image to avoid problems at the border
    #     pad = self.options['pad']
    #     u_t_real = u_t_1[:,:,:,:,0]
    #     u_t_imag = u_t_1[:,:,:,:,1]
        
    #     u_t_real = F.pad(u_t_real,[pad,pad,pad,pad],mode='reflect') #to do: implement symmetric padding
    #     u_t_imag = F.pad(u_t_imag,[pad,pad,pad,pad],mode='reflect')
    #     # split the image in real and imaginary part and perform convolution
    #     u_k_real = F.conv2d(u_t_real,self.conv_kernel[:,:,:,:,0],stride=1,padding=5)
    #     u_k_imag = F.conv2d(u_t_imag,self.conv_kernel[:,:,:,:,1],stride=1,padding=5)
    #     # add up the convolution results
    #     u_k = u_k_real + u_k_imag
    #     #apply activation function
    #     f_u_k = self.activation(u_k)
    #     # perform transpose convolution for real and imaginary part
    #     u_k_T_real = F.conv_transpose2d(f_u_k,self.conv_kernel[:,:,:,:,0],stride=1,padding=5)
    #     u_k_T_imag = F.conv_transpose2d(f_u_k,self.conv_kernel[:,:,:,:,1],stride=1,padding=5)

    #     #Rebuild complex image
    #     u_k_T_real = u_k_T_real.unsqueeze(-1)
    #     u_k_T_imag = u_k_T_imag.unsqueeze(-1)
    #     u_k_T =  torch.cat((u_k_T_real,u_k_T_imag),dim=-1)

    #     #Remove padding and normalize by number of filter
    #     Ru = u_k_T[:,0,pad:-pad,pad:-pad,:] #NxHxWx2
    #     Ru /= self.options['features_out']

    #     if self.options['sampling_pattern'] == 'cartesian':
    #         os = False
    #     elif not 'sampling_pattern' in self.options or self.options['sampling_pattern'] == 'cartesian_with_os':
    #         os = True

    #     Au = self.mri_forward_op(u_t_1[:,0,:,:,:],c,m,os)
    #     At_Au_f = self.mri_adjoint_op(Au - f, c, m,os)
    #     Du = At_Au_f * self.lamb
    #     u_t = u_t_1[:,0,:,:,:] - Ru - Du
    #     output = {'u_t':u_t,'f':f,'coil_sens':c,'sampling_mask':m}
    #     return output #NxHxWx2

class VariationalNetwork(pl.LightningModule):   
    def __init__(self,**kwargs):
        super().__init__()
        options = DEFAULT_OPTS

        for key in kwargs.keys():
            options[key] = kwargs[key]

        self.options = options
        cell_list = []
        for i in range(options['num_stages']):
            options['stage_idx'] = i
            cell_list.append(VnMriReconCell(**options))

        self.cell_list = nn.Sequential(*cell_list)
        self.log_img_count = 0
        

    def forward(self,inputs):
        output = self.cell_list(inputs)
        return output['u_t']
    
    def training_step(self, batch, batch_idx):
        recon_img = self(batch)
        ref_img = batch['reference']

        if self.options['loss_type'] == 'complex':
            loss = F.mse_loss(recon_img,ref_img)
        elif self.options['loss_type'] == 'magnitude':
            recon_img_mag = torch_abs(recon_img)
            ref_img_mag = torch_abs(ref_img)    
            loss = F.mse_loss(recon_img_mag,ref_img_mag)
        loss = self.options['loss_weight']*loss
        if batch_idx % (int(200/self.options['batch_size']/4)) == 0:
            sample_img = save_recon(batch['u_t'],recon_img,ref_img,batch_idx,'save_dir',self.options['error_scale'],False)
            sample_img = sample_img[np.newaxis,:,:]
            self.logger.experiment.add_image('sample_recon',sample_img.astype(np.uint8),self.log_img_count)
            self.log_img_count += 1

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        recon_img = self(batch)
        ref_img = batch['reference']
        recon_img_mag = torch_abs(recon_img)
        ref_img_mag = torch_abs(ref_img)
        loss = F.mse_loss(recon_img_mag,ref_img_mag)
        img_save_dir = Path(self.options['save_dir']) / ('eval_result_img_' + self.options['name'])
        img_save_dir.mkdir(parents=True,exist_ok=True)
        save_recon(batch['u_t'],recon_img,ref_img,batch_idx,img_save_dir,self.options['error_scale'],True)
        return {'test_loss':loss}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}
    

    def configure_optimizers(self):
        if self.options['optimizer'] == 'adam':
            return torch.optim.Adam(self.parameters(),lr=self.options['lr'])
        elif self.options['optimizer'] == 'sgd':
        	return torch.optim.SGD(self.parameters(),lr=self.options['lr'],momentum=self.options['momentum'])
        elif self.options['optimizer'] == 'rmsprop':
        	return torch.optim.RMSprop(self.parameters(),lr=self.options['lr'],momentum=self.options['momentum'])
        elif self.options['optimizer'] == 'iipg':
            iipg = IIPG(torch.optim.SGD,self.parameters(),lr=self.options['lr'],momentum=self.options['momentum'])
            return iipg
