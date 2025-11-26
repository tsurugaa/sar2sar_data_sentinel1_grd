import numpy as np
from numpy.core.defchararray import add
from numpy.lib.arraysetops import isin
from torchvision import transforms
import torch
 
class ToTensor():
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, im):  
        assert isinstance(im,np.ndarray)      
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # im = im.transpose((2, 0, 1))
        if im.ndim ==3 and im.shape[2] > 1:
            im = im.transpose((2, 0, 1))
        elif im.ndim == 3 and im.shape[2] == 1:
            im = np.transpose(im, (2, 0, 1))
        elif im.ndim == 2:
            im = im[np.newaxis, :, :]
        return torch.from_numpy(im)
 
class normalization():
    def __init__(self,m,M,mean) -> None:
 
        assert isinstance(m,(float,int))
        assert isinstance(M,(float,int))
        assert isinstance(mean,(float,int))
        self.m_ = m
        self.M_ = M
        self.mean_ = mean
       
    def __call__(self,im):
       
        assert isinstance(im,np.ndarray)
        # log_im = np.log(im+10**(-3))
        # norm = (log_im + self.mean_ - self.m_)/(self.M_-self.m_)
        # norm = np.clip(norm,0,1)
        log_im = np.log(np.clip(im, 1e-6, None))
 
        den = (self.M_- self.m_)
        if abs(den) < 1e-6:
            den = 1e-6
 
        norm = (log_im + self.mean_ - self.m_) / den
 
        norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
        norm = np.clip(norm, 0, 1)
        return (norm).astype(np.float32)
   
class denormalization():
    def __init__(self,m,M,mean) -> None:
 
        assert isinstance(m,(float,int))
        assert isinstance(M,(float,int))
        assert isinstance(mean,(float,int))
        self.m_ = m
        self.M_ = M
        self.mean_ = mean
       
    def __call__(self,im):
       
        assert isinstance(im,np.ndarray)
        return (np.exp((self.M_ - self.m_) * (np.squeeze(im)).astype('float32') + self.m_ - self.mean_)-10**(-3))
 
class add_speckle():
    def __init__(self, L=None) -> None:
        if L is None:
            self.L_ = np.random.randint(1, 5)  # 4〜6のランダム
        else:
            self.L_ = L
   
    # def __init__(self,L=np.random.randint(3,6)) -> None:
    #    self.L_=L
 
    def __call__(self,im):
             
        dim = im.shape
        s = np.zeros(dim)
       
        for k in range(0,self.L_):
           
            real = np.random.normal(size=dim)
            imag = np.random.normal(size=dim)
            gamma = (np.abs(real + 1j*imag)**2) /2
            s+=gamma
        s = s/self.L_
        speck_im = im**2 * s
        s_amplitude = np.sqrt(speck_im)
        return s_amplitude
   
class add_speckle_pytorch():
    def __init__(self,m,M,L=1) -> None:
        self.L_=L
        self.m_ = m
        self.M_ = M
       
    def __call__(self,im):
       
        dim = im.shape
        s = torch.zeros(dim)
       
        for k in range(0,self.L_):
            comp = torch.randn(size=dim,dtype=torch.cfloat)
            gamma = (torch.square(comp.abs()))/2
            s+=gamma
        s = s/self.L_
        print('Before loggg')
        speck_norm = torch.log(s)
        print('After loggg')
        speck_norm = speck_norm/(self.M_-self.m_)
        speck_im = im+speck_norm
       
        return speck_im