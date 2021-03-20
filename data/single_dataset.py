from data.base_dataset import BaseDataset, get_transform, get_params, get_transform_mask
from data.image_folder import make_dataset
from PIL import Image
import torch
import os


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        if os.path.exists(opt.dataroot):
            self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        else:
            imglistA = 'datasets/list/%s/%s.txt' % (opt.phase+'A', opt.dataroot)
            self.A_paths = sorted(open(imglistA, 'r').read().splitlines())
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        self.opt.W, self.opt.H = A_img.size
        transform_params_A = get_params(self.opt, A_img.size)
        A = get_transform(self.opt, transform_params_A, grayscale=(self.input_nc == 1))(A_img)
        item = {'A': A, 'A_paths': A_path}

        if self.opt.style_control:
            if self.opt.sinput == 'sind':
                B_style = torch.Tensor([0.,0.,0.])
                B_style[self.opt.sind] = 1.
            elif self.opt.sinput == 'svec':
                ss = self.opt.svec.split(',')
                B_style = torch.Tensor([float(ss[0]),float(ss[1]),float(ss[2])])
            elif self.opt.sinput == 'simg':
                self.featureloc = os.path.join('style_features/styles2/', self.opt.sfeature_mode)
                B_style = np.load(self.featureloc, self.opt.simg[:-4]+'.npy')
            B_style = B_style.view(3, 1, 1)
            B_style = B_style.repeat(1, 128, 128)
            item['B_style'] = B_style

        return item

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
