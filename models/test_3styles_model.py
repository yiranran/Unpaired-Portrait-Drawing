from .base_model import BaseModel
from . import networks
import torch

class Test3StylesModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--style_control', type=int, default=0, help='not set style_vec in dataset')
        parser.add_argument('--netga', type=str, default='resnet_style2_9blocks', help='net arch for netG_A')
        parser.add_argument('--model0_res', type=int, default=0, help='number of resblocks in model0')
        parser.add_argument('--model1_res', type=int, default=0, help='number of resblocks in model1 (after insert style, before 2 column merge)')

        return parser

    def __init__(self, opt):
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake1', 'fake2', 'fake3']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G_A']  # only generator is needed.
        print(opt.netga)
        print('model0_res', opt.model0_res)
        print('model1_res', opt.model1_res)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netga, opt.norm,
                                    not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.model0_res, opt.model1_res)
        
        setattr(self, 'netG_A', self.netG)  # store netG in self.

    def set_input(self, input):
        self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']
        self.style1 = torch.Tensor([1, 0, 0]).view(3, 1, 1).repeat(1, 1, 128, 128).to(self.device)
        self.style2 = torch.Tensor([0, 1, 0]).view(3, 1, 1).repeat(1, 1, 128, 128).to(self.device)
        self.style3 = torch.Tensor([0, 0, 1]).view(3, 1, 1).repeat(1, 1, 128, 128).to(self.device)

    def forward(self):
        """Run forward pass."""
        self.fake1 = self.netG(self.real, self.style1)
        self.fake2 = self.netG(self.real, self.style2)
        self.fake3 = self.netG(self.real, self.style3)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
