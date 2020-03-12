from .base_model import BaseModel
from . import networks
import torch

class TestModel(BaseModel):
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
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        parser.add_argument('--style_control', type=int, default=1, help='use style_control')
        parser.add_argument('--sfeature_mode', type=str, default='vgg19_softmax', help='vgg19 softmax as feature')
        parser.add_argument('--sinput', type=str, default='sind', help='use which one for style input')
        parser.add_argument('--sind', type=int, default=0, help='one hot for sfeature')
        parser.add_argument('--svec', type=str, default='1,0,0', help='3-dim vec')
        parser.add_argument('--simg', type=str, default='Yann_Legendre-053', help='drawing example for style')
        parser.add_argument('--netga', type=str, default='resnet_style2_9blocks', help='net arch for netG_A')
        parser.add_argument('--model0_res', type=int, default=0, help='number of resblocks in model0')
        parser.add_argument('--model1_res', type=int, default=0, help='number of resblocks in model1 (after insert style, before 2 column merge)')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake', 'rec']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix, 'G_B']  # only generator is needed.
        if not self.opt.style_control:
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            print(opt.netga)
            print('model0_res', opt.model0_res)
            print('model1_res', opt.model1_res)
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netga, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.model0_res, opt.model1_res)
        
        self.netGB = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.
        setattr(self, 'netG_B', self.netGB)  # store netGB in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']
        if self.opt.style_control:
            self.style = input['B_style']

    def forward(self):
        """Run forward pass."""
        if not self.opt.style_control:
            self.fake = self.netG(self.real)  # G(real)
        else:
            print(torch.mean(self.style,(2,3)),'style_control')
            self.fake = self.netG(self.real, self.style)
        self.rec = self.netG_B(self.fake)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
