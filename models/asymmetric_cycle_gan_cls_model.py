import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import models.dist_model as dm # numpy==1.14.3
import torchvision.transforms as transforms
import os

def truncate(fake_B,a=127.5):#[-1,1]
    return ((fake_B+1)*a).int().float()/a-1

class AsymmetricCycleGANClsModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.set_defaults(dataset_mode='unaligned_mask_stylecls')
        parser.add_argument('--netda', type=str, default='basic_cls')
        parser.add_argument('--netga', type=str, default='resnet_style2_9blocks', help='net arch for netG_A')
        parser.add_argument('--model0_res', type=int, default=0, help='number of resblocks in model0 (before insert style)')
        parser.add_argument('--model1_res', type=int, default=0, help='number of resblocks in model1 (after insert style, before 2 column merge)')
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=5.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=5.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--ntrunc_trunc', type=int, default=1, help='whether use both non-trunc version and trunc version')
            parser.add_argument('--trunc_a', type=float, default=31.875, help='multiply which value to round when trunc')
            parser.add_argument('--lambda_A_trunc', type=float, default=5.0, help='weight for cycle loss for trunc')
            parser.add_argument('--hed_pretrained_mode', type=str, default='./checkpoints/network-bsds500.pytorch', help='path to the pretrained hed model')
            parser.add_argument('--lambda_G_A_l', type=float, default=0.5, help='weight for local GAN loss in G')
            parser.add_argument('--style_loss_with_weight', type=int, default=1, help='whether multiply prob in style loss')
        # for masks
        parser.add_argument('--use_mask', type=int, default=1, help='whether use mask for special face region')
        parser.add_argument('--use_eye_mask', type=int, default=1, help='whether use mask for special face region')
        parser.add_argument('--use_lip_mask', type=int, default=1, help='whether use mask for special face region')
        parser.add_argument('--mask_type', type=int, default=3, help='use mask type, 0 outside black, 1 outside white')
        # for style control
        parser.add_argument('--style_control', type=int, default=1, help='use style_control')
        parser.add_argument('--sfeature_mode', type=str, default='1vgg19_softmax', help='vgg19 softmax as feature')
        parser.add_argument('--one_hot', type=int, default=0, help='use one-hot for style code')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        if self.isTrain:
            visual_names_A.append('real_A_hed')
            visual_names_A.append('rec_A_hed')
        if self.isTrain and self.opt.ntrunc_trunc:
            visual_names_A.append('rec_At')
            visual_names_A.append('rec_At_hed')
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'cycle_A2', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'G']
        if self.isTrain and self.opt.use_mask:
            visual_names_A.append('fake_B_l')
            visual_names_A.append('real_B_l')
            self.loss_names += ['D_A_l', 'G_A_l']
        if self.isTrain and self.opt.use_eye_mask:
            visual_names_A.append('fake_B_le')
            visual_names_A.append('real_B_le')
            self.loss_names += ['D_A_le', 'G_A_le']
        if self.isTrain and self.opt.use_lip_mask:
            visual_names_A.append('fake_B_ll')
            visual_names_A.append('real_B_ll')
            self.loss_names += ['D_A_ll', 'G_A_ll']
        if not self.isTrain and self.opt.use_mask:
            visual_names_A.append('fake_B_l')
            visual_names_A.append('real_B_l')
        if not self.isTrain and self.opt.use_eye_mask:
            visual_names_A.append('fake_B_le')
            visual_names_A.append('real_B_le')
        if not self.isTrain and self.opt.use_lip_mask:
            visual_names_A.append('fake_B_ll')
            visual_names_A.append('real_B_ll')
        self.loss_names += ['D_A_cls','G_A_cls']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        print(self.visual_names)
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
            if self.opt.use_mask:
                self.model_names += ['D_A_l']
            if self.opt.use_eye_mask:
                self.model_names += ['D_A_le']
            if self.opt.use_lip_mask:
                self.model_names += ['D_A_ll']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        if not self.opt.style_control:
            self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            print(opt.netga)
            print('model0_res', opt.model0_res)
            print('model1_res', opt.model1_res)
            self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netga, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.model0_res, opt.model1_res)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netda,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, n_class=3)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            if self.opt.use_mask:
                if self.opt.mask_type in [2, 3]:
                    output_nc = opt.output_nc + 1
                else:
                    output_nc = opt.output_nc
                self.netD_A_l = networks.define_D(output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            if self.opt.use_eye_mask:
                if self.opt.mask_type in [2, 3]:
                    output_nc = opt.output_nc + 1
                else:
                    output_nc = opt.output_nc
                self.netD_A_le = networks.define_D(output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            if self.opt.use_lip_mask:
                if self.opt.mask_type in [2, 3]:
                    output_nc = opt.output_nc + 1
                else:
                    output_nc = opt.output_nc
                self.netD_A_ll = networks.define_D(output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if not self.isTrain:
            self.criterionGAN = networks.GANLoss('lsgan').to(self.device)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionCls = torch.nn.CrossEntropyLoss()
            self.criterionCls2 = torch.nn.CrossEntropyLoss(reduction='none')
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            if not self.opt.use_mask:
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif not self.opt.use_eye_mask:
                D_params = list(self.netD_A.parameters()) + list(self.netD_B.parameters()) + list(self.netD_A_l.parameters())
                self.optimizer_D = torch.optim.Adam(D_params, lr=opt.lr, betas=(opt.beta1, 0.999))
            elif not self.opt.use_lip_mask:
                D_params = list(self.netD_A.parameters()) + list(self.netD_B.parameters()) + list(self.netD_A_l.parameters()) + list(self.netD_A_le.parameters())
                self.optimizer_D = torch.optim.Adam(D_params, lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                D_params = list(self.netD_A.parameters()) + list(self.netD_B.parameters()) + list(self.netD_A_l.parameters()) + list(self.netD_A_le.parameters()) + list(self.netD_A_ll.parameters())
                self.optimizer_D = torch.optim.Adam(D_params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.lpips = dm.DistModel(opt,model='net-lin',net='alex',use_gpu=True)

            self.hed = networks.define_HED(init_weights_=opt.hed_pretrained_mode, gpu_ids_=self.opt.gpu_ids_p)
            self.set_requires_grad(self.hed, False)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if self.opt.use_mask:
            self.A_mask = input['A_mask'].to(self.device)
            self.B_mask = input['B_mask'].to(self.device)
        if self.opt.use_eye_mask:
            self.A_maske = input['A_maske'].to(self.device)
            self.B_maske = input['B_maske'].to(self.device)
        if self.opt.use_lip_mask:
            self.A_maskl = input['A_maskl'].to(self.device)
            self.B_maskl = input['B_maskl'].to(self.device)
        if self.opt.style_control:
            self.real_B_style = input['B_style'].to(self.device)
            self.real_B_label = input['B_label'].to(self.device)
        if self.opt.isTrain and self.opt.style_loss_with_weight:
            self.real_B_style0 = input['B_style0'].to(self.device)
            self.zero = torch.zeros(self.real_B_label.size(),dtype=torch.int64).to(self.device)
            self.one = torch.ones(self.real_B_label.size(),dtype=torch.int64).to(self.device)
            self.two = 2*torch.ones(self.real_B_label.size(),dtype=torch.int64).to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if not self.opt.style_control:
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        else:
            #print(torch.mean(self.real_B_style,(2,3)),'style_control')
            self.fake_B = self.netG_A(self.real_A, self.real_B_style)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        if not self.opt.style_control:
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        else:
            #print(torch.mean(self.real_B_style,(2,3)),'style_control')
            self.rec_B = self.netG_A(self.fake_A, self.real_B_style) # -- cycle_B loss

        if self.opt.use_mask:
            self.fake_B_l = self.masked(self.fake_B,self.A_mask)
            self.real_B_l = self.masked(self.real_B,self.B_mask)
        if self.opt.use_eye_mask:
            self.fake_B_le = self.masked(self.fake_B,self.A_maske)
            self.real_B_le = self.masked(self.real_B,self.B_maske)
        if self.opt.use_lip_mask:
            self.fake_B_ll = self.masked(self.fake_B,self.A_maskl)
            self.real_B_ll = self.masked(self.real_B,self.B_maskl)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D_basic_cls(self, netD, real, fake):
        # Real
        pred_real, pred_real_cls = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        if not self.opt.style_loss_with_weight:
            loss_D_real_cls = self.criterionCls(pred_real_cls, self.real_B_label)
        else:
            loss_D_real_cls = torch.mean(self.real_B_style0[:,0] * self.criterionCls2(pred_real_cls, self.zero) + self.real_B_style0[:,1] * self.criterionCls2(pred_real_cls, self.one) + self.real_B_style0[:,2] * self.criterionCls2(pred_real_cls, self.two))
        # Fake
        pred_fake, pred_fake_cls = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        if not self.opt.style_loss_with_weight:
            loss_D_fake_cls = self.criterionCls(pred_fake_cls, self.real_B_label)
        else:
            loss_D_fake_cls = torch.mean(self.real_B_style0[:,0] * self.criterionCls2(pred_fake_cls, self.zero) + self.real_B_style0[:,1] * self.criterionCls2(pred_fake_cls, self.one) + self.real_B_style0[:,2] * self.criterionCls2(pred_fake_cls, self.two))
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D_cls = (loss_D_real_cls + loss_D_fake_cls) * 0.5
        loss_D_total = loss_D + loss_D_cls
        loss_D_total.backward()
        return loss_D, loss_D_cls

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A, self.loss_D_A_cls = self.backward_D_basic_cls(self.netD_A, self.real_B, fake_B)
    
    def backward_D_A_l(self):
        """Calculate GAN loss for discriminator D_A_l"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A_l = self.backward_D_basic(self.netD_A_l, self.masked(self.real_B,self.B_mask), self.masked(fake_B,self.A_mask))

    def backward_D_A_le(self):
        """Calculate GAN loss for discriminator D_A_le"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A_le = self.backward_D_basic(self.netD_A_le, self.masked(self.real_B,self.B_maske), self.masked(fake_B,self.A_maske))
    
    def backward_D_A_ll(self):
        """Calculate GAN loss for discriminator D_A_ll"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A_ll = self.backward_D_basic(self.netD_A_ll, self.masked(self.real_B,self.B_maskl), self.masked(fake_B,self.A_maskl))

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    
    def update_process(self, epoch):
        self.process = (epoch - 1) / float(self.opt.niter_decay + self.opt.niter)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_G_A_l = self.opt.lambda_G_A_l
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_A_trunc = self.opt.lambda_A_trunc
        if self.opt.ntrunc_trunc:
            lambda_A = lambda_A * (1 - self.process * 0.9)
            lambda_A_trunc = lambda_A_trunc * self.process * 0.9
        self.lambda_As = [lambda_A, lambda_A_trunc]
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        pred_fake, pred_fake_cls = self.netD_A(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        if not self.opt.style_loss_with_weight:
            self.loss_G_A_cls = self.criterionCls(pred_fake_cls, self.real_B_label)
        else:
            self.loss_G_A_cls = torch.mean(self.real_B_style0[:,0] * self.criterionCls2(pred_fake_cls, self.zero) + self.real_B_style0[:,1] * self.criterionCls2(pred_fake_cls, self.one) + self.real_B_style0[:,2] * self.criterionCls2(pred_fake_cls, self.two))
        if self.opt.use_mask:
            self.loss_G_A_l = self.criterionGAN(self.netD_A_l(self.fake_B_l), True) * lambda_G_A_l
        if self.opt.use_eye_mask:
            self.loss_G_A_le = self.criterionGAN(self.netD_A_le(self.fake_B_le), True) * lambda_G_A_l
        if self.opt.use_lip_mask:
            self.loss_G_A_ll = self.criterionGAN(self.netD_A_ll(self.fake_B_ll), True) * lambda_G_A_l
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss  LPIPS( HED(G_B(G_A(A))), HED(A))
        ts = self.real_A.shape
        gpu_p = self.opt.gpu_ids_p[0]
        gpu = self.opt.gpu_ids[0]
        rec_A_hed = (self.hed(self.rec_A.cuda(gpu_p)/2+0.5)-0.5)*2
        real_A_hed = (self.hed(self.real_A.cuda(gpu_p)/2+0.5)-0.5)*2
        self.loss_cycle_A = (self.lpips.forward_pair(rec_A_hed.expand(ts), real_A_hed.expand(ts)).mean()).cuda(gpu) * lambda_A
        self.rec_A_hed = rec_A_hed
        self.real_A_hed = real_A_hed
        if self.opt.ntrunc_trunc:
            self.rec_At = self.netG_B(truncate(self.fake_B,self.opt.trunc_a))
            rec_At_hed = (self.hed(self.rec_At.cuda(gpu_p)/2+0.5)-0.5)*2
            self.loss_cycle_A2 = (self.lpips.forward_pair(rec_At_hed.expand(ts), real_A_hed.expand(ts)).mean()).cuda(gpu) * lambda_A_trunc
            self.rec_At_hed = rec_At_hed

        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        if getattr(self,'loss_cycle_A2',-1) != -1:
            self.loss_G = self.loss_G + self.loss_cycle_A2
        if getattr(self,'loss_G_A_l',-1) != -1:
            self.loss_G = self.loss_G + self.loss_G_A_l
        if getattr(self,'loss_G_A_le',-1) != -1:
            self.loss_G = self.loss_G + self.loss_G_A_le
        if getattr(self,'loss_G_A_ll',-1) != -1:
            self.loss_G = self.loss_G + self.loss_G_A_ll
        if getattr(self,'loss_G_A_cls',-1) != -1:
            self.loss_G = self.loss_G + self.loss_G_A_cls
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        if self.opt.use_mask:
            self.set_requires_grad([self.netD_A_l], False)
        if self.opt.use_eye_mask:
            self.set_requires_grad([self.netD_A_le], False)
        if self.opt.use_lip_mask:
            self.set_requires_grad([self.netD_A_ll], False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        if self.opt.use_mask:
            self.set_requires_grad([self.netD_A_l], True)
        if self.opt.use_eye_mask:
            self.set_requires_grad([self.netD_A_le], True)
        if self.opt.use_lip_mask:
            self.set_requires_grad([self.netD_A_ll], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        if self.opt.use_mask:
            self.backward_D_A_l()# calculate gradients for D_A_l
        if self.opt.use_eye_mask:
            self.backward_D_A_le()# calculate gradients for D_A_le
        if self.opt.use_lip_mask:
            self.backward_D_A_ll()# calculate gradients for D_A_ll
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
