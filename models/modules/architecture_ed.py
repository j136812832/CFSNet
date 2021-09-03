import math
import torch
import torch.nn as nn
import torchvision
import functools
from torch.nn import init
import math
import torch.nn.functional as F
from torch.nn import init
import torchvision
import functools
import numpy as np
from . import main_net_ed
from . import tuning_blocks_ed
from . import block_ed as B


class CFSNet(nn.Module):
    def __init__(self, in_channel, out_channel, num_channels, num_main_blocks, num_tuning_blocks, upscale=4,
                 task_type='sr'):
        super(CFSNet, self).__init__()
        self.num_main_blocks = num_main_blocks
        self.num_tuning_blocks = num_tuning_blocks
        self.task_type = task_type
        self.main = main_net_ed.MainNet(in_channel, out_channel, num_channels, num_main_blocks, task_type=task_type,
                                     upscale=upscale)
        self.tuning_blocks = tuning_blocks_ed.TuningBlockModule(channels=num_channels, num_blocks=num_tuning_blocks,
                                                             task_type=task_type, upscale=upscale)

    def forward(self, x, control_vector):
        out = self.main.head(x)
        head_f = out
        for i, body in enumerate(self.main.body):
            tun_out, tun_alpha = self.tuning_blocks(x=out, alpha=control_vector.cuda(), number=i)
            # print(body(out).shape)
            # print(tun_out.shape)
            # print(tun_alpha.shape)
            out = body(out) * tun_alpha + tun_out

        if self.task_type == 'sr12':
            tun_out, tun_alpha = self.tuning_blocks(x=out + head_f, alpha=control_vector.cuda(), number=-1)
            out = self.main.tail(out + head_f)
            out = self.main.end(out * tun_alpha + tun_out)
        else:
            out = self.main.end(out + head_f)
        return out


####################
# Discriminator input: 1024 x 1024
####################
# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128, self).__init__()
        # [64, 128, 128]
        self.maxpooling0_0 = nn.MaxPool2d(2, 2)
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        #self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        self.maxpooling0 = nn.MaxPool2d(2, 2)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        #self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        self.maxpooling1 = nn.MaxPool2d(2, 2)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False)
        #self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        #self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        self.maxpooling2 = nn.MaxPool2d(2, 2)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False)
        #self.bn3_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        #self.bn3_1 = nn.BatchNorm2d(nf * 4, affine=True)
        self.maxpooling3 = nn.MaxPool2d(2, 2)


        # self.conv4_0 = nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False)
        # #self.bn4_0 = nn.BatchNorm2d(nf * 4, affine=True)
        # self.conv4_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        # #self.bn4_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # self.maxpooling4 = nn.MaxPool2d(2, 2)
        #
        # self.conv5_0 = nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False)
        # #self.bn5_0 = nn.BatchNorm2d(nf * 4, affine=True)
        # self.conv5_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        #self.bn5_1 = nn.BatchNorm2d(nf * 4, affine=True)

        # self.maxpooling2 = nn.MaxPool2d(2, 2)
        #
        # self.conv6_0 = nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False)
        # #self.bn6_0 = nn.BatchNorm2d(nf * 4, affine=True)
        # self.conv6_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        # #self.bn6_1 = nn.BatchNorm2d(nf * 4, affine=True)

        self.linear1 = nn.Linear(64 * 2 * 2, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.maxpooling0_0(x)
        fea = self.lrelu(self.conv0_0(fea))
        fea = self.lrelu(self.conv0_1(fea))
        fea = self.maxpooling0(fea)


        fea = self.lrelu(self.conv1_0(fea))
        fea = self.lrelu(self.conv1_1(fea))
        fea = self.maxpooling1(fea)


        fea = self.lrelu(self.conv2_0(fea))
        fea = self.lrelu(self.conv2_1(fea))
        fea = self.maxpooling2(fea)


        fea = self.lrelu(self.conv3_0(fea))
        fea = self.lrelu(self.conv3_1(fea))
        fea = self.maxpooling3(fea)

        # fea = self.lrelu(self.conv4_0(fea))
        # fea = self.lrelu(self.conv4_1(fea))
        #
        #
        # fea = self.lrelu(self.conv5_0(fea))
        # fea = self.lrelu(self.conv5_1(fea))
        #
        #
        # fea = self.lrelu(self.conv6_0(fea))
        # fea = self.lrelu(self.conv6_1(fea))
        #

        # fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        # fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out
def get_activation(activation_name):
    activ_name = activation_name.lower()

    if activ_name == "relu":
        return nn.ReLU(True)
    elif activ_name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        raise NotImplementedError("Unidentified activation name {}.".format(activ_name))
class BlurLayer(nn.Module):
    """Implements the blur layer used in StyleGAN."""

    def __init__(self,
                 channels,
                 kernel=(1, 2, 1),
                 normalize=True,
                 flip=False):
        super().__init__()
        kernel = np.array(kernel, dtype=np.float32).reshape(1, 3)
        kernel = kernel.T.dot(kernel)
        if normalize:
            kernel /= np.sum(kernel)
        if flip:
            kernel = kernel[::-1, ::-1]
        kernel = kernel.reshape(3, 3, 1, 1)
        kernel = np.tile(kernel, [1, 1, channels, 1])
        kernel = np.transpose(kernel, [2, 3, 0, 1])
        self.register_buffer('kernel', torch.from_numpy(kernel))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1, groups=self.channels)
# blur layer based conv block
def conv_blur_block(in_dim, out_dim, padding_type='zero', norm_layer=None,
                    down_sample=True, use_sn=False, blur_layer=True, activation="leaky_relu"):
    sequence = []
    activation = get_activation(activation)
    if use_sn:
        sequence += [nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1))]
    else:
        sequence += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)]
    if norm_layer is not None:
        pass
        # TODO: add norm layer
    sequence += [activation]
    if blur_layer:
        sequence += [BlurLayer(out_dim)]
    # down sample layer
    if use_sn:
        sequence += [nn.utils.spectral_norm(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1))]
    else:
        sequence += [nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1)]
    sequence += [activation]
    return nn.Sequential(*sequence)

# Define the resnet block in Discriminator
class DResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, padding_type, norm_layer, activation=nn.LeakyReLU(0.2, True),
                 use_dropout=False,
                 wide=True, down_sample=False, use_sn=False):
        super(DResnetBlock, self).__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        self.hidden_channel = self.out_channel if wide else self.in_channel
        self.downsample = down_sample
        self.sn = use_sn

        self.conv_block = self.build_conv_block(padding_type, norm_layer, activation, use_dropout)
        module = []
        if down_sample:
            module += [BlurLayer(in_channel)]
        if use_sn:
            module += [nn.utils.spectral_norm(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2 if down_sample else 1, padding=1))]
        else:
            module += [nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2 if down_sample else 1, padding=1)]
        self.skip = nn.Sequential(*module)

        # add conv1x1 for downsample and channels
        self.learn_able_sc = True if down_sample or in_channel != out_channel else False
        if self.learn_able_sc:
            self.conv_sc = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def build_conv_block(self, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if self.sn:
            conv_block += [
                nn.utils.spectral_norm(nn.Conv2d(self.in_channel, self.hidden_channel, kernel_size=3, padding=p))]
        else:
            conv_block += [nn.Conv2d(self.in_channel, self.hidden_channel, kernel_size=3, padding=p)]

        if norm_layer is not None:
            conv_block += [norm_layer(self.hidden_channel)]
        conv_block += [activation]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        stride = 2 if self.downsample else 1

        if self.downsample:
            conv_block += [BlurLayer(channels=self.hidden_channel)]
        if self.sn:
            conv_block += [
                nn.utils.spectral_norm(nn.Conv2d(self.hidden_channel, self.out_channel, kernel_size=3,
                                                 stride=stride, padding=p))]
        else:
            conv_block += [nn.Conv2d(self.hidden_channel, self.out_channel, kernel_size=3, stride=stride, padding=p)]

        conv_block += [activation]
        if norm_layer is not None:
            conv_block += [norm_layer(self.out_channel)]

        return nn.Sequential(*conv_block)

    def shortcut(self, x):
        if self.downsample:
            x = F.interpolate(x, scale_factor=0.5)
        if self.learn_able_sc:
            x = self.conv_sc(x)
        return x

    def forward(self, x):
        h = self.conv_block(x)
        # if self.downsample:
        #     h = F.interpolate(h, scale_factor=0.5)
        # out = h + self.shortcut(x)
        out = h + self.skip(x)
        return out / np.sqrt(2)
class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation, pool=False, use_sn=False):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.pool = pool
        if not use_sn:
            self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
            self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        else:
            self.query_conv = nn.utils.spectral_norm(
                nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1))
            self.key_conv = nn.utils.spectral_norm(
                nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1))
            self.value_conv = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        if self.pool:
            x = F.max_pool2d(x, 3, 2, 1)

        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        if self.pool:
            out = F.interpolate(out, scale_factor=2, mode='bilinear')

        return out
# Defines the PatchGAN discriminator with the specified arguments.
# input_dim ,output_dim, kernel_size, stride, padding=0, norm='none', activation='relu', pad_type='zero'
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc,
                 ndf=64,
                 n_layers=3,
                 norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False,
                 getIntermFeat=False,
                 use_sn=False,
                 use_attention=False,
                 dresblock=False,
                 not_first_sn=False,  # In order to be compatible with the very old setting
                 realness_gan=True,
                 input_size=1024,
                 dense_connect=False,
                 compress_channel=False,
                 blur_layer=False,
                 stddev=False
                 ):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.attention = use_attention
        # add downsample resoblock for discriminator
        self.resblock = dresblock
        self.realness_gan = realness_gan
        self.dense_connect = dense_connect
        self.use_sigmoid = use_sigmoid
        # add stddev
        self.stddev = stddev
        self.stddev_group = 4
        self.stddev_feat = 1
        kw = 3
        # padw = int(np.ceil((kw - 1.0) / 2))
        padw = 1

        if self.resblock:
            sequence = [[DResnetBlock(input_nc, ndf, padding_type='zero',
                                      norm_layer=norm_layer, wide=False, down_sample=True,
                                      use_sn=use_sn)]]
        elif blur_layer:
            sequence = [conv_blur_block(input_nc, ndf, use_sn=use_sn)]
        else:
            # if use_sn
            # not use SN in the first layer
            if use_sn and not not_first_sn:
                sequence = [[nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf,
                                                              kernel_size=kw, stride=2, padding=padw)),
                             nn.LeakyReLU(0.2, True)]]
            else:
                sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                             nn.LeakyReLU(0.2, True)]]

        nf = ndf
        # move modify layer to here
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * 2, 512)
            wide = (nf_prev != nf)
            if self.resblock:
                sequence += [[DResnetBlock(nf_prev, nf, padding_type='zero', norm_layer=norm_layer, wide=wide,
                                           down_sample=True, use_sn=use_sn)]]
            elif blur_layer:
                sequence += [[conv_blur_block(nf_prev, nf)]]
            else:
                if use_sn:
                    sequence += [[
                        nn.utils.spectral_norm(
                            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                        nn.LeakyReLU(0.2, True)
                    ]]
                else:
                    sequence += [[
                        nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                  stride=2, padding=padw),
                        norm_layer(nf), nn.LeakyReLU(0.2, True)
                    ]]

            # add attention block
            if self.attention and n == (n_layers - 1):
                # process size: 64x64
                sequence += [[SelfAttention(in_dim=nf,
                                            activation=None, pool=False)]]

        out_nf = 1 if compress_channel else nf
        #if self.resblock:
        #    sequence += [[nn.ReLU(True)]]
        if not realness_gan:
            if not use_sn:
                sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
            else:
                sequence += [[nn.utils.spectral_norm(nn.Conv2d(nf, 1,
                                       kernel_size=kw, stride=1, padding=padw))]]
        else:
            out_dim = nf
            if self.dense_connect:
                if not self.stddev:
                    sequence += [[nn.utils.spectral_norm(nn.Conv2d(nf, out_dim, kernel_size=3, stride=1, padding=1))]]
                else:
                    self.final_conv = nn.utils.spectral_norm(
                        nn.Conv2d(nf + 1, out_dim, kernel_size=3, stride=1, padding=1))
                linear_in = (input_size / (2 ** (n_layers + 1))) ** 2 * out_dim
                self.fc = nn.Sequential(
                    nn.utils.spectral_norm(nn.Linear(int(linear_in), nf)), nn.LeakyReLU(0.2, False),
                    nn.utils.spectral_norm(nn.Linear(nf, 51)))
            else:
                # Default as spectral normalization setting
                if not self.stddev:
                    sequence += [[nn.Conv2d(nf, 1, kernel_size=3, stride=1, padding=1)]]
                else:
                    self.final_conv = nn.Conv2d(nf + 1, 1, kernel_size=3, stride=1, padding=1)
                linear_in = (input_size / (2 ** (n_layers + 1))) ** 2
                self.fc = nn.Linear(int(linear_in), 51)

        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def _apply_stddev(self, out):
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
                          )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        return out

    def forward(self, input):
        print(input.shape)
        if not self.realness_gan:
            if self.getIntermFeat:
                res = [input]
                for n in range(self.n_layers + 2):
                    model = getattr(self, 'model' + str(n))
                    res.append(model(res[-1]))
                final_out = res[-1]
                return final_out, res[1:]
            else:
                b, c, h, w = input.size()
                if self.dense_connect:
                    h_out = self.model(input)
                    h_out = h_out.view(b, -1)
                    output = self.final_linear(h_out)
                else:
                    output = self.model(input)
                if self.use_sigmoid:
                    output = self.sigmoid(output)
                return output
        else:
            if self.getIntermFeat:
                res = [input]
                k = 2 if not self.stddev else 1
                for n in range(self.n_layers + k):
                    model = getattr(self, 'model' + str(n))
                    res.append(model(res[-1]))

                batch_size = input.shape[0]
                if not self.stddev:
                    pre_linear = res[-1].view(batch_size, -1)
                else:
                    out = self._apply_stddev(res[-1])
                    pre_linear = self.final_conv(out).view(batch_size, -1)

                final_out = self.fc(pre_linear)
                return final_out, res[1:]

            batch_size = input.shape[0]
            if not self.stddev:
                pre_linear = self.model(input).view(batch_size, -1)
            else:
                out = self._apply_stddev(self.model(input))
                pre_linear = self.final_conv(out).view(batch_size, -1)

            output_digits = self.fc(pre_linear)

            return output_digits

def define_D(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    which_model = opt_net['model_D_type']

    if which_model == 'discriminator_wgan':
        netD = Discriminator_VGG_128(in_nc=opt_net['in_channel'], nf=opt_net['num_channels'])
    elif which_model == 'sgan':
        netD = NLayerDiscriminator(3)
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    init_weights(netD, init_type='kaiming', scale=0.1)
    # if gpu_ids:
    #     netD = nn.DataParallel(netD)
    return netD


####################
# Perceptual Network
####################

# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 opt,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=False)
        else:
            model = torchvision.models.vgg19(pretrained=False)
        if isinstance(model, nn.DataParallel):
            model = model.module
        model.load_state_dict(torch.load(opt["path"]["VGG_model_path"]), strict=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(opt=opt, feature_layer=feature_layer, use_bn=use_bn,
                               use_input_norm=True, device=device)
    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF


####################
# init
####################

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1.0, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))