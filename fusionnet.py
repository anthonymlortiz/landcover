import torch
import torch.nn as nn
import numpy as np
from group_norm import GroupNorm2d

class Conv_residual_conv(nn.modules.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3

def conv_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        GroupNorm2d(out_dim, num_groups=4, affine=True, track_running_stats=True),
        act_fn,
    )
    return model


def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        GroupNorm2d(out_dim, num_groups=4, affine=True, track_running_stats=True),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        GroupNorm2d(out_dim, num_groups=4, affine=True, track_running_stats=True),
    )
    return model


def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        GroupNorm2d(out_dim, num_groups=4, affine=True, track_running_stats=True),
    )
    return model


class Fusionnet(nn.modules.Module):

    def __init__(self, model_opts):
        super(Fusionnet, self).__init__()
        self.opts = model_opts["unet_opts"]
        self.in_dim = self.opts["n_input_channels"]
        self.out_dim = self.opts["n_filters"]
        self.final_out_dim = self.opts["n_classes"]
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        print("\n------Initiating FusionNet------\n")

        # encoder

        self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = maxpool()

        # bridge

        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)

        # decoder

        self.deconv_1 = conv_trans_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = conv_trans_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = conv_trans_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = conv_trans_block(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2)

        # output

        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)
        self.out_2 = nn.Tanh()
        '''
        self.out = nn.Sequential(
            nn.Conv2d(self.out_dim,self.final_out_dim, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(self.final_out_dim),
            nn.Tanh(),
        )
        '''

        # initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):

        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4) / 2
        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3) / 2
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2) / 2
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2
        up_4 = self.up_4(skip_4)

        out = self.out(up_4)
        out = self.out_2(out)
        # out = torch.clamp(out, min=-1, max=1)
        return out

    """
class GroupNorm(nn.Module):
    def __init__(self, num_features, channels_per_group=8, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.channels_per_group = channels_per_group
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = int(C/self.channels_per_group)
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

       # print(var.shape)
        #if n == 4:
            # print("Mean",mean)
           # print("Var REAL", x.var(-1, keepdim=True))


        _,n,_ = var.shape
        if n == 4 and var[0,0,0]>100:
            var[0,:,0] =torch.from_numpy(np.array([3808.6565,
         9257.1025,
         3660.0928,
         5798.0645]))
        elif n==4:
            var[0, :, 0] = torch.from_numpy(np.array([2.6291,
         2.7672,
         4.8078,
         8.1384]))
        if n==8 and var[0,0,0]<10000:
            var[0, :, 0] = torch.from_numpy(np.array([3135.5400,
         2231.0354,
          939.4938,
         1429.2697,
          254.0065,
         4154.1094,
         6880.1182,
          409.2467]))
        elif n==8:
            var[0, :, 0] = torch.from_numpy(np.array([23575.3750,
         99257.5781,
         54378.3047,
          9595.9014,
         20566.9844,
         21890.5137,
         50549.0469,
         20196.9941]))


        if n==16 and var[0,0,0]<50000:
            var[0, :, 0] = torch.from_numpy(np.array([ 6485.2842,
          4982.5801,
          5038.2832,
          3203.1484,
         26720.5547,
         12525.9072,
          6620.0205,
          4009.7852,
         18603.8887,
          3879.6147,
         18651.0000,
          7784.3535,
          3875.3140,
          4989.3140,
         17317.2695,
         12387.6611]))

        elif n==16 :
            var[0, :, 0] = torch.from_numpy(np.array([279357.5938,
         108401.8203,
         148729.7969,
         143381.7188,
         137962.4062,
         508200.0625,
         156837.0938,
         120432.4609,
         224339.5469,
         211053.0781,
         190445.8594,
         125831.5781,
         112907.0000,
         284965.4375,
          98772.5391,
         100268.8594]))


        if n==32 and var[0,0,0]<60000000:
            var[0, :, 0] = torch.from_numpy(np.array([1127342.8750,
          582421.5625,
          417171.0625,
          705904.5000,
          505460.0312,
          737456.1250,
          404555.9062,
          558486.2500,
          728479.1875,
         1161552.8750,
         1449123.6250,
          837998.5000,
          873951.6875,
          881552.7500,
          429365.3438,
          465764.5312,
          390617.6562,
          390902.0000,
          658295.4375,
          411266.1562,
          301606.4688,
          472866.2812,
          719657.3750,
          437742.5625,
          671126.1875,
          409630.2812,
          231775.1094,
          503776.0625,
          364464.2188,
          384450.7188,
          249571.8281,
          749674.0625]))
        elif n==32:
            var[0, :, 0] = torch.from_numpy(np.array([3.6560e+12,
         3.0970e+12,
         4.4772e+12,
         5.3185e+12,
         2.8309e+12,
         2.6579e+12,
         5.8887e+12,
         1.1299e+12,
         1.5420e+12,
         2.0512e+12,
         8.7220e+12,
         1.0344e+12,
         8.4140e+12,
         2.5447e+12,
         1.5298e+12,
         5.0352e+12,
         3.9295e+12,
         3.2397e+12,
         3.5290e+12,
         5.3887e+12,
         2.3891e+12,
         4.7125e+12,
         6.6654e+12,
         4.2175e+12,
         9.0795e+12,
         4.1916e+12,
         5.0819e+12,
         7.3338e+12,
         5.9512e+12,
         4.0978e+12,
         3.1517e+12,
         7.0164e+12]))

        if n==64:
            var[0, :, 0] = torch.from_numpy(np.array([15242029,
         16153170,
         26555114,
         15404270,
         14727756,
         16032939,
         22247736,
         21294150,
         14559857,
         19460650,
         27103658,
         17849042,
         15932710,
         19748860,
         19062124,
         15536215,
         30740708,
         18461288,
         12052730,
         16530036,
         21998762,
         28492356,
         25271440,
         14786707,
         11584878,
         16418677,
         19497330,
         21569332,
         23412836,
         16586966,
         18927684,
         13625426,
         21498374,
         23526502,
         15557969,
         19791828,
         30222362,
         24591754,
         19670958,
         17755498,
         22758854,
         21859234,
         18117132,
         18265260,
         20239548,
         24608548,
         18983918,
         20412978,
         17995068,
         22973714,
         13671396,
         16991116,
         23081160,
         17634158,
         18315092,
         16467814,
         19963372,
         19257464,
         21122746,
         14607588,
         25685710,
         23175112,
         16401916,
         25546240]))

        if n == 4:
           # print("Mean",mean)
            print("Var REAL", x.var(-1, keepdim=True))
        #var[:,:,:] = torch.ones(var.shape) * 2e6

        x = (x) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)

        return x * self.weight + self.bias
        """


"""
class GroupNormRunningStats(nn.Module):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked']
    def __init__(self, num_features, channels_per_group=8, eps=1e-5, momentum=0.1, affine=True, track_running_stats = True):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.channels_per_group = channels_per_group
        self.momentum = momentum
        self.affine = affine
        self.eps = eps
        self.groups = int(channels_per_group / self.channels_per_group)
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.groups))
            self.register_buffer('running_var', torch.ones(self.groups))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.uniform_(self.weight)
            init.zeros_(self.bias)

    @weak_script_method
    def forward(self, input):
        N,C,H,W = input.size()
        G = int(C/self.channels_per_group)
        assert C % G == 0

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        x = input.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

       # x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(GroupNormRunningStats, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
            """

if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(1, 3, 256, 256)
   # model = Fusionnet(model_opts)
   # y = model(im)
    #print(y.shape)
    #del model
    #del y
