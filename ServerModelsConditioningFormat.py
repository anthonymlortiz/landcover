import numpy as np
import os
from ServerModelsAbstract import BackendModel
import torch
import torch.nn as nn
from torch.autograd import Variable
from fusionnet import Fusionnet
from unet import Unet
import json


def softmax(output):
    output_max = np.max(output, axis=2, keepdims=True)
    exps = np.exp(output-output_max)
    exp_sums = np.sum(exps, axis=2, keepdims=True)
    return exps/exp_sums


class Fusionnet_gn_model(BackendModel):

    def __init__(self, model_fn, gpuid):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        self.model_fn = model_fn
       # self.opts = json.load(open("/mnt/blobfuse/train-output/conditioning/models/backup_conditional_superres512/training/params.json", "r"))["model_opts"]
        self.opts = json.load(
            open("/mnt/blobfuse/train-output/conditioning/models/backup_fusionnet32_gn_runningstats8/training/params.json",
                 "r"))["model_opts"]

    def run(self, naip_data, naip_fn, extent, buffer, gammas, betas, dropouts):
        return self.run_model_on_tile(naip_data, gammas, betas, dropouts), os.path.basename(self.model_fn)


    def run_model_on_tile(self, naip_tile, gammas, betas, dropouts, batch_size=32):
        inf_framework = InferenceFramework(Fusionnet, self.opts)
        inf_framework.load_model(self.model_fn)
        y_hat = inf_framework.predict_entire_image_gammas_fusionnet(naip_tile, gammas, betas, dropouts)
        output = y_hat[:, :, 1:5]
        return softmax(output)

class Unet_gn_model(BackendModel):

    def __init__(self, model_fn, gpuid):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        self.model_fn = model_fn
        self.opts = json.load(open("/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_runningstats4/training/params.json", "r"))["model_opts"]

    def run(self, naip_data, naip_fn, extent, buffer, gammas, betas, dropouts):
        return self.run_model_on_tile(naip_data, gammas, betas, dropouts), os.path.basename(self.model_fn)


    def run_model_on_tile(self, naip_tile, gammas, betas, dropouts, batch_size=32):
        inf_framework = InferenceFramework(Unet, self.opts)
        inf_framework.load_model(self.model_fn)
        y_hat = inf_framework.predict_entire_image_gammas_unet(naip_tile, gammas, betas, dropouts)
        output = y_hat[:, :, 1:5]
        return softmax(output)

class Finetuned_unet_gn_model(BackendModel):

    def __init__(self, model_fn, gpuid):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        self.model_fn = model_fn
        self.opts = json.load(open("/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_runningstats4/training/params.json", "r"))["model_opts"]
        self.unet = Unet(self.opts)
        checkpoint = torch.load("/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_runningstats4/training/checkpoint_best.pth.tar")
        self.unet.load_state_dict(checkpoint['model'])
        self.unet.eval()
        for param in self.unet.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        self.model = GroupParams(self.unet)
        checkpoint1 = torch.load(self.model_fn)
        self.model.load_state_dict(checkpoint1)

    def run(self, naip_data, naip_fn, extent, buffer, gammas, betas, dropouts):
        return self.run_model_on_tile(naip_data, gammas, betas, dropouts), os.path.basename(self.model_fn)


    def run_model_on_tile(self, naip_tile, gammas, betas, dropouts, batch_size=32):
        y_hat = self.predict_entire_image_unet_fine(naip_tile)
        output = y_hat[:, :, 1:5]
        return softmax(output)

    def predict_entire_image_unet_fine(self, x):
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        if torch.cuda.is_available():
            self.model.cuda()
        x = np.rollaxis(x, 2, 1)
        x = x[:4, :, :]
        norm_image = x / 255.0
        _, w, h = norm_image.shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = np.zeros((5, w, h))

        norm_image1 = norm_image[:, 130:w - (w % 892) + 130, 130:h - (h % 892) + 130]
        x_c_tensor1 = torch.from_numpy(norm_image1).float().to(device)
        y_pred1 = self.model.forward(x_c_tensor1.unsqueeze(0))
        y_hat1 = (Variable(y_pred1).data).cpu().numpy()
        out[:, 92 + 130:w - (w % 892) + 130 - 92, 92 + 130:h - (h % 892) - 92 + 130] = y_hat1
        pred = np.rollaxis(out, 0, 3)
        pred = np.moveaxis(pred, 0, 1)
        return pred

class Finetuned_fusionnet_gn_model(BackendModel):

    def __init__(self, model_fn, gpuid):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        self.output_channels = 5
        self.input_size = 512
        self.model_fn = model_fn
        self.opts = json.load(open("/mnt/blobfuse/train-output/conditioning/models/backup_fusionnet32_gn_runningstats8/training/params.json", "r"))["model_opts"]
        self.fusionnet = Fusionnet(self.opts)
        checkpoint = torch.load("/mnt/blobfuse/train-output/conditioning/models/backup_fusionnet32_gn_runningstats8/training/checkpoint_best.pth.tar")
        self.fusionnet.load_state_dict(checkpoint['model'])
        self.fusionnet.eval()
        for param in self.fusionnet.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        self.model = GroupParamsFusionnet(self.fusionnet)
        checkpoint1 = torch.load(self.model_fn)
        self.model.load_state_dict(checkpoint1)

    def run(self, naip_data, naip_fn, extent, buffer, gammas, betas, dropouts):
        return self.run_model_on_tile(naip_data, gammas, betas, dropouts), os.path.basename(self.model_fn)


    def run_model_on_tile(self, naip_tile, gammas, betas, dropouts, batch_size=32):
        y_hat = self.predict_entire_image_fusionnet_fine(naip_tile)
        output = y_hat[:, :, 1:5]
        return softmax(output)

    def predict_entire_image_fusionnet_fine(self, x):
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        if torch.cuda.is_available():
            self.model.cuda()
        x = np.rollaxis(x, 2, 1)
        x = x[:4, :, :]
        naip_tile = x / 255.0

        down_weight_padding = 40
        height = naip_tile.shape[1]
        width = naip_tile.shape[2]

        stride_x = self.input_size - down_weight_padding * 2
        stride_y = self.input_size - down_weight_padding * 2

        output = np.zeros((self.output_channels, height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32) + 0.000000001
        kernel = np.ones((self.input_size, self.input_size), dtype=np.float32) * 0.1
        kernel[20:-20, 20:-20] = 1
        kernel[down_weight_padding:down_weight_padding + stride_y,
        down_weight_padding:down_weight_padding + stride_x] = 5

        batch = []
        batch_indices = []

        batch_count = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for y_index in (list(range(0, height - self.input_size, stride_y)) + [height - self.input_size, ]):
            for x_index in (list(range(0, width - self.input_size, stride_x)) + [width - self.input_size, ]):
                naip_im = naip_tile[:, y_index:y_index + self.input_size, x_index:x_index + self.input_size]
                batch.append(naip_im)
                batch_indices.append((y_index, x_index))
                batch_count += 1
        batch_arr = np.zeros((batch_count, 4, self.input_size, self.input_size))
        i = 0
        for im in batch:
            batch_arr[i, :, :, :] = im
            i += 1
        batch = torch.from_numpy(batch_arr).float().to(device)
        model_output = self.model.forward(batch)
        model_output = (Variable(model_output).data).cpu().numpy()
        for i, (y, x) in enumerate(batch_indices):
            output[:, y:y + self.input_size, x:x + self.input_size] += model_output[i] * kernel[np.newaxis, ...]
            counts[y:y + self.input_size, x:x + self.input_size] += kernel

        output = output / counts[np.newaxis, ...]
        pred = np.rollaxis(output, 0, 3)
        pred = np.moveaxis(pred, 0, 1)
        return pred


class InferenceFramework():
    def __init__(self, model, opts):
        self.opts = opts
        self.model = model(self.opts)
        self.output_channels = 5
        self.input_size = 512


    def load_model(self, path_2_saved_model):
        checkpoint = torch.load(path_2_saved_model)

        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()


    def fusionnet_gn_fun(self, x, gamma, beta, dropouts):
        """
        Activations to write for the duke U-net
        """
        down_1 = self.model.down_1(x)
        pool_1 = self.model.pool_1(down_1)
        down_2 = self.model.down_2(pool_1)
        pool_2 = self.model.pool_2(down_2)
        down_3 = self.model.down_3(pool_2)
        pool_3 = self.model.pool_3(down_3)
        down_4 = self.model.down_4(pool_3)
        pool_4 = self.model.pool_4(down_4)

        bridge = self.model.bridge(pool_4)

        deconv_1 = self.model.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4) / 2
        up_1 = self.model.up_1(skip_1)
        deconv_2 = self.model.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3) / 2
        up_2 = self.model.up_2(skip_2)
        deconv_3 = self.model.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2) / 2
        up_3 = self.model.up_3(skip_3)
        deconv_4 = self.model.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2
        up_4 = self.model.up_4(skip_4)

        gammas = np.zeros((1, 32, 1, 1))
        gammas[0, :8, 0, 0] = gamma[0]
        gammas[0, 8:16, 0, 0] = gamma[1]
        gammas[0, 16:24, 0, 0] = gamma[2]
        gammas[0, 24:32, 0, 0] = gamma[3]

        betas = np.zeros((1, 32, 1, 1))
        betas[0, :8, 0, 0] = beta[0]
        betas[0, 8:16, 0, 0] = beta[1]
        betas[0, 16:24, 0, 0] = beta[2]
        betas[0, 24:32, 0, 0] = beta[3]
        for id in dropouts:
            gammas[0, id, 0, 0] = 0
        gammas = torch.Tensor(gammas).to('cuda')
        betas = torch.Tensor(betas).to('cuda')
        up_4 = up_4 * gammas + betas

        out = self.model.out(up_4)

        #out = self.model.out_2(out)

        return out

    def unet_gn_fun(self, x, gamma, beta, dropouts):
        """
        Activations to write for the duke U-net
        """
        x, conv1_out, conv1_dim = self.model.down_1(x)

        gammas = np.zeros((1, 32, 1, 1))
        gammas[0, :8, 0, 0] = gamma[0]
        gammas[0, 8:16, 0, 0] = gamma[1]
        gammas[0, 16:24, 0, 0] = gamma[2]
        gammas[0, 24:32, 0, 0] = gamma[3]

        betas = np.zeros((1, 32, 1, 1))
        betas[0, :8, 0, 0] = beta[0]
        betas[0, 8:16, 0, 0] = beta[1]
        betas[0, 16:24, 0, 0] = beta[2]
        betas[0, 24:32, 0, 0] = beta[3]

        for id in dropouts:
            gammas[0, id, 0, 0] = 0
        gammas = torch.Tensor(gammas).to('cuda')
        betas = torch.Tensor(betas).to('cuda')
        x = x * gammas + betas

        x, conv2_out, conv2_dim = self.model.down_2(x)
        x, conv3_out, conv3_dim = self.model.down_3(x)
        x, conv4_out, conv4_dim = self.model.down_4(x)

        # Bottleneck
        x = self.model.conv5_block(x)

        # up layers
        x = self.model.up_1(x, conv4_out, conv4_dim)
        x = self.model.up_2(x, conv3_out, conv3_dim)
        x = self.model.up_3(x, conv2_out, conv2_dim)
        x = self.model.up_4(x, conv1_out, conv1_dim)

        return self.model.conv_final(x)

    def cond_fusionnet_gn_fun(self, x, gamma, beta, dropouts):
        """
        Activations to write for the duke U-net
        """
        conditioning_info = self.model.conditioning_model.pre_pred(x)

        cbn = self.model.fc_cbn(conditioning_info)
        gammas = cbn[:, :int(cbn.shape[1] / 2)]
        betas = cbn[:, int(cbn.shape[1] / 2):]

        down_1 = self.model.down_1(x, gammas[:, :32], betas[:, :32])
        pool_1 = self.model.pool_1(down_1)
        down_2 = self.model.down_2(pool_1, gammas[:, 32:96], betas[:, 32:96])
        pool_2 = self.model.pool_2(down_2)
        down_3 = self.model.down_3(pool_2, gammas[:, 96:224], betas[:, 96:224])
        pool_3 = self.model.pool_3(down_3)
        down_4 = self.model.down_4(pool_3, gammas[:, 224:480], betas[:, 224:480])
        pool_4 = self.model.pool_4(down_4)

        bridge = self.model.bridge(pool_4, gammas[:, 480:992], betas[:, 480:992])

        deconv_1 = self.model.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4) / 2
        up_1 = self.model.up_1(skip_1, gammas[:, 992:1248], betas[:, 992:1248])
        deconv_2 = self.model.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3) / 2
        up_2 = self.model.up_2(skip_2, gammas[:, 1248:1376], betas[:, 1248:1376])
        deconv_3 = self.model.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2) / 2
        up_3 = self.model.up_3(skip_3, gammas[:, 1376:1440], betas[:, 1376:1440])
        deconv_4 = self.model.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2
        up_4 = self.model.up_4(skip_4, gammas[:, 1440:1472], betas[:, 1440:1472])

        gammas = np.zeros((1, 32, 1, 1))
        gammas[0, :8, 0, 0] = gamma[0]
        gammas[0, 8:16, 0, 0] = gamma[1]
        gammas[0, 16:24, 0, 0] = gamma[2]
        gammas[0, 24:32, 0, 0] = gamma[3]

        betas = np.zeros((1, 32, 1, 1))
        betas[0, :8, 0, 0] = beta[0]
        betas[0, 8:16, 0, 0] = beta[1]
        betas[0, 16:24, 0, 0] = beta[2]
        betas[0, 24:32, 0, 0] = beta[3]
        gammas = torch.Tensor(gammas).to('cuda')
        betas = torch.Tensor(betas).to('cuda')
        up_4 = up_4 * gammas + betas

        out = self.out(up_4)
        out = self.out_2(out)
        # out = torch.clamp(out, min=-1, max=1)

        return out

    def predict_entire_image_unet(self, x):
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        if torch.cuda.is_available():
            self.model.cuda()
        x = np.rollaxis(x, 2, 1)
        x = x[:4, :, :]
        norm_image = x / 255.0
        _, w, h = norm_image.shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = np.zeros((5,w,h))
        r = np.pad(norm_image[0, :, :], ((92, 92), (92, 92)), 'constant', constant_values=((0, 0), (0, 0)))
        g = np.pad(norm_image[1, :, :], ((92, 92), (92, 92)), 'constant', constant_values=((0, 0), (0, 0)))
        b = np.pad(norm_image[2, :, :], ((92, 92), (92, 92)), 'constant', constant_values=((0, 0), (0, 0)))
        ir = np.pad(norm_image[3, :, :], ((92, 92), (92, 92)), 'constant', constant_values=((0, 0), (0, 0)))

        rw, rh = r.shape
        norm_image_padded = np.zeros((3, rw, rh))
        norm_image_padded[0, :, :] = r
        norm_image_padded[1, :, :] = g
        norm_image_padded[2, :, :] = b
        norm_image_padded[3, :, :] = ir
        # print("norm image", norm_image_padded.shape)

        x_chips = self.cunet_chip(norm_image_padded)
        y_hat_chips = []
        for x_c in x_chips:
            x_c_tensor1 = torch.from_numpy(x_c).float().to(device)
            y_pred1 = self.model.forward(x_c_tensor1.unsqueeze(0))
            y_hat1 = (Variable(y_pred1).data).cpu().numpy()
            y_hat_chips.append(y_hat1)
        out = self.cunet_stitch_mask(
            y_hat_chips,w,h
        )
        pred = np.rollaxis(out, 0, 3)
        pred = np.moveaxis(pred, 0, 1)
        return pred

    def predict_entire_image_gammas_fusionnet(self, x, gammas, betas, dropouts):
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        if torch.cuda.is_available():
            self.model.cuda()
        x = np.rollaxis(x, 2, 1)
        x = x[:4, :, :]
        naip_tile = x / 255.0

        down_weight_padding = 40
        height = naip_tile.shape[1]
        width = naip_tile.shape[2]

        stride_x = self.input_size - down_weight_padding * 2
        stride_y = self.input_size - down_weight_padding * 2

        output = np.zeros((self.output_channels, height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32) + 0.000000001
        kernel = np.ones((self.input_size, self.input_size), dtype=np.float32) * 0.1
        kernel[20:-20, 20:-20] = 1
        kernel[down_weight_padding:down_weight_padding + stride_y,
        down_weight_padding:down_weight_padding + stride_x] = 5

        batch = []
        batch_indices = []

        batch_count = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for y_index in (list(range(0, height - self.input_size, stride_y)) + [height - self.input_size, ]):
            for x_index in (list(range(0, width - self.input_size, stride_x)) + [width - self.input_size, ]):
                naip_im = naip_tile[:, y_index:y_index + self.input_size, x_index:x_index + self.input_size]
                batch.append(naip_im)
                batch_indices.append((y_index, x_index))
                batch_count += 1
        batch_arr = np.zeros((batch_count, 4, self.input_size, self.input_size))
        i = 0
        for im in batch:
            batch_arr[i,:,:,:] = im
            i+=1
        batch = torch.from_numpy(batch_arr).float().to(device)
        model_output = self.fusionnet_gn_fun(batch, gammas, betas, dropouts)
        model_output = (Variable(model_output).data).cpu().numpy()
        for i, (y, x) in enumerate(batch_indices):
            output[:,y:y + self.input_size, x:x + self.input_size] += model_output[i] * kernel[np.newaxis, ...]
            counts[y:y + self.input_size, x:x + self.input_size] += kernel

        output = output / counts[np.newaxis, ...]
        pred = np.rollaxis(output, 0, 3)
        pred = np.moveaxis(pred, 0, 1)
        return pred


    def predict_entire_image_gammas_unet(self, x, gammas, betas, dropouts):
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        if torch.cuda.is_available():
            self.model.cuda()
        x = np.rollaxis(x, 2, 1)
        x = x[:4, :, :]
        norm_image = x / 255.0
        _, w, h = norm_image.shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = np.zeros((5, w, h))

        norm_image1 = norm_image[:, 130:w - (w % 892)+130, 130:h - (h % 892)+130]
        x_c_tensor1 = torch.from_numpy(norm_image1).float().to(device)
        y_pred1 = self.unet_gn_fun(x_c_tensor1.unsqueeze(0), gammas, betas, dropouts)
        y_hat1 = (Variable(y_pred1).data).cpu().numpy()
        out[:, 92+130:w - (w % 892)+130-92, 92+130:h - (h % 892)-92+130] = y_hat1
        pred = np.rollaxis(out, 0, 3)
        pred = np.moveaxis(pred, 0, 1)
        return pred

class GroupParamsFusionnet(nn.Module):

    def __init__(self, model):
        super(GroupParamsFusionnet, self).__init__()
        self.gammas = nn.Parameter(torch.ones((1, 32, 1, 1)))
        self.betas = nn.Parameter(torch.zeros((1, 32, 1, 1)))
        self.gammas2 = nn.Parameter(torch.ones((1, 64, 1, 1)))
        self.betas2 = nn.Parameter(torch.zeros((1, 64, 1, 1)))
        self.model = model

    def forward(self, input):


        down_1 = self.model.down_1(input)
        pool_1 = self.model.pool_1(down_1)
        #pool_1 = pool_1 * self.gammas + self.betas
        down_2 = self.model.down_2(pool_1)
        pool_2 = self.model.pool_2(down_2)
        down_3 = self.model.down_3(pool_2)
        pool_3 = self.model.pool_3(down_3)
        down_4 = self.model.down_4(pool_3)
        pool_4 = self.model.pool_4(down_4)

        bridge = self.model.bridge(pool_4)

        deconv_1 = self.model.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4) / 2
        up_1 = self.model.up_1(skip_1)
        deconv_2 = self.model.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3) / 2
        up_2 = self.model.up_2(skip_2)
        deconv_3 = self.model.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2) / 2
        up_3 = self.model.up_3(skip_3)
        up_3 = up_3 * self.gammas2 + self.betas2

        deconv_4 = self.model.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1) / 2
        up_4 = self.model.up_4(skip_4)
        up_4 = up_4 * self.gammas + self.betas

        out = self.model.out(up_4)
        out = self.model.out_2(out)
        # out = torch.clamp(out, min=-1, max=1)
        return out


class GroupParams(nn.Module):

    def __init__(self, model):
        super(GroupParams, self).__init__()
        self.gammas = nn.Parameter(torch.ones((1, 32, 1, 1)))
        self.betas = nn.Parameter(torch.zeros((1, 32, 1, 1)))
        self.model = model

    def forward(self, x):
        x, conv1_out, conv1_dim = self.model.down_1(x)
        x = x * self.gammas + self.betas

        # gammas = np.zeros((1, 32, 1, 1))
        # gammas[0, :8, 0, 0] = self.gammas.detach().numpy()[0]
        # gammas[0, 8:16, 0, 0] = self.gammas.detach().numpy()[1]
        # gammas[0, 16:24, 0, 0] = self.gammas.detach().numpy()[2]
        # gammas[0, 24:32, 0, 0] = self.gammas.detach().numpy()[3]
        #
        # betas = np.zeros((1, 32, 1, 1))
        # betas[0, :8, 0, 0] = self.betas.detach().numpy()[0]
        # betas[0, 8:16, 0, 0] = self.betas.detach().numpy()[1]
        # betas[0, 16:24, 0, 0] = self.betas.detach().numpy()[2]
        # betas[0, 24:32, 0, 0] = self.betas.detach().numpy()[3]

       # gammas = torch.Tensor(gammas).to('cuda')
       # betas = torch.Tensor(betas).to('cuda')


        x, conv2_out, conv2_dim = self.model.down_2(x)
        x = x * self.gammas2 + self.betas2

        x, conv3_out, conv3_dim = self.model.down_3(x)
        x, conv4_out, conv4_dim = self.model.down_4(x)

        # Bottleneck
        x = self.model.conv5_block(x)

        # up layers
        x = self.model.up_1(x, conv4_out, conv4_dim)
        x = self.model.up_2(x, conv3_out, conv3_dim)
        x = self.model.up_3(x, conv2_out, conv2_dim)
        x = self.model.up_4(x, conv1_out, conv1_dim)


        return self.model.conv_final(x)
"""
        x_chips = self.cunet_chip(norm_image)
        y_hat_chips = []
        for x_c in x_chips:
            # 2636x2636
            # 2452x2452
            # get predictions of size 2452x2452
            x_c_tensor1 = torch.from_numpy(x_c).float().to(device)
            y_pred1 = self.unet_gn_fun(x_c_tensor1.unsqueeze(0), gammas, betas, dropouts)
            y_hat1 = (Variable(y_pred1).data).cpu().numpy()
            y_hat_chips.append(y_hat1)
        out = self.cunet_stitch_mask(
            y_hat_chips, w, h
        )
        pred = np.rollaxis(out, 2, 1)
        print(pred.shape)
        pred = np.rollaxis(out, 0, 3)
        pred = np.moveaxis(pred, 0, 1)
        return pred
        """




