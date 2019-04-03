import numpy as np
import os
from ServerModelsAbstract import BackendModel
import torch
from torch.autograd import Variable
from fusionnet import Fusionnet
from unet import Unet
import json
import random

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
        self.opts = json.load(open("/mnt/blobfuse/train-output/conditioning/models/backup_conditional_superres512/training/params.json", "r"))["model_opts"]

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
        self.opts = json.load(open("/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn/training/params.json", "r"))["model_opts"]

    def run(self, naip_data, naip_fn, extent, buffer, gammas, betas, dropouts):
        return self.run_model_on_tile(naip_data, gammas, betas, dropouts), os.path.basename(self.model_fn)


    def run_model_on_tile(self, naip_tile, gammas, betas, dropouts, batch_size=32):
        inf_framework = InferenceFramework(Unet, self.opts)
        inf_framework.load_model(self.model_fn)
        y_hat = inf_framework.predict_entire_image_gammas_unet(naip_tile, gammas, betas, dropouts)
        output = y_hat[:, :, 1:5]
        return softmax(output)

class InferenceFramework():
    def __init__(self, model, opts):
        self.opts = opts
        self.model = model(self.opts)


    def load_model(self, path_2_saved_model):
        checkpoint = torch.load(path_2_saved_model)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def cunet_chip(self, x):
        _, w, h = x.shape
        in_dim = 604
        out_dim = in_dim - 184

        chips = []
        n = int((w-184)/out_dim)
        for i in range(n):
            chips.append(x[:, i * out_dim:i * out_dim + in_dim:, h - in_dim:])

        for j in range(n):
            chips.append(x[:, w - in_dim:, j * out_dim:j * out_dim + in_dim])
        for i in range(n):
            for j in range(n):
                chips.append(x[:, i * out_dim:i * out_dim + in_dim, j * out_dim:j * out_dim + in_dim])
                chips.append(x[:, i * out_dim+92,:i * out_dim + in_dim +92, j * out_dim+92,:j * out_dim + in_dim +92])



        chips.append(x[:, w - in_dim:, h - in_dim:])
        return chips

    def cunet_stitch_mask(self, y_hat_c, w, h):
        [img_width, img_height] = [w, h]
        mask = np.zeros([5, img_width, img_height])
        in_dim = 604
        out_dim = in_dim - 184
        n = int(w / out_dim)
        quarter = 0
        for i in range(n):
            mask[:,i * out_dim:(i + 1) * out_dim, img_height - out_dim:] = y_hat_c[quarter]
            quarter += 1

        for j in range(n):
            mask[:,img_width - out_dim:, j * out_dim:(j + 1) * out_dim] = y_hat_c[quarter]
            quarter += 1
        for i in range(n):
            for j in range(n):
                mask[:,i * out_dim:(i + 1) * out_dim, j * out_dim:(j + 1) * out_dim] = y_hat_c[quarter]
                quarter += 1
                mask[:, i * out_dim+92,:(i + 1) * out_dim+92, j * out_dim+92,:(j + 1) * out_dim + 92] = y_hat_c[quarter]
                quarter += 1


        mask[:,img_width - out_dim:, img_height - out_dim:] = y_hat_c[quarter]
        return mask

    def fusionnet_gn_fun(self, x, gamma, beta):
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
        gammas = torch.Tensor(gammas).to('cuda')
        betas = torch.Tensor(betas).to('cuda')
        up_4 = up_4 * gammas + betas

        out = self.model.out(up_4)

        out = self.model.out_2(out)

        return out

    def unet_gn_fun(self, x, gamma, beta, dropouts):
        """
        Activations to write for the duke U-net
        """
        x, conv1_out, conv1_dim = self.model.down_1(x)
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
        r = np.pad(norm_image[0, :, :], ((92, 92), (92, 92)), 'reflect')
        g = np.pad(norm_image[1, :, :], ((92, 92), (92, 92)), 'reflect')
        b = np.pad(norm_image[2, :, :], ((92, 92), (92, 92)), 'reflect')
        ir = np.pad(norm_image[3, :, :], ((92, 92), (92, 92)), 'reflect')

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
            #2636x2636
            #2452x2452
            #get predictions of size 2452x2452
            x_c_tensor1 = torch.from_numpy(x_c).float().to(device)
            y_pred1 = self.model.forward(x_c_tensor1.unsqueeze(0))
            y_hat1 = (Variable(y_pred1).data).cpu().numpy()
            y_hat_chips.append(y_hat1)
        out = self.cunet_stitch_mask(
            y_hat_chips,w,h
        )
        pred = np.rollaxis(out, 2, 1)
        print(pred.shape)
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
        norm_image = x / 255.0
        _, w, h = norm_image.shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = np.zeros((5,w,h))
        norm_image1 = norm_image[:, :w-(w%128), :h-(h%128)]
        norm_image2 = norm_image[:, (w % 128):w, (h % 128):h ]
        norm_image3 = norm_image[:, :w - (w % 128), (h % 128):h]
        norm_image4 = norm_image[:, (w % 128):w, :h - (h % 128)]
        x_c_tensor1 = torch.from_numpy(norm_image1).float().to(device)
        x_c_tensor2 = torch.from_numpy(norm_image2).float().to(device)
        x_c_tensor3 = torch.from_numpy(norm_image3).float().to(device)
        x_c_tensor4 = torch.from_numpy(norm_image4).float().to(device)
        y_pred1 = self.fusionnet_gn_fun(x_c_tensor1.unsqueeze(0), gammas, betas, dropouts)
        y_pred2 = self.fusionnet_gn_fun(x_c_tensor2.unsqueeze(0), gammas, betas, dropouts)
        y_pred3 = self.fusionnet_gn_fun(x_c_tensor3.unsqueeze(0), gammas, betas, dropouts)
        y_pred4 = self.fusionnet_gn_fun(x_c_tensor4.unsqueeze(0), gammas, betas, dropouts)
        y_hat1 = (Variable(y_pred1).data).cpu().numpy()
        y_hat2 = (Variable(y_pred2).data).cpu().numpy()
        y_hat3 = (Variable(y_pred3).data).cpu().numpy()
        y_hat4 = (Variable(y_pred4).data).cpu().numpy()
        out[:, :w - (w % 128), :h - (h % 128)] = y_hat1
        out[:, (w % 128):w, (h % 128):h ] = y_hat2
        out[:, :w - (w % 128), (h % 128):h] = y_hat3
        out[:, (w % 128):w, :h - (h % 128)] = y_hat4
        pred = np.rollaxis(out, 0, 3)
        pred = np.moveaxis(pred, 0, 1)
        print(pred.shape)
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
        print(norm_image.shape)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = np.zeros((5, w, h))
        r = np.pad(norm_image[0, :, :], ((92, 92), (92, 92)), 'reflect')
        g = np.pad(norm_image[1, :, :], ((92, 92), (92, 92)), 'reflect')
        b = np.pad(norm_image[2, :, :], ((92, 92), (92, 92)), 'reflect')
        ir = np.pad(norm_image[3, :, :], ((92, 92), (92, 92)), 'reflect')

        rw, rh = r.shape
        norm_image_padded = np.zeros((4, rw, rh))
        norm_image_padded[0, :, :] = r
        norm_image_padded[1, :, :] = g
        norm_image_padded[2, :, :] = b
        norm_image_padded[3, :, :] = ir
        # print("norm image", norm_image_padded.shape)

        x_chips = self.cunet_chip(norm_image_padded)
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



