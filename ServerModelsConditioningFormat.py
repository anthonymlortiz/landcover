import numpy as np
import os
from ServerModelsAbstract import BackendModel
import torch
from torch.autograd import Variable
from fusionnet import Fusionnet
import json

def softmax(output):
    output_max = np.max(output, axis=2, keepdims=True)
    exps = np.exp(output-output_max)
    exp_sums = np.sum(exps, axis=2, keepdims=True)
    return exps/exp_sums

class GnPytorchModel(BackendModel):

    def __init__(self, model_fn, gpuid):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
        self.model_fn = model_fn
        self.opts = json.load(open("/mnt/blobfuse/train-output/conditioning/models/backup_conditioning_groups/training/params.json", "r"))["model_opts"]

    def run(self, naip_data, naip_fn, extent, buffer, gammas, betas):
        return self.run_model_on_tile(naip_data, gammas, betas), os.path.basename(self.model_fn)


    def run_model_on_tile(self, naip_tile, gammas, betas, batch_size=32):
        inf_framework = InferenceFramework(Fusionnet, self.opts)
        inf_framework.load_model(self.model_fn)
        y_hat = inf_framework.predict_entire_image_gammas(naip_tile, gammas, betas)
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

    def predict_entire_image(self, x):
        x = np.swapaxes(x, 0, 2)
        x = np.swapaxes(x, 1, 2)
        if torch.cuda.is_available():
            self.model.cuda()
        x = np.rollaxis(x, 2, 1)
        x = x[:4, :, :]
        norm_image = x / 255.0
        _, w, h = norm_image.shape
        out = np.zeros((5,w,h))
        norm_image1 = norm_image[:, :w-(w%128), :h-(h%128)]
        norm_image2 = norm_image[:, (w % 128):w, (h % 128):h ]
        norm_image3 = norm_image[:, :w - (w % 128), (h % 128):h]
        norm_image4 = norm_image[:, (w % 128):w, :h - (h % 128)]
        x_c_tensor1 = torch.from_numpy(norm_image1).float()
        x_c_tensor2 = torch.from_numpy(norm_image2).float()
        x_c_tensor3 = torch.from_numpy(norm_image3).float()
        x_c_tensor4 = torch.from_numpy(norm_image4).float()
        y_pred1 = self.model.forward(x_c_tensor1.unsqueeze(0))
        y_pred2 = self.model.forward(x_c_tensor2.unsqueeze(0))
        y_pred3 = self.model.forward(x_c_tensor3.unsqueeze(0))
        y_pred4 = self.model.forward(x_c_tensor4.unsqueeze(0))
        y_hat1 = (Variable(y_pred1).data).cpu().numpy()
        y_hat2 = (Variable(y_pred2).data).cpu().numpy()
        y_hat3 = (Variable(y_pred3).data).cpu().numpy()
        y_hat4 = (Variable(y_pred4).data).cpu().numpy()
        out[:, :w - (w % 128), :h - (h % 128)] = y_hat1
        out[:, (w % 128):w, (h % 128):h ] = y_hat2
        out[:, :w - (w % 128), (h % 128):h] = y_hat3
        out[:, (w % 128):w, :h - (h % 128)] = y_hat4
        pred = np.rollaxis(out, 2, 1)
        print(pred.shape)
        return pred

    def predict_entire_image_gammas(self, x, gammas, betas):
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
        y_pred1 = self.fusionnet_gn_fun(x_c_tensor1.unsqueeze(0), gammas, betas)
        y_pred2 = self.fusionnet_gn_fun(x_c_tensor2.unsqueeze(0), gammas, betas)
        y_pred3 = self.fusionnet_gn_fun(x_c_tensor3.unsqueeze(0), gammas, betas)
        y_pred4 = self.fusionnet_gn_fun(x_c_tensor4.unsqueeze(0), gammas, betas)
        y_hat1 = (Variable(y_pred1).data).cpu().numpy()
        y_hat2 = (Variable(y_pred2).data).cpu().numpy()
        y_hat3 = (Variable(y_pred3).data).cpu().numpy()
        y_hat4 = (Variable(y_pred4).data).cpu().numpy()
        out[:, :w - (w % 128), :h - (h % 128)] = y_hat1
        out[:, (w % 128):w, (h % 128):h ] = y_hat2
        out[:, :w - (w % 128), (h % 128):h] = y_hat3
        out[:, (w % 128):w, :h - (h % 128)] = y_hat4
        pred = np.rollaxis(out, 0, 3)
        print(pred.shape)
        return pred


class CondFusionnetModel(BackendModel):

    def __init__(self, model_fn, gpuid):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

        # TODO build this

        self.model_fn = model_fn

    def run(self, naip_data, naip_fn, extent, buffer):
        return self.run_model_on_tile(naip_data)

    def run_model_on_tile(self, naip_tile, batch_size=32):
        inf_framework = InferenceFramework(Fusionnet)
        inf_framework.load_model(self.model_fn)
        y_hat = inf_framework.predict_entire_image(naip_tile)
        output = y_hat[:, :, 1:5]
        return output