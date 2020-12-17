import torch
import numpy as np
from torch import nn
from parameters import setup_test_parser, print_args
import data_utils
import seaborn as sb
from argparse import Namespace
from utils import get_model
from scipy.stats import spearmanr

# partially from https://snappishproductions.com/blog/2018/01/03/class-activation-mapping-in-pytorch.html.html

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, l = feature_conv.shape
    if class_idx is None:
        cam = weight_fc.dot(feature_conv.reshape((nc, l)))
    else:
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, l)))
    cam = cam.reshape(l)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]
    
def createCAM(args, model):
    weight_softmax_params = list(model._modules.get('linear').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    
    i = np.random.randint(low=0, high=len(raw_data))
    input = tensor_input[i].reshape(1, *tensor_input.shape[1:])

    final_conv_layer = model._modules.get('tcn')._modules.get('network')._modules.get('7')._modules.get('conv2')
    activated_features = SaveFeatures(final_conv_layer)
    
    prediction = model(input)
    pred_probability = torch.sigmoid(prediction)
    
    activated_features.remove()
    
    class_idx = None
    
    overlay = getCAM(activated_features.features, weight_softmax, class_idx)
    return overlay, i
    
def plot_heatmap(args, raw_data, heatmap, idx):
    fig, axs = plt.subplots(len(args.x_vals)+1, 1)
    for i, x_val in enumerate(args.x_vals):
        x = range(len(raw_data[idx][x_val]))
        y = raw_data[idx][x_val]
        axs[i].plot(x, y)
        axs[i].set_ylabel(x_val)
    axs[i].set_xlabel('Workout time step')
    h = sb.heatmap(overlay, yticklabels = ['Final layer activation'], ax = axs[-1])
    h.tick_params(left=False, bottom=False, labelbottom=False)
    
    fig.suptitle('CAM for '+raw_data[idx][args.y_vals[0]]+', workout ID: '+raw_data[idx]['id'])
    plt.tight_layout()
    plt.savefig(f'workout_{raw_data[idx]["id"]}_cam.png', dpi=300)
    plt.close()

def main(args):
    info = torch.load(args.model_path)
    train_args = Namespace(**info['args'])
    
    model = get_model(train_args)
    model.load_state_dict(info['model_state_dict'])
    
    raw_data, tensor_input, tensor_y = data_utils.get_npy_data(train_args)
    
    heatmap, idx = createCAM(train_args, model)
    plot_heatmap(args, raw_data, heatmap, idx)

if __name__ == "__main__":
    args = setup_test_parser().parse_args()
    print_args(args)
    main(args)
