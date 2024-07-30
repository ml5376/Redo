

import matplotlib.pyplot as plt

import os
import pickle


def save_model(model, file_name, directory="models"):
    """Save model as pickle"""
    model = model.cpu()
    model_dict = {
        "state_dict": model.state_dict(),
        "hparams": model.hparams
    }
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_path = os.path.join(directory, file_name)
    pickle.dump(model_dict, open(model_path, 'wb', 4))
    return model_path


def show_all_keypoints(image, keypoints, pred_kpts=None):
    """Show image with predicted keypoints"""
    image = (image.clone() * 255).view(96, 96)
    plt.imshow(image, cmap='gray')
    keypoints = keypoints.clone() * 48 + 48
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=200, marker='.', c='m')
    if pred_kpts is not None:
        pred_kpts = pred_kpts.clone() * 48 + 48
        plt.scatter(pred_kpts[:, 0], pred_kpts[:, 1], s=200, marker='.', c='r')
    plt.show()
