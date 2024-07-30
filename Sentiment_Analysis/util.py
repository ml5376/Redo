
import os
import shutil
import urllib
import tarfile
import zipfile
import gzip
import tqdm

import matplotlib.pyplot as plt
import torch
from math import sqrt, ceil
import numpy as np


def gen_bar_updater():
    """tqdm report hook for urlretrieve"""
    pbar = tqdm.tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def download_url(url, root, filename):
    """
    Download a file with given filename from a given url to a given directory
    :param url: url from where to download
    :param root: root directory to which to download
    :param filename: filename under which the file should be saved
    """
    file_path = os.path.join(root, filename)
    os.makedirs(root, exist_ok=True)
    if not os.path.exists(file_path):
        print('Downloading ' + url + ' to ' + file_path)
        urllib.request.urlretrieve(
            url,
            file_path,
            reporthook=gen_bar_updater()
        )
    return file_path


def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tgz(filename):
    return filename.endswith(".tgz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")

def extract_archive(from_path, to_path=None, remove_finished=False):
    """
    Extract a given archive
    :param from_path: path to archive which should be extracted
    :param to_path: path to which archive should be extracted
        default: parent directory of from_path
    :param remove_finished: if set to True, delete archive after extraction
    """
    if not os.path.exists(from_path):
        return

    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(
            to_path,
            os.path.splitext(os.path.basename(from_path))[0]
        )
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as zip_:
            zip_.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)

def download_dataset(url, data_dir, dataset_zip_name, force_download=False):
    """
    Download dataset
    :param url: URL to download file from
    :param data_dir: Base name of the current dataset directory
    :param dataset_zip_name: Name of downloaded compressed dataset file
    :param force_download: If set to True, always download dataset
        (even if it already exists)
    """
    if not os.path.exists(data_dir) or not os.listdir(data_dir) or force_download:
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        data_file = download_url(url, data_dir, dataset_zip_name)
        extract_archive(data_file, remove_finished=True)



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def classifier_test(classifier, num_embeddings):
    # Define some constants
    seq_len=10
    batch_size=3
    
    # Create a random sequence
    x = torch.randint(0, num_embeddings-1, (seq_len, batch_size))
    #print("x")
   # print(x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # Test the output format
    # x.to(device)
    # y = classifier(x)

    # print(y)

    # passed = True
    # if not torch.logical_and((y <= 1), (y >= 0)).all():
    #     print('Your model does not output probabilities between 0 and 1!')
    #     passed = False
    # if y.shape != (batch_size, ):
    #     print('Your model does not produce a 1-D output of shape (batch_size, )')
    #     passed = False

    # # Test varying batch sizes
    # assert seq_len-batch_size > 0, "Seq len must be bigger than batch size"
    lengths = torch.tensor([seq_len-i for i in range(batch_size)]).long()
    batched_outputs = classifier(x, lengths)
    print("length numel!!!!!!!!!!!!LOOOOOKKKKKKKK HERE",batched_outputs)
    # print(lengths.numel())

    regular_outputs = torch.stack([
        classifier(x[:lengths[i], i].unsqueeze(1))
        for i in range(lengths.numel())
    ]).squeeze()

    print('regular outputs',regular_outputs)

    if batched_outputs.shape != regular_outputs.shape:
        print('Output with lengths {} produced wrong size argument {} vs {}'.format(
            lengths.tolist(), batched_outputs.shape, regular_outputs.shape
        ))
        print('Make sure you handle lengths argument properly in your classifier!')
        passed = False

    diff = torch.norm(batched_outputs - regular_outputs)
    if diff > 1e-9:
        print('Output with lengths {} has a large error: {}'.format(lengths.tolist(), diff))
        print('Make sure you handle lengths argument properly in your classifier!') 
        passed = False

    # Log the final result
    if passed:
        print('All output tests are passed :)!')
    else:
        print('Some output tests are failed :(!')
    return passed
