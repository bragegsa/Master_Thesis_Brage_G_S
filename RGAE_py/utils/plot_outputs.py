import matplotlib.pyplot as plt
import numpy as np

def plot_loss(epochs_loss, losses):

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[1].set_title("Epoch loss")
    axs[0].plot([loss.detach().numpy() for loss in epochs_loss[:]], linewidth=1.5)

    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Loss')
    axs[1].set_title("Iteration loss")
    axs[1].plot([loss.detach().numpy() for loss in losses[:]], linewidth=1.5)

    plt.show()

def plot_det_map(input, gt, output):
    '''
        input: HSI in 2D (only 1 spectral band)
        gt: ground truth
        output: the output of the network
    '''

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    axs[0].axis('off')
    axs[0].set_title('Input data (HSI)')
    axs[0].imshow(input)
    axs[1].axis('off')
    axs[1].set_title('Ground truth')
    axs[1].imshow(gt)
    axs[2].axis('off')
    axs[2].set_title('Detection map')
    axs[2].imshow(output, cmap='jet')

    plt.show()

def plot_auc_and_epochloss(auc_list, epochs_loss):
    
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('AUC score')
    axs[0].set_title("AUC each epoch")
    axs[0].plot(auc_list, linewidth=1.5)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].set_title("Epoch loss")
    axs[1].plot([loss.detach().numpy() for loss in epochs_loss[:]], linewidth=1.5)

    plt.show()