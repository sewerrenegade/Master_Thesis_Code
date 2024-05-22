import torch
import torch.utils.data
import torch.nn as nn
import os
import torchvision as tv
import torchvision.transforms as transforms
from models.topology_models.distances.custom_space_distances import ReconstructionProjectionModel
import matplotlib.pyplot as plt
import numpy as np
from datasets.MNIST.MNIST_base import baseDataset
import torch.multiprocessing as mp

mp.set_start_method('spawn')





def train_or_show_rec_autoencoder(model_path = 'models/topology_models/reconstruction_distance_parameters/MNIST_Reconstruction_model.pth',device = "cuda"):
    num_epochs = 150
    batch_size = 64
    transform =transforms.Compose([transforms.Grayscale(),transforms.Normalize((0.1307,), (0.3081,))])
    device = torch.device(device)
   
    if not os.path.exists(model_path):
        # trainset = tv.datasets.MNIST(root='./data',  train=True, download=True, transform=transform)
       
        train_set = baseDataset(True)
        dataloader = torch.utils.data.DataLoader(train_set.data, batch_size=batch_size, shuffle=True)
        model = ReconstructionProjectionModel().to(device)
        distance_l2 = nn.MSELoss()
        #distance_l1 = nn.L1Loss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        for epoch in range(num_epochs):
            for data in dataloader:
                img, _ = data
                output = model(img)
                loss = distance_l2(output, img)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch [{}/{}], loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

        torch.save(model.state_dict(), model_path)

    else:
        train_set = baseDataset(False)
        dataloader = torch.utils.data.DataLoader(train_set.data, batch_size=batch_size, shuffle=True)

        # Define the model architecture
        model = ReconstructionProjectionModel().to(device)

        # Load the saved model parameters
        model.load_state_dict(torch.load(model_path))

        # Set the model to evaluation mode
        model.eval()

        # Use the model for inference
        for data in dataloader:
            img, _ = data
            output = model(img)
            # Perform inference with the output
            fig, axes = plt.subplots(1, 2)
            cpu_out = output.cpu()
            img_cpu = img.cpu()

            # Display the images on the axis objectsdetach().numpy()
            axes[0].imshow(img_cpu[0].squeeze().detach().numpy(), cmap='gray')
            axes[0].set_title('OG')
            axes[0].axis('off')

            axes[1].imshow(cpu_out[0].squeeze().detach().numpy(), cmap='gray')
            axes[1].set_title('Output')
            axes[1].axis('off')

            # Show the plot
            plt.show()