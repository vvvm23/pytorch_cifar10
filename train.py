if __name__ == '__main__':
    from model import CNN

    import os

    import torch
    print("Starting Torch version:", torch.__version__)
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.autograd import Variable

    from tqdm import tqdm

    import torchvision
    print("Starting Torchvision version:", torchvision.__version__)
    #import torchvision.datasets
    from torchvision.datasets import CIFAR10

    bs=32 # Batch Size
    epoch=20 # Nb. Epochs

    # Get datasets if they do not exist
    
    train_set = CIFAR10(root="{0}\datasets".format(os.path.dirname(os.path.realpath(__file__))), download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)


    # Print statistics about datasets
    print("MNIST loaded. Train:", len(train_set))

    cnn = CNN()

    #input, t = torch.load("datasets/cifar-10-batches-py/data_batch_1") # Load training data

    #target = torch.zeros(len(train_set), 10)
    #for _ in range(len(train_set)):
    #    target[_, t[_]] = 1 # Explode labels into tensors

    #torch.stack([input], dim=1, out=input) # Explode 2D image to 3D for conv layers

    # Convert to FloatTensor for use with conv layers
    # May be pointless to change type here if also changed at input_batch etc.
    #input = input.type('torch.FloatTensor')
    #target = target.type('torch.FloatTensor')

    print("Defining critic.")
    critic = nn.MSELoss() # Mean Square Error Loss
    print("Done")
    print("Defining Optimiser")
    optimiser = optim.Adam(cnn.parameters(), lr=0.001) # Adam Optimiser
    print("Done")

    print("Begin training")

    with tqdm(total=epoch*len(train_set)/bs) as progress_bar: # Create progress bar object
        for _ in (range(epoch)): # Iterate across number of epochs
            for i, data in enumerate(train_loader, 0): # Iterate across number of batches
                progress_bar.update(1) # Increment progress bar

                # Get batch and convert to FloatTensor
                #input_batch = input[b*bs:(b+1)*bs].type('torch.FloatTensor')
                #target_batch = target[b*bs:(b+1)*bs].type('torch.FloatTensor')
                inputs, t = data

                target_outputs = torch.zeros(t.size()[0], 10)
                for _ in range(target_outputs.size()[0]):
                    target_outputs[_, t[_]] = 1 # Explode labels into tensors

                inputs, target_outputs = inputs.type('torch.FloatTensor'), target_outputs.type('torch.FloatTensor')

                optimiser.zero_grad() # Zero gradients
                outputs = cnn(inputs) # Get output from network
                loss = critic(outputs, target_outputs) # Calculate Loss
                loss.backward() # Backpropogate errors
                optimiser.step() # Apply changes to parameters

    print("Training Complete.")
    torch.save(cnn, "saved_model.pth")