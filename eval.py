if __name__ == '__main__':
    from model import CNN

    import os

    import torch
    print("Starting Torch version:", torch.__version__)
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from tqdm import tqdm
    from torch.autograd import Variable

    import torchvision
    print("Starting Torchvision version:", torchvision.__version__)
    #import torchvision.datasets
    from torchvision.datasets import CIFAR10

    import matplotlib
    import matplotlib.pyplot as plt
    print("Starting matplotlib version:", matplotlib.__version__)

    # Get test set if they do not exist
    test_set = CIFAR10(root="{0}\datasets".format(os.path.dirname(os.path.realpath(__file__))), train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    test_loader = DataLoader(test_set, num_workers=4)

    # Print statistics about datasets
    print("MNIST loaded. Test:", len(test_set))

    # Load model
    print("Loading CNN from file.")
    cnn = torch.load("saved_model.pth")
    print(cnn)
    print("Loaded CNN.")

    print("Evaluating network")
    cnn.eval()
    correct = 0
    total = 0

    with tqdm(total=len(test_loader)) as progress_bar:
        with torch.no_grad():
            for inputs, classes in test_loader:
                progress_bar.update(1)
                output = cnn.forward(inputs)
                prediction = torch.argmax(output)
                # To do, have current evaluation in description of progress bar
                #tqdm.set_description("Prediction %s . Actual %s" % (prediction, labels[_]))
                if prediction == classes:
                    correct += 1
                total += 1

    print("Evaluation complete. Network accuracy:", correct/total * 100, "%")

    dictionary = {
        0 : "Airplane",
        1 : "Automobile",
        2 : "Bird",
        3 : "Cat",
        4 : "Deer",
        5 : "Dog",
        6 : "Frog",
        7 : "Horse",
        8 : "Ship",
        9 : "Truck"
    }

    #for _ in range(16):
    #    plt_input = torch.stack([inputs[_]], dim=1, out=inputs[_])
    #
    #    plt_prediction = torch.argmax(cnn(plt_input))
    #
    #    plt_input = plt_input.view((28, 28))
    #
    #    plt.subplot(4, 4, _+1)
    #    plt.xticks([])
    #    plt.yticks([])
    #    plt.title(dictionary[int(plt_prediction)], fontsize=6)
    #    plt.imshow(plt_input, cmap='gray')

    #plt.axis('off')
    #plt.tight_layout()
    #plt.show()

