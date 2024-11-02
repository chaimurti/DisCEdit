
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torch.nn.utils import prune as prune
import torch
import torch.nn as nn
# import cvxpy as cp # type: ignore
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)


class SubsetDataset(Dataset):
    """ Custom Dataset to handle subsets of CIFAR10 data """
    def __init__(self, dataset, indices):
        """
        Initializes the dataset with the main dataset and indices pointing to specific samples.
        Args:
        dataset (Dataset): The complete dataset.
        indices (list): List of indices pointing to the required samples in the dataset.
        """
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Fetch the dataset item via the stored index
        return self.dataset[self.indices[idx]]

def create_class_conditional_datasets(root, train=True, transform=None):
    # Load the CIFAR10 dataset with transformations
    full_dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    
    # Initialize containers for class-conditional datasets
    class_datasets = {}
    non_class_datasets = {}
    
    # Filter data by class and package into new datasets
    for class_id in range(10):  # CIFAR10 has 10 classes
        # Collecting indices for all samples belonging to the current class_id
        class_indices = [i for i, y in enumerate(full_dataset.targets) if y == class_id]
        non_class_indices = [i for i, y in enumerate(full_dataset.targets) if y != class_id]
        
        # Create a subset for each class using the indices
        class_datasets[class_id] = SubsetDataset(full_dataset, class_indices)
        non_class_datasets[class_id] = SubsetDataset(full_dataset, non_class_indices)
    
    return class_datasets, non_class_datasets

transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
    ])
# class_datasets, non_class_datasets = create_class_conditional_datasets('./data', train=True, transform=transform)

# # DataLoader for class 0 and not class 0
# class_loader = DataLoader(class_datasets[0], batch_size=4, shuffle=True)
# non_class_loader = DataLoader(non_class_datasets[0], batch_size=4, shuffle=True)


def get_layer_activation(model, layer, data):
    """
    Extract output features from a specific layer for a single sample.

    Args:
        layer (torch.nn.Module): The layer from which to extract the output.
        data (torch.Tensor): The input data for the layer (should be properly formatted and include a batch dimension).

    Returns:
        torch.Tensor: The output from the specified layer as activated by the input data.
    """
    with torch.no_grad():
        # Dictionary to store the activations from the forward hook
        activations = {}

        # Define the hook function
        def hook_function(module, input, output):
            activations['output'] = output.detach()

        # Attach the hook to the layer
        handle = layer.register_forward_hook(hook_function)

        # Ensure the layer's containing model (if any) is in evaluation mode if it impacts this layer
        if hasattr(layer, 'training') and layer.training:
            print("Warning: The layer is in training mode, outputs may differ due to this state.")

        # Run the layer on the input data
        with torch.no_grad():  # Temporarily disable gradient calculation
            out = model(data)  # We only care about triggering the hook, not the direct output

        # Remove the hook after use
        handle.remove()
        activation = activations['output']
        t = torch.sum(torch.exp(torch.sum(activation, [2,3])),0)
    #     t1 = torch.exp(t)
        
        # Retrieve and return the captured activation
        return activations['output']

def get_wit_moments(model, layer, data, batch_size):
    """
    Extract output features from a specific layer for a single sample.

    Args:
        layer (torch.nn.Module): The layer from which to extract the output.
        data (torch.Tensor): The input data for the layer (should be properly formatted and include a batch dimension).

    Returns:
        torch.Tensor: The output from the specified layer as activated by the input data.
    """
    with torch.no_grad():
        # Dictionary to store the activations from the forward hook
        activations = {}

        # Define the hook function
        def hook_function(module, input, output):
            activations['output'] = output.detach()

        # Attach the hook to the layer
        handle = layer.register_forward_hook(hook_function)

        # Ensure the layer's containing model (if any) is in evaluation mode if it impacts this layer
        if hasattr(layer, 'training') and layer.training:
            print("Warning: The layer is in training mode, outputs may differ due to this state.")

        # Run the layer on the input data
        with torch.no_grad():  # Temporarily disable gradient calculation
            out = model(data)  # We only care about triggering the hook, not the direct output

        # Remove the hook after use
        activation = activations['output']
        handle.remove()
        
        t1 = torch.sum(torch.exp(torch.sum(activation, [2,3])),0) 
        t2 = torch.sum(torch.exp(2*torch.sum(activation, [2,3])),0)
    #     t1 = torch.exp(t)
        t1.detach().numpy()
        t2.detach().numpy()
        t1 = t1/batch_size
        t2 = t2/batch_size
        # Retrieve and return the captured activation
        return t1,t2

def get_gauss_moments(model, layer, data, batch_size):
    """
    Extract output features from a specific layer for a single sample.

    Args:
        layer (torch.nn.Module): The layer from which to extract the output.
        data (torch.Tensor): The input data for the layer (should be properly formatted and include a batch dimension).

    Returns:
        torch.Tensor: The output from the specified layer as activated by the input data.
    """
    with torch.no_grad():
        # Dictionary to store the activations from the forward hook
        activations = {}

        # Define the hook function
        def hook_function(module, input, output):
            activations['output'] = output.detach()

        # Attach the hook to the layer
        handle = layer.register_forward_hook(hook_function)

        # Ensure the layer's containing model (if any) is in evaluation mode if it impacts this layer
        if hasattr(layer, 'training') and layer.training:
            print("Warning: The layer is in training mode, outputs may differ due to this state.")

        # Run the layer on the input data
        with torch.no_grad():  # Temporarily disable gradient calculation
            out = model(data)  # We only care about triggering the hook, not the direct output

        # Remove the hook after use
        activation = activations['output']
        handle.remove()
        
        t1 = torch.sum(torch.exp(-torch.norm(activation, dim=[2,3])),0) 
        t2 = torch.sum(torch.exp(-2*torch.norm(activation, dim=[2,3])),0)
    #     t1 = torch.exp(t)

        t1 = t1/batch_size
        t2 = t2/batch_size
        # Retrieve and return the captured activation
        return t1,t2


class WitnessPrune:
    def __init__(self, model, layer, budget):
        if not 0 <= budget <= 1:
            raise ValueError("Budget must be a real number between 0 and 1.")

        self.model = model
        self.layer = layer
        self.budget = budget
     

        # Validate if the specified layer is a convolutional layer
        # if not isinstance(self.layer, nn.Conv2d):
        #     raise ValueError("The specified layer is not a convolutional layer.")

    def get_basic_masks(self, saliencies):
        # Get the number of filters in the layer
        num_filters = self.layer.weight.shape[0]

        # Ensure the number of saliencies matches the number of filters
        if len(saliencies) != num_filters:
            raise ValueError("Number of saliencies must match the number of filters in the layer.")

        # Compute the number of filters to prune based on the budget
        num_pruned = int(torch.floor(torch.tensor(self.budget) * num_filters))

        # Sort saliencies and get the indices of the M_f smallest elements
        sorted_indices = torch.argsort(saliencies)

        # Define a basic mask where M_f smallest saliencies are set to 0 and the rest to 1
        basic_mask = torch.ones_like(saliencies)
        basic_mask[sorted_indices[-num_pruned:]] = 0

        return basic_mask

    def build_pruning_mask(self, basic_mask):
        # Get the number of filters in the layer
        num_filters = self.layer.weight.shape[0]

        # Ensure the length of the basic mask matches the number of filters
        if len(basic_mask) != num_filters:
            raise ValueError("Length of basic mask must match the number of filters in the layer.")

        # Create the pruning mask with the same shape as the layer's weight tensor
        pruning_mask = torch.ones_like(self.layer.weight)

        # Set elements corresponding to pruned filters to 0
        for i, mask_value in enumerate(basic_mask):
            if mask_value == 0:
                pruning_mask[i] = 0

        return pruning_mask
    
    def Prune2(self, basic_mask, pruning_mask):
        # Prune the bias of the convolutional layer using basic_mask
        if isinstance(self.layer, nn.Conv2d):
            torch.pruning_utils.prune.custom_from_mask(self.layer, name='bias', mask=basic_mask)

        # Prune the weights of the batch normalization layer associated with the current convolutional layer
        if hasattr(self.layer, 'bn'):
            prune.custom_from_mask(self.layer.bn, name='weight', mask=basic_mask)

        # Prune the biases of the batch normalization layer associated with the current convolutional layer
        if hasattr(self.layer, 'bn'):
            prune.custom_from_mask(self.layer.bn, name='bias', mask=basic_mask)

        # Prune the convolutional filters using pruning_mask
        prune.custom_from_mask(self.layer, name='weight', mask=pruning_mask)

    
    def Prune(self, pruning_mask, basic_mask, model, lnum):
        conv_layer = model.features[lnum]
        bn_layer = model.features[lnum + 1]

        # Prune the bias of the convolutional layer using basic_mask
        # if isinstance(conv_layer, nn.Conv2d):
        m1 = prune.custom_from_mask(conv_layer, name='bias', mask=basic_mask)
        m1 = prune.remove(m1, name='bias')
        # Prune the weights of the batch normalization layer associated with the current convolutional layer
        # if isinstance(bn_layer, nn.BatchNorm2d):
        m2 = prune.custom_from_mask(bn_layer, name='weight', mask=basic_mask)
        m2 = prune.remove(m2, name='weight')
        # Prune the biases of the batch normalization layer associated with the current convolutional layer
        # if isinstance(bn_layer, nn.BatchNorm2d):
        m3 = prune.custom_from_mask(bn_layer, name='bias', mask=basic_mask)
        m3 = prune.remove(m3, name='bias')
        # Prune the convolutional filters using pruning_mask
        m4 = prune.custom_from_mask(conv_layer, name='weight', mask=pruning_mask)
        m4 = prune.remove(m4, name='weight')
        return 
    
    def build_kernel_mask(self, basic_mask, clnum, nclnum, model):
        # Get the current and next convolutional layers from the model
        conv_layer = model.features[clnum]
        next_conv_layer = model.features[nclnum]

        # Get the number of input and output channels in the current and next convolutional layers
        num_input_channels = conv_layer.weight.shape[1]
        num_output_channels = conv_layer.weight.shape[0]

        # Ensure the length of basic_mask matches the number of output channels in the current layer
        if len(basic_mask) != num_output_channels:
            raise ValueError("Length of basic mask must match the number of output channels in the current layer.")

        # Initialize kernel mask with all 1s
        kernel_mask = torch.ones(next_conv_layer.weight.shape)

        # Iterate over the indices of basic_mask
        for i in range(len(basic_mask)):
            if basic_mask[i] == 0:
                # If the output channel is pruned, set the corresponding slice of the kernel mask tensor to 0
                kernel_mask[:, i, :, :] = 0
        return kernel_mask
    

def get_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_labels(dataset):
    # Adjust labels to ensure they are contiguous
    label_map = {old_label: new_label for new_label, old_label in enumerate(set(dataset.targets))}
    adjusted_labels = [label_map[label] for label in dataset.targets]
    dataset.targets = adjusted_labels
    return dataset

# def get_layers(model):

#     keywords = ['relu', 'maxpool', 'avgpool', 'downsample']
#     selected_layers = []

#     # Iterate over all named modules and check if the name contains any of the keywords
#     for name, layer in model.named_modules():
#         if any(keyword in name.lower() for keyword in keywords):
#             selected_layers.append((name, layer))
    
#     return selected_layers


def get_layers(model):

    # keywords = ['relu', 'maxpool', 'avgpool', 'downsample']
    keywords = ['bn']
    selected_layers = []

    # Iterate over all named modules and check if the name contains any of the keywords
    for name, layer in model.named_modules():
        if any(keyword in name.lower() for keyword in keywords):
            selected_layers.append((name, layer))
    
    return selected_layers

def get_conv_layers(model):

    # keywords = ['relu', 'maxpool', 'avgpool', 'downsample']
    keywords = ['conv']
    selected_layers = []

    # Iterate over all named modules and check if the name contains any of the keywords
    for name, layer in model.named_modules():
        if any(keyword in name.lower() for keyword in keywords):
            selected_layers.append((name, layer))
    
    return selected_layers


def main():
    with torch.no_grad():
        f = open('pruning_accs.txt', 'a')
        
        #Load and setup the datasets
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
            ])
        
        class_datasets, non_class_datasets = create_class_conditional_datasets('./data', train=True, transform=transform)
        # test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_class_datasets, test_non_class_datasets = create_class_conditional_datasets('./data', train=False, transform=transform) 

        #load the model
        #run the main loop for each layer


        for class_id in range(10):
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)  # Assuming 'model' is your pruned model



            model.eval()

            layers_relu=  get_conv_layers(model.layer3)
            layers_conv = get_conv_layers(model.layer3)

            class_loader = DataLoader(class_datasets[class_id], batch_size=256, shuffle=True)
            non_class_loader = DataLoader(non_class_datasets[class_id], batch_size=1024, shuffle=True)
            test_class_loader = DataLoader(test_class_datasets[class_id], batch_size=800, shuffle=True)
            test_non_class_loader = DataLoader(test_non_class_datasets[class_id], batch_size=7000, shuffle=True)


            # image,label = next(iter(test_class_loader))
            # image.to(device='cuda:0')
            # label.to(device='cuda:0')
            # print(image.dtype, label.dtype)
            # output = model(image)
            # test_class_acc = accuracy(output, label, topk=(1,))

            # image2, label2 = next(iter(test_non_class_loader))
            # image2.to(device='cuda:0')
            # label2.to(device='cuda:0')
            # output2 = model(image2)
            # test_non_class_acc = accuracy(output2, label2, topk=(1,))

            # Evaluate on non-class-specific data
            # up_non_class_accuracy = get_accuracy(model, test_non_class_loader, device)
            # print(f"Unpruned Accuracy on non-class dataset: {test_class_acc * 100:.2f}%")
            # print(f"Unpruned Accuracy on non-class dataset: {test_non_class_acc * 100:.2f}%")
            # print(test_class_acc, test_non_class_acc)
            # f.write(f"class index is: {class_id}, class_accuracy = {class_accuracy}, non_class_accuracy = {non_class_accuracy}\n")

            for ll in range(len(layers_relu)):
                print(ll,class_id)
                layer = layers_relu[ll]
                
                # print(ll)
                image, label = next(iter(class_loader))
                # image2, label2 = next(iter(non_class_loader))
                
                # activation = get_layer_activation(model, layer, image)
                
                f1c,f2c = get_gauss_moments(model, layer,image, 256)

                
                # image, label = next(iter(class_loader))
                image2, label2 = next(iter(non_class_loader))
                
                # activation = get_layer_activation(model, layer, image2)
                
                f1n,f2n = get_gauss_moments(model, layer, image2, 1024)

                rr = (f1c - f1n)**2 / (f2c + f2n)
                mu = torch.mean(rr)
                std = torch.std(rr)

                thresh = mu+2*std
                large = rr > thresh
                p_ratio = large.sum().item()/len(rr)
                layer2 = layers_conv[ll]
                print(layer2.weight.shape)
                pruner = WitnessPrune(model, layer2, p_ratio)
                basic_mask = pruner.get_basic_masks(rr)
                pruning_mask = pruner.build_pruning_mask(basic_mask)
                pruner.Prune(pruning_mask, basic_mask, model, layer2)
                # kernel_mask = pruner.build_kernel_mask(basic_mask, conv_dict[ll], conv_dict[ll+1], model)
                # filename = './vgg16_c10/vgg16c10_later'+str(ll)+'_class'+str(class_id)
                # plt.plot(rr)
                # plt.savefig(filename)
                # plt.close()
            
            # Evaluate on class-specific data
            class_accuracy = get_accuracy(model, test_class_loader, device)
            print(f"Accuracy on class-specific dataset: {class_accuracy * 100:.2f}%")

            # Evaluate on non-class-specific data
            non_class_accuracy = get_accuracy(model, test_non_class_loader, device)
            print(f"Accuracy on non-class dataset: {non_class_accuracy * 100:.2f}%")
            f.write(f"c_id = {class_id}, cacc = {class_accuracy}, ncacc = {non_class_accuracy}, \n")
        f.close()


if __name__ == "__main__":
    main()