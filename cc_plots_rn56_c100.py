import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)


class SubsetDataset(Dataset):
    """ Custom Dataset to handle subsets of CIFAR100 data """
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
    full_dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    
    # Initialize containers for class-conditional datasets
    class_datasets = {}
    non_class_datasets = {}
    
    # Filter data by class and package into new datasets
    for class_id in range(100):  # CIFAR100 has 100 classes
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

    
def get_layers(model):
    """
    Retrieves all layers from the given model where the layer's name contains
    'relu', 'maxpool', 'avgpool', or 'downsample'.
    
    Args:
    model (torch.nn.Module): The model to inspect.
    
    Returns:
    list of tuples: Each tuple contains the name of the layer and the layer instance
                    that matches the specified name criteria.
    """
    keywords = ['relu', 'maxpool', 'avgpool', 'downsample']
    selected_layers = []

    # Iterate over all named modules and check if the name contains any of the keywords
    for name, layer in model.named_modules():
        if any(keyword in name.lower() for keyword in keywords):
            selected_layers.append((name, layer))
    
    return selected_layers

def main():
    with torch.no_grad():

        #Load and setup the datasets
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
            ])
        
        class_datasets, non_class_datasets = create_class_conditional_datasets('./data', train=True, transform=transform)

        #load the model
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)
        model.eval()

        specific_layers = get_layers(model)

        #run the main loop for each layer
        for ll in range(len(specific_layers)):
            layer_name, layer = specific_layers[ll]
            # print(ll)

            for class_id in range(100):
                print(ll,class_id)
                class_loader = DataLoader(class_datasets[class_id], batch_size=256, shuffle=True)
                non_class_loader = DataLoader(non_class_datasets[class_id], batch_size=1024, shuffle=True)

                image, label = next(iter(class_loader))
                # image2, label2 = next(iter(non_class_loader))
                # print(label.dtype, label.shape, image.shape)
                activation = get_layer_activation(model, layer, image)
                # print(activation.shape)

                f1c,f2c = get_gauss_moments(model, layer,image, 256)

                
                # image, label = next(iter(class_loader))
                image2, label2 = next(iter(non_class_loader))
                # print(label2.dtype, label2.shape, image2.shape)
                activation = get_layer_activation(model, layer, image2)
                # print(activation.shape)

                f1n,f2n = get_gauss_moments(model, layer, image2, 1024)

                rr = (f1c - f1n)**2 / (f2c + f2n)
                filename = './resnet56_c100/resnet56c100_layer'+str(ll)+'_class'+str(class_id)
                plt.plot(rr)
                plt.savefig(filename)
                plt.close()


if __name__ == "__main__":
    main()