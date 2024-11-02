import torch
import torchvision

#
import scipy as sp
from scipy.linalg import svdvals

import numpy as np
import powerlaw
import cvxpy as cp

# import sklearn
# from sklearn.decomposition import TruncatedSVD

import matplotlib

import matplotlib.pyplot as plt

from torch import nn, Tensor
from torchvision.datasets import CIFAR100

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

from torch.nn.utils import prune as prune

from torch.utils.data.sampler import SubsetRandomSampler

import pickle as pickle

from torchsummary import summary


import scipy as sp

from scipy import stats as stats

import seaborn as seaborn

import torchsummary

from torchsummary import summary


class FilterStatsCollector:  
    def __init__(self, model, dataset, layer, device='cpu'):
        self.model = model.to(device)
        self.dataset = dataset
        self.layer = layer
        self.device = device
        self.activations = None
        self.hook = None  # Initialize the hook attribute

        # Check if the layer is a convolutional layer and set the hook
        if isinstance(self.layer, nn.Conv2d):
            self.hook = self.layer.register_forward_hook(self.save_output_hook)
        else:
            raise ValueError("The provided layer is not a convolutional layer.")

    def save_output_hook(self, module, input, output):
        self.activations = output


    def get_z(self):
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False)
        inputs,labels = next(iter(loader))
        inputs = inputs.to(self.device)
        self.model.eval()
        self.activations = None
        self.model(inputs)
        acts = self.activations
        # print(acts.shape)
        # print(acts[0,1,:,:].shape)
        z_values = []
        for i in range(acts.shape[1]):
              # Move inputs to the same device as the model          
            s_j = acts[0,i,:,:].sum()
            z_j = torch.tensor([s_j.item(), s_j.item() ** 2], device=self.device)
            z_values.append(z_j)
            # print(i,z_j)

        return z_values

    def __del__(self):
        # Safely remove the hook if it exists
        if self.hook is not None:
            self.hook.remove()

    def class_dataloader(self, class_label, batch_size=32, shuffle=True):
        # Filter the dataset to include only samples of the specified class
        class_indices = [i for i, (_, label) in enumerate(self.dataset) if label == class_label]

        class_subset = Subset(self.dataset, class_indices)
        return DataLoader(class_subset, batch_size=batch_size, shuffle=shuffle)
    
    def complement_dataloader(self, class_label, batch_size=32, shuffle=True):
        # Filter the dataset to exclude samples of the specified class
        non_class_indices = [i for i, (_, label) in enumerate(self.dataset) if label != class_label]

        non_class_subset = Subset(self.dataset, non_class_indices)
        return DataLoader(non_class_subset, batch_size=batch_size, shuffle=shuffle)
    def compute_average_z_per_filter(self, class_label, use_complement=False, num_samples=None):
        if use_complement:
            loader = self.complement_dataloader(class_label, num_samples=num_samples)
        else:
            loader = self.class_dataloader(class_label, num_samples=num_samples)

        sum_z_values = None
        count = 0

        for inputs, _ in loader:
            inputs = inputs.to(self.device)
            self.model.eval()
            self.activations = None
            _ = self.model(inputs)

            z_values = []
            for i in range(self.activations.size(1)):  # Iterate over the filters
                s_j = self.activations[:, i, :, :].sum()
                z_j = torch.tensor([s_j.item(), s_j.item() ** 2], device=self.device)
                z_values.append(z_j)

            if sum_z_values is None:
                sum_z_values = torch.stack(z_values)
            else:
                sum_z_values += torch.stack(z_values)

            count += 1

        average_z_values = sum_z_values / count
        return average_z_values
    
    def get_mean_z(self, loader, num_samples):
        sum_z_values = None
        count = 0

        for _ in range(num_samples):
            try:
                inputs, _ = next(iter(loader))
            except StopIteration:
                break

            inputs = inputs.to(self.device)
            self.model.eval()
            self.activations = None
            _ = self.model(inputs)

            if self.activations is None:
                continue  # Skip if no activations are recorded

            num_filters = self.activations.size(1)
            z_values = []
            for j in range(num_filters):  # Iterate over the filters
                s_j = self.activations[:, j, :, :].sum()
                z_j = torch.tensor([s_j.item(), s_j.item() ** 2], device=self.device)
                z_values.append(z_j)

            if sum_z_values is None:
                sum_z_values = torch.stack(z_values)
            else:
                sum_z_values += torch.stack(z_values)

            count += 1

        if count == 0:
            raise ValueError("No samples were processed. DataLoader may be empty or num_samples too large.")

        mean_z_values = sum_z_values / count
        return mean_z_values


    # def get_mean_z(self, loader, num_samples):
    #     sum_z_values = None
    #     count = 0

    #     for _ in range(num_samples):
    #         try:
    #             inputs, _ = next(iter(loader))
    #         except StopIteration:
    #             # If the loader runs out of data, break out of the loop
    #             break

    #         inputs = inputs.to(self.device)
    #         self.model.eval()
    #         self.activations = None
    #         _ = self.model(inputs)

    #         z_values = []
    #         for j in range(self.activations.size(1)):  # Iterate over the filters
    #             # print(j)
    #             s_j = self.activations[:, j, :, :].sum()
                
    #             z_j = torch.tensor([s_j.item(), s_j.item() ** 2], device=self.device)
    #             # print(z_j.shape)
    #             z_values.append(z_j)
    #             # print(z_values.shape)

    #         if sum_z_values is None:
    #             sum_z_values = torch.stack(z_values)
    #         else:
    #             sum_z_values += torch.stack(z_values)
    #             print(sum_z_values.shape)

    #         count += 1

    #     if count == 0:
    #         raise ValueError("No samples were processed. DataLoader may be empty or num_samples too large.")

    #     mean_z_values = sum_z_values / count
    #     print(mean_z_values.shape)
    #     return mean_z_values

    def get_mean_Z(self, loader, num_samples):
        sum_Z_values = None
        count = 0

        for _ in range(num_samples):
            try:
                inputs, _ = next(iter(loader))
            except StopIteration:
                break

            inputs = inputs.to(self.device)
            self.model.eval()
            self.activations = None
            _ = self.model(inputs)

            Z_values = []
            for i in range(self.activations.size(1)):  # Iterate over the filters
                s_j = self.activations[:, i, :, :].sum()
                z_j = torch.tensor([s_j.item(), s_j.item() ** 2], device=self.device)
                Z_j = torch.ger(z_j, z_j)  # Outer product to form a 2x2 matrix
                Z_values.append(Z_j)

            if sum_Z_values is None:
                sum_Z_values = torch.stack(Z_values)
            else:
                sum_Z_values += torch.stack(Z_values)

            count += 1

        if count == 0:
            raise ValueError("No samples were processed.")

        mean_Z_values = sum_Z_values / count
        # print(mean_Z_values.shape)
        return mean_Z_values
    
   
    def get_vars_z(self, loader, num_samples, means):
        sum_vars_values = None
        count = 0

        for _ in range(num_samples):
            try:
                inputs, _ = next(iter(loader))
            except StopIteration:
                break

            inputs = inputs.to(self.device)
            self.model.eval()
            self.activations = None
            _ = self.model(inputs)

            vars_values = []
            for i in range(self.activations.size(1)):  # Iterate over the filters
                s_j = self.activations[:, i, :, :].sum()
                z_j = torch.tensor([s_j.item(), s_j.item() ** 2], device=self.device)
                m_j = means[i]  # Get the mean for this filter
                z_j_centered = z_j - m_j
                var_j = torch.ger(z_j_centered, z_j_centered)  # Outer product for variance
                vars_values.append(var_j)

            if sum_vars_values is None:
                sum_vars_values = torch.stack(vars_values)
            else:
                sum_vars_values += torch.stack(vars_values)

            count += 1

        if count == 0:
            raise ValueError("No samples were processed.")

        mean_vars_values = sum_vars_values / count
        return mean_vars_values


    def compute_class_means(self, num_samples_per_class):
        num_classes = 100  
        class_means = {f"Class_{i}": None for i in range(num_classes)}
        complement_means = {f"Class_{i}": None for i in range(num_classes)}

        for class_label in range(num_classes):
            # DataLoader for the specific class
            class_loader = self.class_dataloader(class_label)
            mean_z_class = self.get_mean_z(class_loader, num_samples_per_class)
            # print('class label is', class_label, 'mean z class', mean_z_class.shape)
            class_means[f"Class_{class_label}"] = mean_z_class

            # DataLoader for the complement of the class
            complement_loader = self.complement_dataloader(class_label)
            mean_z_complement = self.get_mean_z(complement_loader, num_samples_per_class)
            complement_means[f"Class_{class_label}"] = mean_z_complement
            # print('comp_class label is', class_label, 'mean z class', mean_z_class.shape)

        self.class_means = class_means
        self.complement_means = complement_means
        # print(self.complement_means.keys())
        # print(self.class_means.keys())
        return class_means, complement_means
    

    def compute_class_means_Z(self, num_samples_per_class):
        num_classes = 100  
        class_means_Z = {f"Class_{i}": None for i in range(num_classes)}
        complement_means_Z = {f"Class_{i}": None for i in range(num_classes)}

        for class_label in range(num_classes):
            # DataLoader for the specific class
            class_loader = self.class_dataloader(class_label)
            mean_Z_class = self.get_mean_Z(class_loader, num_samples_per_class)
            class_means_Z[f"Class_{class_label}"] = mean_Z_class

            # DataLoader for the complement of the class
            complement_loader = self.complement_dataloader(class_label)
            mean_Z_complement = self.get_mean_Z(complement_loader, num_samples_per_class)
            complement_means_Z[f"Class_{class_label}"] = mean_Z_complement
        
        self.class_means_Z = class_means_Z
        self.complement_means_Z = complement_means_Z
        # print(self.class_means_Z.keys())
        # print(self.complement_means_Z.keys())
        return class_means_Z, complement_means_Z
    


    def compute_Sj(self, num_samples_per_class):
        class_means = self.class_means
        complement_means = self.complement_means
        class_means_Z = self.class_means_Z
        complement_means_Z = self.complement_means_Z
        # class_means, complement_means = self.compute_class_means(num_samples_per_class)
        # class_means_Z, complement_means_Z = self.compute_class_means_Z(num_samples_per_class)

        Sj_class = {}
        Sj_complement = {}

        num_classes = 100  # 
        num_filters = 512  # Adjust this to the correct number of filters
        for i in range(num_classes):
            # Reshape E_zj to [num_filters, 2]
            E_zj = class_means[f"Class_{i}"].view(num_filters, 2)
            E_Zj = class_means_Z[f"Class_{i}"]

            Sj_class[f"Class_{i}"] = torch.zeros((num_filters, 2, 2))
            for j in range(num_filters):
                # Compute Sj for each filter
                Sj_class[f"Class_{i}"][j] = E_Zj[j] - torch.ger(E_zj[j], E_zj[j] * (1/num_samples_per_class))

            # Similarly for the complement
            E_zj_complement = complement_means[f"Class_{i}"].view(num_filters, 2)
            E_Zj_complement = complement_means_Z[f"Class_{i}"]

            Sj_complement[f"Class_{i}"] = torch.zeros((num_filters, 2, 2))
            for j in range(num_filters):
                # Compute Sj for each filter
                Sj_complement[f"Class_{i}"][j] = E_Zj_complement[j] - torch.ger(E_zj_complement[j], E_zj_complement[j]*(1/num_samples_per_class))

        self.Sj_class = Sj_class
        self.Sj_complement = Sj_complement
        # print(self.Sj_class.keys())
        # print(self.Sj_complement.keys())
        return Sj_class, Sj_complement
    

    def compute_class_variances(self, num_samples_per_class):
        num_classes = 100 
        class_variances = {f"Class_{i}": None for i in range(num_classes)}
        complement_variances = {f"Class_{i}": None for i in range(num_classes)}

        for class_label in range(num_classes):
            class_loader = self.class_dataloader(class_label)
            complement_loader = self.complement_dataloader(class_label)

            mean_z_class = self.class_means[f"Class_{class_label}"]
            mean_z_complement = self.complement_means[f"Class_{class_label}"]

            variance_class = self.get_vars_z(class_loader, num_samples_per_class, mean_z_class)
            variance_complement = self.get_vars_z(complement_loader, num_samples_per_class, mean_z_complement)

            class_variances[f"Class_{class_label}"] = variance_class
            complement_variances[f"Class_{class_label}"] = variance_complement

        self.class_variances = class_variances
        self.complement_variances = complement_variances
        return class_variances, complement_variances


    

    def get_fish_saliency(self):
        fisher_scores = {}
        num_filters = len(next(iter(self.class_means.values())))
        saliency_scores = torch.full((num_filters,), float('inf'))

        for class_label in self.class_means.keys():
            m_c = self.class_means[class_label]
            m_cp = self.complement_means[class_label]
            S_c = self.class_variances[class_label]
            S_cp = self.complement_variances[class_label]

            T_scores = torch.zeros(num_filters)

            for j in range(num_filters):
                delta_m = m_c[j] - m_cp[j]
                S = S_c[j] + S_cp[j]

                # Ensure S is invertible
                if torch.det(S) == 0:
                    S += torch.eye(2) * 1e-5  # Adding a small value to the diagonal for numerical stability

                # Calculate Fisher Discriminant for each filter
                F_c = torch.dot(delta_m.T, torch.linalg.inv(S) @ delta_m)
                T_c = F_c / (2 + F_c)
                T_scores[j] = T_c
                saliency_scores[j] = min(saliency_scores[j], T_c)

            fisher_scores[class_label] = T_scores

        return fisher_scores, saliency_scores

    
    


# class WitnessPrune:
#     def __init__(self, model, layer, budget, dataset):
#         if not 0 <= budget <= 1:
#             raise ValueError("Budget must be a real number between 0 and 1.")

#         self.model = model
#         self.layer = layer
#         self.budget = budget
#         self.dataset = dataset

#         # Validate if the specified layer is a convolutional layer
#         if not isinstance(self.layer, nn.Conv2d):
#             raise ValueError("The specified layer is not a convolutional layer.")

#     def get_basic_masks(self, saliencies):
#         # Get the number of filters in the layer
#         num_filters = self.layer.weight.shape[0]

#         # Ensure the number of saliencies matches the number of filters
#         if len(saliencies) != num_filters:
#             raise ValueError("Number of saliencies must match the number of filters in the layer.")

#         # Compute the number of filters to prune based on the budget
#         num_pruned = int(torch.floor(torch.tensor(self.budget) * num_filters))

#         # Sort saliencies and get the indices of the M_f smallest elements
#         sorted_indices = torch.argsort(saliencies)

#         # Define a basic mask where M_f smallest saliencies are set to 0 and the rest to 1
#         basic_mask = torch.ones_like(saliencies)
#         basic_mask[sorted_indices[:num_pruned]] = 0

#         return basic_mask
    
class WitnessPrune:
    def __init__(self, model, layer, budget, dataset):
        if not 0 <= budget <= 1:
            raise ValueError("Budget must be a real number between 0 and 1.")

        self.model = model
        self.layer = layer
        self.budget = budget
        self.dataset = dataset

        # Validate if the specified layer is a convolutional layer
        if not isinstance(self.layer, nn.Conv2d):
            raise ValueError("The specified layer is not a convolutional layer.")

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
        basic_mask[sorted_indices[:num_pruned]] = 0

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
        prune.custom_from_mask(conv_layer, name='bias', mask=basic_mask)

        # Prune the weights of the batch normalization layer associated with the current convolutional layer
        # if isinstance(bn_layer, nn.BatchNorm2d):
        prune.custom_from_mask(bn_layer, name='weight', mask=basic_mask)

        # Prune the biases of the batch normalization layer associated with the current convolutional layer
        # if isinstance(bn_layer, nn.BatchNorm2d):
        prune.custom_from_mask(bn_layer, name='bias', mask=basic_mask)

        # Prune the convolutional filters using pruning_mask
        prune.custom_from_mask(conv_layer, name='weight', mask=pruning_mask)

        return model
    def Prune3(self, kernel_mask, model, lnum):
        conv_layer = model.features[lnum]
        
        # Prune the convolutional filters using pruning_mask
        prune.custom_from_mask(conv_layer, name='weight', mask=kernel_mask)

        return model
    
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


if __name__ == '__main__':
    device = torch.device('cuda')



    print('loading model')
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)

    print('setting up dataset')
    # Prepare CIFAR100 test set
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)


  

    #######################



    # train_set = torchvision.datasets.CIFAR10('./datasets', train=True, 
    #                                         download=True, transform=transform)
    # test_set = torchvision.datasets.CIFAR10('./datasets', train=False, 
    #                                         download=True, transform=transform)

        
    num_samples = 2000
    
    # Generate a random list of indices
    indices = torch.randperm(len(test_set)).tolist()[:num_samples]
    
    # Create a subset
    subset = Subset(test_set, indices)
    # Number of subprocesses to use for data loading
    num_workers = 4


    num_samples_per_class = 20
    num_samples_per_class_complement = 150
    budget = 0.5




    # conv_dict_feats = [3,7,10,14,17,21,24,28]           #VGG11
    # conv_dict_feats = [3,7,10,14,17,21,24,28]
    conv_dict_feats = [2,6,9,13,16,19,23,26,29,33,36,39,43]

    # conv_dict = [0,4,8,11,15,18,22,25]
    conv_dict = [0,3,7,10,14,17,20,24,27,30,34,37,40]

    for k in range(len(conv_dict)):
        
        clnum = conv_dict[k]
        nclnum = conv_dict[k+1]
        # Instantiate FilterStatsCollector
        print('set up stats collector')
        collector = FilterStatsCollector(model=model, dataset=testset, layer=model.features[clnum])
        
        # Compute class means, class means Z, and Sj
        print('computing class means')

        collector.compute_class_means(num_samples_per_class)


        # print('computing uncentered variances')
        # collector.compute_class_means_Z(num_samples_per_class)

        print('computing variances')

        collector.compute_class_variances(num_samples_per_class)

        
        # Compute Fisher scores and saliency
        print('computing fisher scores and saliencies')
        fisher_scores, saliency = collector.get_fish_saliency()

        print('setting up pruner')
        pruner = WitnessPrune(model, model.features[clnum], budget, testset)
        
        # Get basic masks and pruning mask
        print('setting up basic mask')
        basic_mask = pruner.get_basic_masks(saliency)
        
        print('setting up pruning mask')
        pruning_mask = pruner.build_pruning_mask(basic_mask)
        
        # Apply pruning
        print('pruning layer l')
        model = pruner.Prune(pruning_mask, basic_mask, model, clnum)
        
        # Optionally, build and apply kernel mask to the next convolutional layer
        if k < len(conv_dict)-1:
            print('set up mask for layer l+1')
            kernel_mask = pruner.build_kernel_mask(basic_mask, clnum, nclnum, model)
            print('pruning layer l+1')    
            model = pruner.Prune2(kernel_mask, model, clnum)

