from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch
import copy


def mlp_init(input_dim, hidden_dim, output_dim, num_layers, activation="relu", bias=True):
    """Initializes a Multi-Layer Perceptron (MLP) with the specified architecture."""
    layers = []

    if activation == "relu":
        act_fn = torch.nn.ReLU
    elif activation == "tanh":
        act_fn = torch.nn.Tanh
    elif activation == "sigmoid":
        act_fn = torch.nn.Sigmoid
    elif activation == "leaky_relu":
        act_fn = torch.nn.LeakyReLU
    elif activation == "softplus":
        act_fn = torch.nn.Softplus
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

    if num_layers == 1:
        layers.append(torch.nn.Linear(input_dim, output_dim, bias=bias))
        layers.append(act_fn())
    else:
        layers.append(torch.nn.Linear(input_dim, hidden_dim, bias=bias))
        layers.append(act_fn())
        for _ in range(num_layers - 2):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim, bias=bias))
            layers.append(act_fn())
        layers.append(torch.nn.Linear(hidden_dim, output_dim, bias=bias))
        #layers.append(act_fn()) #variant with activation at the end as outlined in theory 
            
    return torch.nn.Sequential(*layers)


class BF_GNNLayer(MessagePassing):
    """A single layer of the BF GNN model."""
    def __init__(self,num_layers_mlp, input_dim, hidden_dim, output_dim):
        super(BF_GNNLayer, self).__init__(aggr='min')  # "min" aggregation.
       
        self.update_mlp = mlp_init(input_dim=output_dim,hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers_mlp)
        self.aggr_mlp = mlp_init(input_dim=input_dim+1,hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers_mlp)

    
    def forward(self, x, edge_index, edge_attr):
        return self.update_mlp(self.propagate(edge_index, x=x, edge_attr=edge_attr))

    def message(self, x_j, edge_attr):
        return self.aggr_mlp(torch.cat([x_j, edge_attr], dim=-1))

    def reset_parameters(self):
        return 

    def init_random(self):
        for layer in self.update_mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, 0.0, 0.5)
        for layer in self.aggr_mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, 0.0, 0.5)
                

    def init_random_pos(self):
        for layer in self.update_mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.uniform_(layer.weight, 0.0, 0.5)
        for layer in self.aggr_mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.uniform_(layer.weight, 0.0, 0.5)
    

class BF_Model(torch.nn.Module):
    """The complete BF GNN model consisting of multiple BF_GNNLayer layers."""
    def __init__(self, num_layers, num_layers_mlp, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.module_list = torch.nn.ModuleList([
            BF_GNNLayer(num_layers_mlp, input_dim, hidden_dim, output_dim) for _ in range(num_layers)
        ])
        #for layer in self.module_list:
        #    layer.init_random()

    def forward(self, x, edge_index, edge_attr):
        for layer in self.module_list:
            x = layer(x, edge_index, edge_attr)
        return x
    
    def reset_parameters(self):
        for layer in self.module_list:
            layer.reset_parameters()



def load_model(args):
    """Loads the BF GNN model based on the provided arguments."""
    model = BF_Model(
        num_layers=args.num_layers,
        num_layers_mlp=args.num_layers_mlp,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
    )
    return model