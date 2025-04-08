import argparse
from pathlib import Path
import os
import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# -----------------------------------------------------------------------------------
# 1) Shared Utilities
# -----------------------------------------------------------------------------------
def get_device() -> torch.device:
    """
    Checks if a GPU is available; if not, defaults to CPU.
    Returns:
        torch.device ('cuda' if available, otherwise 'cpu').
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_networkx_directed_graph(n_nodes: int, graph_type: str, bidirectional_prob: float = 0.3) -> nx.DiGraph:
    """
    Creates a directed NetworkX graph given the number of nodes and specified type.

    Supported types:
        - 'barabasi' : Directed Barabási–Albert preferential attachment
        - 'watt'     : Directed Watts–Strogatz small-world
        - 'erdos'    : Directed Erdős–Rényi random graph

    Parameters:
        n_nodes (int): Number of nodes in the graph.
        graph_type (str): The type of graph to create ('barabasi', 'watt', 'erdos').
        bidirectional_prob (float): Probability of making an edge bidirectional.

    Returns:
        nx.DiGraph: The generated directed graph.
    """
    graph_type = graph_type.lower()
    bidirectional_prob = 0.3  # Overriding the default for demonstration

    if graph_type == "barabasi":
        # First create undirected BA graph
        m = min(10, max(1, n_nodes - 1))
        G_undirected = nx.barabasi_albert_graph(n_nodes, m)
        
        # Convert to directed by randomly orienting edges
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        
        for u, v in G_undirected.edges():
            # Always add edge in one direction
            G.add_edge(u, v)
            # With some probability, add edge in reverse direction
            if np.random.random() < bidirectional_prob:
                G.add_edge(v, u)
    
    elif graph_type == "watt":
        # k must be even and < n_nodes
        k = 4 if n_nodes >= 4 else 2
        if k % 2 != 0:
            k += 1
        p = 0.1
        
        # First create undirected WS graph
        G_undirected = nx.watts_strogatz_graph(n_nodes, k, p)
        
        # Convert to directed by randomly orienting edges
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        
        for u, v in G_undirected.edges():
            # Always add edge in one direction
            G.add_edge(u, v)
            # With some probability, add edge in reverse direction
            if np.random.random() < bidirectional_prob:
                G.add_edge(v, u)
    
    elif graph_type == "erdos":
        # Probability of edge
        p = 0.05
        
        # Create directed Erdos-Renyi graph
        G = nx.gnp_random_graph(n_nodes, p, directed=True)
        
        # Ensure graph is weakly connected if n_nodes>1
        while n_nodes > 1 and not nx.is_weakly_connected(G):
            G = nx.gnp_random_graph(n_nodes, p, directed=True)
            p *= 1.1  # Increase probability if not connected
    
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")

    return G


# -----------------------------------------------------------------------------------
# 2) Base Case Functions
# -----------------------------------------------------------------------------------
def initialize_graph_base(n_nodes: int, graph_type: str, device=None):
    """
    Initializes graph for the 'base' method using GPU acceleration.

    1) Builds a directed graph of the chosen type (Barabási, Watt, or Erdős).
    2) Computes adjacency matrix (A) and Laplacian (L) on the GPU.
    3) Calculates:
       - K = L + A
       - epsilon = 1 / (100 * max(K))
       - P = I - epsilon * L   (the Perron-like matrix)
    4) Initial state: diag(K) (as a tensor).

    Parameters:
        n_nodes (int): Number of nodes in the graph.
        graph_type (str): 'barabasi', 'watt', or 'erdos'.
        device (torch.device, optional): GPU or CPU for computations.

    Returns:
        (torch.Tensor, torch.Tensor, nx.DiGraph):
          - initial_state: diag(K)  [n_nodes, ]
          - P:  The Perron-like matrix [n_nodes, n_nodes]
          - G:  The generated directed graph (NetworkX)
    """
    if device is None:
        device = get_device()

    G = create_networkx_directed_graph(n_nodes, graph_type)

    # Create adjacency matrix on CPU then convert to GPU
    A_np = nx.to_numpy_array(G, dtype=float)
    A = torch.tensor(A_np, dtype=torch.float32, device=device)

    # For directed graph, the out-degree is the sum of each row
    out_degrees = A.sum(dim=1)
    
    # Laplacian: L = D - A
    D = torch.diag(out_degrees)
    L = D - A

    # K = L + A
    K = L + A

    # Step size
    max_k = torch.max(K)
    epsilon = 1.0 / (100.0 * max_k) if max_k > 0 else 1e-4

    # Perron matrix: P = I - epsilon * L
    I = torch.eye(n_nodes, device=device)
    P = I - epsilon * L

    # Initial state
    initial_state = torch.diag(K)

    return initial_state, P, G

def generate_sample_base(
    n_nodes: int,
    graph_type: str,
    n_time_samples: int = 50,
    max_iter: int = 1000000000,
    tol: float = 1e-3,
    device=None
):
    """
    Generates the time evolution for the 'base' case until convergence or max_iter.

    1) Initializes the graph (initialize_graph_base).
    2) Iterates phi(t+1) = P @ phi(t) until max diff < tol or max_iter steps.
    3) Returns the first n_time_samples states, the mean of the final state,
       the total number of directed edges, and the number of bidirectional edges.

    Parameters:
        n_nodes (int): Number of nodes.
        graph_type (str): 'barabasi', 'watt', or 'erdos'.
        n_time_samples (int): Number of time steps to keep.
        max_iter (int): Maximum iterations for convergence.
        tol (float): Convergence tolerance.
        device (torch.device, optional): Where to run computations.

    Returns:
        (torch.Tensor, float, int, int):
            - phi_values [n_time_samples, n_nodes]
            - final_mean (float)
            - num_edges (int)
            - num_bidirectional (int)
    """
    if device is None:
        device = get_device()

    phi_init, P, G = initialize_graph_base(n_nodes, graph_type, device=device)
    phi_curr = phi_init.clone()

    phi_values_list = [phi_curr.clone()]
    
    for i in range(max_iter):
        phi_next = P @ phi_curr
        diff = torch.max(torch.abs(phi_next - phi_curr))
        phi_values_list.append(phi_next.clone())
        phi_curr = phi_next
        
        if diff <= tol:
            # Converged
            break

    # Stack states
    phi_values = torch.stack(phi_values_list, dim=0)
    num_steps = phi_values.shape[0]

    # Pad or truncate to n_time_samples
    if num_steps < n_time_samples:
        pad_size = n_time_samples - num_steps
        pad_zeros = torch.zeros((pad_size, n_nodes), device=device)
        phi_values = torch.cat([phi_values, pad_zeros], dim=0)
    else:
        phi_values = phi_values[:n_time_samples]

    # Mean of the final state
    final_mean = torch.mean(phi_curr).item()

    # Compute edge stats from G
    # Number of directed edges = G.number_of_edges()
    # Number of bidirectional edges = count pairs (u,v) where both u->v and v->u exist
    num_edges = G.number_of_edges()
    num_bidirectional = sum(1 for (u, v) in G.edges() if G.has_edge(v, u))

    # Move to CPU
    return phi_values.cpu(), final_mean, num_edges, num_bidirectional


# -----------------------------------------------------------------------------------
# 3) Exponential Case Functions
# -----------------------------------------------------------------------------------
def initialize_graph_exponential(n_nodes: int, graph_type: str, device=None):
    """
    Initializes graph for the 'exponential' method using GPU acceleration.

    1) Builds a directed graph of the chosen type (Barabási, Watt, or Erdős).
    2) Computes adjacency matrix (A) and Laplacian (L).
    3) Precomputes A^k for k up to diameter(G). Sums them with exponential decay
       to form an extended Laplacian, then calculates P = I - epsilon * SLk.
    4) Returns diag(K) and P, plus the graph G for edge stats.

    Parameters:
        n_nodes (int): Number of nodes in the graph.
        graph_type (str): 'barabasi', 'watt', or 'erdos'.
        device (torch.device, optional): GPU or CPU.

    Returns:
        (torch.Tensor, torch.Tensor, nx.DiGraph):
          - K_diag: diag(K)
          - P: The Perron-like matrix with exponential modifications
          - G: The generated directed graph
    """
    if device is None:
        device = get_device()

    # Build the chosen graph
    G = create_networkx_directed_graph(n_nodes, graph_type)

    # Adjacency
    A_np = nx.to_numpy_array(G, dtype=float)
    A = torch.tensor(A_np, dtype=torch.float32, device=device)

    # Check diameter (weakly, but we use the undirected version if connected)
    if n_nodes > 1 and nx.is_weakly_connected(G):
        diameter_val = nx.diameter(G.to_undirected())
    else:
        diameter_val = 1

    # Out-degree for directed graph
    out_degrees = A.sum(dim=1)
    
    # Laplacian: L = D - A
    D = torch.diag(out_degrees)
    L = D - A

    # K = L + A
    K = L + A

    # Precompute adjacency powers up to diameter
    adjacency_powers = [torch.zeros_like(A) for _ in range(diameter_val + 1)]
    if diameter_val >= 1:
        adjacency_powers[1] = A.clone()
        for k in range(2, diameter_val + 1):
            adjacency_powers[k] = adjacency_powers[k - 1].matmul(A)

    # S_list[k] = A + A^2 + ... + A^k
    S_list = [torch.zeros_like(A) for _ in range(diameter_val + 1)]
    for k in range(1, diameter_val + 1):
        S_list[k] = S_list[k - 1] + adjacency_powers[k]

    # Build the extended Laplacian using exponential decay
    SLk = torch.exp(torch.tensor(-0.5, device=device)) * L

    for k in range(2, diameter_val + 1):
        Sk = S_list[k]
        Pk = (Sk != 0).float()  # 0/1 adjacency (at power k)
        row_sums = Pk.sum(dim=1)
        Lk = torch.diag(row_sums) - Pk
        decay_factor = torch.exp(torch.tensor(-0.5 * k, device=device))
        SLk += decay_factor * Lk

    # P = I - epsilon * SLk
    I = torch.eye(n_nodes, device=device)
    epsilon = 1e-6
    P = I - epsilon * SLk

    K_diag = torch.diag(K)

    return K_diag, P, G

def generate_sample_exponential(
    n_nodes: int,
    graph_type: str,
    n_time_samples: int = 50,
    max_iter: int = 1000000000,
    tol: float = 1e-3,
    show_bar: bool = False,
    device=None
):
    """
    Generates the time evolution for the 'exponential' case until convergence or max_iter.

    1) Initializes graph (initialize_graph_exponential).
    2) Iterates phi(t+1) = P @ phi(t) until max diff < tol or max_iter steps.
    3) Returns the time samples, mean of final state,
       number of directed edges, number of bidirectional edges.

    Parameters:
        n_nodes (int): Number of nodes.
        graph_type (str): 'barabasi', 'watt', 'erdos'.
        n_time_samples (int): Number of time steps to keep.
        max_iter (int): Maximum iterations for convergence.
        tol (float): Convergence tolerance.
        show_bar (bool): Whether to display a tqdm bar for the internal loop.
        device (torch.device, optional): Where to run computations.

    Returns:
        (torch.Tensor, float, int, int):
            - phi_values [n_time_samples, n_nodes]
            - final_mean (float)
            - num_edges (int)
            - num_bidirectional (int)
    """
    if device is None:
        device = get_device()

    phi_init, P, G = initialize_graph_exponential(n_nodes, graph_type, device=device)
    phi_curr = phi_init.reshape(-1).clone()

    phi_values_list = [phi_curr.clone()]

    # Optionally show a progress bar
    progress_iter = range(max_iter)
    if show_bar:
        progress_iter = tqdm(progress_iter, desc="Time evolution", leave=False)

    for i in progress_iter:
        phi_next = P @ phi_curr
        diff = torch.max(torch.abs(phi_next - phi_curr))
        phi_values_list.append(phi_next.clone())
        phi_curr = phi_next
        
        if diff <= tol:
            # Converged
            break

        if show_bar:
            progress_iter.set_postfix({"diff": diff.item()})

    # Stack states
    phi_values = torch.stack(phi_values_list, dim=0)
    num_steps = phi_values.shape[0]

    # Pad or truncate
    if num_steps < n_time_samples:
        extra = n_time_samples - num_steps
        padding = torch.zeros((extra, phi_values.size(1)), device=device)
        phi_values = torch.cat([phi_values, padding], dim=0)
    else:
        phi_values = phi_values[:n_time_samples]

    final_mean = torch.mean(phi_curr).item()

    # Number of directed edges
    num_edges = G.number_of_edges()
    # Number of bidirectional edges
    num_bidirectional = sum(1 for (u, v) in G.edges() if G.has_edge(v, u))

    return phi_values.cpu().float(), final_mean, num_edges, num_bidirectional


# -----------------------------------------------------------------------------------
# 4) Dataset and Data Generation
# -----------------------------------------------------------------------------------
def generate_n_samples(
    n_samples: int,
    n_nodes: int,
    graph_type: str,
    case_type: str = "base",
    n_time_samples: int = 50,
    max_iter: int = 1000000000,
    tol: float = 1e-3,
    show_bar: bool = False
):
    """
    Generates multiple samples for either the 'base' or 'exponential' case.

    For each sample in range(n_samples):
      - Calls either generate_sample_base(...) or generate_sample_exponential(...)
      - Collects (phi_values, final_state_mean) in the dataset Tensors.

    Also accumulates:
      - final_state_mean for each sample (to compute mean/variance at the end),
      - number of edges and number of bidirectional edges for each sample.

    Parameters:
        n_samples (int): Number of samples to generate.
        n_nodes (int): Number of nodes.
        graph_type (str): 'barabasi', 'watt', 'erdos'.
        case_type (str): 'base' or 'exponential'.
        n_time_samples (int): Steps recorded per sample.
        max_iter (int): Convergence iteration limit.
        tol (float): Convergence tolerance.
        show_bar (bool): Whether to show the external tqdm bar over samples.

    Returns:
        (torch.Tensor, torch.Tensor):
            - data   [n_samples, n_time_samples, n_nodes]
            - target [n_samples, 1]
    """
    device = get_device()
    data = torch.zeros((n_samples, n_time_samples, n_nodes), dtype=torch.float32)
    targets = torch.zeros((n_samples, 1), dtype=torch.float32)

    # For final stats
    final_means = []
    edges_list = []
    bidir_list = []

    sample_range = range(n_samples)
    if show_bar:
        sample_range = tqdm(sample_range, desc="Generating samples", leave=True)

    for i in sample_range:
        if case_type.lower() == "base":
            x_i, y_i, directed_e, bidir_e = generate_sample_base(
                n_nodes=n_nodes,
                graph_type=graph_type,
                n_time_samples=n_time_samples,
                max_iter=max_iter,
                tol=tol,
                device=device
            )
        elif case_type.lower() == "exponential":
            x_i, y_i, directed_e, bidir_e = generate_sample_exponential(
                n_nodes=n_nodes,
                graph_type=graph_type,
                n_time_samples=n_time_samples,
                max_iter=max_iter,
                tol=tol,
                device=device
            )
        else:
            raise ValueError(f"Unsupported case_type: {case_type}")

        data[i] = x_i
        targets[i] = y_i
        final_means.append(y_i)
        edges_list.append(directed_e)
        bidir_list.append(bidir_e)

    # After generating all samples, compute global stats
    final_means = np.array(final_means)
    edges_list = np.array(edges_list, dtype=np.float32)
    bidir_list = np.array(bidir_list, dtype=np.float32)

    global_mean = final_means.mean()
    global_var = final_means.var()
    avg_edges = edges_list.mean()
    avg_bidir = bidir_list.mean()

    print("\nSummary for this dataset block:")
    print(f"  - Final state means across {n_samples} samples: mean={global_mean:.6f}, var={global_var:.6f}")
    print(f"  - Avg directed edges: {avg_edges:.2f}")
    print(f"  - Avg bidirectional edges: {avg_bidir:.2f}")

    return data, targets


class GraphDataset(Dataset):
    """
    A PyTorch Dataset that pre-generates (x, y) samples for a given evolution method:
    'base' or 'exponential'.

    Each sample is a sequence of node states [seq_len, n_nodes] plus a scalar target.
    Adjacency is NOT stored or saved.
    """
    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        n_nodes: int,
        graph_type: str,
        case_type: str = "base",
        max_iter: int = 1000000000,
        tol: float = 1e-3,
        show_bar: bool = False
    ):
        self.data, self.targets = generate_n_samples(
            n_samples=n_samples,
            n_nodes=n_nodes,
            graph_type=graph_type,
            case_type=case_type,
            n_time_samples=seq_len,
            max_iter=max_iter,
            tol=tol,
            show_bar=show_bar
        )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]


def main():
    # Parameters for dataset generation
    train_samples = 800
    test_samples = 200
    seq_len = 10
    max_iter = 1000000000
    tol = 1e-3
    output_dir = "./data_directed"

    # Define the node sizes, case types, and graph types to iterate over
    node_sizes = [25]
    case_types = ["base", "exponential"]
    graph_types = ["barabasi","watt","erdos"]

    for case_type in case_types:
        for graph_type in graph_types:
            for n_nodes in node_sizes:
                # Prepare the output directory for this combination
                save_dir = Path(output_dir) / f"{case_type}_{graph_type}_n{n_nodes}_seq{seq_len}" \
                                             / f"train{train_samples}_test{test_samples}"
                os.makedirs(save_dir, exist_ok=True)

                print(f"\n{'='*80}")
                print(f"Generating dataset with parameters:")
                print(f"  case_type   : {case_type}")
                print(f"  graph_type  : {graph_type}")
                print(f"  n_nodes     : {n_nodes}")
                print(f"  seq_len     : {seq_len}")
                print(f"  max_iter    : {max_iter}")
                print(f"  tol         : {tol}")
                print(f"  output_dir  : {save_dir}")
                print(f"{'='*80}\n")

                # Generate training dataset
                print(f"Generating training dataset (n_samples={train_samples})...")
                train_dataset = GraphDataset(
                    n_samples=train_samples,
                    seq_len=seq_len,
                    n_nodes=n_nodes,
                    graph_type=graph_type,
                    case_type=case_type,
                    max_iter=max_iter,
                    tol=tol,
                    show_bar=True
                )
                torch.save(train_dataset.data, save_dir / "train_data.pt")
                torch.save(train_dataset.targets, save_dir / "train_targets.pt")
                
                print(f"\nTraining dataset statistics:")
                print(f"  - Data shape: {train_dataset.data.shape}")
                print(f"  - Targets shape: {train_dataset.targets.shape}")

                # Generate test dataset
                print(f"\nGenerating test dataset (n_samples={test_samples})...")
                test_dataset = GraphDataset(
                    n_samples=test_samples,
                    seq_len=seq_len,
                    n_nodes=n_nodes,
                    graph_type=graph_type,
                    case_type=case_type,
                    max_iter=max_iter,
                    tol=tol,
                    show_bar=True
                )
                torch.save(test_dataset.data, save_dir / "test_data.pt")
                torch.save(test_dataset.targets, save_dir / "test_targets.pt")

                print(f"\nTest dataset statistics:")
                print(f"  - Data shape: {test_dataset.data.shape}")
                print(f"  - Targets shape: {test_dataset.targets.shape}")

                print(f"\nDone! Datasets saved in {save_dir}\n")

if __name__ == "__main__":
    main()