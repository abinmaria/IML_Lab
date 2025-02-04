import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Step 1: Load MNIST CSV
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    features = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float)  # Pixels (ignoring 'Label' column)
    
    # Assuming the 'Label' column contains the target variable
    labels = torch.tensor(df['Label'].values, dtype=torch.long)
    
    # Creating fully connected graph (for simplicity)
    num_nodes = len(df)
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()  # Fully connected edges

    data = Data(x=features, edge_index=edge_index, y=labels)
    
    return data

# Step 2: Define GCN Model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))  # First Graph Conv Layer
        x = self.conv2(x, edge_index)  # Second Graph Conv Layer
        return x

# Step 3: Training function
def train(model, data, epochs=100, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)  # Forward pass
        loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Only use training nodes for loss
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

# Step 4: Evaluate function
def evaluate(model, data):
    model.eval()
    out = model(data)
    _, pred = out.max(dim=1)
    accuracy = accuracy_score(data.y.cpu().numpy(), pred.cpu().numpy())
    return accuracy

# Step 5: Visualize Graph
def visualize_graph(data):
    G = nx.Graph()
    
    # Add nodes to the graph
    for i in range(data.num_nodes):
        G.add_node(i, label=str(data.y[i].item()))  # Use label as node attribute
    
    # Add edges to the graph (create an undirected edge for each pair)
    edge_index = data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0][i], edge_index[1][i])
    
    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # Spring layout for visualization
    node_labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, with_labels=True, node_size=50, node_color='skyblue', font_size=8, font_weight='bold', edge_color='gray')
    plt.title("Graph Visualization of GCN")
    plt.show()

# Step 6: Run everything
def main():
    csv_file = '/home/abin/IML lab/mnist1.5k.csv'  # Update this path with the actual CSV location
    data = load_data(csv_file)

    # Splitting data into training (80%) and testing (20%) masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[:int(0.8 * data.num_nodes)] = 1
    test_mask[int(0.8 * data.num_nodes):] = 1

    data.train_mask = train_mask
    data.test_mask = test_mask

    # Initialize GCN model
    input_dim = data.num_node_features  # 784 (28x28 pixels)
    hidden_dim = 64
    output_dim = 10  # 10 classes (0-9 digits)

    model = GCN(input_dim, hidden_dim, output_dim)

    # Visualize the graph (This is optional, can be commented out if you don't want the visualization)
    visualize_graph(data)

    # Train the model
    train(model, data, epochs=200, lr=0.01)

    # Evaluate the model
    accuracy = evaluate(model, data)
    print(f"Final Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
