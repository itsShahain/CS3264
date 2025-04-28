import os
import numpy as np
import mediapipe as mp
import torch
import cv2
from mediapipe.python.solutions.pose import POSE_CONNECTIONS
mp_pose = mp.solutions.pose
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

class ST_GCN_18(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 edge_importance_weighting=True,
                 data_bn=True,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      A.size(1)) if data_bn else iden
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels,
                         64,
                         kernel_size,
                         1,
                         residual=False,
                         **kwargs0),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, kernel_size, 2, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 256, kernel_size, 2, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # Input shape: (N, C, T, V)
        N, C, T, V = x.size()

        # Data normalization
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
        x = x.view(N, V * C, T)  # (N, V*C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)  # (N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)

        # Forward through ST-GCN blocks
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # Global pooling
        x = F.avg_pool2d(x, x.size()[2:])  # (N, 256, 1, 1)
        x = x.view(N, -1, 1, 1)  # (N, 256, 1, 1)

        # Prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)  # (N, num_class)
        return x


def zero(x):
    return 0


def iden(x):
    return x

class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
    

class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """
    def __init__(self,
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge()
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self):
        # edge is a list of [child, parent] paris

        self.num_node = 33
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)]
        self.edge = self_link + neighbor_link
        self.center = 24

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                    i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

class CognitiveReasoningModuleImproved(nn.Module):
    def __init__(self, num_frames=500, num_joints=33, feature_dim=2):
        super().__init__()
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.feature_dim = feature_dim # Number of cognitive features we are calculating

        self.joint_indices = {
            "head": 0,
            "left_foot": 32,
            "right_foot": 31,
            "spine": 24
        }

        # Optional: Normalize the raw continuous features
        self.feature_normalizer = nn.LayerNorm(feature_dim)

        # Reasoning head: MLP operates on continuous, normalized features
        # Input size is feature_dim
        self.reasoning_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output: Binary fall / no-fall logits per frame
        )

    def forward(self, x):
        """
        x: Tensor of shape [B, C, T, V] (e.g., C=4 for x, y, z, conf)
        Returns:
            logits: Frame-wise logits [B, T, 2]
            cognitive_feats: Frame-wise continuous cognitive features [B, T, feature_dim]
        """
        B, C, T, V = x.shape
        assert C >= 2, "Expected at least 2D coordinates (X, Y)"

        coords = x[:, :2, :, :].permute(0, 2, 3, 1)  # [B, T, V, C=2] (x, y)

        # Step 1: Mask padded frames (all joints = 0 for both x and y)
        # Check if *any* coordinate in a frame is non-zero
        valid_mask = (coords.abs().sum(dim=(2, 3)) > 1e-6).float()  # [B, T]

        cognitive_feats_list = []

        # Rule 1: Head-ground proximity (continuous distance)
        head_y = coords[:, :, self.joint_indices["head"], 1]
        # Use min foot y as ground reference for robustness if one foot is lifted
        foot_y = torch.min(coords[:, :, self.joint_indices["left_foot"], 1],
                           coords[:, :, self.joint_indices["right_foot"], 1])
        head_to_ground = head_y - foot_y # Smaller (more negative) means closer to ground relative to feet
        # Clamp potential extreme values if needed, or normalize later
        # head_to_ground = torch.clamp(head_to_ground, -2.0, 2.0) # Example clamp
        cognitive_feats_list.append(head_to_ground) # Shape [B, T]

        # Rule 2: Torso vertical velocity (continuous)
        spine_y = coords[:, :, self.joint_indices["spine"], 1]
        # Calculate difference between consecutive frames, pad first frame diff with 0
        spine_velocity_y = spine_y[:, 1:] - spine_y[:, :-1] # [B, T-1]
        spine_velocity_y = F.pad(spine_velocity_y, (1, 0), value=0) # [B, T] (negative indicates downward motion)
        # Clamp potential extreme values if needed, or normalize later
        # spine_velocity_y = torch.clamp(spine_velocity_y, -0.5, 0.5) # Example clamp
        cognitive_feats_list.append(spine_velocity_y) # Shape [B, T]

        # --- Add more continuous rules here ---
        # Example: Rule 3: Body aspect ratio (Height/Width) - might indicate lying down
        # Requires calculating bounding box or min/max x/y coordinates

        # Stack and permute features
        cognitive_feats = torch.stack(cognitive_feats_list, dim=-1) # [B, T, feature_dim]

        # Zero out features for padded frames BEFORE normalization
        cognitive_feats = cognitive_feats * valid_mask.unsqueeze(-1)

        # Normalize features across the feature dimension for each frame
        # Only apply norm where mask is valid to avoid normalizing zeros?
        # Safer: Normalize all, then re-apply mask if needed, or just normalize valid parts.
        # Let's normalize all for simplicity, LayerNorm handles zero inputs reasonably.
        cognitive_feats_norm = self.feature_normalizer(cognitive_feats)

        # Get frame-wise logits from normalized features
        logits = self.reasoning_head(cognitive_feats_norm)  # [B, T, feature_dim] -> [B, T, 2]

        # Zero out logits for padded frames
        logits = logits * valid_mask.unsqueeze(-1)

        # Return both frame-wise logits and the raw (but masked) features
        return logits, cognitive_feats
    
class STGCNWithReasoningImproved(nn.Module):
    def __init__(self, stgcn_model, reasoning_feature_dim=2, lstm_hidden_dim=64):
        super().__init__()
        self.stgcn = stgcn_model
        # Use the improved reasoning module
        self.reasoning = CognitiveReasoningModuleImproved(feature_dim=reasoning_feature_dim)

        self.reasoning_feature_dim = reasoning_feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # Temporal aggregation for reasoning features using LSTM
        self.reasoning_lstm = nn.LSTM(input_size=reasoning_feature_dim,
                                      hidden_size=lstm_hidden_dim,
                                      num_layers=1, # Or more layers
                                      batch_first=True) # Input shape: (batch, seq_len, features)

        # Get the output dimension from the STGCN model's fcn layer
        stgcn_output_dim = stgcn_model.fcn.out_channels # Should be num_class (e.g., 2)

        # Fusion MLP: Takes concatenated features from STGCN and Reasoning LSTM
        # Input size = STGCN output dim + LSTM hidden dim
        fusion_input_dim = stgcn_output_dim + lstm_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2), # Intermediate layer
            nn.ReLU(),
            nn.Dropout(0.5), # Add dropout for regularization
            nn.Linear(fusion_input_dim // 2, stgcn_output_dim) # Final output classes
        )

    def forward(self, x):
        # ST-GCN branch
        # Assume stgcn outputs logits directly [B, num_class]
        stgcn_logits = self.stgcn(x)  # Shape: [B, 2]

        # Reasoning branch
        # Get frame-wise features [B, T, feature_dim] (we ignore frame-wise logits now)
        _, cognitive_feats = self.reasoning(x) # Shape: [B, T, feature_dim]

        # Temporal aggregation using LSTM
        # Input to LSTM: [B, T, feature_dim]
        # We only need the final hidden state or the output of the last time step
        lstm_out, (h_n, c_n) = self.reasoning_lstm(cognitive_feats)
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # Get the hidden state of the last layer
        reasoning_aggregated_features = h_n[-1] # Shape: [B, lstm_hidden_dim]
        # Alternatively, use the output of the last time step:
        # reasoning_aggregated_features = lstm_out[:, -1, :] # Shape: [B, lstm_hidden_dim]

        # Fusion
        # Concatenate the STGCN logits and the aggregated reasoning features
        fusion_input = torch.cat([stgcn_logits, reasoning_aggregated_features], dim=-1) # Shape: [B, 2 + lstm_hidden_dim]

        # Pass through fusion MLP
        final_logits = self.fusion(fusion_input) # Shape: [B, 2]

        return final_logits

# --- Example Usage (in __main__ block) ---
if __name__ == "__main__":
    # --- Data loading code remains the same ---
    with open('train_falls_le2_v2.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('test_falls_le2_v2.pkl', 'rb') as f:
        test_data = pickle.load(f)

    # open pkl 
    # with open('train_ur_fall_50_frames_4_features.pkl', 'rb') as f:
    #     train_data_new_set = pickle.load(f)
    # with open('val_ur_fall_50_frames_4_features.pkl', 'rb') as f:
    #     test_data_new_set = pickle.load(f)
    
    # train_data = [(torch.tensor(data[0].numpy()).permute(2, 0, 1), data[1]) for data in train_data_new_set]
    # test_data = [(torch.tensor(data[0].numpy()).permute(2, 0, 1), data[1]) for data in test_data_new_set]

    batch_size = 32
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build base STGCN model
    graph_cfg = {'strategy': 'spatial'}
    base_model = ST_GCN_18(
        in_channels=4, # Make sure this matches your data (x, y, z, conf?)
        num_class=2,
        graph_cfg=graph_cfg,
        data_bn=True, # Keep data_bn=True usually
        edge_importance_weighting=True, # Keep edge importance
        # dropout=0.5 # Dropout is applied within st_gcn_block, no need here
    ).to(device)

    # Wrap with IMPROVED reasoning and fusion
    # Adjust reasoning_feature_dim if you add more rules
    model = STGCNWithReasoningImproved(base_model,
                                       reasoning_feature_dim=2,
                                       lstm_hidden_dim=64).to(device)
    # model = base_model

    # --- Training loop code remains largely the same ---
    # (Loss, optimizer, scheduler, training/test loops)
    # You might need to adjust the learning rate or optimizer for the combined model
    criterion = nn.CrossEntropyLoss()
    # Use AdamW for potentially better convergence with complex models
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    # Adjust scheduler if needed, e.g., ReduceLROnPlateau or CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1) # Adjusted step_size

    # Training loop
    num_epochs = 30
    best_test_acc = 0.0

    dataloader = DataLoader(train_data, batch_size=batch_size)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (batch_data, batch_labels) in enumerate(dataloader):
            batch_data = batch_data.to(device, dtype=torch.float) # Ensure float type
            batch_labels = batch_labels.to(device, dtype=torch.long)
            optimizer.zero_grad()

            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            # Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

            # Print progress within epoch (optional)
            # if (i + 1) % 10 == 0:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')


        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")

        scheduler.step() # Step the scheduler after validation

                # Validation step after each epoch
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        for batch_test_data, batch_test_labels in test_dataloader:
            batch_test_data = batch_test_data.to(device, dtype=torch.float) # Ensure float type
            batch_test_labels = batch_test_labels.to(device, dtype=torch.long)

            outputs = model(batch_test_data)
            loss = criterion(outputs, batch_test_labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_test_labels.size(0)
            correct += (predicted == batch_test_labels).sum().item()
            
            all_preds.append(predicted.cpu())
            all_labels.append(batch_test_labels.cpu())


        test_loss /= len(test_dataloader)
        test_acc = 100 * correct / total
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        test_f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average='weighted')

        print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Test F1 Score: {test_f1:.4f}")

        # Save model if it's the best so far
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # torch.save(model.state_dict(), 'best_stgcn_reasoning_model.pth')
            print(f"Saved new best model with Test Accuracy: {best_test_acc:.2f}% and F1 Score: {test_f1:.4f}")

    print("Training completed.")
    print(f"Best Test Accuracy achieved: {best_test_acc:.2f}%")
