#ideas prototyped with the help of Gemini
import pickle
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split 
from torch_geometric.data import Data, Batch
from torch_geometric.nn import TransformerConv, GATConv, GATv2Conv, GCNConv, global_mean_pool, GraphNorm, BatchNorm, GlobalAttention
import torch.nn.functional as F
import time
import os
import glob 
import math
import traceback

from sklearn.metrics import confusion_matrix, f1_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch will use device: {device}")


# We learn the graph on top of these connections
MEDIAPIPE_POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
    (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
    (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
    (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32), (12, 23), (24, 11),
    (14, 0), (14, 11), (14, 23), (14, 25),
    (12, 13), (12, 25), (12, 26),
    (0, 12), (0, 11), (0, 14), (0, 13), (0, 26), (0, 25),
    (11, 26), (11, 25),
    (13, 24), (13, 26),
    (14, 18), (14, 28), (14, 27), (14, 17),
    (13, 17), (13, 18), (13, 27), (13, 28),
    (18, 26), (18, 25),
    (17, 26), (17, 25),
    (26, 31), (25, 32)
    ])

class MediaPipeGraphBuilder:
    MP_EDGES = MEDIAPIPE_POSE_CONNECTIONS
    @staticmethod
    def create_graph(keypoints_np, velocities_np):
        if not isinstance(keypoints_np, np.ndarray) or keypoints_np.shape != (33, 4):
            return None
        if not isinstance(velocities_np, np.ndarray) or velocities_np.shape != (33, 3):
            return None

        try:
            node_features_np = np.concatenate((keypoints_np, velocities_np), axis=1)
        except ValueError as e:
             return None

        node_features = torch.tensor(node_features_np, dtype=torch.float32) # Shape: [33, 7]
        edge_index = torch.tensor(list(MediaPipeGraphBuilder.MP_EDGES), dtype=torch.long).t().contiguous()
        valid_edge_mask = (edge_index[0] < 33) & (edge_index[1] < 33)
        edge_index = edge_index[:, valid_edge_mask]
        return Data(x=node_features, edge_index=edge_index)

class PoseSequenceDataset(Dataset):
    def __init__(self, master_pickle_path):
        self.master_pickle_path = master_pickle_path
        self.all_video_data = [] 

        try:
            with open(master_pickle_path, 'rb') as f:
                loaded_data = pickle.load(f)

            if not isinstance(loaded_data, list):
                raise TypeError(f"Check data types...expected master pickle file to contain a list of video data.")

            self.all_video_data = loaded_data
            print("Successfully loaded videos")

        except FileNotFoundError:
            print("pkl file not found")
            raise
        except pickle.UnpicklingError as e:
            print("cannot unpickle pkl")
            raise
        except TypeError as e:
             print(f"FATAL ERROR: Data structure validation failed. {e}")
             raise
        except Exception as e:
            print(f"FATAL ERROR: An unexpected error occurred while loading {self.master_pickle_path}: {e}")
            traceback.print_exc()
            raise

        self.graph_builder = MediaPipeGraphBuilder()

    def __len__(self):
        return len(self.all_video_data)

    def __getitem__(self, idx):
        if idx >= len(self.all_video_data):
             raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.all_video_data)}")

        video_data = self.all_video_data[idx]
        graphs = []
        label = None

        try:
            if not isinstance(video_data, dict):
                print(f"Warning: Data for index {idx} is not a dict ({type(video_data)}). Skipping.")
                return [], -1 

            if 'label' not in video_data or 'frames_data' not in video_data:
                print(f"Warning: Data for index {idx} missing 'label' or 'frames_data' key. Skipping.")
                return [], -1

            label = video_data['label']
            frames_data = video_data['frames_data']

            if not isinstance(label, int):
                 try:
                     label = int(label)
                 except ValueError:
                     return [], -1

            if not isinstance(frames_data, list):
                print(f"Warning: 'frames_data' for index {idx} is not a list, got {type(frames_data)}. Skipping graph creation.")
                return [], label

            if not frames_data:
                 return [], label


            for i, frame_data in enumerate(frames_data):
                if not isinstance(frame_data, dict):
                    continue

                if 'keypoints' not in frame_data or 'velocities' not in frame_data:
                    continue

                keypoints_np = frame_data['keypoints']
                velocities_np = frame_data['velocities']

                graph = self.graph_builder.create_graph(keypoints_np, velocities_np)

                if graph is not None:
                    if graph.x.shape[1] == 7:
                        graphs.append(graph)
                # else: GraphBuilder might print a warning internally

        except KeyError as e:
             print(f"Error: Missing key {e} in video data structure for index {idx}. Skipping.")
             return [], -1
        except Exception as e:
            print(f"Error processing item at index {idx}: {e}")
            traceback.print_exc()
            return [], -1 
        if label is None:
            print(f"Error: Label extraction failed for index {idx} despite processing. Skipping.")
            return [], -1

        return graphs, label


def collate_fn(batch):
    filtered_batch = [(seq, lbl) for seq, lbl in batch if seq and lbl != -1]
    if not filtered_batch:
        return [], torch.tensor([], dtype=torch.long)
    sequences, labels = zip(*filtered_batch)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return list(sequences), labels_tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe_to_add = self.pe[:x.size(1), :].unsqueeze(0) 
        x = x + pe_to_add 
        return self.dropout(x)


class TemporalGaitGCN(nn.Module):
    def __init__(self, num_node_features=7, feature_dim=64, gcn_hidden=128, gcn_out_dim=128,
                 transformer_hidden=128, nhead=4, num_layers=2, num_classes=2, dropout=0.1):
        super().__init__()
        self.gcn_out_dim = gcn_out_dim
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_node_features, feature_dim), nn.ReLU(),
            nn.Linear(feature_dim, feature_dim), nn.ReLU() )
        self.heads = 8
        self.gcn1 = TransformerConv(feature_dim, gcn_hidden, heads=self.heads, concat=True, dropout=dropout)
        self.bn1 = BatchNorm(gcn_hidden * self.heads)
        self.gcn2 = TransformerConv(gcn_hidden * self.heads, gcn_hidden, heads=self.heads, concat=True, dropout=dropout)
        self.bn2 = BatchNorm(gcn_hidden * self.heads)
        
        self.gcn3 = TransformerConv(gcn_hidden * self.heads, gcn_hidden, heads=self.heads, concat=True, dropout=dropout)
        self.bn3 = BatchNorm(gcn_hidden * self.heads)
        self.gcn10 = TransformerConv(gcn_hidden * self.heads, gcn_out_dim, heads=1, concat=False, dropout=dropout)
        self.bn10 = BatchNorm(gcn_out_dim)

        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(nn.Linear(gcn_out_dim, gcn_out_dim // 2), nn.ReLU(), nn.Linear(gcn_out_dim // 2, 1)) )

        self.pos_encoder = PositionalEncoding(gcn_out_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=gcn_out_dim, nhead=nhead, dim_feedforward=transformer_hidden, dropout=dropout, batch_first=True )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(gcn_out_dim, gcn_out_dim // 2), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(gcn_out_dim // 2, num_classes) )

        self.d_h = feature_dim 
        self.d_k = self.d_h   
        self.lin_Q = nn.Linear(self.d_h, self.heads * self.d_k)
        self.lin_K = nn.Linear(self.d_h, self.heads * self.d_k)

    def forward(self, batch_data):
        if not isinstance(batch_data, tuple) or len(batch_data) != 2:

            if isinstance(batch_data, list): sequences = batch_data
            else: raise ValueError("Input batch_data must be a tuple of (sequences, labels) or list")
        else: sequences, _ = batch_data

        current_device = next(self.parameters()).device
        if not sequences: return torch.empty((0, self.fc[-1].out_features), device=current_device)

        batch_temporal_features = []
        sequences_processed_indices = []

        for seq_idx, sequence in enumerate(sequences):
            if not sequence or not all(isinstance(g, Data) for g in sequence): continue
            sequence_on_device = [graph.to(current_device) for graph in sequence]

            try:
                graph_batch = Batch.from_data_list(sequence_on_device)
                x, edge_index, batch_idx = graph_batch.x, graph_batch.edge_index, graph_batch.batch
                expected_features = self.feature_encoder[0].in_features
                if x is None or x.shape[1] != expected_features:
                    continue

                x = self.feature_encoder(x) 
                N_total = x.shape[0]      

                Q = self.lin_Q(x)  
                K = self.lin_K(x)  

 
                #fix
                Q = Q.view(N_total, self.heads, self.d_k)
                K = K.view(N_total, self.heads, self.d_k)

                Q_bmm = Q.permute(0, 2, 1) 
                Q_bmm = Q.permute(1, 0, 2) 
                K_bmm = K.permute(1, 2, 0) 

                scores = torch.matmul(Q_bmm, K_bmm) 
                scores = scores / (self.d_k ** 0.5)


                scores = scores.mean(dim=0)

                #be careful, need to mask out other nodes in other graphs of the same batch
                batch_idx_row = batch_idx.unsqueeze(1).expand(N_total, N_total)
                batch_idx_col = batch_idx.unsqueeze(0).expand(N_total, N_total)
                same_graph_mask = (batch_idx_row == batch_idx_col)
                scores_masked = scores.masked_fill(same_graph_mask == False, -float('inf'))


                #ST-Gumbel start
                logits_non_exist = torch.zeros_like(scores_masked)
                logits_exist = scores_masked

                logits_cat = torch.stack([logits_non_exist, logits_exist], dim=-1)

                tau = 1.0 
                adj_soft_one_hot = F.gumbel_softmax(logits_cat, tau=tau, hard=True, dim=-1)

                A_hard = adj_soft_one_hot[..., 1] 

                edge_indices_tuple = A_hard.nonzero(as_tuple=True)
                
                learned_edge_index = torch.stack(edge_indices_tuple, dim=0)

                edge_index = learned_edge_index


                x1 = F.relu(self.bn1(self.gcn1(x, edge_index)))
                x2_identity = x1
                x2 = F.relu(self.bn2(self.gcn2(x1, edge_index)))
                if x2.shape == x2_identity.shape: x2 = x2 + x2_identity 
                x3 = F.relu(self.bn3(self.gcn3(x2, edge_index)))
                x10 = F.relu(self.bn10(self.gcn10(x3, edge_index)))


                frame_features = self.pool(x10, batch_idx)
                batch_temporal_features.append(frame_features)
                sequences_processed_indices.append(seq_idx)

            except Exception as e:
                print(f"Error processing graph batch for sequence index {seq_idx}: {e}")
                traceback.print_exc()
                continue

        if not batch_temporal_features:
            return torch.empty((0, self.fc[-1].out_features), device=current_device)

        try:
            padded_sequences = torch.nn.utils.rnn.pad_sequence(
                batch_temporal_features, batch_first=True, padding_value=0.0 )
        except RuntimeError as e:
            print(f"Error during padding: {e}")
            return torch.empty((0, self.fc[-1].out_features), device=current_device)

        x = self.pos_encoder(padded_sequences)
        transformer_output = self.transformer_encoder(x)
        last_step_output = transformer_output[:, -1, :]
        output = self.fc(last_step_output)
        return output


if __name__ == '__main__':
    TRAIN_PICKLE_PATH = '/content/train_ur_fall_50_frames_7_features.pkl'
    VAL_PICKLE_PATH = '/content/val_ur_fall_50_frames_7_features.pkl'


    train_dataset = PoseSequenceDataset(master_pickle_path=TRAIN_PICKLE_PATH)
    val_dataset = PoseSequenceDataset(master_pickle_path=VAL_PICKLE_PATH)

    batch_size = 4 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    print(f"Created DataLoader with batch size {batch_size}")

    model = TemporalGaitGCN(num_node_features=7, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 5000 
    print("\nStarting Training...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        processed_batches = 0
        correct_preds = 0
        total_samples = 0

        start_time = time.time()
        for batch_idx, batch_data in enumerate(train_loader):
            sequences, labels = batch_data 

            if not sequences: continue
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model((sequences, labels))
            if outputs.shape[0] == 0: continue

            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                print(f"!!! NaN loss detected. Skipping backprop. !!!")
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            processed_batches += 1
            _, predicted = torch.max(outputs.data, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_time = time.time() - start_time
        avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
        accuracy = (correct_preds / total_samples) * 100 if total_samples > 0 else 0
        print(f'--- Epoch {epoch+1}/{num_epochs} Summary ---')
        print(f'Time: {epoch_time:.2f}s')
        print(f'Average Training Loss: {avg_loss:.4f}')
        print(f'Training Accuracy: {accuracy:.2f}% ({correct_preds}/{total_samples})')


        model.eval()
        val_loss = 0
        val_correct_preds = 0
        val_total_samples = 0
        val_processed_batches = 0
        all_val_labels = []   
        all_val_preds = []       

        with torch.no_grad():
            for batch_data in val_loader:
                sequences, labels = batch_data
                if not sequences: continue


                all_val_labels.extend(labels.cpu().numpy()) 

                labels = labels.to(device)
                outputs = model((sequences, labels)) 
                if outputs.shape[0] == 0: continue

                loss = criterion(outputs, labels)
                if torch.isnan(loss):
                    print(f"!!! NaN loss detected.Skipping batch metrics. !!!")
                    del all_val_labels[-outputs.shape[0]:] 
                    continue 

                val_loss += loss.item()
                val_processed_batches += 1
                _, predicted = torch.max(outputs.data, 1)

                all_val_preds.extend(predicted.cpu().numpy())

                val_correct_preds += (predicted == labels).sum().item()
                val_total_samples += labels.size(0)

        avg_val_loss = val_loss / val_processed_batches if val_processed_batches > 0 else 0
        val_accuracy = (val_correct_preds / val_total_samples) * 100 if val_total_samples > 0 else 0

        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}% ({val_correct_preds}/{val_total_samples})')


        if val_total_samples > 0 and len(all_val_labels) == len(all_val_preds): 

            y_true = np.array(all_val_labels)
            y_pred = np.array(all_val_preds)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) 
            if cm.shape == (2, 2): 
                tn, fp, fn, tp = cm.ravel()
            else: 
                tn, fp, fn, tp = 0, 0, 0, 0
                if 0 in y_true or 0 in y_pred: 
                    tn = ((y_true == 0) & (y_pred == 0)).sum()
                    fp = ((y_true == 0) & (y_pred == 1)).sum()
                if 1 in y_true or 1 in y_pred:
                    fn = ((y_true == 1) & (y_pred == 0)).sum()
                    tp = ((y_true == 1) & (y_pred == 1)).sum()

            f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)

            print(f'Validation TN: {tn}')
            print(f'Validation FP: {fp}')
            print(f'Validation FN: {fn}')
            print(f'Validation TP: {tp}')
            print(f'Validation F1 Score: {f1:.4f}')

        else:
            print("Validation set empty or prediction/label count mismatch. Skipping detailed metrics.")
        # --- END ADDED METRIC CALCULATION ---

        print('-----------------------------')

    print("\nTraining Finished.")

    # save_path = 'fall_detection_gcn.pth'
    # torch.save(model.state_dict(), save_path)
    # print(f"Model saved to {save_path}")
