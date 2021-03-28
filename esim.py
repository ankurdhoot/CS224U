from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import vocab
from dataset import SNLITrainDataset, SNLIDevDataset
import torch.optim as optim
from sklearn.metrics import classification_report
import numpy as np
from typing import List


LABEL_MAP = {'neutral' : 0, 'contradiction' : 1, 'entailment' : 2}
VOCAB = vocab.build_vocab()
# setting device on GPU if available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

class ESIM(nn.Module):
    def __init__(self, embed_size=100, hidden_size=100, dropout_rate=0.4):
        """
        Args:
            embed_size : int
                Size of input embeddings
            hidden_size : int
                Size of the LSTM hidden state
            dropout_rate: float
                dropout probability for linear layers
        """
        super(ESIM, self).__init__()
        
        self.device = DEVICE
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = VOCAB
        
        self.premise_repr = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, bias=True, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.hypothesis_repr = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, bias=True, batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        # TODO(dhoota): Implement the bidirectional version.
        # self.premise_inference = nn.LSTM(input_size=2 * self.hidden_size, hidden_size=self.hidden_size, bias=True, batch_first=True)
        # lf.hypothesis_inference = nn.LSTM(input_size=2 * self.hidden_size, hidden_size=self.hidden_size, bias=True, batch_first=True)
        
        self.representation_projection_layer = nn.Linear(in_features=self.hidden_size * 4,out_features=self.hidden_size, bias=True)
        
        # TODO(dhoota): Implement the bidirectional version.
        self.premise_LSTM = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, bias=True, batch_first=True, dropout=dropout_rate)
        
        self.hypothesis_LSTM = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, bias=True, batch_first=True, dropout=dropout_rate)
        
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
        self.final_layer = nn.Sequential(nn.Linear(in_features=2*self.hidden_size,
                                                   out_features=self.hidden_size),
                                         nn.Dropout(),
                                         nn.ReLU(),
                                        nn.Linear(in_features=self.hidden_size, out_features=3),
                                         nn.Tanh()
                                        )
        
        self.relu = nn.ReLU()
        
        
        # TODO(ankur): Maybe set freeze=False
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.vocab.embedding, dtype=torch.float), padding_idx=self.vocab.pad_id)
        
    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.
        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(), dtype=torch.float)
        # enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, :, src_len:] = 1
        return enc_masks.to(self.device)
    
    def forward(self, premises, hypotheses) -> torch.Tensor:
        """
        Args:
            premises : List[Tensor]
            hypotheses: List[Tensor]
        """
        premise_lengths = [t.shape[0] for t in premises]
        hypothesis_lengths = [t.shape[0] for t in hypotheses]
        
        # print(premise_lengths)
        # print(hypothesis_lengths)
        
        # (batch_len, max_seq_len)
        premises_padded = torch.nn.utils.rnn.pad_sequence(premises, batch_first=True, padding_value=self.vocab.pad_id)
        hypotheses_padded = torch.nn.utils.rnn.pad_sequence(hypotheses, batch_first=True, padding_value=self.vocab.pad_id)
        
        # Get the embeddings
        # (batch_len, max_seq_len, embed_size)
        distributed_repr_premise = self.embedding(premises_padded)
        distributed_repr_hypothesis = self.embedding(hypotheses_padded)
        
        distributed_repr_premise = torch.nn.utils.rnn.pack_padded_sequence(distributed_repr_premise, lengths=premise_lengths,batch_first=True, enforce_sorted=False)
        
        distributed_repr_hypothesis = torch.nn.utils.rnn.pack_padded_sequence(distributed_repr_hypothesis, lengths=hypothesis_lengths,batch_first=True, enforce_sorted=False)
        
        # Pass the distributed representations through the ESIM model.
        
        # Pass the premise and hypothesis through the LSTM to represent a word and its context. These
        # representations will be used to compute attention weights.
        
        # (batch, seq_len, num_directions * hidden_size), (batch, num_directions, hidden_size), (batch, num_directions, hidden_size)
        premise_output, (premise_h_n, premise_c_n) = self.premise_repr(distributed_repr_premise)
        hypothesis_output, (hypothesis_h_n, hypothesis_c_n) = self.hypothesis_repr(distributed_repr_hypothesis)
        
        # (batch, n, k)
        premise_output, p_lengths = torch.nn.utils.rnn.pad_packed_sequence(premise_output, batch_first=True)
        
        # (batch, m, k)
        hypothesis_output, h_lengths = torch.nn.utils.rnn.pad_packed_sequence(hypothesis_output, batch_first=True)
        
        # We want to compute e_ij = dot(u_i, v_j) where u_i is the LSTM representation of the ith premise token
        # and v_j is the LSTM representation of the jth hypothesis token.
        
        # Let n denote the seq_len of premise_output and m denote the seq_len of hypothesis_output
        # Let k = num_directions * hidden_size
        # (batch, m, k) -> (batch, k, m)
        hypothesis_output_permute = hypothesis_output.permute(0, 2, 1)
        
        # (batch, n, k) * (batch, k, m) -> (batch, n, m)
        unnormalized_attention_weights_premise = torch.bmm(premise_output, hypothesis_output_permute)
        
         # (batch, m, n)
        unnormalized_attention_weights_hypothesis = unnormalized_attention_weights_premise.clone().detach().permute(0, 2, 1)
        
        copy = unnormalized_attention_weights_premise.clone().detach()
        
        # Mask those vectors that are padding vectors in the hypothesis.
        premise_mask = self.generate_sent_masks(unnormalized_attention_weights_premise, hypothesis_lengths)
        unnormalized_attention_weights_premise.data.masked_fill_(premise_mask.bool(), -float('inf'))
        
        # (batch, n, m)
        normalized_attention_weights_premise = torch.nn.functional.softmax(unnormalized_attention_weights_premise, dim=2)
        
        # (batch, n, m) * (batch, m, k) --> (batch, n, k)
        # The extracted relevant information from the hypothesis for each word in the premise.
        premise_attention_vectors = torch.bmm(normalized_attention_weights_premise, hypothesis_output)
        
        # (batch, m, n)
        # unnormalized_attention_weights_hypothesis = unnormalized_attention_weights_premise.permute(0, 2, 1)
        
        # Mask those vectors that are padding vectors in the premise.
        hypothesis_mask = self.generate_sent_masks(unnormalized_attention_weights_hypothesis, premise_lengths)
        
        unnormalized_attention_weights_hypothesis.data.masked_fill_(hypothesis_mask.bool(), -float('inf'))
        
        # (batch, m, n)
        normalized_attention_weights_hypothesis = torch.nn.functional.softmax(unnormalized_attention_weights_hypothesis, dim=2)
        
        # (batch, m, n) * (batch, n, k) --> (batch, m, k)
        # The extracted relevant information from the premise for each word in the hypothesis.
        hypothesis_attention_vectors = torch.bmm(normalized_attention_weights_hypothesis, premise_output)
        
        # TODO(dhoota): Add the subtraction and dot product.
        # Combine the premise_output with the premise_attention_vectors. This represents our full representation of the premise words.
        # (batch, n, 2 * k)
        full_premise_representation = torch.cat((premise_output, premise_attention_vectors), dim=2)
        
        # Combine the hypothesis_output with the hypothesis_attention_vectors. This represents our full representation of the hypothesis.
        # (batch, m, 2 * k)
        full_hypothesis_representation = torch.cat((hypothesis_output, hypothesis_attention_vectors), dim=2)
        
        # Feed the representations through a projection layer to get back to self.hidden_size dimensionality.
        # (batch, n, k)
        full_premise_representation = self.relu(self.representation_projection_layer(full_premise_representation))
        # (batch, m, k)
        full_hypothesis_representation = self.relu(self.representation_projection_layer(full_hypothesis_representation))
        
        # Pack the sequences before passing through LSTM. This is necessary so that the last hidden state used is the correct one.
        full_premise_representation = torch.nn.utils.rnn.pack_padded_sequence(full_premise_representation, lengths=premise_lengths, batch_first=True, enforce_sorted=False)
        
        full_hypothesis_representation = torch.nn.utils.rnn.pack_padded_sequence(full_hypothesis_representation, lengths=hypothesis_lengths, batch_first=True, enforce_sorted=False)
        
        
        # Feed the representations through another LSTM
        # (batch, n, k), (batch, num_directions, hidden_size), (batch, num_directions, hidden_size)
        p_output, (p_h_n, p_c_n) = self.premise_LSTM(full_premise_representation)
        # (batch, m, k), (batch, num_directions, hidden_size), (batch, num_directions, hidden_size)
        h_output, (h_h_n, h_c_n) = self.hypothesis_LSTM(full_hypothesis_representation)
        
        # Combine the last hidden states from the premise and hypothesis.
        # (batch, 2 * hidden_size)
        final_output_representation = torch.cat((torch.reshape(p_h_n, (-1, 1*self.hidden_size)), torch.reshape(h_h_n, (-1,               1*self.hidden_size))), dim=1) 
        
        pred_weights = self.final_layer(final_output_representation)
        return pred_weights        
        
        
def collate_fn(batch):
    """
    Args:
        batch : List[NLIExample]
    Returns:
    
    data_list, label_list
    """
    
    label_list = []
    premise_word_list = []
    hypothesis_word_list = []
    for ex in batch:
        premise = ex[0]
        hypothesis = ex[1]
        
        # TODO(dhoota): Strip away punctuation at the end.
        premise_words = premise.lower().split()
        hypothesis_words = hypothesis.lower().split()
        
        premise_word_list.append(premise_words)
        hypothesis_word_list.append(hypothesis_words)
       
        label_list.append(LABEL_MAP[ex[2]])
        
    premise_tensors = VOCAB.to_input_tensor(premise_word_list, device=DEVICE)
    hypothesis_tensors = VOCAB.to_input_tensor(hypothesis_word_list, device=DEVICE)
    
    label_list = torch.tensor(label_list, device=DEVICE)
        
    return premise_tensors, hypothesis_tensors, label_list
    
    
def train():
    
    # Create the model
    print("Creating model")
    model = ESIM()
    print(model)
    print("Trying to move model to GPU")
    model.to(DEVICE)
    print("Put the model on GPU")
    print(model)
    print(next(model.parameters()).is_cuda)
    
    # TODO(ankur): Maybe implement a toy dataset for quick iteration.
    train_dataset = SNLITrainDataset()
    dev_dataset = SNLIDevDataset()
    
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_data_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    
    # TODO(dhoota): Switch to adam optimizer
    # Create the optimizer
    opt = optim.Adam(model.parameters(), lr=.0004)
    
    # Create the loss function
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    for epoch in range(1):
        model.train()
        running_loss = 0.0
        y_pred = []
        y_true = []
        for i, data in enumerate(train_data_loader, 1):
            # print("Training")
            # get the inputs; data is a list of [inputs, labels]
            premise, hypothesis, labels = data
            # print(premise)
            # print(hypothesis)
            # print(labels)
            
            # zero the parameter gradients
            opt.zero_grad()
            
            # forward pass
            # (batch_size, 3)
            outputs = model(premise, hypothesis)
            
            outputs_list = outputs.tolist()
            y_pred += np.argmax(outputs_list, axis=1).tolist()
            y_true += labels.tolist()
            
            assert len(y_pred) == len(y_true)
            
            # print(outputs)
            
            # calculate loss
            loss = cross_entropy_loss(outputs, labels)
            loss.backward()
            opt.step()
            # print(loss.item())
                
            running_loss += loss.item()
            
            
            if i % 1000 == 0:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i, running_loss / 1000))
                print(classification_report(y_true, y_pred, target_names=['neutral', 'contradiction', 'entailment']))
                running_loss = 0.0
                
        print("Running classification report on dev set")
        y_pred = []
        y_true = []
        model.eval()
        with torch.no_grad():
            for data in dev_data_loader:
                
                premise, hypothesis, labels = data
                outputs = model(premise, hypothesis)
                
                outputs_list = outputs.tolist()
                y_pred += np.argmax(outputs_list, axis=1).tolist()
                y_true += labels.tolist()
                
                
                
            print(classification_report(y_true, y_pred, target_names=['neutral', 'contradiction', 'entailment']))
          
                
    print("Finished Training!")
                
if __name__ == '__main__':
    train()