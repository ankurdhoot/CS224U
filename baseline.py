from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import vocab
from dataset import SNLITrainDataset, SNLIDevDataset
import torch.optim as optim
from sklearn.metrics import classification_report
import numpy as np


LABEL_MAP = {'neutral' : 0, 'contradiction' : 1, 'entailment' : 2}
VOCAB = vocab.build_vocab()
# setting device on GPU if available, else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

class Baseline(nn.Module):
    def __init__(self, embed_size=100, hidden_size=100, projection_dim=200, dropout_rate=0.2):
        """
        Args:
            embed_size : int
                Size of input embeddings
            hidden_size : int
                Size of the LSTM hidden state
            projection_dim: int
                Size of the linear projection layers
            dropout_rate: float
                dropout probability for linear layers
        """
        super(Baseline, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.projection_dim=200
        self.vocab = VOCAB
        
        self.premise_rnn = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, bias=True, batch_first=True)
        self.hypothesis_rnn = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, bias=True, batch_first=True)
        
        self.proj1 = nn.Linear(in_features=2*self.hidden_size, out_features=self.projection_dim, bias=True)
        self.proj2 = nn.Linear(in_features=self.projection_dim, out_features=self.projection_dim, bias=True);
        self.proj3 = nn.Linear(in_features=self.projection_dim, out_features=3, bias=True)
        
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
        # TODO(ankur): Maybe set freeze=False
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(self.vocab.embedding, dtype=torch.float), padding_idx=self.vocab.pad_id)
        
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
        
        # Pass the distributed representations through the LSTMS.
        # (seq_len, batch, hidden_size) (1, batch_size, hidden_size) (1, batch_size, hidden_size)
        premise_output, (premise_h_n, premise_c_n) = self.premise_rnn(distributed_repr_premise)
        hypothesis_output, (hypothesis_h_n, hypothesis_c_n) = self.hypothesis_rnn(distributed_repr_hypothesis)
        
        # (batch_size, hidden_size)
        premise_h_n = torch.squeeze(premise_h_n, dim=0)
        hypothesis_h_n = torch.squeeze(hypothesis_h_n, dim=0)
        
        # (batch_size, 2 * hidden_size)
        concat = torch.cat((premise_h_n, hypothesis_h_n), dim=1)
        
        # (batch_size, self.projection_dim)
        proj1 = torch.tanh(self.proj1(concat))
        # (batch_size, self.projection_dim)
        proj2 = torch.tanh(self.proj2(proj1))
        # (batch_size, 3)
        proj3 = torch.tanh(self.proj3(proj2))
        
        return proj3
        
        
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
    model = Baseline()
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
    
    # Create the optimizer
    opt = optim.Adam(model.parameters(), lr=.0004)
    
    # Create the loss function
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    for epoch in range(100):
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
            # print(loss)
                
            running_loss += loss.item()
            
            
            if i % 2000 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i, running_loss / 2000))
                print(classification_report(y_true, y_pred, target_names=['neutral', 'contradiction', 'entailment']))
                running_loss = 0.0
                
        print("Saving model")
        path = "baseline/epoch_{}.pt".format(epoch)
        torch.save(model.state_dict(), path)
        print("Model saved")
                
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
            file = open("baseline_report.txt", "a")  # append mode
            file.write(classification_report(y_true, y_pred, target_names=['neutral', 'contradiction', 'entailment']))
            file.write("\n")
            file.close()
          
                
    print("Finished Training!")
                
if __name__ == '__main__':
    train()
    