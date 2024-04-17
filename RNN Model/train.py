import torch
import time
import sys

from src.Tokenizer import Tokenizer
from model import RNNModel

class DataGenerator:
    """ A class for batching text data 

    Attributes: 
        tokenizer (Tokenizer): An instance of a white-space tokenizer
        fnames (list[str]): A list of file names for generating batches
        batchSize (int): Size of batch
        maxSeqLength (int): Size of sequence length 
    """

    def __init__(self, tokenizer:Tokenizer, fnames:list[str], 
                 batchSize:int=40, maxSeqLength:int=35):

        self.tokenizer = tokenizer
        self.fnames = fnames
        self.batchSize = batchSize
        self.maxSeqLength = maxSeqLength

    def batches(self):
        """ Yields batchSize batches of maxSeqLength
        Note: Final batch yield can be before the end of the document if the
        shape doesn't work out
        Yields: 
            batch_data (torch.tensor): tokenized input of maxSeqLength
            batch_targets (torch.tensor): gold targets for tokenized input 
        """
        for fname in self.fnames:
            with open(fname, 'r') as f:

                batch_data = []
                batch_targets = []

                data = []
                targets = []

                sents = f.read()
                encoded = self.tokenizer.encode(sents)
                
                # Targets are offset by one, so skipping final token's
                # predictions
                for d, t in zip(encoded[:-1], encoded[1:]):
                    data.append(d)
                    targets.append(t)

                    if len(data) == self.maxSeqLength:
                        batch_data.append(data)
                        batch_targets.append(targets)
                        data = []
                        targets = []
                    if len(batch_data) == self.batchSize:
                        # Make tensors of the correct shape
                        batch_data = torch.as_tensor(batch_data).unsqueeze(-1)
                        batch_targets = torch.as_tensor(batch_targets).flatten()

                        yield batch_data, batch_targets
                        batch_data = []
                        batch_targets = []

class LMTrainer:
    """ A class for training a RNN language model 

    Attributes: 
        generator (DataGenerator): An instance of for batching DataGenerator
        tokenizer (Tokenizer): Tokenizer used for model
        model (RNNModel): An instance of an RNNModel
        epochs (int): Number of training epochs
        optimizer (torch.optim): A PyTorch optimizer
        criterion (torch.nn.CrossEntropyLoss): CrossEntropyLoss function
    """
    def __init__(self, fnames:list[str], tokenizer: Tokenizer,
                 model: RNNModel, epochs:int=20,
                 batchSize:int=40, maxSeqLength:int=35, 
                 lr=1e-1):

        self.generator = DataGenerator(tokenizer, fnames, batchSize, 
                                       maxSeqLength)
        self.tokenizer = tokenizer
        self.model = model
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def repackage_hidden(self, hidden: torch.tensor) -> torch.tensor:
        """ Wraps hidden states in new Tensors, to detach them from their
        history. 
        
        Args: 
            hidden (torch.tensor): a tensor from a hidden state
        """
        if isinstance(hidden, torch.Tensor):
            return hidden.detach()
        else:
            return tuple(repackage_hidden(h) for h in hidden)

    def train(self):
        """ Function to train model for some number of epochs. The recipe goes
        something like the following: 
            1. Loop over epochs. For each epoch, 
            2. Loop over the input and output pairs using self.generator
            3. Reset optimizer and repackage_hidden your hidden !!
            4. get predictions and new hidden states from your model 
            5. calculate the loss for the model's predictions
            6. calculate the gradient 
            7. update your weights
            8. repeat 2-8

        Note: it is helpful to have a value for the total loss over an epoch to
        compare as you go. If your loss isn't decreasing, mistakes have been
        made. Look at the eval function and the class notes if you need help! 
        """

        for epoch in range(self.epochs):
            print("Training:" + epoch)

            totalloss = 0
            hidden = None


            for data, targets in self.generator.batches():  #PARSING
                if hidden is None:
                    logits, hidden = self.model(data)
                else:
                    hidden = self.repackage_hidden(hidden) 
                    logits, hidden = self.model(data, hidden)
            

                self.optimizer.zero_grad()  #RESET OPTIMIZER 


                loss = self.criterion(logits.reshape(-1, self.model.vocabSize),
                                    targets)                
                
                totalloss += loss.item()

                totalloss.backward() #GRADIENT 
                self.optimizer.step() #UPDATES WEIGHTS 

            print(totalloss + '\n') 
                
                #self.model.forward()


            
    @torch.no_grad()
    def evaluate(self, fname:str, model: RNNModel) -> float:
        """ Calculates the perplexity for a document 
        
        Args: 
            fname (str): Eval file name
            model (RNNModel): A RNN model
        Returns:
            ppl (float): Perplexity assigned by the model
        """ 
        generator = DataGenerator(self.tokenizer, [fname], batchSize=1, 
                                  maxSeqLength = self.generator.maxSeqLength)
        model.eval()
        hidden = None
        losses = []
        for data, targets in generator.batches():
            if hidden is None:
                logits, hidden = self.model(data)
            else:
                hidden = self.repackage_hidden(hidden)
                logits, hidden = self.model(data, hidden)

            loss = self.criterion(logits.reshape(-1, self.model.vocabSize),
                                  targets)

            # Make base 2
            loss = loss/torch.log(torch.tensor(2.0))
            losses.append(loss.item())

        ppl = 2**(sum(losses)/len(losses))
        return ppl

def main(train=True):

    sys.stderr.write('Running on CPU...\n')

    vocab_file = './vocab.txt'
    fnames = ['train/train.txt']

    # Set tokenizer
    tokenizer = Tokenizer(vocab_file)
    vocab_size = len(tokenizer)

    # These are all hyperparameters, you should try a variety
    input_dim = 50
    hidden_dim = 50
    n_layers = 2
    n_epochs = 1
    batchSize=30
    maxSeqLength=40

    model_fname = f'saved_models/RNN_{input_dim}i_{hidden_dim}h'\
                    f'_{n_layers}l_{batchSize}b_{maxSeqLength}s'\
                    f'_{n_epochs}e.pt'

    # Create model
    model = RNNModel(vocab_size, input_dim, 
            hidden_dim, n_layers)

    if train:

        trainer = LMTrainer(fnames, tokenizer, 
                            model, n_epochs, batchSize, maxSeqLength)
        trainer.train()

        #Save model
        torch.save(model.state_dict(), model_fname)

    else:
        model.load_state_dict(torch.load(model_fname))
        fname = './eval/eval.txt'
        evaluater = LMTrainer([fname], tokenizer, 
                              model, batchSize=1)
        ppl = evaluater.evaluate(fname, model)
        print(model_fname, ppl)

if __name__ == '__main__':
    main()
