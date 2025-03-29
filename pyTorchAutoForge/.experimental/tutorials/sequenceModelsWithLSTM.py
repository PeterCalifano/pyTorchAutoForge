'''
Script to demonstrate how to use LSTM for sequence modeling task using PyTorch.
Created by PeterC on 2024-07-19, from PyTorch tutorial: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
'''

# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as torchFcn
import torch.optim as optim

torch.manual_seed(1)

# %% ----------------------------------- LSTM model example -----------------------------------
def RunLSTMmodelExample():
    # Define LSTM layer with
    lstm = nn.LSTM(input_size=3, hidden_size=3)  
    # SIGNATURE: torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
    #               dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)
    # NOTE: Input size is the size of the input vector at one instant of the sequence
    #       Hidden size is the size of the hidden state vector. 
    #       LSTM module can directly stack multiple layers one after the other, by specifying the third input
    # ACHTUNG: LSTM layer does not use batch axis as first axis by default, but as second: (sequence, batch, feature) 
    # in constrast with all other modules.

    # Generate random input sequence of 5 elements, with 3 features each
    inputs = [torch.randn(1, 3)
              for _ in range(5)]  # make a sequence of length 5


    # Initialize the hidden states, randomly --> why is this of size 1x1x3?
    hidden = (torch.randn(1, 1, 3),
              torch.randn(1, 1, 3))
    
    hidden_original = hidden
    # METHOD 1: Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    print('\nMethod 1: Step through the sequence one element at a time')
    for i in inputs:
        out, hidden = lstm(i.view(1, 1, -1), hidden) 
        print(out)

    # METHOD 2: Single call, whole sequence
    print('\n\nMethod 2: Single call, whole sequence')
    # Alternatively, we can do the entire sequence all at once by applying lstm(inputSequence)
    # the first value returned by LSTM is all of the hidden states throughout
    # the sequence. The second is just the most recent hidden state

    # The reason for this is that:
    # "out" will give you access to all hidden states in the sequence
    # "hidden" will allow you to continue the sequence and backpropagate, by passing it as an argument  to the lstm at a later time

    # Combine the input sequence into a single tensor of the correct size to process it all at once
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    hidden = hidden_original  # Reset the hidden state to original state
    out, hidden = lstm(inputs, hidden)
    print(out)
    print(hidden)
    # NOTE that given the same initial state, the output is the same in both methods.

# %% Example LSTM model for Part-of-Speech Tagging
class LSTMTagger(nn.Module):
    # NOTE: the trained LSTM is a classifier returning a 1x3 vector with softmax probabilities
    # for each tag, predicting which of the three classes of tags is the most likely for each word.
    # Note that the lowest absolute value (negative) is the prediction (given by the index of the tag)
    # This is because the loss function is a log softmax.
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # Create embedding module to store word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space: linear readout layer to map
        # the output of the LSTM (in the embeddings space to the tag space, namely the output space)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        # Construct embedding from input sentence
        embeds = self.word_embeddings(sentence)
        # Pass the embeddings through the LSTM layer to get the output
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # Pass the output of the LSTM through the linear layer to get the tag scores
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = torchFcn.log_softmax(tag_space, dim=1)
        return tag_scores


def RunSpeechTaggingWithLSTM():
    print("Running LSTM model example")
    
    def prepare_sequence(seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    # Create training data for Part-of-Speech Tagging
    training_data = [
        # Tags are: DET - determiner; NN - noun; V - verb
        # For example, the word "The" is a determiner
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]

    word_to_ix = {}
    
    # For each words-list (sentence) and tags-list in each tuple of training_data
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:  # word has not been assigned an index yet
                # Assign each word with a unique index
                word_to_ix[word] = len(word_to_ix)
    print(word_to_ix)
    
    tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index

    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6

    # Define model
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # See what the scores are before training
    # Note that element i,j of the output is the score for tag j for word i.
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    # NOTE: construct training labels
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        print(tag_scores)

    # Training loop
    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = model(inputs)
        

        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        print(tag_scores)

if __name__ == "__main__":
    print('-------------------- TUTORIAL: Sequence Models and LSTM ----------------------')
    print('LSTM use example:')
    RunLSTMmodelExample()

    print('LSTM for Part-of-Speech Tagging example:')
    RunSpeechTaggingWithLSTM()
    print('-------------------- END ----------------------')



