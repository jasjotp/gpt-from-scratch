"""
This GPT-like (decoder-only) model is implemented from scratch for learning purposes.

The source code and concepts are based on Andrej Karpathy's video:
"Let's build GPT: from scratch, in code, spelled out" from: https://www.youtube.com/watch?v=kCc8FmEb1nY

This project is meant to help me understand how LLMs work under the hood. The text corpus I used is a compilation of Jensen Huang's interviews from the YouTube Transcript API. 

Our goal is to model how the characters in jensen_huang.txt follow each other. Ex) Given a chunk of the characters from the transcript, given some context of characters in the past, the transformer neural network will look at the characters and predict what is likely to come next. We will train the model on Jensen Huang's transcripts to try and produce character sequences that look like the characters in jensen_huang.txt
"""
import torch 

# hyperparameters 
batch_size = 64 # how many independent sequences will we process in paralell, every forward/backward pass of the transformer
block_size = 64 # the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 128
n_head = 4
n_layer = 4
dropout = 0.2
# -------------------------------------------------------

torch.manual_seed(1337) # set the seed, as we will start sampling random locations in the dataset to pull chunks from, so that the numbers we see are reproducible 

# read the Jensen Huang text into inpt to inspect it 
with open('jensen_huang.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

# exploring the data 
print(f"Length of dataset in characters: {len(text)}")
print(text[:1000])

# find all the unique characters that occur in the text (text is a sequence of chars in Python) - which are all of the possible characters that our model can see/emit
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Unique Characters in our dataset: {''.join(chars)}")
print(f"Vocab Size: {vocab_size}")

'''
Tokenization: to tokenize the input text we send to our model 
 - tokenization means to convert the raw text (from a string format) into some sequence of integers, according to some vocabulary of possible elements 
 - since we are building a character level language model, we will be translating individual characters into integers
'''

# create a mapping from characters to integers 
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: takes a string, outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: takes a list of integers, outputs a string

str_to_encode_decode = "Hi there"
print(f'Encoded {str_to_encode_decode}: {encode(str_to_encode_decode)}')
print(f'Decoded {str_to_encode_decode}: {decode(encode(str_to_encode_decode))}')

'''
OpenAI has a library called tiktoken that uses a Byte-Pair Encoding tokeniser for use with OpenAI's models
we are using a character level tokenize, where each char is a token, and is encoded for simplicity, so we have very small codebooks but we hve very long integer sequences for each token for simplicity 
'''
# encode the entire text dataset and store it into a torch.Tensor 
data = torch.tensor(encode(text), dtype = torch.long)
print(f'Tensor shape: {data.shape}, Tensor dtype: {data.dtype}')
print(f'Tensor of first 1000 chars: \n{data[:1000]}') # this sequence of integers maps to each of the first 1,000 chars above ex) the 0th index in chars maps to a space 

# split data up into a train and test/validation set 
n = int(0.9 * len(data)) # train on 90% of our data 
train_data = data[:n]
val_data = data[n:] # validate on the last 10% of our data, wich will help us understand how much our model is overfitting - we don't want a perfect memorization of jensen_huang.txt, we want a neural network that creates Jensen Huang like-text 

print(f"Length of Training Data: {len(train_data)}")
print(f'Length of Validation Data: {len(val_data)}')

'''
Data Loading: Plug our text/integer sequences into the transformer so that our netural network can train/learn those Jensen Huang-like patterns.
We do not feed the entire text into the transformer all at once, as that would be very computaitonally expensive 
When we train the traansformer, we typically only work with chunks of the dataset. When we train the transformer, we sample random little chunks out of the training set and train our neural network on these chunks at a time.
These chunks of our training data that we train on typically have a max capacity/length (typically called block size or context length)
The firsst integer sequence in the first block for example, has multiple examples packed into it as all of these characters follow each other. The tensor of the first block size's sequence will be trained to make a prediction at every one of the positions in the tensor. 
We use block size + 1 to train the model, as in a chunk of 9 characters, there are 8 individual examples packed into there. 
ex) tensor([51, 37, 73, 71, 61, 55, 52,  0, 61]) -> the model can learn from 8 examples ex) In the context of 51, 17 likely comes next, in the context of 51 AND 37, 73 comes next, in the context of 51 AND 37 AND 73, 71 comes next. 
We train on all 8 examples above, with the context size between 1, all the way up to block size. We train on the context size of 1 to all the way up to block size for computational reasons as we have the previous sequence already, and ***to make sure that the transformer network is used to seeing contexts from all the way as little as 1 all the way to block size. This lets the transformer be used to seeing all context sizes between 1 and block size, which will be useful during inference, as while we are sampling, we can start a sampling generation with as little as 1 character of context, and the transformer knows how to predict the next character with only 1 character of context. So the transformer can predict the characters on a context window of up to block size.
After block size, we have to start truncating, as the transformer will never recieve more than block size inputs when it is predicting the next character. 
'''
print({f'First Block Size + 1 Chars: {train_data[:block_size + 1]}'}) # we use block size + 1, as in a chunk of 0 characters, there are 8 individual examples packed into there

# We use block size + 1 to train the model, as in a chunk of 9 characters, there are 8 individual examples packed into there. 
x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f'When input is {context}, the target is {target}')

'''
As we sample these above chunks of text, everytime we feed them into a transformer, we'll have mini batches of multiple chunks of text that are all stacked up in a single tensor for efficiency. 
The below code helps batch together the chunks of data that we will send to our transformer. 
'''

# function to get the batch for any arbitrary split 
def get_batch(split):
    # generate a small batch of data of inputs x and targets y 
    data = train_data if split == 'train' else val_data 
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 4 numbers that are randomly generated between 0 and len(data) - block_size, to get random offsets into the data
    x = torch.stack([data[i:i+block_size] for i in ix]) # x is a stack of 4 sequences, all becomes 4 rows of 8 columns (one row with each tensor)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # offset by 1 of x 
    x, y = x.to(device), y.to(device)
    return x, y 

# example output of the above get_batch function, we return a (4, 8) tensor, with each row being a tensor from a specific index above 
xb, yb = get_batch('train') # the tensor for x feeds into the transformer and the transformer simaltaneoulsy processes all the examples in x and look up the correct integers to predict in every one of these positions in the tensor y
print(f'Inputs for xb: {xb.shape}')
print(f'Data for xb: {xb}') # each row is a chunk of the training set 
print(f'Inputs for yb: {yb.shape}') # targets come in to the transformer all the way at the end, when creating the loss function, to give us the correct answer for every single position inside x 
print(f'Data for yb: {yb}')

print('__________')
for b in range(batch_size): # batch dimension 
    for t in range(block_size): # time dimension
        context =  xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is: {context.tolist()}, the target: {target}")

# now that we have our batch of input, we feed it into the transformer (xb), the below section feeds the batches into our neural network
# we will start with the simplest neural network when it comes to languague modelling (the bigram language model)
import torch.nn as nn 
from torch.nn import functional as F 
torch.manual_seed(1337)

# function that estimates/averages the loss over multuple batches for the train and val split 
@torch.no_grad() # context manager that tells PyTorch that everything that happens in this function, we will not call .backward on, so PyTorch can be much more effcieint with its memory, as it doesn't have to store all the intermediate variables, as we do not intend to do backpropagation
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# class definition of an attention head - one head of self-attention 
class Head(nn.Module):
    '''
    one head of self-attention
    '''
    def __init__(self, head_size): 
        super().__init__()
        # define the key, query, and value linear layers. (THe linear projections we will apply to all of our nodes)
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # create a trill variable - tril is not a paramter of the module, in PyTorch, this is a buffer 
        self.dropout = nn.Dropout(dropout)

    # forward pass with single-head self-attention
    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores (affinities for each token) - using scaled attention by normalizing using C**-0.5
        weights = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) - makes the transformer a decoder block, so each token only attends to itself and tokens before it. Masking where it is a 0 ti -inf, makes it so when we apply softmax, the probabiltiies of the future tokens = 0, so the model won't pay attention to the future
        weights = F.softmax(weights, dim = -1) # (B, T, T)
        weights = self.dropout(weights) # can randomly add dropout to randomly prevent some of the nodes from communicating to try and prevent overfitting
        # perform the weighted aggregation of the values (multiply each vector for each token above with the values vector)
        v = self.value(x) # (B, T, C)
        out = weights @ v 
        return out 

# class definitiion for Multi-head attention in paralell
class MultiHeadAttention(nn.Module):
    '''
    multiple heads of self-attention in paralell
    '''
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) # concatenate over the channel dimension
        out = self.proj(out) # a linear transformation of the outcome of the above layer
        return out

# definition for the Feed Forward part of the transformer 
class FeedForward(nn.Module):
    '''
    a simple linear layer followed by a non-linearity
    we add computation by multiplying the inner layers of the FFN by 4, and growing that Linear layer that is in the residual block on the side of the residual pathway
    '''
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # the inner layer of the Feed Forward Network should be multiplied by 4 in terms of channel sizes, according to the Attention is all you need Paper
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout), # can add dropout right before the connection back into the residual pathway
        )
    
    def forward(self, x): # this is computed on a per token basis 
        return self.net(x)
    
# definition for a transformer Block
class Block(nn.Module):
    '''
    Transformer block: communication followed by computation.
    '''
    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_headL number of heads we want
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # add residual connections and layer norm to the MultiHeadAttention and the FFN so we keep the original token info and can add in the new learned info from attention or our transformer
        x = x + self.ffwd(self.ln2(x)) # add and layer norm part of Add & Norm in Transformer model
        return x

# definition of the Layer Normalization of Add & Norm
class LayerNorm1d:
    '''
    Layer Norm helps stabilize and speed up training by normalizing each input vector for each token, so that the mean is 0 and the variance is 1, and then scales and shifts it using trainable parameters (gamma and beta)
    Layer Norm is computed across the input vectors of a single token (so BatchNorm is great for where we care about each token independently) 
    params: 
        dim: number of dimensions in your token embeddings 
        eps: small number to prevent dividing by 0 
        momentum: how fast the running mean and variance are updatedd 
    '''
    def __init__(self, dim, eps = 1e-5):
        self.eps = eps 
        # learnable paramaeters in Layer Normalization (trained wtih backprop)
        self.gamma = torch.ones(dim) # for scaling (multiplies the normalized values (xhat) so the model learns how much to scale each feature)
        self.beta = torch.zeros(dim) # for shifting: beta lets the model learn how much to shift each feature
    
    def __call__(self, x):
        # calculate the forward pass 
        xmean = x.mean(1, keepdim = True) # token wise mean
        xvar = x.var(1, keepdim = True) # token wise variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance 
        self.out = self.gamma * xhat + self.beta
        return self.out 
    
    def parameters(self):
        return [self.gamma, self.beta]

# class definition of a Bigram Language Model 
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # size of vocab size x number of embedding dimensions to create vectors that are trainable, to give each token a vector that captrues its meaning in context
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # positiion embedding table that tells the model where each token is in the sequence, since order matters in transformers - for each of our 8 positions in block_size, set a learned vector of 32 floats (8, 32)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # there is typically a LayerNorm at the end of the transformer before the linear layer 
        
        #self.sa_heads = MultiHeadAttention(4, n_embed // 4) # creating a multi-head attention model - 4 heads of 8-dimensional self-attention
        #self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) # to go from token embeddings to the logits, we need a linear layer right before the final linear layer that decodes into vocabulary
    def forward(self, idx, targets = None):
        B, T = idx.shape         
        # idx and targets are both (B, T) tensor of integers (Batch, Time)
        token_embeddings = self.token_embedding_table(idx) # (B - Batch, T - Time, C - Channel = n_embed) formatted tensor, batch is 4, time is 8, and channels are n_embed/32 - Passing idx in the token embedding table means every single integer in our input (xb), is going to refer to our embedding table and is going to pluck out a row of that embedding table that corresponds to its index ex) 24 will go to the embedding table and pluck out the 24th row 
    
        pos_embeddings = self.position_embedding_table(torch.arange(T, device = device)) # (T, C) - gets right aligned, a new dimension of 1 gets added, and it gets broadcasted along all batches - all those integers from 0 to T - 1 get embedded through the table to create (T, C)
        x = token_embeddings + pos_embeddings # (B, T, C) - now x also holds the positional identities 
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        #x = self.sa_heads(x) # apply one head of self-attention (B, T, C)
        #x = self.ffwd(x) # (B, T, C), after we self attend- use the feed forward network 
        logits = self.lm_head(x) # (B, T, C = vocab_size)

        # reshape our logits to match the format of cross entropy needed in PyTorch (Logits: N, C - where N is number of examples and C  is the vocab size, targets: N. where N is the number of examples)
        if targets is None: 
            loss = None
        else: 
            B, T, C = logits.shape 
            logits = logits.view(B * T, C) # strech out the array so it is 2 dimensional: 32 rows of 83 columns each (83 is vocab size), so it is better conformed to what cross_entropy in PyTorch expects 
            targets = targets.view(B * T)

            # the correct index containing the correct character for each logit should ideally have a very high number, and all the other dimensions should be very low
            # evaluate the loss function using PyTorch's Cross Entropy (uses negative log likliehood & softmax to score the model on how it is predicting so we can use it to improve our neural network 
            loss = F.cross_entropy(logits, targets) # since we have the targets / the identity of the next character, the loss tells us how well we are predicting the next character based on the logits

        # PyTorch arranges the loookup values from the token embedding table as a Batch (4) by Time (8) by Channel (Vacab Size or 65) tensor, so we interpret these B by T by C values as our logits, which are basically the scores for the next characters in the seuqnnce. 
        return logits, loss # The logits help us predict/score the next character in the sequence, as they help us predict what is coming next, with a single character 

    # after evaluating the quality of the model on some data, we want to also generate from the model
    def generate(self, idx, max_new_tokens): 
        # idx is (B - Batch (Batch Size = 4), T - Time (Block size = 8))
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens so we never pass in more than block size elements
            idx_cond = idx[:, -block_size:]
            # get the predictions (logits, loss) from each character sequence 
            logits, loss = self(idx_cond)
            # focus only on the last time step / token, as we already have the first 8 tokens as context, and we only want the models prediction for the very next token 
            logits = logits[:, -1, :] # becomes (B, C) - ex) 4 seq. x 83 chars
            # apply softmax to get the probabiltiies of the next character from the logits 
            probs = F.softmax(logits, dim = -1) # (B, C) - ex) 4 seq. x 83 chars 
            # sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples = 1) # shape: (B, 1) torch.multinomial samples from the above probabiltieis and ask PyTorch to give us one sample at random (higher probability tokens are still more likely but lower probability tokens can still be picked)
            # append the sampled index to the running sequence 
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1) - concatenate the prediction with the current idx
        return idx 

# make predictions about what characters come next from our batch 
model = BigramLanguageModel()
m = model.to(device)

logits, loss = model(xb, yb)
print(f'Bigram Model Shape: {logits.shape}')
print(f'Bigram Model Output: {logits}')

# evaluate our loss *in negative log liliehood, since we knowe we have 83 possible characters, we can guess what the loss should be by doing -ln(1/83) ~= 4.42, we are currently getting 4.67, so the initial predictions are diffused and we are guessing quite wrong
print(f'Loss: {loss}')

# test the generate function above - (B. T is 1 x 1, as a 0 to kick off the generation)
context = torch.zeros((1, 1), dtype = torch.long, device = device)
print(f'Output of Bigram model function (initial): {decode(model.generate(context, max_new_tokens = 100000)[0].tolist())}')

'''
Next, we want to train the above model so it gives us actual output that makes snese when generating, tha is less random/
'''
# create a PyTorch optimization object with the Adamw optimizer with learning rate: 1e-3. The typical learning rate is 3e - 4 but for our smaller network, we can use a higher learning rate of 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate) # takes the gradients and updates the models parameters 

for iter in range(max_iters):
    # every once in a while, evaluate the loss on train and validation sets 
    if iter % eval_interval == 0: 
        losses = estimate_loss()
        print(f"Step {iter}: train loss - {losses['train']:.4f}, val loss - {losses['val']:.4f}")

    # sample a batch of data 
    xb, yb = get_batch('train')

    # evaluate the loss 
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True) # clear .grad before the next batch 
    loss.backward() # loss.backward calculates gradients: slopes that tell you how much changing each paramater will change the loss, gives each paramter in model.parameters a .grad attribute
    optimizer.step() # takes the gradients computed by loss.backward() and uses them to update the model's weights 

print(f'Loss after training loop (of final minibatch) ({iter}): {loss.item()}')

# save the output text of the model after optimizing the loss and adding LayerNorm, Residual Connections and Dropout
output_text = decode(model.generate(context, max_new_tokens = 100000)[0].tolist())

with open('output.txt', 'w', encoding = 'utf-8') as f:
    f.write(output_text)
print('Output saved to output.txt')

print(f'Output of Bigram model (after optimizing): {decode(model.generate(context, max_new_tokens = 100000)[0].tolist())}')

'''
the above Bigram model only looks at the previous token to make the prediction about what comes next, Now, our goal is to get these tokens to talk to each other and figuring out what is in the context so that theey can make better predictions for what characters come next 
'''
torch.manual_seed(1337)
B, T, C = 4, 8, 2 # batch, time, channels 
x = torch.randn(B, T, C)
print(x.shape)

'''
For the above time dimension, the 8 tokens above are not talking to each other, and we want to couple them so they can talk to each other. 
ex) The token at position 5 should learn from / talk to the tokens at position 1 to 4 and should NOT communicate with tokens in the 6th, 7th, and 8th location.
    Therefore, information only flows from the previous context to the current timestep, and we cannot get any information from a future timestep.
    The simplest way for a token to see the inputs/integer values of the previous tokens is to do an average of all the preceding elements. 
        ex) If you are the 5th token, you we want to take the logits/channels at the 5th step, and also use the channels from the 1st, 2nd, 3rd, and 4th step and average those up, which gives us a feature vector for time step 5 that gives us context of the previous 4 channels, including the current one
    Just doing an avergae is a weak form of interaction, like the communication between tokens is extremely lossy, as we lose the spatial arrangements / ordering context of the previous tokens if we take the average. The averaged vector does not know what token/character caame first, second, or third so the model loses the ordering context.

'''
# we want x[b, t] = mean_{i <= t} x[b, i]
# we are doing a bag of words computation, as we are averaging the logits/tokens of each timestep up until, and including that timestep
xbow = torch.zeros((B, T, C)) # initize the bag of words at (4, 8, 2)
# go through each batch and timestep in each batch and get the previous tokens (every channel from start to current token)
for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1] # (t, C)
        xbow[b, t] = torch.mean(xprev, 0) # (1, C)
print(f"X (at 0): {x[0]}, \n\nxbow (at 0): {xbow[0]}")
print(f"xbow (last value at 0 - avg of all elements): {xbow[0][-1][:]}")

# the above is very inefficient, we can be more efficient with the above by using matrix multiplication like below, where we do a @ b 
'''
Neat ltitle trick below where we can use matrix multiplicaiton to get the average for each timestep, up until and including that timestep. 
If you multiply the below matrix a with any given matrix b, we get the average of each row, including that row. Mulipying any matrix b with the matrix a below will give us the 
    cumulative averages for each timestep, in an incrementral fashion, instead of our above double for loop appraoch which is very inefficient. 

a: tensor([[1.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000],
        [0.3333, 0.3333, 0.3333]])
'''
triangular_ones = torch.tril(torch.ones(3, 3)) # returns only the lower triangular portion as ones
torch.manual_seed(1337)
a = triangular_ones
a = a / torch.sum(a, 1, keepdim = True) # using a / the sum of that row and then using matrix multiplication gives us the cumulative average for each time step and channel in each time step
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b 
print(f"a = \n{a}")
print('-----')
print(f"b = \n{b}")
print('-----')
print(f"c = \n{c}")

# use the above matrix multiplication method on our example above (version 2)
weights = torch.tril(torch.ones(T, T)) # matrix A in the above example
weights = weights / weights.sum(1, keepdim = True) 
xbow2 = weights @ x # our b in this example is x from above - weights: (B, T, T) @ (B, T, C) - computation happens for each batch in paralell 
# confirm that xbow 2 and xbow from previous manual double for loop example match for their values 
print(f"Do xbow and xbow2 match: {torch.allclose(xbow, xbow2)}")

# version 3: use Softmax 
'''
We use the masked fill method below in self-attention later. 
The weights begin with an T x T matrix of 0's - which tells you the interaction strength, or basically how many of each token from the past do we want to aggregate. 
The weights then are assigned to a masked fill, where the weights of 0's, are replaced by -inf, where trill = 0. This makes each row in the tensor have a sequential number of ones for each following row. This line is saying that tokens from the past cannot communicate, by setting them to -infinity, we are saying that we won't aggregate anything from those tokens that are -inf
Then the weights go through Softmax. Since Softmax takes each value in each row, takes the exponential of it and dividies it by the sum of the row, to get the resulting weights matrix (that is the same as the weight matrix above obtained from: weights = weights / weights.sum(1, keepdim = True) 
'''
tril = torch.tril(torch.ones(T, T))
weights = torch.zeros((T, T))
weights = weights.masked_fill(tril == 0, float('-inf')) # same way to produce matrix a above to matrix multiply with, same as doing: a / torch.sum(a, 1, keepdim = True)
weights = F.softmax(weights, dim = -1) # take softamx along every row 
xbow3 = weights @ x 
print(f"Do xbow and xbow3 match: {torch.allclose(xbow, xbow3)}")

'''
The weights above are set by us to be a T by T matrix of 0's. But we want the affinities between the tokens to be data dependnet, so the tokens look at each other - this allows some tokens to find other tokens more or less interesting. Depending on the token's values, they're going going to find each interesting to differnent amounts (affinities). 
Then we will clamp the tokens, so the future tokens cannot communicate with the past tokens just like we did above. 
Then when we use softmax to normalize and sum the weights, we are going to aggregate their values depending on how interesting they find each other. 
This is the basis of the self-attention method for LLMs, which is spelled out below.
    The code we had before does a simple average of all the past tokens, including the current token. 
Self-Attention makes each token data-dependent on each other, or learn from each other.
    Self-Attention gathers information/context from the past tokens, in a data-dependent way, where the model can gauge the relevance of past tokens depending on the current input. 
    ***Ultimately, Self-Attention helps the tokens talk to each other and decide what past tokens to specifically focus on, and what tokens to not focus on.

    ----------------
    Way that Self-Attention decides what past tokens to specifically focus on: 
    ----------------
    Every single node or token at each position emits 2 vectors: a Q (query) vector and a K (key) vector. THe value vector comes a little after the self-attenstion scores are computed.
    The query (Q) vector basically asks: What information am I looking for (depending on the current token you are at)
    The key (K) vector basically asks: What information do I contain?  
    We get the affinitites/attention scores between each token is to do a dot product between the Query vector and the Key vector for each token, up until the current token, to get attention scores for all tokens (including the current token)
    That dot product for each token's Q and K vectors becomes the weights vector for that token.
    If the K and Q are aligned, they will interac to a very high amount, and you will get to learn more about that specific token.
    The V (value vector) represents the vector that we aggregate instead of our raw data (x which is private information to the token). V tells us
        ex) IF you are at the 5th token, your key/identity you emit is that you are a consonant at index 4. The information for token 5 is kept in vector x. For single head attention, this tokem emits Q: Query (vector representing what this token is interested in/looking for), Key (vector representing what this token contains), and Value (vector that if the token finds the token interesting, emits what it will communicate to you). 
'''
# ***version 4: self-attention! (single-head self-attention)
torch.manual_seed(1337)
B, T, C = 4, 8, 32 # batch, time, channels
x = torch.randn(B, T, C)

# single head performing self-attention 
head_size = 32

# Linear layers for Keys that takes in vector of size C and outputs a vector of size head_size
key = nn.Linear(C, head_size, bias = False) 
query = nn.Linear(C, head_size, bias = False)
value = nn.Linear(C, head_size, bias = False)
k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)

# all the queries dot product with the keys to get the raw self-attention scores/affinities
weights = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ----> (B, T, T)

# add & norm is applied after the multi-head atention transformation in the Attention in All you Need Paper, but now it is a bit more common to apply LayerNorm before the multi-head transformation, so there is a normalization/re-shuffling of the layer norms. We will slightly deviate from the paper and implement the pre-norm formulation
tril = torch.tril(torch.ones(T, T))
weights = weights.masked_fill(tril == 0, float('-inf')) # *this line is key as it fills the triangular lateral portion of 0s, sequentially with -inf, so each token only focuses on the tokens before or at its current position. Using the upper triangular masking to fill the 0s with -inf, so the tokens past the current token's position are not allowed to communicate with the previous tokens
weights = F.softmax(weights, dim = -1) # take softamx along every row 
v = value(x)
out = weights @ v
print(f'Weights output with Single-Head Attention: {weights}') # now every single batch element will have different weights, as every single batch element contains different tokens at different positions, so this is now data depedendent. These weights are now telling us in a data dependent manner, how much information to aggregate from any of these tokens in the past, aka, the weights are telling us how much to focus on each token in the past or at the ucrrent token. 

'''
ex) in the last row of weights[0]: [0.0210, 0.0843, 0.0555, 0.2297, 0.0573, 0.0709, 0.2423, 0.2391]]
    The 8th token: 0.2391 for example, knows what content it has and knows what position its in, and based on that creates a query for what it is looking for. 
    The 8th token, based on that, for example, could emit that it is a vowel (content/key), its in the 8th position, and is looking for any consonants at positions up to 4 (query). Then, all of the tokens emit keys, and maybe one of these keys's channels could represent that this token is a consonant and is in a position up to 4. That key would have a high number in that specific channel. This lets the query and the key find each other when they dot produuct. There would be higher values in token 8's query vector for a specific channel that represent that this token 8 is looking for tokens that is a consonant in a position less than or equal to 4. Token 2's keys, because its a consonant and at postiion 2, also has a high value at that same chnnel. When you do the dot product between query and key, their high values in the same channel multiply and contribute a large number. So token 2 gets a high attention score from token 8. THis happens across all other tokens and the softmax then picks the most relevant tokens to focus on, by turning the scores into probabilities. 
        Ultimately, in the above example, token 8 attends more to token 2 (and others like it), as it uses their values/learns more from token 2 in the final output. 

Notes on Attention: 
- 1. Attention is a communcation mechanism:
    - Attention is a communication mechanism where you have a number of nodes in a directed graph, where you have edges pointing between each node. Each node has some vector of information, and it gets to aggreagete information via a weighted sum from all of the nodes that point to it. This is done in a data dependent manner, so depending on however data is actually sorted in a point in time. 
    - In our example, we have 8 nodes/tokens. The first node is only pointed to by itself. The second node is pointed to by the first node and itself. The 8th node is pointed to by ALL the previous nodes and itself. This is typically the structure in autoregressive modelling like language modelling. 

- 2. Attention has no notion of space, it operates over sets. 
    - Attention has no notion of space. Attention simply acts as a set of vectors over space in this graph. By default, these nodes have no idea where they are positioned in the space. That's why we need to encode them positionally and give them some information that is anchored to a specific condition, so that each node knows where they are in space. 
    - Attention is just a set of vectors out in space. Each vector for each node communicates with each other, but if you want them to have a notion of space, you need to specifically add it (which is what we did above with the positional encodings so each token knows its position)

- 3. The elements across the batch dimensions never talk to each other and are processed completlely independently. In our example above, we had 4 seperate batches/pools of 8 nodes, and those 8 nodes only talked to each other.
     The tokens in one batch only communicate with the tokens in that batch, and not other batches.

- 4. In languague modelling, we have a speicific structure of a directed graph where future tokens will not communicate with past tokens. 
    This does not have to be the general case though. In many cases, all the tokens/nodes can talk to each other fully, regardless of if you are at token n and and want to be looked at by token n + 1. ex) If you are doing sentiment analysis with a transformer, you might have a number of tokens, and you might want to have them talk to each other fully, as you want to predict the entire sentiment of the sentence. This can be done by creating an 'encoder' attention block. 
    An encoder attention block just deletes the single line of code above that does masking with tril, allowing all the tokens to talk to each other. 
    A decoder attention block is what we are implementing above with the masking with trill (replacing 0s with -inf). It is called a decoder because it is sort of decoding language, and has an autoregressive format, where you have use a triangular matrix so nodes from the future never talk to nodes of the past, because this would give away the answer. 

    TLDR: 
    - Encoder blocks delete: weights = weights.masked_fill(tril == 0, float('-inf')) - allowing all the nodes/tokens to talk to each other. 
    - Decoder blocks keep: weights = weights.masked_fill(tril == 0, float('-inf')) - so you have a trinagular structure, allowing no future nodes to talk to the current node. 

- 5. In attention, there is also cross-attention. 
    Self-Attention: The keys, querys, and the value vectors all come from the same source (from x - which are our token embeddings of our tokens in each batch). The same source (x) produces keys, querys and values, so these tokens are self attending, as they look at their own token's key, query, and value to produce attention scores. 
       
    Cross-Attention: is used in the example below, when there is a seperate source of nodes we'd like to pull information from into our nodes. Vs. Self Attention: When we have nodes that would jsust like to look at each other and talk to each other, using the information (querys, keys, values) from the same source (x)
        In encoder, deocder transformers, you can have a case where the queries are produced from x (the encoded representation of our tokens). But the keys and the values come from a different external seperate source (sometimes from the encoder blocks that encode some contenxt that we would like to condition on). The keys and the values actually come from a whole seperate source. x just produces query vectors for each token and reads the key and values from the side source. 

- 6. In The Scaled Dot-Product Attentioon formula: Attention (Q, K, V), we also divide the softmax(Q.K^T)V vector with 1 / sqrt(head size). 
    Scaled Attention addtiionally divides the weights by 1 / sqrt(head_size). So that wehen the input QK are unit variance, weights will be unit variance too, and Softmax will stay diffused and not saturate too much.
    The weights feed into softmax, so weights should be evenly diffused. If weights has very postiive and negative numbers, softmax will converge towards one hot vectors. The softmax values will sharpen towards the maximum number. This results in the values being too extreme, and you are then basically aggregating information from a single node, which is not what we want. So the scaling is used to control the variance when initializing the weights.    
'''

'''
Multi-head Attention: Applying multiple attentions in paralell and concatenating their results
'''



