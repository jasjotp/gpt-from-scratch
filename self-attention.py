# self attention toy example 
import torch 
from torch.nn import functional as F 

torch.manual_seed(1337)
B, T, C = 4, 8, 2 # batch, time, channels 
x = torch.randn(B, T, C)
print(x.shape)

'''
For the above time dimension, the 8 tokens above are not talking to each other, and we want to couple them so they can talk to each other. 
ex) The token at position 5 should learn from / talk to the tokens at position 1 to 4 and should NOT communicate with tokens in the 6th, 7th, and 8th location.
    Therefore, information only flows from the previous context to the current timestep, and we cannot get any information from a future timestep.
    
'''
