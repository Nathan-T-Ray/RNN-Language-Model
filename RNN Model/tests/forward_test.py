import os
import sys
import torch
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M

def testCellForward():

    model = M.RNNCell(4, 3)

    x = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=float)

    hidden = torch.tensor([1, 2, 3], dtype=float)

    inputU = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                          dtype=float)
    hiddenW = torch.tensor([[11, 12, 13], [15, 16, 17], 
                            [19, 20,21]],
                          dtype=float)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            if module.in_features == 4:
                module.weight = torch.nn.Parameter(inputU)
                module.bias = torch.nn.Parameter(torch.tensor(
                                    [1, 2, 3], dtype=float))
            else:
                module.weight = torch.nn.Parameter(hiddenW)
                module.bias = torch.nn.Parameter(torch.tensor(
                                    [1, 2, 3], dtype=float))
    
    assert newHidden.equal(torch.tensor([[76, 102, 128], [86, 128, 170]],dtype=float)), "Forward for RNNCell not working."

def testModelForward():

    model = M.RNNModel(10, 4, 3, 2)

    x = torch.tensor([[[0], [1], [2], [3], [0], [4]], 
                              [[0], [5], [2], [6], [7], [8]]], dtype=int)
                              
    batch_size, seqlen, output_size = output.shape
    assert batch_size == 2 and seqlen == 6 and output_size == 10
