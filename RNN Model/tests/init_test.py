import os
import sys
import torch
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model as M

def testCellInit():

    cell = M.RNNCell(4, 10)

    numLinear = 0
    act = 0
    for module in cell.modules():
        if isinstance(module, torch.nn.Linear):
            numLinear += 1
        else:
            act += 1

    assert numLinear == 2, "You need to create two linear layers for the cell "\
                            "(one for hidden one for input)"
    assert act, "You need an activation function for the cell"

def testModelInit():

    model = M.RNNModel(10, 4, 20, 2)
    
    foundLinear = 0
    hasDecoderRight = False
    hasEncoderRight = False
    hasNLayers = False

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            if module.in_features == 20 and module.out_features == 10:
                hasDecoderRight = True
            else:
                foundLinear += 1
        elif isinstance(module, torch.nn.Embedding):
            shape = list(module.weight.shape)
            if shape[0] == 10 and shape[1] == 4:
                hasEncoderRight = True

    if foundLinear == 2:
        hasNLayers = True

    assert hasDecoderRight, "Your decoder shape is incorrect"
    assert hasEncoderRight, "You embedding/encoder shape is incorrect"
    assert hasNLayers, "You don't have the correct number of hidden layers"
