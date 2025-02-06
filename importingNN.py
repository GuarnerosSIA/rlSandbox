import torch
import onnx2torch
import onnx

def importMatlabNN():
    path = "C:/Users/guarn/Dropbox/Alejandro/DoctoradoITESM/ReinforcementLearningTD3/actorRLcart.onnx"
    model_actor = onnx.load(path)
    actor = onnx2torch.convert(model_actor)

    path = "C:/Users/guarn/Dropbox/Alejandro/DoctoradoITESM/ReinforcementLearningTD3/criticRLcart.onnx"
    model_critic = onnx.load(path)
    critic = onnx2torch.convert(model_critic)
    return actor, critic