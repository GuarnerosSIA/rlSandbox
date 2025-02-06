import torch
import onnx2torch
from onnx2pytorch import ConvertModel
import onnx

def importMatlabNN(mode='ConvertModel'):
    path = "C:/Users/guarn/Dropbox/Alejandro/DoctoradoITESM/ReinforcementLearningTD3/actorRLcart.onnx"
    model_actor = onnx.load(path)
    if mode == 'ConvertModel':
        actor = ConvertModel(model_actor)
    else:
        actor = onnx2torch.convert(model_actor)

    path = "C:/Users/guarn/Dropbox/Alejandro/DoctoradoITESM/ReinforcementLearningTD3/criticRLcart.onnx"
    model_critic = onnx.load(path)
    if mode == 'ConvertModel':
        critic = ConvertModel(model_critic)
    else:
        critic = onnx2torch.convert(model_critic)
    return actor, critic

if __name__ == "__main__":
    importMatlabNN()