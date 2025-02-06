import torch
import onnx2torch
from onnx2pytorch import ConvertModel
import onnx

class newConvertModel(ConvertModel):
    def __init__(self,*args, **kwargs):
        super(newConvertModel, self).__init__(*args, **kwargs)
    
    def set_training_mode(self,mode: bool)->None:
        self.train(mode)


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
    actor, critic = importMatlabNN(mode='ConvertModel')

    
    optimizer = torch.optim.Adam(actor.parameters())