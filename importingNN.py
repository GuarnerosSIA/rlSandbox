import torch
import onnx2torch
import onnx

path = "C:/Users/guarn/Dropbox/Alejandro/DoctoradoITESM/ReinforcementLearningTD3/actorRLcart.onnx"
model_actor = onnx.load(path)
actor = onnx2torch.convert(model_actor)
x = torch.zeros((1,4))
y = actor(x)
print(y)


path = "C:/Users/guarn/Dropbox/Alejandro/DoctoradoITESM/ReinforcementLearningTD3/criticRLcart.onnx"
model_critic = onnx.load(path)
critic = onnx2torch.convert(model_critic)

input1 = torch.ones((1,4))
input2 = torch.zeros((1,1))

y = critic(input1,input2)
print(y)