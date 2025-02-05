import torch
import onnx2torch
import onnx

path = "C:/Users/guarn/Dropbox/Alejandro/DoctoradoITESM/rlSandbox/rlSandbox/actorRLSGP.onnx"
model_actor = onnx.load(path)
actor = onnx2torch.convert(model_actor)
x = torch.ones((1,10))
y = actor(x)
print(y)


path = "C:/Users/guarn/Dropbox/Alejandro/DoctoradoITESM/rlSandbox/rlSandbox/criticRLSGP1.onnx"
model_critic = onnx.load(path)
critic = onnx2torch.convert(model_critic)

input1 = torch.zeros((1,10))
input2 = torch.ones((1,10))

y = critic(input1,input2)
print(y)