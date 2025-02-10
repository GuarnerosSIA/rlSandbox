import torch
import scipy.io
import numpy as np
import torch.nn as nn

data = scipy.io.loadmat("actorParams.mat")

fc1_W = data["fc_1_W"]
fc1_b = data["fc_1_b"]

fc2_W = data["fc_2_W"]
fc2_b = data["fc_2_b"]

fc3_W = data["fc_3_W"]
fc3_b = data["fc_3_b"]

scaleW = data["scaleW"]

class customActor(nn.Module):
    def __init__(self):
        super(customActor, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4,128)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(128,128)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(128,1)
        self.output = nn.Tanh()
    def forward(self, x):
        x = self.flatten(x)
        out1 = self.fc1(x)
        act1 = self.act1(out1)
        out2 = self.fc2(act1)
        act2 = self.act2(out2)
        out3 = self.fc3(act2)
        logits = self.output(out3)
        return logits*10


fc1_W_tensor = torch.from_numpy(fc1_W).float()
fc1_b_tensor = torch.from_numpy(fc1_b[:,0]).float()

fc2_W_tensor = torch.from_numpy(fc2_W).float()
fc2_b_tensor = torch.from_numpy(fc2_b[:,0]).float()

fc3_W_tensor = torch.from_numpy(fc3_W).float()
fc3_b_tensor = torch.from_numpy(fc3_b[:,0]).float()


model = customActor()
with torch.no_grad():
    model.fc1.weight.copy_(fc1_W_tensor)
    model.fc1.bias.copy_(fc1_b_tensor)
    model.fc2.weight.copy_(fc2_W_tensor)
    model.fc2.bias.copy_(fc2_b_tensor)
    model.fc3.weight.copy_(fc3_W_tensor)
    model.fc3.bias.copy_(fc3_b_tensor)

X = torch.ones(1,4)

logits = model(X)
print(logits)

#####

# data = scipy.io.loadmat("criticParams1.mat")

# fc1_W = data["fc1_W"]
# fc1_b = data["fc1_b"]

# fc2_W = data["fc2_W"]
# fc2_b = data["fc2_b"]

# fcbody_W = data["fcBody_W"]
# fcbody_b = data["fcBody_b"]

# fcoutput_W = data["fcoutput_W"]
# fcoutput_b = data["fcoutput_b"]

# class customCritic(nn.Module):
#     def __init__(self):
#         super(customCritic, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fcin1 = nn.Linear(4,128)
#         self.fcin2 = nn.Linear(1,128)
        

#         self.fc2 = nn.Linear(128,128)
#         self.act2 = nn.ReLU()
#         self.fc3 = nn.Linear(128,1)
#         self.output = nn.Tanh()
#     def forward(self, x):
#         x = self.flatten(x)
#         out1 = self.fc1(x)
#         act1 = self.act1(out1)
#         out2 = self.fc2(act1)
#         act2 = self.act2(out2)
#         out3 = self.fc3(act2)
#         logits = self.output(out3)
#         return logits*10


# fc1_W_tensor = torch.from_numpy(fc1_W).float()
# fc1_b_tensor = torch.from_numpy(fc1_b[:,0]).float()

# fc2_W_tensor = torch.from_numpy(fc2_W).float()
# fc2_b_tensor = torch.from_numpy(fc2_b[:,0]).float()

# fc3_W_tensor = torch.from_numpy(fc3_W).float()
# fc3_b_tensor = torch.from_numpy(fc3_b[:,0]).float()


# model = customActor()
# with torch.no_grad():
#     model.fc1.weight.copy_(fc1_W_tensor)
#     model.fc1.bias.copy_(fc1_b_tensor)
#     model.fc2.weight.copy_(fc2_W_tensor)
#     model.fc2.bias.copy_(fc2_b_tensor)
#     model.fc3.weight.copy_(fc3_W_tensor)
#     model.fc3.bias.copy_(fc3_b_tensor)

# X = torch.ones(1,4)

# logits = model(X)
# print(logits)


