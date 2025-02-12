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

        self.actorWeights()
    def forward(self, x):
        x = self.flatten(x)
        out1 = self.fc1(x)
        act1 = self.act1(out1)
        out2 = self.fc2(act1)
        act2 = self.act2(out2)
        out3 = self.fc3(act2)
        logits = self.output(out3)
        return logits
    
    def actorWeights(self):
        data = scipy.io.loadmat("actorParams.mat")

        fc1_W = data["fc_1_W"]
        fc1_b = data["fc_1_b"]

        fc2_W = data["fc_2_W"]
        fc2_b = data["fc_2_b"]

        fc3_W = data["fc_3_W"]
        fc3_b = data["fc_3_b"]

        fc1_W_tensor = torch.from_numpy(fc1_W).float()
        print(fc1_W_tensor[0])
        fc1_b_tensor = torch.from_numpy(fc1_b[:,0]).float()
        print(fc1_b_tensor[0])
        fc2_W_tensor = torch.from_numpy(fc2_W).float()
        fc2_b_tensor = torch.from_numpy(fc2_b[:,0]).float()

        fc3_W_tensor = torch.from_numpy(fc3_W).float()
        fc3_b_tensor = torch.from_numpy(fc3_b[:,0]).float()

        self.fc1.weight = torch.nn.Parameter(fc1_W_tensor)
        self.fc1.bias = torch.nn.Parameter(fc1_b_tensor)
        
        self.fc2.weight = torch.nn.Parameter(fc2_W_tensor)
        self.fc2.bias = torch.nn.Parameter(fc2_b_tensor)

        self.fc3.weight = torch.nn.Parameter(fc3_W_tensor)
        self.fc3.bias = torch.nn.Parameter(fc3_b_tensor)
    
    

model = customActor()

X = torch.ones(1,4)
logits = model(X)
print(logits)

X = torch.zeros(1,4)
logits = model(X)
print(logits)
############################################
############################################
############################################


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

#         self.reluBody = nn.ReLU()

#         self.fcBody = nn.Linear(256,128)
#         self.fcBodyOutput = nn.ReLU()

#         self.output = nn.Linear(128,1)
#     def forward(self, in1,in2):

#         x1 = self.flatten(in1)
#         x2 = self.flatten(in2)
#         fc1 = self.fcin1(x1)
#         fc2 = self.fcin2(x2)
#         concat = torch.cat((fc1,fc2),1)
#         reluBody = self.reluBody(concat)
#         fcBody = self.fcBody(reluBody)
#         bodyOutput = self.fcBodyOutput(fcBody)
#         logits = self.output(bodyOutput)
#         return logits


# fc1_W_tensor = torch.from_numpy(fc1_W).float()
# fc1_b_tensor = torch.from_numpy(fc1_b[:,0]).float()

# fc2_W_tensor = torch.from_numpy(fc2_W).float()
# fc2_b_tensor = torch.from_numpy(fc2_b[:,0]).float()

# fcbody_W_tensor = torch.from_numpy(fcbody_W).float()
# fcbody_b_tensor = torch.from_numpy(fcbody_b[:,0]).float()

# fcoutput_W_tensor = torch.from_numpy(fcoutput_W).float()
# fcoutput_b_tensor = torch.from_numpy(fcoutput_b[:,0]).float()

# modelC = customCritic()
# with torch.no_grad():
#     modelC.fcin1.weight.copy_(fc1_W_tensor)
#     modelC.fcin1.bias.copy_(fc1_b_tensor)
#     modelC.fcin2.weight.copy_(fc2_W_tensor)
#     modelC.fcin2.bias.copy_(fc2_b_tensor)
#     modelC.fcBody.weight.copy_(fcbody_W_tensor)
#     modelC.fcBody.bias.copy_(fcbody_b_tensor)
#     modelC.output.weight.copy_(fcoutput_W_tensor)
#     modelC.output.bias.copy_(fcoutput_b_tensor)

# X1 = torch.zeros(1,4)
# X2 = torch.zeros(1,1)

# logits = modelC(X1,X2)
# print(logits)


