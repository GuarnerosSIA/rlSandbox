import torch as th
import scipy.io
import numpy as np
import torch.nn as nn





class customCritic(nn.Module):
    def __init__(self,path):
        super(customCritic, self).__init__()
        
        self.fcin1 = nn.Linear(4,128)
        self.fcin2 = nn.Linear(1,128)

        self.reluBody = nn.ReLU()

        self.fcBody = nn.Linear(256,128)
        self.fcBodyOutput = nn.ReLU()

        self.output = nn.Linear(128,1)

        self.loadMyWeights(path)
    def forward(self, x):
        x1,x2 = th.split(x,[4,1],dim=1)
        fc1 = self.fcin1(x1)
        fc2 = self.fcin2(x2)
        concat = th.cat((fc1,fc2),1)
        reluBody = self.reluBody(concat)
        fcBody = self.fcBody(reluBody)
        bodyOutput = self.fcBodyOutput(fcBody)
        logits = self.output(bodyOutput)
        return logits

    def loadMyWeights(self,path):
        data = scipy.io.loadmat(path)

        fc1_W = data["fc1_W"]
        fc1_b = data["fc1_b"]

        fc2_W = data["fc2_W"]
        fc2_b = data["fc2_b"]

        fcbody_W = data["fcBody_W"]
        fcbody_b = data["fcBody_b"]

        fcoutput_W = data["fcoutput_W"]
        fcoutput_b = data["fcoutput_b"]

        fc1_W_tensor = th.from_numpy(fc1_W).float()
        fc1_b_tensor = th.from_numpy(fc1_b[:,0]).float()

        fc2_W_tensor = th.from_numpy(fc2_W).float()
        fc2_b_tensor = th.from_numpy(fc2_b[:,0]).float()

        fcbody_W_tensor = th.from_numpy(fcbody_W).float()
        fcbody_b_tensor = th.from_numpy(fcbody_b[:,0]).float()

        fcoutput_W_tensor = th.from_numpy(fcoutput_W).float()
        fcoutput_b_tensor = th.from_numpy(fcoutput_b[:,0]).float()

        self.fcin1.weight = th.nn.Parameter(fc1_W_tensor)
        self.fcin1.bias = th.nn.Parameter(fc1_b_tensor)

        self.fcin2.weight = th.nn.Parameter(fc2_W_tensor)
        self.fcin2.bias = th.nn.Parameter(fc2_b_tensor)

        self.fcBody.weight = th.nn.Parameter(fcbody_W_tensor)
        self.fcBody.bias = th.nn.Parameter(fcbody_b_tensor)

        self.output.weight = th.nn.Parameter(fcoutput_W_tensor)
        self.output.bias = th.nn.Parameter(fcoutput_b_tensor)


path = "criticParams1.mat"
modelC = customCritic(path)

features = th.ones(1,4)
actions = th.zeros(1,1)
X = th.cat([features, actions], dim=1)

print(X)

in1,in2 = th.split(X,[4,1],dim=1)

print(in1,in2)

logits = modelC(X)
print(logits)


