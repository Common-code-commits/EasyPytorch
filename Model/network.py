from base import BaseModel
from torch import nn


class NetWork(BaseModel):

    def __init__(self):
        super().__init__()
        self.b1 = nn.BatchNorm1d(4)
        self.l1 = nn.Linear(4, 784)
        self.l2 = nn.Linear(784, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, 64)
        self.l5 = nn.Linear(64, 8)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.b1(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.softmax(x)
        return x
