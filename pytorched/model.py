from torch import nn

class F1RacePrediction(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential (
        nn.Flatten(),
        nn.Linear(96, 75),
        nn.ReLU(),
        nn.Linear(75, 25),
        nn.ReLU(),
        nn.Linear(25,50),
        nn.ReLU(),
        nn.Linear(50,10)
    )

  def forward(self, x):
    return self.layers(x)
