class Module:
    def __init__(self, name):
        self.name = name
        self.cache = None
        self.phase = 'Train'

    def test(self):
        self.phase = 'Test'

    def train(self):
        self.phase = 'Train'

    def forward(self, x, **kwargs):
        pass

    def backward(self, dout):
        pass
