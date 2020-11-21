import numpy as np
import torch

class Encoder(object):
    def __init__(self, characters):
        self.length = len(characters)
        self.characters = characters
    
    def __call__(self, content: str) -> torch.IntTensor:
        output = np.array([self.characters.index(c) for c in content])
        return torch.IntTensor(output)


class Reducer(object):
    def __init__(self, target_size, blank: int = 0):
        self.blank = blank
        self.target_size = target_size
    
    def __call__(self, inputs: torch.IntTensor) -> torch.IntTensor:
        patch = torch.IntTensor(
            [
                inputs[i]
                for i in range(inputs.size()[0]-1)
                if inputs[i] != inputs[i+1] and inputs[i] != 0
            ] + [inputs[inputs.size()[0]-1]]
        )
        reduced = torch.IntTensor([
            patch[i] if i<patch.size()[0] else -1
            for i in range(self.target_size)
        ])
        return reduced


class Decoder(object):
    def __init__(self, characters):
        self.characters = characters
    
    def __call__(self, inputs) -> str:
        arr = inputs.numpy()
        decoded ="".join([self.characters[idx] for idx in arr])
        return decoded


if __name__ == "__main__":
    text = "A_000000077__HA550"
    characters = [
        '_', 'A', 'B', 'C', 'D', 'E',
        'F', 'G', 'H', 'I', 'J', 'K',
        'L', 'M', 'N', 'O', 'P', 'Q',
        'R', 'S', 'T', 'U', 'V', 'W',
        'X', 'Y', 'Z', '0', '1', '2',
        '3', '4', '5', '6', '7', '8',
        '9'
    ]
    
    encoder = Encoder(characters)
    reducer = Reducer(8)
    decoder = Decoder(characters)
    encoded = encoder(text)
    reduced = reducer(encoded)
    decoded = decoder(reduced)
    print(encoded)
    print(reduced)
    print(decoded)