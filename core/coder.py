import numpy as np
import torch

class Encoder(object):
    def __init__(self, characters):
        self.length = len(characters)
        self.characters = characters
    
    def __call__(self, content):
        output = np.array([self.characters.index(c) for c in content])
        return torch.tensor(output)


class Decoder(object):
    def __init__(self, characters):
        self.characters = characters
    
    def __call__(self, inputs) -> str:
        arr = inputs.numpy()
        output = "".join([self.characters[idx] for idx in arr])
        return output


if __name__ == "__main__":
    text = "A001KB09"
    characters = [
        ' ', 'A', 'B', 'C', 'D', 'E',
        'F', 'G', 'H', 'I', 'J', 'K',
        'L', 'M', 'N', 'O', 'P', 'Q',
        'R', 'S', 'T', 'U', 'V', 'W',
        'X', 'Y', 'Z', '0', '1', '2',
        '3', '4', '5', '6', '7', '8',
        '9'
    ]
    
    encoder = Encoder(characters)
    decoder = Decoder(characters)
    output = encoder(text)
    output = decoder(output)
    print(output)