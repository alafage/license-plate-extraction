# lp-extraction

License plate content extraction from pictures.

* Text detection: Work in progress.
* Text recognition: Convolutional Recurrent Neural Network (CRNN) (network architecture based on [this paper](https://arxiv.org/pdf/1507.05717.pdf), codes for setting the neural network from [ocr.pytorch](https://github.com/courao/ocr.pytorch))

*Future Developement: Should be able to extract written content from any pictures.*

## Dependencies

* PyTorch: `torch`, `torchvision`
* OpenCV
* Pillow

## Structure

(WARNING - This will change when the text detection will be implemented)

* core:
    * coders.py: transform functions to encode and decode texts. 
    * datasets.py: custom dataset classes
    * networks.py: 
* save: contains model back-ups.
* testing.ipynb: notebook to test a model
* training.ipynb: notebook to train a CRNN
