# Interactive Visual Feature Search
Devon Ulrich and Ruth Fong

This repo contains the code for our 2022 preprint "[Interactive Visual Feature Search](https://arxiv.org/abs/2211.15060)".

Many visualization techniques have been created to help explain the behavior of convolutional neural networks (CNNs), but they largely consist of static diagrams that convey limited information.
Interactive visualizations can provide more rich insights and allow users to more easily explore a model's behavior; however, they are typically not easily reusable and are specific to a particular model. 

We introduce Interactive Visual Feature Search, a novel interactive visualization that is generalizable to any CNN and can easily be incorporated into a researcher's workflow. 
Our tool allows a user to highlight an image region and search for images from a given dataset with the most similar CNN features, which can provide new insights into how a model processes the geometric and semantic details in images.


## Example
Choose region             |   Imagenet nearest neighbors
:-------------------------:|:-------------------------:
<img src="images/spia_highlighting.gif" alt="animation of highlighting widget showing the SPIA building" height="200" /> |     <img src="images/search_spia_results.png" alt="top 2 nearest neighbors to SPIA region" height="200" />

## Notebooks

Please see the following notebooks for demos of our interactive tool:
* [Interactive Article](https://colab.research.google.com/drive/1B7SLFWCPiqFYf4kp-tQaLeL5tzVzQkyY?usp=sharing) (contains most qualitative visualizations from preprint)
* [Basic Demo](https://colab.research.google.com/drive/1sIio238NDRiGqxy0jto7O6VCFvqtmnYA?usp=sharing) (search over 50k ImageNet validation images)
* [Out-of-domain Comparison](https://colab.research.google.com/drive/1D7wmXVI9C8Ul_11aGKtQ6UL0hfY38NSr?usp=sharing) (comparing a model's performance on in-domain and out-of-domain queries)

## Implementation Overview

Interactive Visual Feature Search performs similarity search between free-form regions of images. Our method for implementing this can be broken down into a few steps:

1. Choose a model for computing feature data and a dataset to search through (e.g. ResNet50 and the ImageNet validation set).
2. Select a convolutional layer from the model to extract features from (e.g. from ResNet50's conv5 block, with an output tensor of shape 7x7x512).
3. Compute the feature tensors for all images in the search datset.
3. Choose any query image and highlight a region of interest to search for (see above figure, left).
4. Compute the feature tensor for the query image, downsample the highlighted mask to be the same size as the feature data (e.g. 7x7), and multiply the feature tensor by the mask element-wise to obtain the query features.
5. Compute the *k*-nearest neighbors between the query features and all regions of the same size from the dataset features via cosine similarity. 
6. Display the most similar images & corresponding regions within them (see above figure, right). 

This library is designed to make it simple to use Interactive Visual Feature Search on a local laptop/desktop or on a cloud-based notebook environment such as [Google Colab](https://colab.research.google.com/). 

## Library Details

To edit the above notebooks or create your own visualizations with Interactive Visual Feature Search, the following modules are necessary:

* [widgets.py](https://github.com/lookingglasslab/VisualFeatureSearch/blob/main/vissearch/widgets.py): the HighlightWidget and MultiHighlightWidget classes create interactive widgets that can be used in Jupyter/Colab notebooks to select an image and highlight a region within it with the mouse. 
* [searchtool.py](https://github.com/lookingglasslab/VisualFeatureSearch/blob/main/vissearch/searchtool.py): the CachedSearchTool computes the cosine similarities between the query image and the searchable dataset in a region-based manner as described above. 
* [caching.py](https://github.com/lookingglasslab/VisualFeatureSearch/blob/main/vissearch/caching.py): the above notebook demos use precomputed feature data to speed up the runtime of Interactive Visual Feature Search. To create your own feature caches for custom experiments, `precompute()` can be used to save a [Zarr](https://zarr.readthedocs.io/en/stable/) archive that can be used by CachedSearchTool. 

## Reference
If you find this visualization useful, please cite it as follows:
```
@article{ulrich2022search,
  author = {Devon Ulrich and Ruth Fong},
  title = {Interactive Visual Feature Search},
  journal = {CoRR},
  volume = {abs/2211.15060},
  year={2022}
}
```

## Acknowledgements
This visualization arose from discussions with [David Bau](https://baulab.info/) and his initial [prototype](https://github.com/davidbau/gpwidget/blob/master/notebooks/ExploreVggByClick.ipynb) of a similar visualization.
