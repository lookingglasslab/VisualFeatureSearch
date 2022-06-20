# Interactive Visual Feature Search

An interactive tool for analyzing intermediate features in any computer vision model. 
This repo is actively being developed, so our code is subject to change. Stay tuned for more details!

## Example
Choose region             |   Imagenet nearest neighbors
:-------------------------:|:-------------------------:
<img src="images/spia_highlighting.gif" alt="animation of highlighting widget showing the SPIA building" height="200" /> |     <img src="images/search_spia_results.png" alt="top 2 nearest neighbors to SPIA region" height="200" />

## Notebooks
Please see the following notebooks for demos of our interactive tool:
* [Small-scale demo](https://colab.research.google.com/drive/1wG92p-BHrwWBt_03Qw3bO4o0jL9vgyqK?usp=sharing) (no caching; search over 1k ImageNet val images)
* [Caching demo](https://colab.research.google.com/drive/18xIac1wiNaJh_hcqEKRsNUr9EmToAQRT?usp=sharing) (search over 50k ImageNet val images)

## Reference
If you find this visualization useful, please cite it as follows:
```
@online{ulrich2022search,
  author = {Devon Ulrich and Ruth Fong},
  title = {Interactive Visual Feature Search},
  year = 2022,
  url = https://github.com/lookingglasslab/VisualFeatureSearch,
}
```

## Acknowledgements
This visualization arose from discussions with [David Bau](https://baulab.info/) and his initial [prototype](https://github.com/davidbau/gpwidget/blob/master/notebooks/ExploreVggByClick.ipynb) of a similar visualization.
