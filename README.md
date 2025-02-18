# HumorDB

Welcome to the HumorDB Repository! This repository contains a comprehensive collection of humorous images gathered from various sources on the internet, as well as images generated using cutting-edge AI techniques. To complement the funny images, we have also included non-funny images derived from the original funny images through sophisticated editing using tools like Photoshop and other generative AI techniques. The primary objective of this dataset is to serve as a valuable resource for testing complex scene understanding.

### Load Huggingface Dataset

```python
humor_db = load_dataset("kreimanlab/HumorDB")
```

Each item in the dataset has the following keys: **image, range_ratings_mean, comparison_ratings, binary_ratings, words**.


## Dataset Description
### Task Types

1. Binary Classification: Annotators were asked to classify each image pair as either "Funny" or "Not Funny."

2. Regression: In addition to the binary classification, annotators were requested to provide a numerical rating on a scale from 1 to 10, with 1 indicating "Not Funny" and 10 representing "Very Funny."

3. Comparison: Annotators were asked which among two images is funnier.

### Dataset Summary

There are

- 1771 images are rated as "Funny."
- 1771 images are rated as "Not Funny."

As a result of the dataset processing, the final dataset composition is as follows:

- Funny Images: 50.0%
- Not Funny Images: 50.0%

These are split into Train, Valid, Test sets as we want to keep the slightly modified image pairs with differing humor ratings in one set and not across sets.


## Running baselines and Reproducing Paper Results

    Please refer to the folder and README in the benchmarks folder of binary classification, regression, and comparison for more details on how to run the baselines and reproduce the paper results. There are detailed instructions on how to run the baselines and reproduce the paper results in the README of each folder. For training LlaVA, please use the json files similar to the ones in the LlaVa_files folder with the training instructions from the official LlaVA repository.

## Data Sources

The dataset is organized into two main categories:

* Funny Images: This subset comprises a diverse range of humorous images collected from various internet sources, including websites, social media platforms, and forums. To ensure copyright compliance, all non-open-source images are linked to their respective sources in the `all_images_links.csv` file.

* Non-Funny Images: To create this subset, we utilized the original funny images and applied advanced image editing techniques using tools like Photoshop and other generative AI models. This process resulted in forming pairs of images, where each pair consists of two images that are similar in content but differ in their perceived funniness levels.
* This resulted in a set of pairs of images that were similar in appearance but differed in their funniness.
* Not all images have a modified version.

### Training, Validation, and Test Sets
The most critical aspect of the dataset is the pairs of images with differing ratings. For the training, validation, and test sets, we added 698 pairs of images to the train set, 273 pairs to the validation set, and 300 pairs to the test set. The remaining images were randomly allocated to achieve a 60/20/20 split of the dataset for training, validation, and testing, respectively.

### Raw User Data and Processing steps
The raw user data is present in directories: user_binary, user_range, user_comp. Refer to analyse_data.ipynb for details on how the results were aggregated, all details are mentioned in the paper too.

### Words about images
Our crowd-source annotators for comparison tasks were asked to write a word or a phrase about the funnier image. We use these results to get a list of words that give humor to images, by selecting those that occur in at least 30% of the responses. These are present in image_common_words.txt. This is useful to fine tune vision-language models like BLiP and LlaVA.

## Using Raw Participant Data
Refer to `analyse_data.ipynb` to see how the data from participants was aggregated, the `user_binary`, `user_range`, `user_compare` folders give participants' data that was used. Note that this does not include the discarded data caused due to participants unreliability(for details refer to the paper). 
