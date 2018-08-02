# Preprocessing files

This folder contains some C++ scripts I've been using to preprocess the dataset from the Google's Text Normalization Challenge in Kaggle (**available [here](https://www.kaggle.com/google-nlu/text-normalization/data)**). 

The scripts make use of [this](https://www.kaggle.com/google-nlu/text-normalization/data) UTF-8 header-only library. You don't need to download anything from its repository since the files are copied here, but you can go and check his work.

To **compile** any of the scripts you only need to a modern-enough compiler and use the `--std=c++14` option. In my case I have been enabling the optimizations as well, i.e., `-O3`.

The steps I've been following to preprocess the data are the following:

* `preprocess.cc` to chain all the words of the same phrase into a single string.
* `cut.cc` to discard all the entries whose output is longer than 126 (which is hardcoded by now).
* `sort.cc` to sort the entries of a file in decreasing order.
* `random_sample.cc` to extract a subset of the whole dataset with random samples.
* Then, I merge all the files into a single one making use of the `cat` bash command and pipe redirection.

Having at the end three files: `output_train.csv`, `output_val.csv` and `output_test.csv`.
