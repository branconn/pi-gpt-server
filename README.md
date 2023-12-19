# Autocomplete Feature - Decoder on RPi3 served with gRPC

## Motivation

The motivation for this project is to combine a little of what I learned from my Machine Learning and Graduate Intro to Operating Systems courses this past semester... also to finally have a purpose for my raspberry pi 3 aside from being an idle headless server.

## Goal

Use all the papers I've written in the masters program as training data for a text autocompleting feature. I've implemented three layers of complexity:

### Previous character context: Probabilistic prefix trie with loop-backs

**Prefix tries** are structures that can efficiently store a corpus of words and are [simple to implement](https://leetcode.com/problems/implement-trie-prefix-tree/). The root node is blank and each child is the first letter in a word. Subsequent child nodes are subsequent letters in the word. 

**Probabilistic** While building the tree from a list of words, if we keep a count of how many times a node has been traversed, then we can use these frequency counts as a probability distribution and sample from them accordingly. There are two straightforward ways to do this, depending on whether you want to optimize building or sampling:

1) To optimize building, simply increment a child's count each time it is traversed to from a parent while building ***O(1)***. When it's time to sample, generate a random number in the range of counts among the children, and sum from left to right until the sum is larger than the random number ***O(N)***.

2) To optimize sampling, you can do the summing up front by maintaining a cumulative sum array. Which means during building, each time a child is traverse you would increment its count and the counts of all children to its right ***O(N)***. When sampling, since the cumulative sum array is sorted in ascending order, you can use binary search to find the first sum that is larger than the random number ***O(logN)***.

**Loop-backs** are just what I call connecting the leaf nodes to the children of the root node (first letters) with the same frequent counts. This lets us build probability distributions from the last letter of a word the the first letter of the next, but after the first letter is selected, there's no memory retained and the word is generated from same static p dist.

### Previous word context: 1st-Order Markov Chain

This is perhaps even easier to implement than the prefix trie - it is simply a count dictionary of whole words rather than characters. The dictionary can be looped, passing values back as keys, to generate text on a word-to-word posterior distribution based on the counts observed.

The prefix trie can complete a valid word from a subword input, then that word can feed into the Markov Chain dictionary as a key to sample the subsequent word.

The memory here is carried from one word to the next. Using the previous sentence, once "to" generates "the", "the" has equal probability of choosing "memory" or "next".

### Overkill: Decoder model (Karpathy's nanoGPT)

Prefix tries and Markov Chains are simple and fast yet restrictive. A prefix trie will only ever return a word that's encountered in the training text. A Markov Chain will only return a word that has followed the input word. To get more sophisticated results we can start to build in language features with SpaCy like POS tagging, but why not jump on the hype train and make a Transformer Decoder like our pals at OpenAI.

This project was largely inspired by [Andrej Karpathy's video tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) on building a tiny decoder to produce Shakespeare-like text. In fact, my first step was to follow along with him but using my own dataset to produce Brandon-like text.

## Implementation

### Raspberry Pi 3 Server

#### OS

Searching for an OS that is lightweight but still somewhat user-friendly, I decided to switch my Ubuntu Server 20.1 to [Raspberry OS Lite (Legacy) 64-bit](https://www.raspberrypi.com/software/operating-systems/). Legacy, which uses Debian Bullseye, is still recommended for Pi 3s over the new Debian Bookworm-based OS. 

OS changes are made easy with an SD card and the [Raspberry Pi Imager](https://www.raspberrypi.com/software/). You can set up config details like network access, hostname and password, and toggle ssh access from the Imager GUI.

#### Packages

The two main packages we want to install are gRPC for talking to the pi-server with remote procedure calls and [TensorFlow Lite](https://www.tensorflow.org/lite/guide/python) for runtime inference on lower power resources (like an RPi3).
Raspberry OS Lite comes with Python 3.9.2 preloaded, which is sufficient for both TFLite and gRPC.

1) Create an env:

```
mkdir pi-server // create project directory
cd pi-server
sudo apt-get install python3-venv // install venv package
python -m venv env // create virtual environment
source env/bin activate // activate the venv
```

2) Install gRPC and TFLite:

```
pip install grpcio
pip install grpcio-tools
pip install tflite-runtime
```

