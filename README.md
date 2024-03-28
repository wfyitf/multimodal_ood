# Multi-modal OOD Detection
This repo contains the materials for multi-modal OOD detection.


## Project Structure

```
├─── dataset                 <- Main dataset folder
│   ├─── coco                <- COCO2014 Dataset
│   ├─── visdial             <- Visual Dialog Dataset
│   └─── mmd                 <- Multimodal Detection Dataset
│
├─── notebooks               <- Notebooks for the project
│   ├─── a.ipynb             <- Notebook a
│   └───  b.ipynb            <- Notebook b
│   
├─── models                  <- Trained and serialized models, model predictions, or model summaries
└─── reports                 <- Generated analysis / reports 
```

## Project Description
Given a dataset with $((i_k, t_k), y_k)$ with $k = 1, 2, \cdots, n$ and the label $y_k$, we could use the vision-language model to embed the image $i_k$ with the abstract feature $x_{i,k}$, and also we could embed the dialogues with another text embedding $x_{t,k}$. To classify the relevance of an image to a dialogue according to the label $y_k$, we use a scoring function, $s(x, x')$, which evaluates the similarity or relevance between two hidden inputs, and produces a numeric score. A threshold value, denoted as $\lambda$, helps us to determine the classification. Based on this setup, we define two categories:

### Definitions of In-Domain (ID) and Out-Of-Domain (OOD)
- **In-domain**: given both measures $t_i$ from the images and $t_d$ from the dialogue, we say the image is in-domain with the dialogue if $s(t_i, t_d) \geq \lambda$. 
- **Out-of-domain**: given both measures $t_i$ from the images and $t_d$ from the dialogue, we say the image is out-of-domain with the dialogue if $s(t_i, t_d) \leq \lambda$. 

With the above definition, given an image-text data pair $(I^d, T^d)$, we are going to examine whether it is ID or OOD per image-dialogue in the given label set $\mathcal{Y}$. To this end, we draw the overall workflow in the figure below.

![The workflow of dialogue OOD](figures/OOD_figure2.png "The workflow of dialogue OOD")

In the given scenario, we employ a strategy that leverages advanced vision models like CLIP to derive meaningful descriptors or feature embedding from images. Similarly, for dialogues, we extract pivotal descriptors using techniques such as semantic analysis and we also do a feature embedding for the dialogues. These processes yield embeddings $x_{i,k}$ for images and $x_{t,k}$ for dialogues. Utilizing these embeddings, we apply a scoring function $s(x_{i,k}, x_{t,k})$ to assess the relevance between an image and a dialogue. The outcome of this function helps us determine whether the image-dialogue pair falls within the in-distribution categories, indicating a high relevance, or the out-of-distribution category, suggesting low or no relevance.



## Dataset

### QA OOD Dataset

We used the Visdial dataset [^1] to create the OOD dataset with QA systems. The dataset has over 120K images from the COCO2014 image dataset and collected QA dialogues with one-to-one mapping. We created a sample dataset that contains 9915 pairs, with each pair having an image, a conversation, and a set of labels. The data contains the following fields:

| Fields         | Meaning                              | Example                                          |
|----------------|--------------------------------------|--------------------------------------------------|
| image_id       | Image index                          | 265744                                           |
| dialog         | The whole dialogue in dictionary     | [{'answer': 71412, 'gt_index': 0, 'question': ...] |
| caption        | The image caption                    | a person standing next to a fence with horses    |
| dialog_full    | The full dialog text                 | Q:is there only 1 person, A:yes, Q:how many ho... |
| categories     | Categories in the image              | [dog, chair]                                     |
| supercategories | High-level categories in the image  | [animal, furniture]                              |

*Table: Descriptions for QA MMD Dataset*

The dataset stat is summarized as follows.

| Stats               | Figures |
|---------------------|---------|
| # Pair              | 9915    |
| # Image             | 9915    |
| # Dialogue          | 9915    |
| # Turn per dialog   | 10      |
| # Categories        | 80      |
| # Supercategories   | 12      |

*Table: Dataset Stats of QA mmd dataset*

The dataset encompasses a total of 12 higher-level categories, namely `animal`, `person`, `kitchen`, `food`, `sports`, `electronic`, `accessory`, `furniture`, `indoor`, `appliance`, `vehicle`, and `outdoor`. These supercategories are further subdivided into 80 more specific categories, providing a detailed classification framework. One example pair is shown as follows.

![Example pair of QA mmd dataset](figures/example_qa_mmd.png)

[^1]: https://visualdialog.org/

### Real MMD OOD Dataset
Lee et al. (2021) presents a 45k multi-modal dialogue dataset and the dataset creation method. This dataset is meant for training and evaluating multi-modal dialogue systems. Each multi-modal dialogue instance consists of a textual response and a dialogue context with multiple text utterances and images. The dataset details can be found in the [GitHub repository][^2] and the paper is available in the [ACL Anthology][^3]. The data contains the following fields:

| Fields          | Meaning                           | Example                                 |
|-----------------|-----------------------------------|-----------------------------------------|
| dialog          | The whole dialogue                | ['hello, how are you tonight ?', 'i am tired.', ...] |
| dialog_dataset  | The dialogue source dataset       | persona, dailydialog, or empathy       |
| img_dataset     | The image source dataset          | COCO                                    |
| img_file        | Original image file name          | COCO_val2014_000000283210.jpg           |
| set_source      | Dataset version                   | val2014                                 |
| image_id        | Transformed Internal image index  | 2000000283210                           |
| image_idx       | Original image index              | 283210                                  |
| categories      | Categories in the image           | [dog, chair]                            |
| supercategories | High-level categories in the image | [animal, furniture]                     |
| score           | The similarity score between image and dialogue | 0.580191          |

*Table: Descriptions for Real MMD Dataset*

Note that in this real mmd dataset, the dialogue and images are collected from different sources and they could be paired across different images and dialogues. That is, we could have one image pairing with different dialogues or one dialogue pairing with different images. The dataset stat is summarized as follows.

| Stats             | Figures |
|-------------------|---------|
| # Pair            | 16980   |
| # Image           | 6912    |
| # Dialogue        | 7452    |
| # Turn per dialog | 1 - 18  |
| # Categories      | 80      |
| # Supercategories | 12      |

*Table: Dataset Stats of Real MMD Dataset*

One example pair is given below.

![Example pair of real mmd dataset](figures/example_real_mmd.png)

[^2]: https://github.com/shh1574/multi-modal-dialogue-dataset
[^3]: https://aclanthology.org/2021.acl-short.113.pdf

