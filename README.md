# NLP Reading Group in MSRA

This is the reading group homepage for Natural Language Computing Group, Microsoft Research Asia (MSRA). We regularly have interns to present interesting (and usually recent) papers on a wide range of topics including but not limited to Natural Language Processing, Computer Vision, Speech Processing and Machine Learning.

# Schedule
3:00 pm to 4:00 pm  (GMT+8) on Friday in Conf Rm BJW 2/13158, Microsoft Beijing.

# What's in this Repo?

* Slides for Presented Papers

# Reading List
Sometimes there are too many interesting papers and we are not able to present all of them. <br>
So here is a [reading list](readinglist.md), which includes interesting papers not covered by reading group presenters.

# Meetings

## 05/27/2022

**Paper**: [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)

**Abstract**

Building models that can be rapidly adapted to numerous tasks using only a handful of annotated examples is an open challenge for multimodal machine learning research. We introduce Flamingo, a family of Visual Language Models (VLM) with this ability. Flamingo models include key architectural innovations to: (i) bridge powerful pretrained vision-only and language-only models, (ii) handle sequences of arbitrarily interleaved visual and textual data, and (iii) seamlessly ingest images or videos as inputs. Thanks to their flexibility, Flamingo models can be trained on large-scale multimodal web corpora containing arbitrarily interleaved text and images, which is key to endow them with in-context few-shot learning capabilities. We perform a thorough evaluation of the proposed Flamingo models, exploring and measuring their ability to rapidly adapt to a variety of image and video understanding benchmarks. These include open-ended tasks such as visual question-answering, where the model is prompted with a question which it has to answer, captioning tasks, which evaluate the ability to describe a scene or an event, and close-ended tasks such as multiple choice visual question-answering. For tasks lying anywhere on this spectrum, we demonstrate that a single Flamingo model can achieve a new state of the art for few-shot learning, simply by prompting the model with task-specific examples. On many of these benchmarks, Flamingo actually surpasses the performance of models that are fine-tuned on thousands of times more task-specific data.

**Presenter**: Zewen Chi [slides](slides/zewenchi/TBD.pptx)

## 05/20/2022

Skip due to NeurIPS ddl

## 05/13/2022

**Paper**: [Wav2Seq: Pre-training Speech-to-Text Encoder-Decoder Models Using Pseudo Languages](https://arxiv.org/pdf/2205.01086)

**Abstract**

We introduce Wav2Seq, the first self-supervised approach to pre-train both parts of encoder-decoder models for speech data. We induce a pseudo language as a compact discrete representation, and formulate a self-supervised pseudo speech recognition task — transcribing audio inputs into pseudo subword sequences. This process stands on its own, or can be applied as low-cost second-stage pre-training. We experiment with automatic speech recognition (ASR), spoken named entity recognition, and speech-to-text translation. We set new state-of-the-art results for end-to-end spoken named entity recognition, and show consistent improvements on 20 language pairs for speech-to-text translation, even when competing methods use additional text data for training. On ASR, our approach enables encoder-decoder methods to benefit from pre-training for all parts of the network, and shows comparable performance to highly optimized recent methods.

**Presenter**: Zhuoyuan Yao [slides](slides/zhuoyuanyao/paper_share_zhuoyuanyao.pptx)


## 05/06/2022

**Paper**: [Autoregressive Search Engines: Generating Substrings as Document Identifiers](https://arxiv.org/abs/2204.10628)

**Abstract** 

Knowledge-intensive language tasks require NLP systems to both provide the correct answer and retrieve supporting evidence for it in a given corpus. Autoregressive language models are emerging as the de-facto standard for generating answers, with newer and more powerful systems emerging at an astonishing pace. In this paper we argue that all this (and future) progress can be directly applied to the retrieval problem with minimal intervention to the models' architecture. Previous work has explored ways to partition the search space into hierarchical structures and retrieve documents by autoregressively generating their unique identifier. In this work we propose an alternative that doesn't force any structure in the search space: using all ngrams in a passage as its possible identifiers. This setup allows us to use an autoregressive model to generate and score distinctive ngrams, that are then mapped to full passages through an efficient data structure. Empirically, we show this not only outperforms prior autoregressive approaches but also leads to an average improvement of at least 10 points over more established retrieval solutions for passage-level retrieval on the KILT benchmark, establishing new state-of-the-art downstream performance on some datasets, while using a considerably lighter memory footprint than competing systems. Code and pre-trained models at this https URL.

**Presenter**: Xingxing Zhang [slides](slides/xingxingzhang/genenerative_search.pdf)

## 04/27/2022

**Invited Talk**: Diversity-Driven Reinforcement Learning: Win the Game with Creativity

**Time**: April 27 (Wednesday) 2022, 4:30pm - 5:30pm

**Location**: Conf Rm BJW 2/12257


**Abstract** 

Advances in deep reinforcement learning (RL) have helped solve a large number of AI open challenges and defeat human professionals in a variety of testbeds. The objective of RL algorithms is to achieve the highest rewards as fast as possible, which typically yields straightforward strategies that can directly win the game with the highest efficiency. This is a very different paradigm compared with humans, who often invented interesting behaviors not only for winning the game but also for fun. Is it possible for RL agents to also discover novel behaviors that are interesting under human judgment? In this talk, we present an in-depth analysis of this open question and present a novel paradigm, diversity-driven RL, which forces the RL agents to win the game with as many distinct strategies as possible. We will also introduce two practical algorithmic solutions and demonstrate the interesting emergent behaviors discovered in a collection of environments.

**Bio**:

Yi Wu is now an assistant professor at institute for interdisciplinary information sciences (IIIS), Tsinghua university. He obtained his PhD degree from UC Berkeley in 2019 under the supervision of Prof. Stuart Russell and was a researcher at OpenAI before moving back to Tsinghua. His research focuses on improving the generalization capabilities of learning agents. He is broadly interested in a variety of topics in AI, including deep reinforcement learning, multi-agent systems, natural language understanding and robot learning. His representative works include value iteration network, the MADDPG algorithm and OpenAI's hide-and-seek project. He also received the best paper award at NIPS 2016.




## 04/22/2022

**Paper**: [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/abs/2202.03555)

**Abstract** 

While the general idea of self-supervised learning is identical across modalities, the actual algorithms and objectives differ widely because they were developed with a single modality in mind. To get us closer to general self-supervised learning, we present data2vec, a framework that uses the same learning method for either speech, NLP or computer vision. The core idea is to predict latent representations of the full input data based on a masked view of the input in a self-distillation setup using a standard Transformer architecture. Instead of predicting modality-specific targets such as words, visual tokens or units of human speech which are local in nature, data2vec predicts contextualized latent representations that contain information from the entire input. Experiments on the major benchmarks of speech recognition, image classification, and natural language understanding demonstrate a new state of the art or competitive performance to predominant approaches.

**Presenter**: Ziqiang Zhang [slides](slides/ziqiangzhang/paper_reading_data2vec.pptx)



## 04/15/2022

**Paper**: [Efficient Language Modeling with Sparse all-MLP](https://arxiv.org/abs/2203.06850)

**Abstract** 

All-MLP architectures have attracted increasing interest as an alternative to attention-based models. In NLP, recent work like gMLP shows that all-MLPs can match Transformers in language modeling, but still lag behind in downstream tasks. In this work, we analyze the limitations of MLPs in expressiveness, and propose sparsely activated MLPs with mixture-of-experts (MoEs) in both feature and input (token) dimensions. Such sparse all-MLPs significantly increase model capacity and expressiveness while keeping the compute constant. We address critical challenges in incorporating conditional computation with two routing strategies. The proposed sparse all-MLP improves language modeling perplexity and obtains up to 2x improvement in training efficiency compared to both Transformer-based MoEs (GShard, Switch Transformer, Base Layers and HASH Layers) as well as dense Transformers and all-MLPs. Finally, we evaluate its zero-shot in-context learning performance on six downstream tasks, and find that it surpasses Transformer-based MoEs and dense Transformers.


**Presenter**: Yuan Xie [slides](slides/yuanxie/Paper_reading.pptx)


## 04/08/2022
**Paper**: [Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models](https://arxiv.org/abs/2010.05874)

**Abstract** 

Massively multilingual models subsuming tens or even hundreds of languages pose great challenges to multi-task optimization. While it is a common practice to apply a language-agnostic procedure optimizing a joint multilingual task objective, how to properly characterize and take advantage of its underlying problem structure for improving optimization efficiency remains under-explored. In this paper, we attempt to peek into the black-box of multilingual optimization through the lens of loss function geometry. We find that gradient similarity measured along the optimization trajectory is an important signal, which correlates well with not only language proximity but also the overall model performance. Such observation helps us to identify a critical limitation of existing gradient-based multi-task learning methods, and thus we derive a simple and scalable optimization procedure, named Gradient Vaccine, which encourages more geometrically aligned parameter updates for close tasks. Empirically, our method obtains significant model performance gains on multilingual machine translation and XTREME benchmark tasks for multilingual language models. Our work reveals the importance of properly measuring and utilizing language proximity in multilingual optimization, and has broader implications for multi-task learning beyond multilingual modeling.

**Presenter**: Liang Chen [slides](slides/liangchen/Gradient_Vaccine_0408.pdf)

## 04/01/2022
Skip

## 03/25/2022

**Paper**: [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)

**Abstract** 

In this paper, we propose a simple yet effective method to stabilize extremely deep Transformers. Specifically, we introduce a new normalization function (DeepNorm) to modify the residual connection in Transformer, accompanying with theoretically derived initialization. In-depth theoretical analysis shows that model updates can be bounded in a stable way. The proposed method combines the best of two worlds, i.e., good performance of Post-LN and stable training of Pre-LN, making DeepNorm a preferred alternative. We successfully scale Transformers up to 1,000 layers (i.e., 2,500 attention and feed-forward network sublayers) without difficulty, which is one order of magnitude deeper than previous deep Transformers. Remarkably, on a multilingual benchmark with 7,482 translation directions, our 200-layer model with 3.2B parameters significantly outperforms the 48-layer state-of-the-art model with 12B parameters by 5 BLEU points, which indicates a promising scaling direction.

**Presenter**: Hongyu Wang [slides](slides/hongyuwang/DeepNet.pdf)


## 03/18/2022

**Paper**: [Generating Training Data with Language Models: Towards Zero-Shot Language Understanding](https://arxiv.org/abs/2202.04538)

**Abstract** 

Pretrained language models (PLMs) have demonstrated remarkable performance in various natural language processing tasks: Unidirectional PLMs (e.g., GPT) are well known for their superior text generation capabilities; bidirectional PLMs (e.g., BERT) have been the prominent choice for natural language understanding (NLU) tasks. While both types of models have achieved promising few-shot learning performance, their potential for zero-shot learning has been underexplored. In this paper, we present a simple approach that uses both types of PLMs for fully zero-shot learning of NLU tasks without requiring any task-specific data: A unidirectional PLM generates class-conditioned texts guided by prompts, which are used as the training data for fine-tuning a bidirectional PLM. With quality training data selected based on the generation probability and regularization techniques (label smoothing and temporal ensembling) applied to the fine-tuning stage for better generalization and stability, our approach demonstrates strong performance across seven classification tasks of the GLUE benchmark (e.g., 72.3/73.8 on MNLI-m/mm and 92.8 on SST-2), significantly outperforming zero-shot prompting methods and achieving even comparable results to strong few-shot approaches using 32 training samples per class.

**Related Papers:**

https://arxiv.org/abs/2109.09193

**Presenter**: Jian Yang [slides](slides/jianyang/zero-label.pptx)


## 03/11/2022

**Paper**: [Toward Robust VQA: Overcoming the Out-of-distribution Problem in VQA](slides/qingyisi/OOD_VQA.pptx)

**Abstract** 

Background: Visual Question Answering (VQA) is a multi-modal task, involving the comprehension of vision, language and strong reasoning skills. Despite the remarkable performance on many VQA datasets such as VQA v2 , models for VQA have been criticized for their tendency to depend on dataset biases introduced by human annotators. For example, since about 40% of questions that begin with “what sport” have the answer “tennis”, systems tend to learn to output “tennis” for these questions regardless of image content. The models output the answer relying on the language priors rather than the reasoning ability. To evaluate the VQA models’ reasoning ability beyond the shortcuts in a training set, VQA-CP v2 dataset is constructed by re-organizing the VQA v2 dataset to make the answer distributions of the same question type are different between the training set and test set, and it has become the current out-of-distribution(OOD) benchmark in the VQA community.

**Presenter**: Qingyi Si [slides](slides/qingyisi/OOD_VQA.pptx)



## 03/04/2022

**Paper**: [REVISITING OVER-SMOOTHING IN BERT FROM THE PERSPECTIVE OF GRAPH](https://openreview.net/pdf?id=dUV91uaXm3)

**Abstract** 

Recently over-smoothing phenomenon of Transformer-based models is observed in both vision and language fields. However, no existing work has delved deeper to further investigate the main cause of this phenomenon. In this work, we make the attempt to analyze the over-smoothing problem from the perspective of graph, where such problem was first discovered and explored. Intuitively, the self-attention matrix can be seen as a normalized adjacent matrix of a corresponding graph. Based on the above connection, we provide some theoretical analysis and find that layer normalization plays a key role in the over-smoothing issue of Transformer-based models. Specifically, if the standard deviation of layer normalization is sufficiently large, the output of Transformer stacks will converge to a specific low-rank subspace and result in over-smoothing. To alleviate the over-smoothing problem, we consider hierarchical fusion strategies, which combine the representations from different layers adaptively to make the output more diverse. Extensive experiment results on various data sets illustrate the effect of our fusion method.

**Presenter**: Haiteng Zhao [slides](slides/haitengzhao/2022_3_3.pptx)



## 02/25/2022

**Paper**: [Open-vocabulary Object Detection via Vision and Language Knowledge Distillation](https://arxiv.org/abs/2104.13921)

**Abstract** 

We aim at advancing open-vocabulary object detection, which detects objects described by arbitrary text inputs. The fundamental challenge is the availability of training data. Existing object detection datasets only contain hundreds of categories, and it is costly to scale further. To overcome this challenge, we propose ViLD, a training method via Vision and Language knowledge Distillation. Our method distills the knowledge from a pretrained open-vocabulary image classification model (teacher) into a two-stage detector (student). Specifically, we use the teacher model to encode category texts and image regions of object proposals. Then we train a student detector, whose region embeddings of detected boxes are aligned with the text and image embeddings inferred by the teacher. We benchmark on LVIS by holding out all rare categories as novel categories not seen during training. ViLD obtains 16.1 mask APr, even outperforming the supervised counterpart by 3.8 with a ResNet-50 backbone. The model can directly transfer to other datasets without finetuning, achieving 72.2 AP50, 36.6 AP and 11.8 AP on PASCAL VOC, COCO and Objects365, respectively. On COCO, ViLD outperforms previous SOTA (Zareian et al., 2021) by 4.8 on novel AP and 11.4 on overall AP.

**Presenter**: Hongcheng Guo [slides](slides/hongchengguo/Open_Voc_detection.pptx)



## 02/18/2022

**Paper**: [Are Vision-Language Transformers Learning Multimodal Representations? A Probing Perspective](https://hal.archives-ouvertes.fr/hal-03521715/document)

**Abstract** 

In recent years, joint text-image embeddings have significantly improved thanks to the development of transformer based Vision-Language models. Despite these advances, we still need to better understand the representations produced by those models. In this paper, we compare pre-trained and finetuned representations at a vision, language and multimodal level. To that end, we use a set of probing tasks to evaluate the performance of state-of-the-art Vision-Language models and introduce new datasets specifically for multimodal probing. These datasets are carefully designed to address a range of multimodal capabilities while minimizing the potential for models to rely on bias. Although the results confirm the ability of Vision-Language models to understand color at a multimodal level, the models seem to prefer relying on bias in text data for object position and size. On semantically adversarial examples, we find that those models are able to pinpoint finegrained multimodal differences. Finally, we also notice that fine-tuning a Vision-Language model on multimodal tasks does not necessarily improve its multimodal ability. We make all datasets and code available to replicate experiments.

**Presenter**: Haoyu Song [slides](slides/haoyusong/paper_reading_20220218@HaoyuSong.pptx)




## 02/11/2022

**Paper**: [BLIP: Bootstrapping Language-Image Pre-training for
Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)

**Abstract** 

Vision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. In this paper, we propose BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. We achieve state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). BLIP also demonstrates strong generalization ability when directly transferred to video-language tasks in a zero-shot manner. Code, models, and datasets are released at https://github.com/salesforce/BLIP.


**Presenter**: Yupan Huang [slides](slides/yupanhuang/reading_group_yupanhuang_20220211.pptx)


## 02/04/2022
Skip due to Spring Festival Holidays!



## 01/28/2022
Skip due to Spring Festival Holidays!



## 01/21/2022

**Paper**: [Vector-based navigation using grid-like representations in artificial agents](https://www.nature.com/articles/s41586-018-0102-6.pdf)

**Abstract** 

Deep neural networks have achieved impressive successes in fields ranging from object recognition to complex games such as Go. Navigation, however, remains a substantial challenge for artificial agents, with deep neural networks trained by reinforcement learning failing to rival the proficiency of mammalian spatial behaviour, which is underpinned by grid cells in the entorhinal cortex. Grid cells are thought to provide a multi-scale periodic representation that functions as a metric for coding space and is critical for integrating self-motion (path integration) and planning direct trajectories to goals (vector-based navigation). Here we set out to leverage the computational functions of grid cells to develop a deep reinforcement learning agent with mammal-like navigational abilities. We first trained a recurrent network to perform path integration, leading to the emergence of representations resembling grid cells, as well as other entorhinal cell types. We then showed that this representation provided an effective basis for an agent to locate goals in challenging, unfamiliar, and changeable environments—optimizing the primary objective of navigation through deep reinforcement learning. The performance of agents endowed with grid-like representations surpassed that of an expert human and comparison agents, with the metric quantities necessary for vector-based navigation derived from grid-like units within the network. Furthermore, grid-like representations enabled agents to conduct shortcut behaviours reminiscent of those performed by mammals. Our findings show that emergent grid-like representations furnish agents with a Euclidean spatial metric and associated vector operations, providing a foundation for proficient navigation.


**Presenter**: Weizhi Wang [slides](slides/weizhiwang/2022-1-13-Weizhi-Reading_Group.pptx)

## 01/14/2022

Skip due to New Year Team Building!

## 01/07/2022

**Paper**: [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)

**Abstract** 

This is actually a tutorial of the wav2vec serial work from Facebook. This
tutorial focuses on the wav2vec 2.0 model, but will also mention its earlier
variants such as CPC, wav2vec, vq-wav2vec and models built upon it (e.g.,
wav2vec-U, BigSSL and W2V-BERT). The following is the abstract of the wav2vec
2.0 paper, which may give you some ideas of the wav2vec serial models.

We show for the first time that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler. wav2vec 2.0 masks the speech input in the latent space and solves a contrastive task defined over a quantization of the latent representations which are jointly learned. Experiments using all labeled data of Librispeech achieve 1.8/3.3 WER on the clean/other test sets. When lowering the amount of labeled data to one hour, wav2vec 2.0 outperforms the previous state of the art on the 100 hour subset while using 100 times less labeled data. Using just ten minutes of labeled data and pre-training on 53k hours of unlabeled data still achieves 4.8/8.2 WER. This demonstrates the feasibility of speech recognition with limited amounts of labeled data.

**References** 

[1] CPC: https://arxiv.org/abs/1807.03748 , https://arxiv.org/abs/2002.02848 <br>
[2] wav2vec：https://arxiv.org/abs/1904.05862 <br>
[3] vq-wav2vec：https://arxiv.org/abs/1910.05453 <br>
[4] wav2vec-U：https://arxiv.org/abs/2105.11084 <br>
[5] BigSSL： https://arxiv.org/abs/2109.13226 <br>
[6] W2V-BERT：https://arxiv.org/abs/2108.06209 <br>

**Presenter**: Sanyuan Chen [slides](slides/sanyuanchen/wav2vec_presentation.pptx)



## 12/31/2021

**Paper**: [Opening the Black Box of Deep Neural Networks via Information](https://arxiv.org/abs/1703.00810)

**Abstract** Despite their great success, there is still no comprehensive theoretical understanding of learning with Deep Neural Networks (DNNs) or their inner organization. Previous work proposed to analyze DNNs in the \textit{Information Plane}; i.e., the plane of the Mutual Information values that each layer preserves on the input and output variables. They suggested that the goal of the network is to optimize the Information Bottleneck (IB) tradeoff between compression and prediction, successively, for each layer.
In this work we follow up on this idea and demonstrate the effectiveness of the Information-Plane visualization of DNNs. Our main results are: (i) most of the training epochs in standard DL are spent on {\emph compression} of the input to efficient representation and not on fitting the training labels. (ii) The representation compression phase begins when the training errors becomes small and the Stochastic Gradient Decent (SGD) epochs change from a fast drift to smaller training error into a stochastic relaxation, or random diffusion, constrained by the training error value. (iii) The converged layers lie on or very close to the Information Bottleneck (IB) theoretical bound, and the maps from the input to any hidden layer and from this hidden layer to the output satisfy the IB self-consistent equations. This generalization through noise mechanism is unique to Deep Neural Networks and absent in one layer networks. (iv) The training time is dramatically reduced when adding more hidden layers. Thus the main advantage of the hidden layers is computational. This can be explained by the reduced relaxation time, as this it scales super-linearly (exponentially for simple diffusion) with the information compression from the previous layer.



**Presenter**: Yaru Hao [slides](slides/yaruhao/IB_pre.pptx)


## 12/24/2021
Skip due to Christmas Party! Merry Christmas!

## 12/17/2021

**Paper**: [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)

**Abstract** Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as TF-IDF or BM25, are the de facto method. In this work, we show that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dual-encoder framework. When evaluated on a wide range of open-domain QA datasets, our dense retriever outperforms a strong Lucene-BM25 system largely by 9%-19% absolute in terms of top-20 passage retrieval accuracy, and helps our end-to-end QA system establish new state-of-the-art on multiple open-domain QA benchmarks.


**Presenter**: Ting Jiang [slides](slides/tingjiang/DPR.pptx)



## 12/10/2021

**Paper**: [Vector Quantized Diffusion Model for Text-to-Image Synthesis](https://arxiv.org/abs/2111.14822)

**Abstract** We present the vector quantized diffusion (VQ-Diffusion) model for text-to-image generation. This method is based on a vector quantized variational autoencoder (VQ-VAE) whose latent space is modeled by a conditional variant of the recently developed Denoising Diffusion Probabilistic Model (DDPM). We find that this latent-space method is well-suited for text-to-image generation tasks because it not only eliminates the unidirectional bias with existing methods but also allows us to incorporate a mask-and-replace diffusion strategy to avoid the accumulation of errors, which is a serious problem with existing methods. Our experiments show that the VQ-Diffusion produces significantly better text-to-image generation results when compared with conventional autoregressive (AR) models with similar numbers of parameters. Compared with previous GAN-based text-to-image methods, our VQ-Diffusion can handle more complex scenes and improve the synthesized image quality by a large margin. Finally, we show that the image generation computation in our method can be made highly efficient by reparameterization. With traditional AR methods, the text-to-image generation time increases linearly with the output image resolution and hence is quite time consuming even for normal size images. The VQ-Diffusion allows us to achieve a better trade-off between quality and speed. Our experiments indicate that the VQ-Diffusion model with the reparameterization is fifteen times faster than traditional AR methods while achieving a better image quality.

**Presenter**: Minghao Li [slides](slides/minghaoli/VQ-Diffusion.pptx)


## 12/03/2021

**Tutorial**: [Vector Quantised Variational Auto-Encoder](slides/junlongli/VQ-VAE.pptx)

**Abstract** This is a tutorial on Vector Quantised Variational Auto-Encoder (VQ-VAE). It covers the original formulation of VQ-VAE [1][2] and its recent development VQ-GAN [3] and dVAE [4] as well as its applications to speech processing [5], text-to-image generation [4] and vision representation learning [6][7].

**References** <br>
[1] Neural Discrete Representation Learning https://arxiv.org/abs/1711.00937 <br>
[2] Generating Diverse High-Fidelity Images with VQ-VAE-2 https://arxiv.org/abs/1906.00446 <br>
[3] Taming Transformers for High-Resolution Image Synthesis https://arxiv.org/abs/2012.09841 <br>
[4] Zero-Shot Text-to-Image Generation https://arxiv.org/abs/2102.12092 <br>
[5] vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations https://arxiv.org/abs/1910.05453 <br>
[6] BEiT: BERT Pre-Training of Image Transformers https://arxiv.org/abs/2106.08254 <br>
[7] PeCo: Perceptual Codebook for BERT Pre-training of Vision Transformers https://arxiv.org/abs/2111.12710 <br>

**Presenter**: Junlong Li [slides](slides/junlongli/VQ-VAE.pptx)


## 11/26/2021

**Paper**: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf) 

**Abstract** This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task. Coupling these two designs enables us to train large models efficiently and effectively: we accelerate training (by 3× or more) and improve accuracy. Our scalable approach allows for learning high-capacity models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy (87.8%) among methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pretraining and shows promising scaling behavior.

**Presenter**: Hangbo Bao [slides](slides/hangbobao/mae.pdf)

## 11/19/2021

**Paper**: [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447)

**Abstract**: Self-supervised approaches for speech representation learning are challenged by three unique problems: (1) there are multiple sound units in each input utterance, (2) there is no lexicon of input sound units during the pre-training phase, and (3) sound units have variable lengths with no explicit segmentation. To deal with these three problems, we propose the Hidden-Unit BERT (HuBERT) approach for self-supervised speech representation learning, which utilizes an offline clustering step to provide aligned target labels for a BERT-like prediction loss. A key ingredient of our approach is applying the prediction loss over the masked regions only, which forces the model to learn a combined acoustic and language model over the continuous inputs. HuBERT relies primarily on the consistency of the unsupervised clustering step rather than the intrinsic quality of the assigned cluster labels. Starting with a simple k-means teacher of 100 clusters, and using two iterations of clustering, the HuBERT model either matches or improves upon the state-of-the-art wav2vec 2.0 performance on the Librispeech (960h) and Libri-light (60,000h) benchmarks with 10min, 1h, 10h, 100h, and 960h fine-tuning subsets. Using a 1B parameter model, HuBERT shows up to 19% and 13% relative WER reduction on the more challenging dev-other and test-other evaluation subsets.


**Presenter**: Junyi Ao [slides](slides/junyiao/Junyi_20211119_nlp_reading_group.pptx)


## 11/12/2021
Skip due to ACL deadline

## 11/5/2021

**Paper**: [When Attention Meets Fast Recurrence: Training Language Models with Reduced Compute](https://arxiv.org/abs/2102.12459)

**Abstract**: Large language models have become increasingly difficult to train because of the growing computation time and cost. In this work, we present SRU++, a highly-efficient architecture that combines fast recurrence and attention for sequence modeling. SRU++ exhibits strong modeling capacity and training efficiency. On standard language modeling tasks such as Enwik8, Wiki-103 and Billion Word datasets, our model obtains better bits-per-character and perplexity while using 3x-10x less training cost compared to top-performing Transformer models. For instance, our model achieves a state-of-the-art result on the Enwik8 dataset using 1.6 days of training on an 8-GPU machine. We further demonstrate that SRU++ requires minimal attention for near state-of-the-art performance. Our results suggest jointly leveraging fast recurrence with little attention as a promising direction for accelerating model training and inference.


**Presenter**: Chengyi Wang [slides](slides/chengyiwang/paper_reading_ChengyiWang.pptx)


## 10/29/2021

**Paper**: [Sparse MoEs meet Efficient Ensembles](https://arxiv.org/abs/2110.03360)

**Abstract**: Machine learning models based on the aggregated outputs of submodels, either at the activation or prediction levels, lead to strong performance. We study the interplay of two popular classes of such models: ensembles of neural networks and sparse mixture of experts (sparse MoEs). First, we show that these two approaches have complementary features whose combination is beneficial. Then, we present partitioned batch ensembles, an efficient ensemble of sparse MoEs that takes the best of both classes of models. Extensive experiments on fine-tuned vision transformers demonstrate the accuracy, log-likelihood, few-shot learning, robustness, and uncertainty calibration improvements of our approach over several challenging baselines. Partitioned batch ensembles not only scale to models with up to 2.7B parameters, but also provide larger performance gains for larger models. 

**Presenter**: Bo Zheng [slides](slides/bozheng/bo_zheng_reading_group.pptx)


## 10/22/2021

**Paper**: [Towards a Unified View of Parameter-Efficient Transfer Learning](https://arxiv.org/abs/2110.04366)

**Abstract**: Fine-tuning large pre-trained language models on downstream tasks has become the de-facto learning paradigm in NLP. However, conventional approaches fine-tune all the parameters of the pre-trained model, which becomes prohibitive as the model size and the number of tasks grow. Recent work has proposed a variety of parameter-efficient transfer learning methods that only fine-tune a small number of (extra) parameters to attain strong performance. While effective, the critical ingredients for success and the connections among the various methods are poorly understood. In this paper, we break down the design of state-of-the-art parameter-efficient transfer learning methods and present a unified framework that establishes connections between them. Specifically, we re-frame them as modifications to specific hidden states in pre-trained models, and define a set of design dimensions along which different methods vary, such as the function to compute the modification and the position to apply the modification. Through comprehensive empirical studies across machine translation, text summarization, language understanding, and text classification benchmarks, we utilize the unified view to identify important design choices in previous methods. Furthermore, our unified framework enables the transfer of design elements across different approaches, and as a result we are able to instantiate new parameter-efficient fine-tuning methods that tune less parameters than previous methods while being more effective, achieving comparable results to fine-tuning all parameters on all four tasks.


**Presenter**: Yiheng Xu [slides](slides/yihengxu/a_unified_view_of_param_efficient_by_yiheng.pptx.pptx)


## 10/15/2021 

**Paper**: [SimVLM: Simple Visual Language Model Pretraining with Weak Supervision](https://arxiv.org/abs/2108.10904)

**Abstract**: With recent progress in joint modeling of visual and textual representations, Vision-Language Pretraining (VLP) has achieved impressive performance on many multimodal downstream tasks. However, the requirement for expensive annotations including clean image captions and regional labels limits the scalability of existing approaches, and complicates the pretraining procedure with the introduction of multiple dataset-specific objectives. In this work, we relax these constraints and present a minimalist pretraining framework, named Simple Visual Language Model (SimVLM). Unlike prior work, SimVLM reduces the training complexity by exploiting large-scale weak supervision, and is trained end-to-end with a single prefix language modeling objective. Without utilizing extra data or task-specific customization, the resulting model significantly outperforms previous pretraining methods and achieves new state-of-the-art results on a wide range of discriminative and generative vision-language benchmarks, including VQA (+3.74% vqa-score), NLVR2 (+1.17% accuracy), SNLI-VE (+1.37% accuracy) and image captioning tasks (+10.1% average CIDEr score). Furthermore, we demonstrate that SimVLM acquires strong generalization and transfer ability, enabling zero-shot behavior including open-ended visual question answering and cross-modality transfer.


**Presenter**: Hangbo Bao [slides](slides/hangbobao/simvlm.pdf)


## 10/8/2021 

**Paper**: [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795)

**Abstract**: The recently-proposed Perceiver model obtains good results on several domains (images, audio, multimodal, point clouds) while scaling linearly in compute and memory with the input size. While the Perceiver supports many kinds of inputs, it can only produce very simple outputs such as class scores. Perceiver IO overcomes this limitation without sacrificing the original's appealing properties by learning to flexibly query the model's latent space to produce outputs of arbitrary size and semantics. Perceiver IO still decouples model depth from data size and still scales linearly with data size, but now with respect to both input and output sizes. The full Perceiver IO model achieves strong results on tasks with highly structured output spaces, such as natural language and visual understanding, StarCraft II, and multi-task and multi-modal domains. As highlights, Perceiver IO matches a Transformer-based BERT baseline on the GLUE language benchmark without the need for input tokenization and achieves state-of-the-art performance on Sintel optical flow estimation.

**Presenter**: Guanhua Chen [slides](slides/guanhuachen/NLP_reading_group_Oct_08.pdf)


## 9/24/2021 

**Paper**: [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)

**Abstract**: Since the introduction of the transformer model by Vaswani et al. (2017), a fundamental question remains open: how to achieve extrapolation at inference time to longer sequences than seen during training? We first show that extrapolation can be improved by changing the position representation method, though we find that existing proposals do not allow efficient extrapolation. We introduce a simple and efficient method, Attention with Linear Biases (ALiBi), that allows for extrapolation. ALiBi does not add positional embeddings to the word embeddings; instead, it biases the query-key attention scores with a term that is proportional to their distance. We show that this method allows training a 1.3 billion parameter model on input sequences of length 1024 that extrapolates to input sequences of length 2048, achieving the same perplexity as a sinusoidal position embedding model trained on inputs of length 2048, 11% faster and using 11% less memory. ALiBi's inductive bias towards recency allows it to outperform multiple strong position methods on the WikiText-103 benchmark. Finally, we provide analysis of ALiBi to understand why it leads to better performance.

**Presenter**: Zekun Wang [slides](slides/zekunwang/RG-ALiBi.pptx)


## 9/17/2021 

**Paper**: [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/pdf/2109.01652.pdf)

**Abstract**: This paper explores a simple method for improving the zero-shot learning abilities of language models. We show that instruction tuning -- finetuning language models on a collection of tasks described via instructions -- substantially boosts zero-shot performance on unseen tasks.
We take a 137B parameter pretrained language model and instruction-tune it on over 60 NLP tasks verbalized via natural language instruction templates. We evaluate this instruction-tuned model, which we call FLAN, on unseen task types. FLAN substantially improves the performance of its unmodified counterpart and surpasses zero-shot 175B GPT-3 on 19 of 25 tasks that we evaluate. FLAN even outperforms few-shot GPT-3 by a large margin on ANLI, RTE, BoolQ, AI2-ARC, OpenbookQA, and StoryCloze. Ablation studies reveal that number of tasks and model scale are key components to the success of instruction tuning.

**Presenter**: Xin Sun [slides](slides/xinsun/slides_flan_zero_shot.pdf)


## 9/10/2021
  planning meeting

