# NLP Reading Group in MSRA

This is the reading group homepage for Natural Language Computing Group, Microsoft Research Asia (MSRA). We regularly have interns to present interesting (and usually recent) papers on a wide range of topics including but not limited to Natural Language Processing, Computer Vision, Speech Processing and Machine Learning.

# Schedule
3:00 pm to 4:00 pm  (GMT+8) on Friday in Conf Rm BJW 2/13158, Microsoft Beijing.

# What's in this Repo?

* Slides for Presented Papers

# Meetings


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

