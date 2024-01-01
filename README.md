These are some of my projects in Deep-Learning:


  Brain Cancer MRI Image Classification: 



Currently, anomaly detection through MRI is manual mostly and clinicians have to spend a lot of time to detect and segment the tumor for treatment and surgical purpose. This manual technique is also prone to errors and can compromise life. Also, diversity of Tumor types, makes the detection more difficult due to the complex structure of the brain. In order to resolve these issues, studies have started to focus on various machine learning and Deep Learning techniques for computer-based tumor detection and segmentation.

 [This paper](https://www.sciencedirect.com/science/article/abs/pii/S0895611121000896?via%3Dihub) reviews different papers that tried to do the task of brain cancer MRI image classification. In this homework, we will implement and compare some of these methods.
          



  Brain Abnormality Classification With CNN:




  
  CNN-based model for a multi-class classification task, brain abnormality classification.(Resnet50 and Vgg16 + Transfer Learning)




Image Captioning:

Here will be creating and training a neural network to generate captions for images using the [MS-COCO dataset](https://cocodataset.org/#home). Image captioning is a task that requires techniques from both computer vision and natural language processing. The COCO dataset is a large-scale object detection, segmentation, and captioning dataset that will be used to train our model. Our neural network architecture will consist of both convolutional neural networks (CNNs) and recurrent nural networks (RNNs) to automatically generate captions from images. We will be using an Encoder-Decoder architecture that combines the outputs of an image classification CNN (resnet, inception, ... ) and an RNN model (GRU, LSTM) to produce the relevant caption. The accuracy of our model will be evaluated using widely used evaluation metrics in the image captioning field such as BLEU and Perplexity metrics.





  Image Semantic Segmentation:





  Here, we will develope a U-Net for semantis segmentation of lung CT-scans.For this task, we will use a dataset consisting of CT-scans from lungs of COVID-19 patients. The regions affected by COVID-19 are marked with labels of ground glass opacity. we will be training and testing the U-Net model on this dataset.









Medical Image Registration Using Voxelmorph:




  Here, we are going to train a voxelmorph network to do the unsupervised image registration task. Dataset is here [CHAOS MR T2 dataset](https://chaos.grand-challenge.org/).





MLP:





  Multilayer Perceptron for DOROTHEA. (a drug discovery dataset) 







Single-Cell RNA Sequencing Analysis:








  Single-cell RNA sequencing (scRNA-seq) is a powerful technique used in molecular biology to study gene expression at the single-cell level. It allows researchers to profile the transcriptome of individual cells, providing insights into cellular heterogeneity and identifying distinct cell types within a tissue or sample.
By analyzing gene expression at the single-cell level, scRNA-seq enables researchers to gain insights into the diversity of cell types, identify rare cell populations, characterize cell states, and study cellular dynamics in various biological processes, such as development, disease progression, or response to treatments.
It's important to note that scRNA-seq is a rapidly evolving field, and new methods and technologies are continuously being developed to improve sensitivity, throughput, and the ability to capture additional molecular features beyond gene expression, such as chromatin accessibility or protein levels.
Here, we are going to get familiar with the general steps of analyzing scRNA-seq data, through different steps of preprocessing and clustering.







VAE:








 Generating MNIST digits with variational autoencoder. 





 Xray Classification:










 Classification and Interpretation for Xray Images





MRI Classification:







Classification of MRI images with Alexnet plus contrastive learning.