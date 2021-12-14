![HurricaneHarvey](https://github.com/allankapoor/hurricane_damage/blob/master/readme_images/cover.png)

# Post-Hurricane Structure Damage Assessment Leveraging Aerial Imagery and Convolutional Neural Networks

This project is an effort to automate structure damage assessments based on post-hurricane aerial imagery. A labelled training dataset of images of structures from the Houston area just after Hurricane Harvey was acquired from the University of Washington Disaster Data Science Lab. Several neural network architectures were trained and evaluated, from basic architectures to deep networks via transfer learning. Models were trained on Google Public Cloud virtual machines leveraging multiple GPUs to enable fast and efficient iteration. **The final model achieves an accuracy of 0.9775 on test data**. This model could be used by local, state, and federal natural disaster responders to quickly develop damage assessment maps based on a single fly-over by imaging aircraft after a hurricane.</p>

View the [presentation](https://github.com/allankapoor/hurricane_damage/blob/master/PRESENTATION.pdf), full [report](https://github.com/allankapoor/hurricane_damage/blob/master/REPORT.pdf), or check out the summary below.</p>

<h2>Why Post-Hurricane Damage Assessment?</h2>

Natural disasters, especially hurricanes and extreme storms, cause widespread destruction and loss of life and cost the U.S billions of dollars each year. Effective response after a natural disaster strikes is critical to reducing harm and facilitating long term recovery but manual damage assessments are time and resource intensive. A model that can perform automated damage assessment based on remote sensing imagery quickly captured by a few planes flying over a disaster zone would greatly reduce the effort and increase speed in which useful data (location and extent of damaged structures) could be put in the hands of responders.</p>

<h1> Project Summary </h1>

The following sections summarize key steps in my process. Check out the Jupyter notebooks to see how it was done!</p>

<h2>Data Wrangling + Exploratory Data Analysis</h2>

For code/more detail, see: [EDA Notebook](https://github.com/allankapoor/hurricane_damage/blob/master/Step1_EDA.ipynb)

<h3> Data Source </h3>

Data for this project was sourced from a [labeled training dataset](https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized) made publicly available by Quoc Dung Cao and Youngjun Choe at the University of Washington’s Disaster Data Science Lab. The dataset consists of images of structures cropped from aerial imagery collected in the days after Hurricane Harvey in Houston and other nearby cities. There are 12,000 images total: 8,000 for training, 2,000 for validation, and 2,000 for testing. </p>

<h3> Visual Inspection </h3>

The images are color, in RGB format. The figure below compares 3 randomly selected damage images to 3 randomly selected no damage images. From viewing these examples along with others, some patterns emerged:</p>
* Many (but not all) images have flood waters surrounding the structures, so there is a difference in ground texture/color.
 * Other damage images have small objects strewn across the ground. However, many images of no damage also display this visual pattern.
 * Images of structures with no damage seem more likely to feature visible ground  or pools with blue-ish water (uncovered by flood waters).
 * In many cases it isn't obvious to the human eye if an image shows damage or no damage.

![Example_images](https://github.com/allankapoor/hurricane_damage/blob/master/readme_images/examples.png)

<h3> Summary Images </h3>

In classification problems based on structured (tabular) data, it is common to explore summary statistics for each feature (column). For this image classification problem I used the numeric RGB color values in the images to calculate summary statistics by pixel (feature) and then plot them visually. By plotting visual summaries of the two classes, differences can be identified between them that could inform decisions about neural network architecture.  </p>

<h4> Mean Value by Class </h4>

The figures below depict each pixel’s mean value across all images by class. For both damage and no damage images we see that a structure tends to be located within the center of the image. For damage images, the pixel values around the structure tend to be lower in value than around the no damage images. Perhaps this is because damage images tend to have flood waters around the structures, which are a different color than unflooded ground. </p>

![mean_byclass](https://github.com/allankapoor/hurricane_damage/blob/master/readme_images/mean_byclass.png)

<h4>Standard Deviation by Class </h4>

The figures below are similar to those above but instead of depicting the mean pixel value they depict the standard deviation for each pixel by class. Standard deviation around the edges of the image appears to be (slightly) greater for the no damage images. Perhaps this is because visible ground around the structures creates more variation between images in that class. </p>

![mean_byclass](https://github.com/allankapoor/hurricane_damage/blob/master/readme_images/std_byclass.png)


<h3> Geographic Distribution </h3>

I investigated the geographic distribution of the training data by class. This is possible because the GPS coordinates are contained in the filename for each image file. The figure below depicts the locations of each image in the training dataset and whether it is damaged or not damaged. </p>

![map](https://github.com/allankapoor/hurricane_damage/blob/master/readme_images/map.png)
 
We can see that the training data appears to come from 3 distinct areas. In one area, all structures in the training dataset are damaged, while in the other two there is a mix. Within those two there are clear spatial patterns of where flooding/damage occurred and where it didn't. </p>

This is not surprising given that the hurricane (and flooding) impacted areas differently based on topography, urban form, etc. However, it does also indicate that the examples from each class often come from entirely different areas. There is a danger that when training the neural networks models, they may learn to identify differences that are more to due with differences in appearance of structures between those areas, rather than differences that are actually due to hurricane damage. </p>

<h2>Modeling</h2>

This section describes the general approach used to train and iterate on neural network models: </p>
 * Platform: Tensorflow + Keras API
 * Training on Google Cloud VM: 8 vCPUs (30 GB RAM) + 1 NVIDIA T4 GPU (16 GB)
 * Convolutional neural networks assumed to be best approach given image classification problem
 * Training for 50 epochs with early stopping if model does not improve after 10 epochs
 * **Evaluation metric: accuracy** (training/validation/test datasets are balanced)
 
Code for [uploading data to a Google Cloud bucket](https://github.com/allankapoor/hurricane_damage/blob/master/Step2a_GoogleCloudUpload.ipynb) and then [transfering to a virtual machine.](https://github.com/allankapoor/hurricane_damage/blob/master/Step2b_GoogleCloudVMTransfer.ipynb) </p>

For code/more detail on model architecture, see: [Modeling Notebook](https://github.com/allankapoor/hurricane_damage/blob/master/Step3_Modeling.ipynb) </p>

<h3>Baseline Model</h3>
I first created a simple baseline model with three convolution layers, three dense layers, and an output layer with 2 nodes (corresponding to the two classes). Some initial modeling decisions were made here and carry through to the other models:  </p>

* The ReLU activation function is used for all layers except the final output layer because it is computationally efficient and known to lead good results in most cases
* Batch normalization was included after the convolution layers but before activation based on experimentation (when batch normalization came after activation, validation accuracy dropped substantially). 
* Adam optimizer was used for updating network weights because it is known to perform well on image classification problems. Adam adjusts the learning rate over time. 

The baseline model trained on data with no image augmentation achieved a validation accuracy of 0.94650. When the pipeline was updated to include image augmentation, validation accuracy increased to 0.95650. </p>

This was surprisingly good performance for such a simple model, suggesting that there are clear differences between the damage and no damage classes that are relatively easy for the network to learn. However, there was still substantial room for improvement.  </p>

<h3>Model Improvement and Refinement</h3>

<h4>Reducing Overfitting</h4>

During training of the baseline model, training validation scores regularly exceeded 0.99, while the validation accuracy was substantially lower. This suggested that the model was overfitting, even with variation in training data introduced by the image augmentation. To address this, I added:</p>

* **Max pooling layers** which downsample the outputs of the convolution layers by sliding a filter of the outputs and calculating the maximum pixel value within each window. This reduces the sensitivity of the model to exactly where features are located in an input image. Max pooling layers were added before the activation functions as this reduces the number of features that the activation function has to work on, increasing computational efficiency. 
* **Dropout layers** which have the effect of making the training process noisy, forcing nodes within a layer to probabilistically take on more or less responsibility for the inputs - reducing the chances that the model will overfit to noise in the training data. 

<h4>Improving Convergence</h4>

While most of the models performed relatively well, a recurring issue was that models would achieve high accuracy after only a few training epochs but never converge. The figure below summarizes training and validation accuracy by epoch for a model that demonstrates this trend: </p>

![no_converge](https://github.com/allankapoor/hurricane_damage/blob/master/readme_images/no_converge.png)

Model convergence was ultimately achieved through a combination of several updates to the model architecture:</p>
 * Reduction in kernel size and stride for the first convolution layer
 * Reduction in number of filters in first convolution layer
 * Reduction in number of nodes in each dense layer by 50%
 * Reduction in initial learning rate for the Adam optimizer from the default (0.001) to 0.0001
 
As summarized in the figure below, the updated model converges much better than before:</p>

![converge](https://github.com/allankapoor/hurricane_damage/blob/master/readme_images/converge.png)

<h4>Model Architecture</h4>

Several combinations of hyperparameters were tested including smaller and larger convolution filters and more or less nodes in the dense layers. The model below achieved a **validation accuracy of 0.9735**, a substantial improvement over the baseline model. </p>

| Layer                                               | Output Shape   | \# of Params |
| --------------------------------------------------- | -------------- | ------------ |
| Rescaling                                           | (128, 128, 3)  | 0            |
| Convolution (filters=32, kernel\_size=3, strides=1) | (128, 128, 32) | 896          |
| Max Pooling (pool size=2, strides=2)                | (64, 64, 32)   | 0            |
| Batch Normalization                                 | (64, 64, 32)   | 128          |
| Activation (ReLU)                                   | (64, 64, 32)   | 0            |
| Convolution (filters=32, kernel\_size=3, strides=1) | (32, 32, 64)   | 18,496       |
| Max Pooling (pool size=2, strides=2)                | (16, 16, 64)   | 0            |
| Batch Normalization                                 | (16, 16, 64)   | 256          |
| Activation (ReLU)                                   | (16, 16, 64)   | 0            |
| Convolution (filters=32, kernel\_size=3, strides=1) | (8, 8, 64)     | 36,928       |
| Max Pooling (pool size=2, strides=2)                | (4, 4, 64)     | 0            |
| Batch Normalization                                 | (4, 4, 64)     | 256          |
| Activation (ReLU)                                   | (4, 4, 64)     | 0            |
| Flattening                                          | 1024           | 0            |
| Dense (512 nodes, ReLU activation)                  | 512            | 524,800      |
| Dropout (rate=0.3)                                  | 512            | 0            |
| Dense (256 nodes, ReLU activation)                  | 256            | 131,328      |
| Dropout (rate=0.2)                                  | 256            | 0            |
| Dense (128 nodes, ReLU activation)                  | 128            | 32,896       |
| Dropout (rate=0.1)                                  | 128            | 0            |
| Dense (2 nodes, Softmax activation)                 | 2              | 258          |


<h3>Deep Network Leveraging Transfer Learning</h3>

After training and evaluating the models described above, I next leveraged transfer learning to find out if pre-trained deep learning models could produce better and/or more stable results. Out of the many deep learning models available I decided to use Resnet50 because it: </p>
 1) Offers a good balance of accuracy and training efficiency
 2) Was used in the baseline model for the xView2 competition, which suggested that it would perform well for a similar aerial imagery classification task.

Starting with the same hyperparameter settings for the convolution and dense layers as the best performing model from the previous section, several iterations based on different combinations of convolution filters size, dense layer node density, dropout rate, and learning rate were tested. The best performing deep network achieved a **validation accuracy of 0.9765**, slightly better than the standard convolutional network.</p>

<h3>Comparing Models</h3>

The table below summarizes a select subset of the iterations evaluated (not all are included for brevity). Note that all models except the initial baseline model included image augmentation in their pipeline (rotate, flip, and zoom). </p>

| Model                                                                           | Validation Accuracy |
| ------------------------------------------------------------------------------- | ------------------- |
| Baseline (no image augmentation)                                                | 0.9365              |
| Baseline                                                                        | 0.9440              |
| With Max Pooling (kernel=5)  & Dropout Layers                                   | 0.9500              |
| With Max Pooling (kernel=10) & Dropout Layers                                   | 0.9230              |
| With Max Pooling (kernel=3) & Dropout Layers (dense layers with 50% less nodes) | 0.9735              |
| Transfer Learning (with Max Pooling kernel=5)                                   | 0.9765              |
| Transfer Learning (Max Pooling kernel=3)                                        | 0.9735              |


While the transfer learning model did perform slightly better than the best performing standard model, the difference in validation accuracy was only 0.003 - which corresponds to 6 images in the validation set. This is well within the potential variation that would be expected if a different set of images had been randomly selected for the validation set.</p>

During testing it was noted that the standard model is significantly smaller in size (11 mb vs. 500 mb) and prediction is much more time and computationally efficient. Because the validation accuracy rates were essentially the same, and the standard model was more computational efficient, it was selected as the final model.</p>

<h2>Seleced Model Performance</h2>

The final model was then evaluated based on test data (an additional 2,000 images which had been kept separate from the training and validation sets). The final model achieved a **test accuracy of 0.9775**, which was even higher than the validation accuracy (0.9735). Similar performance on the validation and test sets suggest that the model would be generalizable to additional unseen data (see caveats in Conclusion below).</p>

A confusion matrix of the models predictions on test data reveal that the false positives were twice as common as false negatives (0=no damage, 1=damage):</p>

![confusion_matrix](https://github.com/allankapoor/hurricane_damage/blob/master/readme_images/confusion_matrix.png)

Examining the misclassified images reveals some insights about the model:</p>

**False Positives** tend to have surfaces that are mistaken for flood waters or junk around them that is mistaken for damage. False positives also tend to be rural structures:</p>

![false_positives](https://github.com/allankapoor/hurricane_damage/blob/master/readme_images/false_positives.png)

**False negatives** appear to be mostly large or non-residential structures, have a lot of variation in the ground surface, and/or no obvious flood water:</p>

![false_negatives](https://github.com/allankapoor/hurricane_damage/blob/master/readme_images/false_negatives.png)


<h2>Conclusion</h2>

Some final thoughts on potential improvements and how the model could be used going forward are included below:

<h3>Caveats and Potential Improvements</h3>

Training data improvements:</p>
* More labelled data - a larger validation set would enable more rigorous tuning
* Imagery from after floods have subsided - would enable train a model that can detect structure damage only, rather than presence of flood water
* Imagery from neighborhoods with mixed impacts - would ensure that the model is not learning to identify other differences between flooded/unflooded neighborhoods tother than damage such as urban form, density, etc.
* Imagery from different cities - would ensure that the model works for situations where the urban form is substantially different than Houston (i.e., dense urban areas)</p>
Modeling improvements:</p>
* Cross validation - would reduce chance that model evaluation/comparison si not impacted by the images in a particular validation set.
* Hyperparameter optimization - would ensure that best vlaues for various hyperparemters are selected, but would require much more computation time.

<h3>Using the Model </h3>

This damage classification model could be inserted as a step in a fully automated pipeline:</p>
1) Ingest/clean aerial imagery
2) Crop images of structures based on [MS Building Footprints data](https://github.com/microsoft/USBuildingFootprints )
3) Classify images (this model)
4) Plot locations/damage assessment on interactive map

This pipeline would be much faster than a crowdsourcing/manual review: the final model classified 2,0000 structures in seconds on a desktop computer.

<h2>Credits</h2>

Thanks to Shmuel Naaman for mentorship and advice on modeling approach and architecture.