# Fashion Product Attribute Classification

## Problem Statement

The goal of this project is to classify multiple product attributes (such as color, pattern, sleeve length) from fashion product images across various categories. The dataset consists of 70k product images, but some categories and attributes have limited representation, making the task challenging due to class imbalance. Additionally, inference speed must be optimized to ensure predictions are made within 500ms per image, meeting real-world deployment constraints.

## Dataset

The dataset contains fashion products spanning categories such as women’s tops, sarees, kurtis, and men’s t-shirts. Each product is associated with attributes like color, pattern, sleeve type, and length, among others.

- **Train.csv**: Contains product IDs, category information, and attributes (`attr_1` to `attr_10`). Attributes from `attr_6` to `attr_10` may contain NaN values, especially for some categories where certain attributes are irrelevant.
- **Test.csv**: Contains product IDs and categories for which attributes need to be predicted.
- **Images**: The dataset contains around 71,000 product images. Image dimensions vary (height between 259-1080px, width is constant at 512px), necessitating pre-processing for consistency.

### Key Dataset Challenges:
- **Class Imbalance**: Certain attributes, such as "color" or "pattern," have significantly fewer samples for some categories.
- **Missing Values**: Attributes like sleeve length may not be applicable to all categories (e.g., sarees), resulting in missing values.

## Solution Overview

To solve the problem effectively, I employed a combination of advanced deep learning techniques, data augmentation, and model optimization strategies.

### Approach

1. **Data Preprocessing**:
   - Resized all images to a uniform size (e.g., 224x224) for model input consistency.
   - Normalized images using ImageNet mean and standard deviation values.
   - Imputed missing attributes using predefined 'dummy values' for consistency, especially when fewer attributes were required for specific categories.

2## Detailed Augmentation Techniques

- **Random Crop and Resize**:  
  Applied to simulate real-world variability in image framing and scaling, improving the model's robustness to different image resolutions.

- **Color Jittering**:  
  Introduced random changes to brightness, contrast, saturation, and hue to account for varied lighting conditions in product photography.

- **Gaussian Noise Addition**:  
  Simulated sensor noise to make the model more resilient to low-quality images.

- **Random Erasing**:  
  Masked random parts of the image to encourage the model to focus on key features rather than irrelevant regions.

- **Perspective Transformations**:  
  Applied to simulate different viewing angles, making the model robust to perspective changes in images.

- **AugMix**:  
  Combined multiple augmentations into a single pipeline, diversifying the dataset while maintaining semantic consistency.

- **Category-Aware Augmentations**:  
  Specific transformations tailored to distinct product categories:
  - **Sarees**: Enhanced color diversity and added random shearing to mimic fabric flow.
  - **T-shirts**: Used horizontal flips, rotations, and slight distortions to simulate real-world usage variations.
  - **Kurtis**: Introduced random cropping near the neckline to focus on intricate embroidery details.
  - **Men's Wear**: Augmented textures by applying random sharpness and blur to better capture material differences.
 
# Augmentation Pipeline for Class Imbalance in Image Datasets

This repository provides a Python-based augmentation pipeline to handle class imbalance in image datasets. It uses the Albumentations library to perform various augmentation strategies based on class proportions, ensuring better balance in the dataset for improved model training.

## Features
- Computes class proportions and applies augmentation strategies to under-represented classes.
- Supports basic, intermediate, and advanced augmentation pipelines.
- Balances datasets by generating new augmented images for under-represented classes.
- Saves augmented data and metadata to a CSV file for further use.

## Augmentation Strategy

### Class Proportion Calculation
- The function `calculate_class_proportions` computes the frequency of each class in the dataset for a specific attribute (`attr_1`, `attr_2`, ..., `attr_10`).
- A target frequency is calculated as the mean class count.
- A multiplier is determined for each class to indicate how much augmentation is needed:
  - If a class is under-represented, its multiplier will be higher, indicating more augmentation is required.
  - The multiplier is capped at `3.0` to avoid excessive augmentation for any single class.

### Defining Augmentation Pipelines
Three augmentation pipelines are defined using Albumentations:

1. **Basic Augmentations**:
   - Horizontal flipping, vertical flipping, random cropping, and 90-degree rotations.
2. **Intermediate Augmentations**:
   - Adjustments to color (brightness, contrast, saturation, hue) and geometric distortions (shift, scale, rotate).
3. **Advanced Augmentations**:
   - Techniques like Coarse Dropout, Elastic Transform, and Grid Distortion.

### Augmentation Strategy Based on Multipliers
- Classes with higher multipliers receive more augmentations and more complex transformations:
  - **Multiplier > 3.0**: All three augmentation levels are applied.
  - **Multiplier > 1.5**: Basic and intermediate augmentations are applied.
  - **Multiplier ≤ 1.5**: Only basic augmentations are applied.
- The `aug_percentage` determines the fraction of the class's images to augment.

### Selecting Images for Augmentation
- A subset of images for the target class is selected using `pandas.DataFrame.sample` based on the `aug_percentage`.
- Each selected image is augmented using the appropriate augmentation pipelines.

### Augmented Image Generation
- Augmented images are generated using the selected pipelines and saved with unique filenames.
- A new record corresponding to the augmented image is created by copying metadata from the original record and updating the image filename.

### Combining Original and Augmented Data
- The original dataset and augmented dataset are concatenated into a single DataFrame.
- The final DataFrame includes both the original and augmented images, ensuring better class balance.

### Output
- The augmented dataset is saved to a CSV file (`augmented_train.csv`) in the output directory.
- The total number of images after augmentation is logged.

## Usage
1. Set up the input image directory, output directory, and path to the training CSV file.
2. Run the pipeline to generate augmented images and save the updated dataset.

```python
if __name__ == "__main__":
    input_dir = "/path/to/input/images"
    output_dir = "/path/to/output/images"
    train_csv_path = "/path/to/train.csv"
    
    pipeline = AugmentationPipeline(input_dir, output_dir, train_csv_path)
    augmented_df = pipeline.run_pipeline()



3. **Model Architecture**:
I used a combination of state-of-the-art deep learning models, focusing on architectures that excel in visual tasks:

**ResNet18:** Chosen for its simplicity and efficiency in extracting deep hierarchical features, making it an ideal base model for ensembling.

**DenseNet121:** Integrated for its feature reuse capability, which ensures efficient use of parameters and enhances gradient flow, crucial for detailed fashion **attribute extraction.

**EfficientNetB0:** Used as a secondary model for ensembling due to its speed and strong performance in visual classification tasks, achieving a balance between accuracy and computational efficiency.

**XceptionNet:** Leveraged for its superior performance in handling spatial hierarchies with depthwise separable convolutions, providing detailed feature maps for complex patterns.
4. **Hyperparameter Tuning**:
   To optimize the model’s performance, I used **Optuna** for hyperparameter tuning. The following parameters were optimized:
   - Learning rate for both the backbone and custom heads.
   - Dropout rates to prevent overfitting.
   - Batch size, which was set between 32-64 based on GPU memory constraints.

5. **Training Strategy**:
   - **Mixed Precision Training**: Enabled to speed up training and reduce memory usage.
   - **Gradual Unfreezing**: Initially trained with a frozen backbone, allowing the custom heads to learn first. The backbone was gradually unfrozen and fine-tuned with a lower learning rate.
   - **Early Stopping**: Monitored validation F1-scores to prevent overfitting, stopping training early when improvements plateaued.

6. **Ensembling**:
   To further enhance prediction accuracy and robustness, I employed the following ensembling techniques:
   - Combined predictions from all models using weighted averaging and later on used Siberian Tiger optimization to furthur optimkze the weights.
   - Stacked models trained from different cross-validation folds to reduce variance in predictions.

7. **Model Optimization for Inference**:
   - **ONNX Conversion**: Converted the model to ONNX format for optimized inference speed.
   - **Batch Inference**: Implemented batch inference to speed up predictions when handling multiple images.

### Final Model Performance

The final model achieved competitive results in the challenge, with strong Macro and Micro F1-scores across all product attributes. The use of data augmentation, hyperparameter tuning, and ensembling ensured the model handled class imbalance effectively while maintaining inference time under 500ms, meeting the performance constraints.

