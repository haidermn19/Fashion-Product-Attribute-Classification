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

2. **Data Augmentation**:
   To improve the representation of under-represented classes and enhance model generalization, I applied the following augmentation techniques:
   - **CutMix**: Mixes patches from multiple images to generate new samples, helping the model learn spatially complex patterns.
   - **MixUp**: Blends two images and their corresponding labels, encouraging the model to be less sensitive to specific image artifacts.
   - **Category-Specific Augmentations**: Tailored augmentations to specific product categories. For example, augmenting color variations for sarees and applying random flips and rotations for t-shirts.

3. **Model Architecture**:
I used a combination of state-of-the-art deep learning models, focusing on architectures that excel in visual tasks:

**ResNet18:** Chosen for its simplicity and efficiency in extracting deep hierarchical features, making it an ideal base model for ensembling.
**DenseNet121:** Integrated for its feature reuse capability, which ensures efficient use of parameters and enhances gradient flow, crucial for detailed fashion **attribute extraction.
**EfficientNetB0:** Used as a secondary model for ensembling due to its speed and strong performance in visual classification tasks, achieving a balance between **accuracy and computational efficiency.
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
   - Combined predictions from Swin Transformer and EfficientNet models using weighted averaging.
   - Stacked models trained from different cross-validation folds to reduce variance in predictions.

7. **Model Optimization for Inference**:
   - **ONNX Conversion**: Converted the model to ONNX format for optimized inference speed.
   - **Quantization**: Applied model quantization to reduce the model size without compromising accuracy.
   - **Batch Inference**: Implemented batch inference to speed up predictions when handling multiple images.

### Final Model Performance

The final model achieved competitive results in the challenge, with strong Macro and Micro F1-scores across all product attributes. The use of data augmentation, hyperparameter tuning, and ensembling ensured the model handled class imbalance effectively while maintaining inference time under 500ms, meeting the performance constraints.

