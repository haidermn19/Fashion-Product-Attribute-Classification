# Fashion-Product-Attribute-Classification
Predict multiple product attributes, such as color, pattern, and sleeve length, from a dataset of fashion product images.

Problem Statement
The goal of this project is to classify multiple product attributes (such as color, pattern, sleeve length) from fashion product images across various categories. The dataset consists of 70k product images, but some categories and attributes have limited representation, making the task challenging due to class imbalance. Additionally, inference speed must be optimized to ensure predictions are made within 500ms per image, meeting real-world deployment constraints.

Dataset
The dataset contains fashion products spanning categories such as women’s tops, sarees, kurtis, and men’s t-shirts. Each product is associated with attributes like color, pattern, sleeve type, and length, among others.

Train.csv: Contains product IDs, category information, and attributes (attr_1 to attr_10). Attributes from attr_6 to attr_10 may contain NaN values, especially for some categories where certain attributes are irrelevant.
Test.csv: Contains product IDs and categories for which attributes need to be predicted.
Images: The dataset contains around 71,000 product images. Image dimensions vary (height between 259-1080px, width is constant at 512px), necessitating pre-processing for consistency.
Key Dataset Challenges:
Class Imbalance: Certain attributes, such as "color" or "pattern," have significantly fewer samples for some categories.
Missing Values: Attributes like sleeve length may not be applicable to all categories (e.g., sarees), resulting in missing values.
Solution Overview
To solve the problem effectively, I employed a combination of advanced deep learning techniques, data augmentation, and model optimization strategies.

Approach
Data Preprocessing:

Resized all images to a uniform size (e.g., 224x224) for model input consistency.
Normalized images using ImageNet mean and standard deviation values.
Imputed missing attributes using predefined 'dummy values' for consistency, especially when fewer attributes were required for specific categories.
Data Augmentation: To improve the representation of under-represented classes and enhance model generalization, I applied the following augmentation techniques:

CutMix: Mixes patches from multiple images to generate new samples, helping the model learn spatially complex patterns.
MixUp: Blends two images and their corresponding labels, encouraging the model to be less sensitive to specific image artifacts.
Category-Specific Augmentations: Tailored augmentations to specific product categories. For example, augmenting color variations for sarees and applying random flips and rotations for t-shirts.
Model Architecture: I used a combination of state-of-the-art deep learning models, focusing on architectures that excel in visual tasks:

Swin Transformer: Chosen for its ability to capture long-range dependencies and complex patterns, which is essential for understanding fashion products.
EfficientNet: Used as a secondary model for ensembling due to its speed and strong performance in visual classification tasks.
Multi-Task Learning Head: A shared backbone was used, followed by separate fully connected layers (heads) for each attribute. This allowed simultaneous predictions for multiple attributes.
Hyperparameter Tuning: To optimize the model’s performance, I used Optuna for hyperparameter tuning. The following parameters were optimized:

Learning rate for both the backbone and custom heads.
Dropout rates to prevent overfitting.
Batch size, which was set between 32-64 based on GPU memory constraints.
Training Strategy:

Mixed Precision Training: Enabled to speed up training and reduce memory usage.
Gradual Unfreezing: Initially trained with a frozen backbone, allowing the custom heads to learn first. The backbone was gradually unfrozen and fine-tuned with a lower learning rate.
Early Stopping: Monitored validation F1-scores to prevent overfitting, stopping training early when improvements plateaued.
Ensembling: To further enhance prediction accuracy and robustness, I employed the following ensembling techniques:

Combined predictions from Swin Transformer and EfficientNet models using weighted averaging.
Stacked models trained from different cross-validation folds to reduce variance in predictions.
Model Optimization for Inference:

ONNX Conversion: Converted the model to ONNX format for optimized inference speed.
Quantization: Applied model quantization to reduce the model size without compromising accuracy.
Batch Inference: Implemented batch inference to speed up predictions when handling multiple images.
Final Model Performance
The final model achieved competitive results in the challenge, with strong Macro and Micro F1-scores across all product attributes. The use of data augmentation, hyperparameter tuning, and ensembling ensured the model handled class imbalance effectively while maintaining inference time under 500ms, meeting the performance constraints.

Instructions to Run the Project
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/fashion-product-classification.git
cd fashion-product-classification
Install dependencies: Ensure you have the required packages installed:

bash
Copy code
pip install -r requirements.txt
Dataset Preparation:

Download the dataset from Kaggle and place it in the appropriate directory (data/).
Organize the dataset as follows:
kotlin
Copy code
data/
  train_images/
  test_images/
  train.csv
  test.csv
  category_attributes.parquet
  sample_submission.csv
Training the Model: To train the model, execute the following command:

bash
Copy code
python train.py --epochs 30 --batch_size 64 --model swin_transformer
Hyperparameters can be adjusted as per requirement.

Generating Predictions: After training, use the trained model to generate predictions for the test set:

bash
Copy code
python predict.py --model checkpoint/model.pth --output sample_submission.csv
Model Evaluation: Use the validation set to evaluate model performance:

bash
Copy code
python evaluate.py --model checkpoint/model.pth --data_dir data/val
Key Learnings
Data Augmentation: The use of advanced augmentation techniques like CutMix and MixUp significantly improved model generalization and performance, particularly for under-represented classes.
Hyperparameter Tuning: Tuning parameters like learning rate, dropout, and batch size using Optuna proved vital in improving model accuracy.
Model Optimization: Techniques like mixed precision training and ONNX conversion helped maintain fast inference times while ensuring accurate predictions.
Ensembling: Combining multiple models boosted robustness, ensuring the model performed well across a variety of fashion product categories.
Conclusion
This project showcases an end-to-end deep learning pipeline for multi-label classification of fashion product attributes. By tackling challenges such as class imbalance and missing data through advanced augmentation and ensembling strategies, the model achieved competitive results while adhering to strict inference time constraints.
