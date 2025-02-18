# Regression
 * Regression tasks involve predicting numerical values rather than binary outcomes. The process for regression is largely similar to binary classification, so please refer to the corresponding section for detailed steps. However, there's a key difference – the incorporation of ratings for the images.

To fine-tune a transformer model for regression on the dataset, follow these steps:

1. **Training Process**: Most of the commands for regression are akin to those used for binary classification, so you can refer to the binary_classification section for details, please do so.

2. **Rating Flag**: The distinction lies in an additional flag representing the ratings associated with the images.

3. **Fine-Tuning Command**: To fine-tune a transformer model for regression, execute the following command:

    ```bash
    accelerate launch --config_file on.yaml fine_tune_transformers_regression.py --data_dir Dataset/  --num_epochs 5 --save_steps 200 --batch_size 4 --model_path facebook/dinov2-large --output_dir dinov2_large_Funny_Range --LR 0.00001 --labels_path range_ratings.csv
    ```

    In this command:
    - `--data_dir`: Specifies the directory containing the dataset.
    - `--num_epochs`: Sets the number of training epochs.
    - `--save_steps`: Determines after how many steps model checkpoints should be saved.
    - `--batch_size`: Defines the batch size used during training.
    - `--model_path`: Specifies the pre-trained model to use.
    - `--output_dir`: Sets the directory for saving training outputs.
    - `--LR`: Defines the learning rate for training.
    - `--labels_path`: Points to the file containing ratings data (`range_ratings.csv` in this case).
    
    Note: The labels ratings file is assumed to be within the data directory.