# Running Scripts from this directory

This repository contains a collection of scripts that perform the benchmarks mentioned in the paper. This guide will help you get started with running these scripts on your local machine. Please follow the instructions below:

## Prerequisites

Before you begin, ensure you have the following installed:

1. **Git**: Make sure you have Git installed on your system. You can download it from [https://git-scm.com/](https://git-scm.com/).

2. **Programming Language**: Make sure you are using Python 3.10

3. **Dependencies**: Install using the requirements.txt file. 
Most important dependencies: [transformers](https://huggingface.co/transformers), [deepspeed](https://github.com/microsoft/deepspeed), [accelerate](https://huggingface.co/docs/accelerate/index), [PyTorch](https://pytorch.org/)

## Instructions

1. **Clone the Repository**

2. **Navigate to the Repository Directory**

3. **Running models with Huggingface transformers implementation**
    * When working with the models implemented in this repository, you can take advantage of the **Hugging Face Transformers** library, which hosts many state-of-the-art models. Here's how to get started:

    * **Multiple GPU Configuration**: To utilize multiple GPUs using the **DeepSpeed** and **Accelerate** configuration, you can either use a provided configuration file from the `scripts` folder that is set up for three GPUs on the same machine. Alternatively, you can create a config file using:
        ```bash
        accelerate config --config_file config_file.yaml
        ```
    * **Fine-Tuning a Model**: To fine-tune a model using the aforementioned configuration file and the **Accelerate** library, execute the following command:
        ```bash
        accelerate launch --config_file ds_config.yaml fine_tune_accel.py --data_dir /../HumorDataset/Dataset/  --num_epochs 5 --save_steps 200 --batch_size 4 --model_path facebook/dinov2-large --output_dir dinov2_large_Funny_Range --LR 0.00001 --weight_decay 0.001 --performance_log performance_logs
        ```
        In this command:
    - `--data_dir`: Specifies the directory containing the dataset.
    - `--num_epochs`: Sets the number of training epochs.
    - `--save_steps`: Determines after how many gradient steps model checkpoints should be saved.
    - `--batch_size`: Defines the batch size used during training.
    - `--model_path`: Specifies the path to the pre-trained model to use.
    - `--output_dir`: Sets the directory for saving training outputs.
    - `--LR`: Defines the learning rate for training.
    - `--weight_decay`: Specifies the weight decay value.
    - `--performance_log`: Indicates where performance logs will be stored.
    * **Performance Results**: The best test and validation accuracies will be stored in the directory specified by `outdir_dir/performance_log/performance.txt`. For example, above the location would be `dinov2_large_Funny/performance_logs/performance.txt`.
4.  **Running ConvNets from torchvision**
     * If you're interested in training convolutional neural networks (ConvNets) using torchvision for the task at hand, we offer scripts to assist you in this process.

     * **Fine-Tuning ConvNets**: To fine convnets for the task using a single machine, use:
     ```bash
        fine_tune.py --data_dir /../HumorDataset/Dataset/  --num_epochs 5 --save_steps 200 --batch_size 4 --model convnext_large --output_dir convnext_large output_file best_model.bin --LR 0.001 --weight_decay 0.001 --performance_log performance_logs --pretrained --train_full --cuda_visible_devices '0,1' --seed 32412432
     ``` 
        In this command:
    - `--data_dir`: Specifies the directory containing the dataset.
    - `--num_epochs`: Sets the number of training epochs.
    - `--save_steps`: Determines after how many steps model checkpoints should be saved.
    - `--batch_size`: Defines the batch size used during training.
    - `--model`: Specifies the ConvNet model to use (in this case, `convnext_large`). The choices are (`convnext_large`, `resnet152`, `inceptionv3`, `convnext_base`)
    - `--output_dir`: Sets the directory for storing outputs.
    - `output_file`: Specifies the name of the output file.
    - `best_model.bin`: Specifies the name of the best model file.
    - `--LR`: Defines the learning rate for training.
    - `--weight_decay`: Specifies the weight decay value.
    - `--performance_log`: Indicates where performance logs will be stored.
    - `--pretrained`: Uses a pretrained model.
    - `--train_full`: Trains the entire model.
    - `--cuda_visible_devices '0,1'`: Specifies which GPUs to use.
    - `--seed 32412432`: Sets the random seed for reproducibility.
     * **Model and Performance**: The trained model will be stored in `output_dir/output_file`, and performance metrics will be available in `output_dir/performance_logs/performance.txt`.

5. **Running other models**
    * In addition to models available through Hugging Face Transformers, we also provide an example of training a different model, specifically the **ViT-g/14** model used in **EVA-CLIP** and **BLIP2** implementations, on our dataset.

    * **Example with ViT-g/14**: We demonstrate how a model not available on Hugging Face can be trained using our dataset as an example.

    * The majority of the steps are similar, with the primary distinction being how the model is instantiated.
    * **Navigating to Directory**: Move to the `vitg_14_scripts` directory, where we've organized the scripts for this specific model.
    * **Creating DeepSpeed Configuration**: Create a DeepSpeed configuration file, following the same procedure as described in the previous steps.
    * **Fine-Tuning the Model**: To fine-tune the ViT-g/14 model with the above configuration file using Accelerate, run the following comand:
        ```bash
        accelerate launch --config_file ds_config.yaml fine_tune_eva_vit_g_14.py --data_dir /../HumorDataset/Dataset/  --num_epochs 5 --save_steps 200 --batch_size 4 --LR 0.00001 --weight_decay 0.001
        ```
        In this command:
    - `--data_dir`: Specifies the directory containing the dataset.
    - `--num_epochs`: Sets the number of training epochs.
    - `--save_steps`: Determines after how many steps model checkpoints should be saved.
    - `--batch_size`: Defines the batch size used during training.
    - `--LR`: Defines the learning rate for training.
    - `--weight_decay`: Specifies the weight decay value.

    * **Performance Results**: The best test and validation accuracies will be stored in the directory specified by `outdir_dir/performance_log/performance.txt`. By default, the location is `vit_g14/performance_logs/performance.txt`.

6. **Notebooks**
    * In addition to the provided scripts, we also offer a Jupyter notebook that demonstrates how to utilize the **Hugging Face Trainer** to train models for our specific task. This notebook is located in the `notebooks` folder. It showcases the process of fine-tuning a ViT_large model for our task using the Hugging Face ecosystem.