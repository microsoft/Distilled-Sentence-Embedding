# Scalable Attentive Sentence-Pair Modeling via Distilled Sentence Embedding (AAAI 2020) - PyTorch Implementation
This repository contains a PyTorch implementation of the Distilled Sentence Embedding (DSE) method, presented in this [paper](https://arxiv.org/abs/1908.05161). The code is published to allow reproduction of the model.

## Method Description
DSE distills knowledge from a finetuned state-of-the-art transformer model (BERT) to to create high quality sentence embeddings. For a complete description, as well as 
 implementation details and hyperparameters, please refer to the paper. 


## Usage
Follow the instructions below in order to run the training procedure of the Distilled Sentence Embedding (DSE) method. For all of the below python scripts you can 
also run them with the -h run parameter to get more information regarding the run parameters they accept.

### 1. Install Requirements
```
pip install -r requirements.txt
```

### 2. Download GLUE Datasets
Run the download_glue_data.py script to download the GLUE datasets.
```
python download_glue_data.py
```

### 3. Finetune BERT on a Specific Task
Finetune a standard BERT model on a specific task (e.g. MRPC, MNLI, etc.). Below is an example for the MRPC dataset.
```
python run_classifier.py \
--gpu_id 0 \
--bert_model bert-large-uncased-whole-word-masking \
--task_name mrpc \
--do_train \
--do_eval \
--do_lower_case \
--data_dir glue_data/MRPC \
--max_seq_length 128 \
--train_batch_size 32 \
--gradient_accumulation_steps 2 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--output_dir outputs/large_uncased_finetuned_mrpc \
--overwrite_output_dir \
--no_parallel
```

Note: To use the AllNLI dataset (a combination of the MNLI and SNLI datasets) with the given code, you can simply create a folder where the downloaded GLUE datasets are and 
copy the MNLI and SNLI datasets into it.

### 4. Create the Logits from the Finetuned BERT
Execute the following command to create the logits which will be used for distillation from the BERT model we finetuned. The run parameters match those from the previous step and 
need to be adapted according to the directories and task used.
```
python run_distillation_logits_creator.py \
--gpu_id 0 \
--task_name mrpc \
--data_dir glue_data/MRPC \
--do_lower_case \
--train_features_path data/MRPC/train_bert-large-uncased-whole-word-masking_128_mrpc \
--bert_checkpoint_dir outputs/large_uncased_finetuned_mrpc
```

### 5. Train the DSE Model using the Finetuned BERT Logits
Train the DSE model using the extracted logits. Again, the run parameters match those from the previous step and 
need to be adapted according to the directories and task used.
```
python siamese_bert_train_runner.py \
--gpu_id 0 \
--task_name mrpc \
--data_dir glue_data/MRPC \
--distillation_logits_path outputs/logits/large_uncased_finetuned_mrpc_logits.pt \
--do_lower_case \
--file_log \
--store_checkpoints
```

Note: To store checkpoints for the model make sure that the --store_checkpoints flag is passed as shown above.

### 6. Loading of the Trained DSE Model
During training checkpoints of the Trainer object which contains the model will be saved. You can load these checkpoints and extract from the the model 
itself.

## Acknowledgements
- We based our implementation on the BERT pretrained model from the [HuggingFace transformers repository](https://github.com/huggingface/transformers).

- The script for downloading the GLUE datasets is taken from [here](https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py).

## Reference
If you find this code useful, please cite the following paper:
```
@article{barkan2019scalable,
  title={Scalable Attentive Sentence-Pair Modeling via Distilled Sentence Embedding},
  author={Barkan, Oren and Razin, Noam and Malkiel, Itzik and Katz, Ori and Caciularu, Avi and Koenigstein, Noam},
  journal={arXiv preprint arXiv:1908.05161},
  year={2019}
}
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.