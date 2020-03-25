# Scalable Attentive Sentence-Pair Modeling via Distilled Sentence Embedding
PyTorch implementation for the [Scalable Attentive Sentence-Pair Modeling via Distilled Sentence Embedding](https://arxiv.org/abs/1908.05161) (AAAI 2020) paper.

## Method Description
Distilled Sentence Embedding (DSE) distills knowledge from a finetuned state-of-the-art transformer model (BERT) to create high quality sentence embeddings. For a complete description, as well as implementation details and hyperparameters, please refer to the paper. 


## Usage
Follow the instructions below in order to run the training procedure of the Distilled Sentence Embedding (DSE) method. The python scripts below can be run with the -h parameter to get more information.

### 1. Install Requirements
Tested with Python 3.6+.
```
pip install -r requirements.txt
```

### 2. Download GLUE Datasets
Run the download_glue_data.py script to download the GLUE datasets.
```
python download_glue_data.py
```

### 3. Finetune BERT on a Specific Task
Finetune a standard BERT model on a specific task (e.g., MRPC, MNLI, etc.). Below is an example for the MRPC dataset.
```
python finetune_bert.py \
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

Note: For our code to work with the AllNLI dataset (a combination of the MNLI and SNLI datasets), you simply need to create a folder where the downloaded GLUE datasets are and copy the MNLI and SNLI datasets into it.

### 4. Create Logits for Distillation from the Finetuned BERT
Execute the following command to create the logits which will be used for the distillation training objective. Note that the bert_checkpoint_dir parameter has to match the output_dir from the previous command.
```
python run_distillation_logits_creator.py \
--task_name mrpc \
--data_dir glue_data/MRPC \
--do_lower_case \
--train_features_path glue_data/MRPC/train_bert-large-uncased-whole-word-masking_128_mrpc \
--bert_checkpoint_dir outputs/large_uncased_finetuned_mrpc
```

### 5. Train the DSE Model using the Finetuned BERT Logits
Train the DSE model using the extracted logits. Notice that the distillation_logits_path parameter needs to be changed according to the task.
```
python dse_train_runner.py \
--task_name mrpc \
--data_dir glue_data/MRPC \
--distillation_logits_path outputs/logits/large_uncased_finetuned_mrpc_logits.pt \
--do_lower_case \
--file_log \
--epochs 8 \
--store_checkpoints \
--fc_dims 512 1
```

__Important Notes:__ 
- To store checkpoints for the model make sure that the --store_checkpoints flag is passed as shown above.
- The fc_dims parameter accepts a list of space separated integers, and is the dimensions of the fully connected classifier that is put on top of the extracted features from the Siamese DSE model. The output dimension (in this case 1) needs to be changed according to the wanted output dimensionality. For example, for the MNLI dataset the fc_dims parameter should be 512 3 since it is a 3 class classification task.

### 6. Loading the Trained DSE Model
During training, checkpoints of the Trainer object which contains the model will be saved. You can load these checkpoints and extract the model state dictionary from them. Then you can load the state into a created DSESiameseClassifier model. The load_dse_checkpoint_example.py script contains an example of how to do that.

To load the model trained with the example commands above  you can use:
```
python load_dse_checkpoint_example.py \
--task_name mrpc \
--trainer_checkpoint <path_to_saved_checkpoint> \
--do_lower_case \
--fc_dims 512 1
```

## Acknowledgments
- We based our implementation on the BERT pretrained model from the [HuggingFace transformers repository](https://github.com/huggingface/transformers).

- The script for downloading the GLUE datasets is taken from [here](https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py).

## Reference
If you find this code useful, please cite the following paper:
```
@article{barkan2019scalable,
  title={Scalable Attentive Sentence-Pair Modeling via Distilled Sentence Embedding},
  author={Barkan, Oren and Razin, Noam and Malkiel, Itzik and Katz, Ori and Caciularu, Avi and Koenigstein, Noam},
  journal={AAAI},
  year={2020}
}
```
