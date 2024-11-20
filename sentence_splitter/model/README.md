---
library_name: transformers
license: mit
base_model: FacebookAI/xlm-roberta-base
tags:
- generated_from_trainer
metrics:
- precision
- recall
- f1
- accuracy
model-index:
- name: xlmr_base2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# xlmr_base2

This model is a fine-tuned version of [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0334
- Precision: 0.8981
- Recall: 0.9242
- F1: 0.9109
- Accuracy: 0.9921

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 24
- eval_batch_size: 128
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 10.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:------:|:----:|:---------------:|:---------:|:------:|:------:|:--------:|
| 0.0467        | 0.3521 | 50   | 0.0301          | 0.8723    | 0.9070 | 0.8893 | 0.9889   |
| 0.0354        | 0.7042 | 100  | 0.0275          | 0.8875    | 0.9008 | 0.8941 | 0.9903   |
| 0.0271        | 1.0563 | 150  | 0.0248          | 0.8816    | 0.9072 | 0.8943 | 0.9909   |
| 0.0292        | 1.4085 | 200  | 0.0261          | 0.9019    | 0.8892 | 0.8955 | 0.9903   |
| 0.024         | 1.7606 | 250  | 0.0213          | 0.8648    | 0.9233 | 0.8931 | 0.9908   |
| 0.0132        | 2.1127 | 300  | 0.0244          | 0.8645    | 0.9211 | 0.8919 | 0.9909   |
| 0.0216        | 2.4648 | 350  | 0.0222          | 0.8593    | 0.9281 | 0.8924 | 0.9909   |
| 0.0213        | 2.8169 | 400  | 0.0226          | 0.8646    | 0.9273 | 0.8948 | 0.9913   |
| 0.0162        | 3.1690 | 450  | 0.0236          | 0.8800    | 0.9137 | 0.8965 | 0.9913   |
| 0.0165        | 3.5211 | 500  | 0.0213          | 0.9065    | 0.9101 | 0.9083 | 0.9926   |
| 0.0145        | 3.8732 | 550  | 0.0264          | 0.8588    | 0.9261 | 0.8912 | 0.9908   |
| 0.0139        | 4.2254 | 600  | 0.0223          | 0.8926    | 0.9230 | 0.9076 | 0.9919   |
| 0.0129        | 4.5775 | 650  | 0.0235          | 0.8835    | 0.9256 | 0.9040 | 0.9916   |
| 0.0093        | 4.9296 | 700  | 0.0231          | 0.8972    | 0.9129 | 0.9050 | 0.9917   |
| 0.008         | 5.2817 | 750  | 0.0283          | 0.8819    | 0.9264 | 0.9036 | 0.9917   |
| 0.0066        | 5.6338 | 800  | 0.0274          | 0.8927    | 0.9194 | 0.9058 | 0.9919   |
| 0.0083        | 5.9859 | 850  | 0.0260          | 0.8862    | 0.9225 | 0.9040 | 0.9917   |
| 0.0062        | 6.3380 | 900  | 0.0265          | 0.9024    | 0.9228 | 0.9125 | 0.9921   |
| 0.0063        | 6.6901 | 950  | 0.0288          | 0.8888    | 0.9216 | 0.9049 | 0.9918   |
| 0.0084        | 7.0423 | 1000 | 0.0262          | 0.8934    | 0.9213 | 0.9071 | 0.9918   |
| 0.0051        | 7.3944 | 1050 | 0.0303          | 0.8955    | 0.9154 | 0.9053 | 0.9919   |
| 0.0028        | 7.7465 | 1100 | 0.0318          | 0.8951    | 0.9259 | 0.9102 | 0.9919   |
| 0.0047        | 8.0986 | 1150 | 0.0300          | 0.8982    | 0.9250 | 0.9114 | 0.9922   |
| 0.0042        | 8.4507 | 1200 | 0.0321          | 0.8982    | 0.9259 | 0.9118 | 0.9920   |
| 0.0032        | 8.8028 | 1250 | 0.0325          | 0.8930    | 0.9228 | 0.9077 | 0.9920   |
| 0.0034        | 9.1549 | 1300 | 0.0351          | 0.8908    | 0.9244 | 0.9073 | 0.9918   |
| 0.0025        | 9.5070 | 1350 | 0.0341          | 0.8966    | 0.9264 | 0.9113 | 0.9921   |
| 0.002         | 9.8592 | 1400 | 0.0334          | 0.8985    | 0.9239 | 0.9110 | 0.9921   |


### Framework versions

- Transformers 4.47.0.dev0
- Pytorch 2.5.1+cu121
- Datasets 3.1.0
- Tokenizers 0.20.3
