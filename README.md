# A4_NLI
## TASK 1:
1) Implemented BIdirection Encoder Reperesentations from Transformers (BERT). Trained the model with following hyperparameters. 
## Model Hyperparameters

| Hyperparameter | Value |
|---------------|-------|
| **Batch Size** | 6 |
| **Max Mask** | 5 |
| **Max Length** | 1000 |
| **Number of Layers (n_layers)** | 6 |
| **Number of Heads (n_heads)** | 8 |
| **Embedding Size (d_model)** | 768 |
| **FeedForward Dimension (d_ff)** | 3072 (4 * 768) |
| **Key/Query/Value Dimension (d_k, d_v)** | 64 |
| **Number of Segments (n_segments)** | 2 |

The training process is divided into increments of 100 epochs, with updates at 20%, 40%, 60%, 80%, and 100% completion. Initially, at epoch 0, the loss is 125.69, which is expected since the model starts with random weights. As training progresses, the loss steadily decreases, reaching 3.47 at epoch 100, 4.32 at epoch 200, 5.85 at epoch 300, and 3.49 at epoch 400, indicating that the model is learning effectively. The estimated total training time is around 2 hours and 56 minutes, with each iteration taking approximately 19-21 seconds.. 

![Training Diagram](/images/BERT_SS.png)

2) The BookCorpus dataset is a large collection of free books used for training natural language processing (NLP) models. It contains over 11,000 books covering various genres and topics, making it a valuable resource for language modeling, text generation, and pretraining transformer-based models such as BERT and GPT. In the code, only the first 1% of the training split is loaded using the Hugging Face Datasets library.

3) Saved model as bert_model.pth. 
Note: Due to large size of model, it cannot be pushed to Github

## TASK 2
1) Used snli and mnli dataset from HuggingFace.
2) Trained the Sentence-BERT
![S-BERT](/images/S_BERT2.png)
The image shows the training log for Sentence-BERT (S-BERT), where the loss values are recorded for two epochs. In Epoch 1, the loss is 38.16, and by Epoch 2, it significantly drops to 17.57, indicating that the model is learning and improving sentence representations.

For training the model saved in Task 1 was used. Due to architecture there were few changes made in the training loop of S-BERT.

## TASK 3
1) 
| Model Type  | Dataset   | Accuracy | Precision | Recall | F1-Score |
|------------|----------|----------|-----------|--------|----------|
| Our Model  | Test Data | 0.15     | 0.598571  | 0.15   | 0.058643 |

2) The challenge I encoutered was to implement the saved model weights. Also GPU becomes out of memory so I wasn't able to use some more data and more epochs. 
So to improve the model, training should have more extended epochs with early stopping, and hyperparameters like learning rate and optimizer settings should be fine-tuned. Using different embedding strategies, such as the [CLS] token instead of mean pooling, may provide better feature representations. Furthermore, unfreezing some BERT layers for fine-tuning can enhance generalization. 

## TASK 4
WEB APP
![Web app](/images/SS2.png)

Demo:
https://drive.google.com/file/d/1szbn5QV8ISuJUrnjfqChwSbK-ITLuNeg/view?usp=sharing
