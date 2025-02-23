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

The training process is divided into increments of 100 epochs, with updates at 20%, 40%, 60%, 80%, and 100% completion. Initially, at epoch 0, the loss is 125.69, which is expected since the model starts with random weights. As training progresses, the loss steadily decreases, reaching 3.47 at epoch 100, 4.32 at epoch 200, 5.85 at epoch 300, and 3.49 at epoch 400, indicating that the model is learning effectively. The estimated total training time is around 2 hours and 56 minutes, with each iteration taking approximately 19-21 seconds. 
![Training Diagram](/images/BERT_SS.png)

2)
3)

## TASK 2
1)
2)
3)

## TASK 3
1)
2)
3)

## TASK 4
1)
2)
3)