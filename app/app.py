import torch
from flask import Flask, render_template, request
from transformers import AutoTokenizer

# Import your BERT model definition
from model_arch import BERT  # Ensure model.py contains your BERT definition

# Define device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Change if needed

# Define model parameters (same as training)
n_layers = 12
n_heads = 12
d_model = 768
d_ff = 768 * 4
d_k = 64
n_segments = 2
vocab_size = 60305  # Match tokenizer vocab size
max_len = 256

# Initialize the BERT model
model = BERT(n_layers, n_heads, d_model, d_ff, d_k, n_segments, vocab_size, max_len, device)

# Define the classifier head (same as training)
class ClassifierHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(ClassifierHead, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


classifier_head = ClassifierHead(d_model * 3)  # Since we concatenate 3 vectors

# Load checkpoint
model_save_path = "model/sbert_nli_model_updated.pth"
checkpoint = torch.load(model_save_path, map_location=device)

model.load_state_dict(checkpoint["sbert_model_state_dict"])
classifier_head.load_state_dict(checkpoint["classifier_head_state_dict"])

# Move models to device
model.to(device)
classifier_head.to(device)

# Set to evaluation mode
model.eval()
classifier_head.eval()

app = Flask(__name__)

def mean_pooling(hidden_state, attention_mask):
    """Mean Pooling - Take the average of hidden states based on attention mask."""
    attention_mask = attention_mask.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
    
    # Ensure hidden_state has the expected dimension (batch_size, seq_len, hidden_dim)
    if len(hidden_state.shape) == 3:
        hidden_dim = hidden_state.shape[-1]
        mask_expanded = attention_mask.expand(-1, -1, hidden_dim)  # Match hidden_state dim
    else:
        raise ValueError(f"Unexpected hidden state shape: {hidden_state.shape}")

    sum_hidden = torch.sum(hidden_state * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # Avoid division by zero

    return sum_hidden / sum_mask  # Returns [batch_size, hidden_dim]


def predict_nli(premise, hypothesis):
    """Function to predict NLI (Entailment, Neutral, Contradiction)."""

    # Tokenize input
    inputs = tokenizer(
        premise, hypothesis, padding=True, truncation=True, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Create segment IDs
    segment_ids = torch.zeros_like(input_ids).to(device)

    # Extract last hidden states
    with torch.no_grad():
        u = model.get_last_hidden_state(input_ids, segment_ids)

    # Apply mean pooling
    u_mean_pool = mean_pooling(u, attention_mask)

    # Concatenate three embeddings as done during training
    x = torch.cat([u_mean_pool, u_mean_pool, torch.abs(u_mean_pool)], dim=-1)  # [batch_size, 2304]

    # Predict using classifier head
    x = classifier_head(x)  # [batch_size, 3]

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(x, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()

    # Mapping output to label
    label_map = {0: "Contradiction", 1: "Neutral", 2: "Entailment"}
    return label_map[predicted_label]


@app.route("/", methods=["GET", "POST"])
def home():
    """Home page with input form."""
    result = None
    if request.method == "POST":
        premise = request.form["premise"]
        hypothesis = request.form["hypothesis"]
        
        if premise and hypothesis:
            result = predict_nli(premise, hypothesis)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
