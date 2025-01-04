import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
from lime.lime_text import LimeTextExplainer


# Path for input dataset, bert model, output model, output figures and train epoch
data_path = "/data/liuyu/final/Tweets.csv"
bert_path = "/data/liuyu/final/bert_model/"
model_path ="/data/liuyu/final/output_model/sentiment_classifier.pth"
figure_dir = "figure"
num_epochs = 10


# Data Analysis and Visualization
def data_analysis(df):
    print("Dataset Overview:")
    print(df.head())
    print("\nDataset Information:")
    print(df.info())
    print("\nSentiment Distribution:")
    print(df['airline_sentiment'].value_counts())

    # Plot sentiment distribution
    sns.countplot(data=df, x='airline_sentiment', order=df['airline_sentiment'].value_counts().index)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig(os.path.join(figure_dir, 'sentiment_distribution.png'))
    plt.close()

    # Plot airline sentiment distribution
    sns.countplot(data=df, x='airline', hue='airline_sentiment', order=df['airline'].value_counts().index)
    plt.title('Airline Sentiment Distribution')
    plt.xlabel('Airline')
    plt.ylabel('Count')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(figure_dir, 'airline_sentiment_distribution.png'))
    plt.close()


    plt.figure(figsize=(6,6))
    plt.pie(x=df['airline_sentiment'].value_counts(),
            labels=df['airline_sentiment'].value_counts().index,
            autopct="%f1%%")
    plt.title('Airline Sentiment Distribution')
    plt.savefig(os.path.join(figure_dir, 'PieChart_airline_sentiment_distribution.png'))
    plt.close()


# Data Preprocessing
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text


# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
    

# Model
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        output = self.drop(pooled_output)
        return self.out(output)


# Single train preocess
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


# Evaluate model
def eval_model(model, data_loader, criterion, device):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    return accuracy, f1, precision, recall, np.mean(losses), all_preds, all_labels


# Generate probability predictions for LIME explanations
def predict_proba(texts):
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    encodings.pop('token_type_ids', None)
    encodings = {key: val.to(model.device) for key, val in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


# Function to save LIME explanations as images
def save_lime_explanation(text, actual, pred, index):
    """
    Generate and save LIME explanation for a single text.
    Args:
        text: Input text.
        actual: True label.
        pred: Predicted label.
        index: Index for saving the explanation.
    """
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=10,
        labels=[pred]
    )
    # Save explanation as image
    fig = exp.as_pyplot_figure(label=pred)
    plt.title(f"Sample {index + 1}: LIME Explanation\nActual: {class_names[actual]}, Predicted: {class_names[pred]}")
    plt.savefig(f'figure/lime_explanation_{index + 1}.png')
    plt.close()


# Function to generate and save LIME explanations for multiple samples
def save_multiple_lime_explanations(texts, labels, preds, num_samples=3):
    for i in range(num_samples):
        idx = np.random.randint(0, len(texts))
        text = texts.iloc[idx]
        actual = labels.iloc[idx]
        pred = preds[idx]
        print(f"\nSample {i+1}")
        print(f"Original Tweet: {df.iloc[X_test.index[idx]]['text']}")
        print(f"Cleaned Tweet: {text}")
        print(f"Actual Sentiment: {class_names[actual]}")
        print(f"Predicted Sentiment: {class_names[pred]}")
        save_lime_explanation(text, actual, pred, i)



# Create figure dir
os.makedirs(figure_dir, exist_ok=True)
# Load dataset
df = pd.read_csv(data_path)
data_analysis(df)
# Preprocess
df['text'] = df['text'].apply(preprocess_text)

# Encode labels
label_encoder = LabelEncoder()
df['sentiment_label'] = label_encoder.fit_transform(df['airline_sentiment'])
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment_label'], test_size=0.2, random_state=42
)

# Tokenization
tokenizer = BertTokenizer.from_pretrained(bert_path)

# Datasets and Dataloaders
train_dataset = SentimentDataset(X_train, y_train, tokenizer)
test_dataset = SentimentDataset(X_test, y_test, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Training and Evaluation
model = SentimentClassifier(n_classes=len(label_encoder.classes_))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.device = device
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Train
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
val_f1, val_precision, val_recall = [], [], []

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_acc, f1, precision, recall, val_loss, all_preds, all_labels = eval_model(model, test_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc.item())
    val_accuracies.append(val_acc)
    val_f1.append(f1)
    val_precision.append(precision)
    val_recall.append(recall)
    final_preds, final_labels = all_preds, all_labels

    print(f'Train loss: {train_loss}, Train accuracy: {train_acc}')
    print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}, F1: {f1}, Precision: {precision}, Recall: {recall}')

# Save the model
torch.save(model.state_dict(), model_path)

# Plot and save loss graph
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(figure_dir, 'loss_graph.png'))
plt.close()

# Plot and save accuracy graph
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(figure_dir, 'accuracy_graph.png'))
plt.close()

# Plot Confusion matrix
cm = confusion_matrix(final_labels, final_preds, labels=np.arange(len(label_encoder.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix')
plt.savefig('figure/confusion_matrix.png')
plt.close()

# Initialize LIME explainer
class_names = label_encoder.classes_
explainer = LimeTextExplainer(class_names=class_names)

# Generate LIME explanations for random samples after evaluation
final_preds = np.array(final_preds)  # Ensure preds is in numpy array format
save_multiple_lime_explanations(X_test, y_test, final_preds, num_samples=5)

