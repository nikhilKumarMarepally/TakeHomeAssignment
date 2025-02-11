from transformers import BertModel, BertTokenizer
import torch

'''     
    For task1, we have to implement a sentence transformer, I chose pre-trained BERT model (works well for most the NLP tasks)
	The model processes input text, extracts token-level embeddings from BERT, and applies an adaptive average pooling layer to obtain fixed-size        sentence representations. The tokenizer converts input sentences into tensors with padding and truncation for uniform input processing. 
    Finally, the model generates sentence embeddings, which can be used for NLP tasks like similarity detection and classification.

'''
class SentenceTransformer(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(SentenceTransformer, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.pooling = torch.nn.AdaptiveAvgPool1d(1) #torch.nn.AdaptiveAvgPool1d(1) is used to convert variable-length token embeddings into a fixed-size sentence embedding.

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  
        embeddings = self.pooling(last_hidden_state.permute(0, 2, 1)).squeeze(2)  
        return embeddings

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = SentenceTransformer()

sentences = ["AI is transforming every Industry", "AI Agents are going to be the next big thing"]
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
embeddings = model(inputs['input_ids'], inputs['attention_mask'])


print("----TASK1 Embedddings----------")
for sentence, embeddings in zip(sentences, embeddings):
	print(f"Sentence: {sentence}, Sentence Embeddings: {embeddings}")
