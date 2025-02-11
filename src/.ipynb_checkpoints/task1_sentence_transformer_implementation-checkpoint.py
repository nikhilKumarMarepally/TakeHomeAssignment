from transformers import BertModel, BertTokenizer
import torch


'''     
	This script defines a SentenceTransformer model using a pre-trained BERT model to generate sentence embeddings. 
	The model processes input text, extracts token-level embeddings from BERT, and applies an adaptive average pooling layer to obtain fixed-size sentence representations. The tokenizer converts input sentences into tensors with padding and truncation for uniform input processing. Finally, the model generates sentence embeddings, which can be used for NLP tasks like similarity detection and classification.'''
class SentenceTransformer(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(SentenceTransformer, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.pooling = torch.nn.AdaptiveAvgPool1d(1) 

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

print("Sentence Embeddings:", embeddings)
