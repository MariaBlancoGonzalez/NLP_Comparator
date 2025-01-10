import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.data import RandomSampler

import numpy as np
from time import perf_counter
import core.config as config
from tqdm.auto import trange
from dataframe import Dataframe

train = pd.read_csv(config.TRAIN, sep=",")
test = pd.read_csv(config.TEST, sep=",")
val = pd.read_csv(config.VALID, sep=",")

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
epochs = 50

model = BertForSequenceClassification.from_pretrained('bert-base-cased') # Pre-trained model
optimizer = AdamW(model.parameters(), lr=1e-5) # Optimization function
loss_fn = torch.nn.CrossEntropyLoss() # Loss function

label_to_id = {
            lab: i for i, lab in enumerate(train["type"].unique())
        }
    
#"source_field": "source",
train_dataset = Dataframe(train, label_to_id, text_field="description", label_field="type")
train_sampler = RandomSampler(train_dataset)
test_dataset = Dataframe(test, label_to_id, text_field="description", label_field="type")
test_sampler = RandomSampler(test_dataset)
val_dataset = Dataframe(val, label_to_id, text_field="description", label_field="type")
val_sampler = RandomSampler(val_dataset)


train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=16)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=16)
        
device = "cuda"
criterion = torch.nn.CrossEntropyLoss()

def flat_accuracy(self, preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

stats = []

for epoch in trange(epochs, desc="Epochs"):

    # -------------------- #
    # ------- Train ------ #
    # -------------------- #

    start = perf_counter()
    print("\n\n")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Training...')
    total_epoch_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        text, labels = batch
        text = tokenizer(text, padding='longest', truncation=True,  return_tensors="pt")

        inputs = {k: v.to(device) for k, v in text.items()}

        labels = labels.to(device)
                    
        optimizer.zero_grad()

        # -- Use model -- #
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    

        loss = criterion(outputs, labels)
        loss.backward()

   
        optimizer.step()
            #scheduler.step()

        total_epoch_loss += loss.item()

    avg_train_loss = total_epoch_loss / len(train_dataloader) 

    end = perf_counter()
    training_time = end - start
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # -------------------- #
    # ---- Validation ---- #
    # -------------------- #

    start_val = perf_counter()
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Evaluation...')
            
    total_eval_accuracy = 0
    total_eval_loss = 0
    model.eval()
    for batch in val_dataloader:
       
        text, labels = batch
        text = tokenizer(text, padding='longest', truncation=True,  return_tensors="pt")
        inputs = {k: v.to(device) for k, v in text.items()}

        
        labels = labels.to(device)

        with torch.no_grad():        

            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])            
            loss = criterion(logits, labels)

        # Accumulate the validation loss.
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)
                

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(val_dataloader)
            
    # Measure how long the validation run took.
    end_val = perf_counter()
    validation_time = end_val - start_val
            
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))
    stats.append(
                    {
                    'epoch': epoch + 1,
                    'Training Loss': avg_train_loss,
                    'Training Time': training_time,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy*10,
                    'Valid. Time': validation_time
                    }
                )
print("")
print("Training complete!")
print(avg_train_loss, avg_val_accuracy)
