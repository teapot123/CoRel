# training/validation/testing function of relation classifer 

import torch
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F


def train_func(sub_train_, model, BATCH_SIZE, optimizer, scheduler, generate_batch):
    train_loss = 0
    train_acc = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
    for i, (input_ids, entity1_mask, entity2_mask, attention_mask, labels) in enumerate(data):
        optimizer.zero_grad()
        input_ids, entity1_mask, entity2_mask, attention_mask, labels = input_ids.to(device), entity1_mask.float().to(device), entity2_mask.float().to(device), attention_mask.to(device), labels.to(device)
        output, loss = model(input_ids, attention_mask=attention_mask, entity1_mask=entity1_mask, entity2_mask=entity2_mask, labels=labels)
        train_acc += (output.argmax(1) == labels).sum().item()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    scheduler.step()
    
    return train_loss/len(sub_train_), train_acc/len(sub_train_)

def valid_func(data_, model, BATCH_SIZE, generate_batch):
    valid_loss = 0
    valid_acc = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for i, (input_ids, entity1_mask, entity2_mask, attention_mask, labels) in enumerate(data):
        with torch.no_grad():
            input_ids, entity1_mask, entity2_mask, attention_mask, labels = input_ids.to(device), entity1_mask.float().to(device), entity2_mask.float().to(device), attention_mask.to(device), labels.to(device)
            output, loss = model(input_ids, attention_mask=attention_mask, entity1_mask=entity1_mask, entity2_mask=entity2_mask, labels=labels)
            valid_acc += (output.argmax(1) == labels).sum().item()
            valid_loss += loss.item()
            

    return valid_loss / len(data_), valid_acc/len(data_)

def test(data_, model, BATCH_SIZE, generate_test_batch):
    logits = []
    entities = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_test_batch)
    for i, (input_ids, entity1_mask, entity2_mask, attention_mask, entity) in enumerate(data):
        entities.extend(entity)
        input_ids, entity1_mask, entity2_mask, attention_mask = input_ids.to(device), entity1_mask.float().to(device), entity2_mask.float().to(device), attention_mask.to(device)
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask, entity1_mask=entity1_mask, entity2_mask=entity2_mask)
            logits.extend(F.softmax(output, dim=1).cpu().numpy())
    return logits, entities 
        