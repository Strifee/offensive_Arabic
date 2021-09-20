import pickle
from model import initialize_model, set_seed, train
from model import train_dataloader, val_dataloader


set_seed(42)    # Set seed for reproducibility
bert_classifier, optimizer, scheduler = initialize_model(epochs=3)
train(bert_classifier, train_dataloader, val_dataloader, epochs=3, evaluation=True)

filename = 'trained_model.sav'
pickle.dump(bert_classifier, open(filename, 'wb'))