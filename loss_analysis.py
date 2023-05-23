import json
import os
import matplotlib.pyplot as plt

sda_log_file_path = 'outputs/paperspace/sda/logs/'
sda_log_files = os.listdir(sda_log_file_path)
sda_log_dictionaries = []
print("SDA log files: ")
for file in sda_log_files:
    print(file)

no_sda_log_file_path = 'outputs/paperspace/no_sda/logs/'
no_sda_log_files = os.listdir(no_sda_log_file_path)
no_sda_log_dictionaries = []

print("No SDA log files: ")
for file in no_sda_log_files:
    print(file)

for file in sda_log_files:
    with open((sda_log_file_path+file), 'r') as f:
    
        for line in f:
            log_entry = json.loads(line)            
            sda_log_dictionaries.append(log_entry)

sda_losses = []
sda_head_losses = []
for log_entry in sda_log_dictionaries:
    #print(log_entry)
    #check if type entry exist
    if 'type' in log_entry:
        # check if type is train-epoch
        if log_entry['type'] == 'train-epoch':
            sda_losses.append(log_entry['loss'])
            sda_head_losses.append(log_entry['head_losses'])


for file in no_sda_log_files:
    with open((no_sda_log_file_path+file), 'r') as f:
    
        for line in f:
            log_entry = json.loads(line)            
            no_sda_log_dictionaries.append(log_entry)

no_sda_losses = []
no_sda_head_losses = []
for log_entry in no_sda_log_dictionaries:
    #print(log_entry)
    #check if type entry exist
    if 'type' in log_entry:
        # check if type is train-epoch
        if log_entry['type'] == 'train-epoch':
            no_sda_losses.append(log_entry['loss'])
            no_sda_head_losses.append(log_entry['head_losses'])



sda_epochs = [x for x in range(200, len(sda_losses)+200)]
no_sda_epochs = [x for x in range(200, len(no_sda_losses)+200)]

print(len(sda_losses))
plt.xlim(200, len(sda_losses)+200)
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses per Epoch')
plt.plot(sda_epochs, sda_losses, label='with sda')
plt.plot(no_sda_epochs, no_sda_losses, label='without sda')
#show label
plt.legend()
plt.show()