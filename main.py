# -*- coding: utf-8 -*-
import os
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms



from google.colab import drive
drive.mount('/content/drive')

dataset_path = "YOUR_DATASET_PATH"

image_files = []
text_files = []

for file_name in os.listdir(dataset_path):
    if file_name.endswith(".png"):
        image_files.append(file_name)
    elif file_name.endswith(".txt"):
        text_files.append(file_name)

if len(image_files) != len(text_files):
    print("match error!")
    exit()
image_text_mapping = {}
for text_file in text_files:
    with open(os.path.join(dataset_path, text_file), "r") as f:
        image_text_mapping[text_file.replace(".txt", ".png")] = f.read().strip()

min = 100
for image_file, text_content in image_text_mapping.items():
    #print("image file:", image_file)
    #print("text content:", text_content)
    #print(len(text_content.split(",")))
    if len(text_content.split(",")) < min:
      min = len(text_content.split(","))
    #print("-----------------------------------------")
    
print(min)



image_text_pairs = list(image_text_mapping.items())

random.shuffle(image_text_pairs)

train_size = int(0.7 * len(image_text_pairs))
val_size = int(0.15 * len(image_text_pairs))

train_set = image_text_pairs[:train_size]
val_set = image_text_pairs[train_size:train_size+val_size]
test_set = image_text_pairs[train_size+val_size:]

print("train set length:", len(train_set))
print("val set length:", len(val_set))
print("test set length:", len(test_set))

dataset = image_text_pairs

# i dont want split data

train_set = dataset



def load_image(image_file):
    image_path = 'YOUR_DATASET_PATH/' + image_file
    image = Image.open(image_path)
    return image


label_set = []

for text_file in text_files:
    with open(os.path.join(dataset_path, text_file), "r") as f:
        labels = f.read().strip().split(", ")
        #print(labels)
        for label in labels:
          #print(label)
          if label not in label_set:
            label_set.append(label)

        #label_set.update(labels)

num_labels = len(label_set)
print("different labels:", num_labels)
your_num_classes = num_labels

label_index_map = {

}

label_set = list(label_set)
for i in range(len(label_set)):
  label_index_map[label_set[i]] = i

print(label_index_map)

for i in train_set:
  print(i)

def text_encoder(labels, label_index_map):
  tag_list = labels.split(", ")[1:]
  #print(tag_list)

  tag_indices = [label_index_map[tag] for tag in tag_list]

  selected_tags = tag_list[:24]

  selected_tag_indices = [label_index_map[tag] for tag in selected_tags]

  #print(selected_tags)
  return selected_tag_indices



class CustomDataset(Dataset):
    def __init__(self, data, data_dir, transform=None, max_tags=20, label_index_map={}):
        self.data = data
        self.data_dir = data_dir
        self.transform = transform
        self.max_tags = max_tags
        self.label_index_map = label_index_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, labels = self.data[idx]
        img = Image.open(self.data_dir + "/" + img_name)

        if self.transform:
            img = self.transform(img)

        #print(type(labels), labels)
        tag_tensor = torch.tensor(text_encoder(labels, self.label_index_map))

       # print(type(tag_tensor), tag_tensor.values)

        return img, tag_tensor

data_dir = dataset_path



transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    transforms.Resize((48, 24)),  
    transforms.ToTensor(),           
])

custom_dataset = CustomDataset(train_set, data_dir, transform=transform, label_index_map=label_index_map)

print(label_index_map)

data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=True)


for images, labels in data_loader:
  print(images.shape)
  print(labels.shape)




def decode_result(draw=None):

  #1024, 1536
  if draw == None:
    tensor = torch.randn(1, 3, 48, 24)
  else:
    tensor = draw
    print(type(tensor))


  image = tensor.squeeze().permute(1, 2, 0)

  image = (image - image.min()) / (image.max() - image.min())

  #tensor.detach().numpy()
  plt.imshow(image.detach().numpy())

  plt.axis('off')
  plt.show()

print(images.size())
decode_result(images)


class NoraV1(nn.Module):
    def __init__(self):
        super(NoraV1, self).__init__()
        self.fc = nn.Linear(24, 3 * 48 * 24)  


    def forward(self, x):
        x = x.view(-1, 24)
        x = self.fc(x)
        x = x.view(-1, 3, 48, 24)
        return x

model = NoraV1()

input_tensor = torch.randn(1, 24)

output_tensor = model(input_tensor)

print("Output tensor shape: ", output_tensor.size())



criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 50 
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in data_loader:
        optimizer.zero_grad()
 
        outputs = model(labels.float())

        loss = criterion(outputs, labels.float())

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(data_loader)}")

print("")

def main():
  generationString = "noraV1, 1girl, solo, long hair, looking at viewer, brown hair, shirt, black hair, long sleeves, brown eyes, standing, full body, white shirt, outdoors, parted lips, shoes, shorts, lips, sleeves past wrists, short shorts, bare legs, sandals, denim, clothes writing, blue shorts, denim shorts, cutoffs, torn shorts"
  encoded_input = text_encoder(generationString, label_index_map)
  input = torch.tensor(encoded_input).float()
  
  output = model(input)
  output_numpy = output.detach().numpy()
  
  print(output_numpy)
  decode_result(output)
