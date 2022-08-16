import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
import pandas as pd

classes = ('Egg', 'Fried Food', 'Meat', 'Rice', 'Seafood')

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pth")
model.eval()
model.to(DEVICE)


path = 'dataset/test/'
testList = os.listdir(path)
preds = []
for file in testList:
    img = Image.open(path + file).convert('RGB')
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    print('name:{},predicted:{}'.format(file, classes[pred.data.item()]))
    preds.append(pred)


path_list = list(map(lambda x: os.path.join(path,x), os.listdir(path)))
test_df = pd.DataFrame({'file':path_list})

# predictions_list = list(map(pred))

output_df = pd.DataFrame({'file':test_df.file.apply(lambda x:x.split('/')[2])})
output_df['predicted'] = pd.DataFrame(preds)


output_df.to_csv('output.csv', index = False)
