import code
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class cifar10(Dataset):
	def __init__(self,path, transform = None, is_train = True):
		self.path = path

		if is_train:
			main_path = os.path.join(self.path,"train")
		else:
			main_path = os.path.join(self.path,"test")

		file_names = os.listdir(main_path)

		self.data_path = [os.path.join(main_path,file_name) for file_name in file_names]
		if (transform is not None):
			self.transform = transform
		else:
			transform = transforms.Compose([transforms.ToTensor()])
			self.transform = transform
		#code.interact(local = dict(globals(),**locals()))

		self.label_map ={"airplane":0,"automobile":1,"bird":2,"cat":3,"deer":4,"dog":5,"frog":6,"horse":7,"ship":8,"truck":9}
		pass
	def __len__(self):
		return len(self.data_path)
		pass

	def __getitem__(self, index):
		image = Image.open(self.data_path[index])
		image = self.transform(image)
		label_name = self.data_path[index].split('_')[-1].split(".")[0]
		label = self.label_map[label_name]
		return image,label
		pass

"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = cifar10("cifar",transform = transform,is_train = True)
for i in range(10):
	image, label = dataset[i]
	code.interact(local = dict(globals(),**locals()))
"""
