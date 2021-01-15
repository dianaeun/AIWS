import code,os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
class cifar10(Dataset):
	def __init__(self, path, is_train = True, transform = None):
		self.classes = {'airplane':0, 'car':1, 'bird':2, 'cat':3,'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
		self.train_path  = os.path.join(path,"train")
		self.test_path  = os.path.join(path,"test")

		if (is_train == True):
			self.main_path = self.train_path  = os.path.join(path,"train")
		else:
			self.main_path  = os.path.join(path,"test")

		self.data = [os.path.join(self.main_path,item) for item in os.listdir(self.main_path)]
		
		self.tranform = transform

		pass

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		image = Image.open(self.data[0])
		#code.interact(local = dict(globals(),**locals()))
		if self.tranform is not None:
			image = self.tranform(image)
		label_name = self.data[0].split('_')[-1].split(".")[0]
		label = self.classes[label_name]

		return image, label
		pass
"""
if __name__ == "__main__":
	dataset = cifar10("cifar")
	dataset[0]

"""