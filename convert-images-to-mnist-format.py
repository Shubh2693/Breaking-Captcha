import os
from PIL import Image
from array import *
from random import shuffle

# images are 50 by 60 in size
# Load from and save to
def gendata():

	# output_file = 'train_data.csv'
	output_file = 'test_data.csv'
	csv_delimiter = ','
	ff = open(output_file,"a")

	target = './test_extracted_letter_images'

	FileList = []
	
	for folder in os.listdir('./test_extracted_letter_images/'):
		for filename in os.listdir('./test_extracted_letter_images/'+folder+'/'):
			if filename.endswith(".png"):
				FileList.append(os.path.join(target + '/' + folder + '/',str(filename)))

	shuffle(FileList) # Usefull for further segmenting the validation set

	column_headers = []
	for i in range(1,3001):
		column_headers.append("P" + str(i))
		# print("Pixel" + str(i))
	column_headers.append("Label")

	ff.write(csv_delimiter.join(column_headers) + "\n")

	for filename in FileList:
		data_image = []
		try:
			Im = Image.open(filename)
		except:
			print("Skipping Malformed png")
			continue

		pixel = Im.load()

		width, height = Im.size

		for x in range(0,width):
			for y in range(0,height):
				data_image.append(pixel[x,y])

		# data_im2 = map(lambda x:str(x), data_image)
		data_im2 = [str(i) for i in data_image]
		data_im2.append(filename.split('/')[2])

		print(len(data_image))
		# ff.write(csv_delimiter.join(map(lambda x:str(x), data_image)) + filename.split('/')[2] + "\n")
		ff.write(csv_delimiter.join(data_im2) + "\n")
		Im.close()

	ff.close()



if __name__ == '__main__':
	gendata()
