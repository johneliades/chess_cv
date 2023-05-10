import cv2
import numpy as np
import os

# Splitting the dataset found here 
# https://universe.roboflow.com/chess-project/2d-chessboard-and-chess-pieces

pieces_labels = ['B', 'K', 'N', 'P', 'Q', 'R', 'b_', 'board', 'k_', 'n_', 'p_', 'q_', 'r_']

image_path = "./images/"
label_path = "./labels/"
data  = "./data/"

name_counter = 0

for root, dirs, files in os.walk(image_path):
	for file in files:
		img = cv2.imread(os.path.join(root, file))
		dh, dw, _ = img.shape

		with open(label_path + file[:-4] + ".txt", "r") as txt_file:
			for line in txt_file:
				class_id, x_center, y_center, w, h = line.strip().split()
				x_center, y_center, w, h = float(x_center), float(y_center), float(w), float(h)
				x_center = round(x_center * dw)
				y_center = round(y_center * dh)
				w = round(w * dw)
				h = round(h * dh)
				x = round(x_center - w / 2)
				y = round(y_center - h / 2)
				piece = pieces_labels[int(class_id)]
				if(piece!="board"):
					cropped_object = img[y:y+h, x:x+w]
					cv2.imwrite(data + piece + "/" + str(name_counter) + ".jpg", cropped_object)
					name_counter += 1
