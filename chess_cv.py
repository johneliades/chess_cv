import cv2
import numpy as np
import itertools
import chess
import chess.engine
import os
from url_normalize import url_normalize
from PIL import ImageGrab
import win32gui
import numpy as np
import cv2
import time
# import pytesseract
import sys
import pyautogui as pg
import random

piece_names = {
	'bk': 'k',
	'bq': 'q',
	'br': 'r',
	'bb': 'b',
	'bn': 'n',
	'bp': 'p',
	'wn': 'N',
	'wp': 'P',
	'wk': 'K',
	'wq': 'Q',
	'wr': 'R',
	'wb': 'B'
}

def load_templates():
	# Load all images in the folder
	template_images = []
	template_names = []
	
	# Path to the folder containing the images
	paths = ["./chess_pieces/chess", "./chess_pieces/lichess"]
	for folder_path in paths:
		for filename in os.listdir(folder_path):
			if filename.endswith(".png"):
				# Define the color to replace transparent pixels with (in this case, white)
				color = (255, 255, 255)

				# Load the image
				img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_UNCHANGED)

				# Split the image into color channels
				b, g, r, a = cv2.split(img)

				# Create a mask for the transparent pixels
				mask = (a == 0)

				# Replace the transparent pixels with the desired color
				b[mask] = color[0]
				g[mask] = color[1]
				r[mask] = color[2]

				# Merge the color channels back into an image without alpha channel
				img = cv2.merge((b,g,r))

				template_images.append(img)
				template_names.append(filename[:-4]) # Remove the '.jpg' extension

	return template_images, template_names

def find_chessboard(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Apply a threshold to the image to segment the chessboard
	_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

	# Find the contours in the thresholded image
	contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) == 0:
		print("No chessboard contour found.")
		exit()

	# Find the contour with the largest area (assumed to be the chessboard)
	chessboard_contour = max(contours, key=cv2.contourArea)
	x, y, w, h = cv2.boundingRect(chessboard_contour)
	cropped_img = img[y-1:y+h+1, x-1:x+w+1]
	
	# cv2.imshow("Cropped board", cropped_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# Finds the center of each square and save it in square_to_coords
	
	square_to_coords = []

	CELL_SIZE = w/8
	BOARD_LEFT_COORD = x
	BOARD_TOP_COORD = y

	# board top left corner coords
	x = BOARD_LEFT_COORD
	y = BOARD_TOP_COORD

	# loop over board rows
	for row in range(8):
		# loop over board columns
		for col in range(8):
			# init square
			square = row * 8 + col

			# associate square with square center coordinates
			square_to_coords.append((int(x + CELL_SIZE / 2), int(y + CELL_SIZE / 2)))

			# increment x coord by cell size
			x += CELL_SIZE

		# restore x coord, increment y coordinate by cell size
		x = BOARD_LEFT_COORD
		y += CELL_SIZE

	return cropped_img, square_to_coords

def find_lines(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 50, 150, apertureSize=3)
	lines = cv2.HoughLines(edges, 1, np.pi/180, 300)
	
	horizontal_lines = []
	vertical_lines = []
	for line in lines:
		rho, theta = line[0]
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

		# check for vertical lines
		if abs(np.cos(theta)) > 0.9:
			vertical_lines.append((x1, y1, x2, y2, rho))
		
	# sort the lines based on their length
	vertical_lines.sort(key=lambda x: abs(x[3] - x[1]))

	# keep only the first 9 lines
	vertical_lines = vertical_lines[:9]
	
	if(len(vertical_lines) != 9):
		raise ValueError("Didn't find 9 lines")

	for line in vertical_lines:
		x1, y1, x2, y2, rho = line
		cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.line(img, (y1, x1), (y2, x2), (0, 0, 255), 2)

	vertical_lines = [line[4] for line in vertical_lines]

	# sort the lines by their y-coordinates (for horizontal lines) or x-coordinates (for vertical lines)
	vertical_lines.sort()

	horizontal_lines = vertical_lines

	# cv2.imshow("Cropped board with lines", img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	return horizontal_lines, vertical_lines

def find_squares(img, horizontal_lines, vertical_lines):
	squares = np.empty((8,8), dtype=object)

	first_size = 0

	for i in range(len(horizontal_lines) - 1):
		for j in range(len(vertical_lines) - 1):
			# calculate the coordinates of the top left and bottom right corners of the square
			top_left = (int(vertical_lines[j]), int(horizontal_lines[i]))
			bottom_right = (int(vertical_lines[j + 1]), int(horizontal_lines[i + 1]))

			# crop the square from the original image
			square = img[top_left[1]+2:bottom_right[1]-2, top_left[0]+2:bottom_right[0]-2]

			if(first_size == 0):
				first_size = square.shape[0]

			# Make width same as height if different
			square = cv2.resize(square, (first_size, first_size), interpolation=cv2.INTER_AREA)

			squares[i, j] = square
			# cv2.imwrite(f'pieces\\{i}_{j}.jpg', square)

	return squares

def coordinates_to_notation(row, column, is_black):
	column_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
	if is_black:
		row = 0 + row
		column = column_map[7 - column]
	else:
		row = 7 - row
		column = column_map[column]
	
	return f"{column}{row+1}"

def notation_to_coordinates(notation, is_black):
	column_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
	column = column_map[notation[0]]
	row = int(notation[1])-1
	if is_black:
		column = 7 - column
		row = row
	else:
		column = column
		row = 7 - row
	
	return row, column

def pattern_matcher(input_img, templates):
	gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
	_, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
	input_img_gray = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

	confidence = 0.2

	# Loop through each template and match it with the input image
	best_match = None
	best_match_name = None
	best_match_value = 0
	for template, name in zip(templates[0], templates[1]):
		# Resize the image to the desired shape
		template = cv2.resize(template, (input_img.shape[0], input_img.shape[0]), interpolation=cv2.INTER_AREA)

		result = cv2.matchTemplate(input_img_gray, template, cv2.TM_CCOEFF_NORMED)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
		if max_val > best_match_value and max_val > confidence:
			best_match = template
			best_match_name = piece_names[name]
			best_match_value = max_val

	# if(best_match is not None):
	# 	horizontal_stacks = cv2.hconcat([input_img_gray, best_match])
	# 	cv2.imshow("Stacked Images", horizontal_stacks)
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()

	if(best_match is not None):
		return best_match, best_match_name
	else:
		return input_img, " "

def predict_board(squares, templates, h_row_down):
	images = []
	names = []
	for row in range(8):
		for column in range(8):
			image, name = pattern_matcher(squares[row, column], templates)

			images.append(image)
			names.append(name)

	# Convert the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Apply thresholding to convert the image to binary format
	_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	# text = pytesseract.image_to_string(thresh, config='--psm 11')

	# print(text)

	# if('a' in text or '8' in text):
	# 	h_row_down = True
	# 	print("h row down")
	# elif('h' in text or '1' in text):
	# 	h_row_down = False
	# 	print("a row down")
	# else:
	# 	print("Couldn't find h_row")

	# cv2.imshow("Image", thresh)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	if(h_row_down):
		names.reverse()

	# Stack the images horizontally in groups of 8
	# Stack the horizontal stacks vertically
	# Display the stacked image
	# horizontal_stacks = [cv2.hconcat(images[i:i+8]) for i in range(0, 64, 8)]
	# vertical_stack = cv2.vconcat(horizontal_stacks)
	# cv2.imshow("Stacked Images", vertical_stack)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# print(names)

	return names

def calculate_fen(pieces, turn):
	fen = ""
	empty_cells = 0

	array = np.array(pieces)
	array = array.reshape((8, 8))

	# Iterate over the 2D array
	for i in range(8):
		for j in range(8):
			current_piece = array[i, j]
			if(current_piece==" "):
				empty_cells+=1
			else:
				if(empty_cells!=0):
					fen+=str(empty_cells)
					empty_cells=0
				fen+=current_piece
		
		if(empty_cells!=0):
			fen+=str(empty_cells)
			empty_cells=0
		
		fen+="/"

	fen = fen[:-1]

	fen += " " + turn

	return fen

def analyze_position(fen):
	# Create an instance of the Stockfish engine
	engine = chess.engine.SimpleEngine.popen_uci(".\\stockfish.exe")

	# Set the position for the engine to evaluate
	board = chess.Board(fen)

	info = engine.analyse(board, chess.engine.Limit(time=1))

	# Remember to close the engine after use
	engine.quit()

	print(f"{info['pv'][0]}", end="")
	if(info["score"].relative.score() != None):
		print(f" ({info['score'].white().score() / 100})")
	else:
		print(f"({info['score'].white()})")
	print()
	
	return str(info['pv'][0])

def move_and_click(x, y):
	pg.PAUSE = 0.01

	current_x, current_y = pg.position()
	num_points = 50

	delta_x = (x - current_x) / (num_points - 1)
	delta_y = (y - current_y) / (num_points - 1)
	points = [(current_x + i * delta_x, current_y + i * delta_y) for i in range(num_points)]
	for point in points:
		pg.moveTo(int(point[0]), int(point[1]))
	pg.click()

def main():
	if(len(sys.argv) < 2):
		print("Example: python chess_cv.py b")
		exit()

	turn = sys.argv[1]
	# h_row_down = sys.argv[2].lower() == 'true'
	h_row_down = False
	if(turn == "b"):
		h_row_down = True

	template_images, template_names = load_templates()
	templates = (template_images, template_names)

	while True:
		# Take a screenshot of the entire screen
		screenshot = ImageGrab.grab()
		# Convert the screenshot to a numpy array
		img = np.array(screenshot)
		# Convert the color space from RGB to BGR (OpenCV format)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		# Convert the PIL Image to a numpy array
		img = np.array(img)

		# Convert the color format from RGB to BGR (which is what OpenCV uses)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		# img = cv2.imread("lichess.png")

		try:
			chessboard, square_to_coords = find_chessboard(img)
			horizontal_lines, vertical_lines = find_lines(chessboard)
			squares = find_squares(chessboard, horizontal_lines, vertical_lines)
			pieces = predict_board(squares, templates, h_row_down)
			fen = calculate_fen(pieces, turn)

			print()
			print("link: " + url_normalize("https://lichess.org/analysis/fromPosition/" + fen))
			print()

			best_move = analyze_position(fen)

			# extract source and destination square coordinates
			row, column = notation_to_coordinates(best_move[0:2], h_row_down)
			from_sq = square_to_coords[row * 8 + column]
			row, column = notation_to_coordinates(best_move[2:4], h_row_down)
			to_sq = square_to_coords[row * 8 + column]
			
			# move_and_click(from_sq[0], from_sq[1])
			# move_and_click(to_sq[0], to_sq[1])

			time.sleep(1)
		except KeyboardInterrupt:
			break
		except Exception as e:
			pass 
			# print(e)

main()