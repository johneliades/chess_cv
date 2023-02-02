import cv2
import numpy as np
import itertools
import chess
import chess.engine
import os
from url_normalize import url_normalize

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
	cropped_img = img[y-2:y+h+2, x-2:x+w+2]
	
	# cv2.imshow("Cropped board", cropped_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	
	return cropped_img

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

		# check for horizontal lines
		if abs(np.sin(theta)) > 0.9:
			horizontal_lines.append((x1, y1, x2, y2, rho))
		# check for vertical lines
		elif abs(np.cos(theta)) > 0.9:
			vertical_lines.append((x1, y1, x2, y2, rho))

	# sort the lines based on their length
	horizontal_lines.sort(key=lambda x: abs(x[2] - x[0]))
	vertical_lines.sort(key=lambda x: abs(x[3] - x[1]))

	# keep only the first 9 lines
	horizontal_lines = horizontal_lines[:9]
	vertical_lines = vertical_lines[:9]

	# draw the lines on the image
	for line in horizontal_lines:
		x1, y1, x2, y2, rho = line
		cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
	
	for line in vertical_lines:
		x1, y1, x2, y2, rho = line
		cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

	horizontal_lines = [line[4] for line in horizontal_lines]
	vertical_lines = [line[4] for line in vertical_lines]

	# cv2.imshow("Cropped board with lines", img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	return horizontal_lines, vertical_lines

def find_squares(img, horizontal_lines, vertical_lines):
	squares = np.empty((8,8), dtype=object)

	# sort the lines by their y-coordinates (for horizontal lines) or x-coordinates (for vertical lines)
	horizontal_lines.sort()
	vertical_lines.sort()

	for i in range(len(horizontal_lines) - 1):
		for j in range(len(vertical_lines) - 1):
			# calculate the coordinates of the top left and bottom right corners of the square
			top_left = (int(vertical_lines[j]), int(horizontal_lines[i]))
			bottom_right = (int(vertical_lines[j + 1]), int(horizontal_lines[i + 1]))

			# crop the square from the original image
			square = img[top_left[1]+2:bottom_right[1]-2, top_left[0]+2:bottom_right[0]-2]
			square = cv2.resize(square, (60, 60), interpolation = cv2.INTER_AREA)
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

def pattern_matcher(input_img):
	# Path to the folder containing the images
	paths = ["./chess_pieces/chess", "./chess_pieces/lichess"]

	gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

	# Threshold the grayscale image to create a black and white image
	_, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

	# Convert the binary image back to RGB
	input_img_gray = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

	# Load all images in the folder
	images = []
	names = []
	
	for folder_path in paths:
		for filename in os.listdir(folder_path):
			if filename.endswith(".png"):
				# Load the image
				img = cv2.imread(os.path.join(folder_path, filename))

				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

				# Threshold the grayscale image to create a black and white image
				_, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

				# Convert the binary image back to RGB
				img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

				images.append(img)
				names.append(filename[:-4]) # Remove the '.jpg' extension

	threshold = 0.2

	# Loop through each template and match it with the input image
	best_match = None
	best_match_name = None
	best_match_value = 0
	for template, name in zip(images, names):
		result = cv2.matchTemplate(input_img_gray, template, cv2.TM_CCOEFF_NORMED)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
		if max_val > best_match_value and max_val > threshold:
			best_match = template
			best_match_name = piece_names[name]
			best_match_value = max_val

	if(best_match is not None):
		return best_match, best_match_name
	else:
		return input_img, " "

def predict_board(squares):
	images = []
	names = []
	for row in range(8):
		for column in range(8):
			image, name = pattern_matcher(squares[row, column])

			images.append(image)
			names.append(name)

	h_lichess = cv2.imread("./chess_pieces/h_lichess.png")
	h_lichess = cv2.resize(h_lichess, (5, 6), interpolation = cv2.INTER_AREA)

	h_chess = cv2.imread("./chess_pieces/h_chess.png")
	h_chess = cv2.resize(h_chess, (6, 7), interpolation = cv2.INTER_AREA)

	last_image = images[len(images)-1]

	result = cv2.matchTemplate(last_image, h_lichess, cv2.TM_CCOEFF_NORMED)
	_, val1, _, _ = cv2.minMaxLoc(result)

	result = cv2.matchTemplate(last_image, h_chess, cv2.TM_CCOEFF_NORMED)
	_, val2, _, _ = cv2.minMaxLoc(result)

	if(max(val1, val2)>0.7):
		pass
		#names.reverse()
		#print("Reversed board for fen calculation")
		#print("lichess: " + str(val1))
		#print("chess: " + str(val2))

	# cv2.imshow("Image", last_image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# Stack the images horizontally in groups of 8
	# Stack the horizontal stacks vertically
	# Display the stacked image
	horizontal_stacks = [cv2.hconcat(images[i:i+8]) for i in range(0, 64, 8)]
	vertical_stack = cv2.vconcat(horizontal_stacks)
	cv2.imshow("Stacked Images", vertical_stack)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
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
	info = engine.analyse(board, chess.engine.Limit(time=0.1))

	# Remember to close the engine after use
	engine.quit()

	return info["pv"][0], info["score"].relative.score() / 100

def main():
	turn = "w"

	img = cv2.imread("lichess.png")

	chessboard = find_chessboard(img)
	horizontal_lines, vertical_lines = find_lines(chessboard)
	squares = find_squares(chessboard, horizontal_lines, vertical_lines)
	pieces = predict_board(squares)
	fen = calculate_fen(pieces, turn)

	print()
	print("link: " + url_normalize("https://lichess.org/analysis/fromPosition/" + fen))
	print()

	best_move, advantage = analyze_position(fen)

	print("Best move: ", best_move)
	print("Advantage: ", advantage)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

main()