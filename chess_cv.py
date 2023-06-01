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
from keras.models import load_model
from keras.optimizers import Adadelta
import tkinter as tk

classes = [' ', 'B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r']

model = load_model('model/model.h5')
model.compile(optimizer=Adadelta(),
	loss = 'sparse_categorical_crossentropy',
	metrics = ['sparse_categorical_accuracy'])

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

	return cropped_img, square_to_coords, CELL_SIZE

def find_lines(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 50, 150, apertureSize=3)
	lines = cv2.HoughLines(edges, 1, np.pi/180, 300)
	
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

	# cv2.imshow("Cropped board with lines", img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	return vertical_lines

def find_squares(img, vertical_lines):
	squares = np.empty((8,8), dtype=object)

	first_size = 0

	for i in range(len(vertical_lines) - 1):
		for j in range(len(vertical_lines) - 1):
			# calculate the coordinates of the top left and bottom right corners of the square
			top_left = (int(vertical_lines[j]), int(vertical_lines[i]))
			bottom_right = (int(vertical_lines[j + 1]), int(vertical_lines[i + 1]))

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

def predict_board(squares, black_perspective):
	images = []
	for row in range(8):
		for column in range(8):
			gray = cv2.cvtColor(squares[row, column], cv2.COLOR_BGR2GRAY)
			img = cv2.resize(gray, (47,47))
			img_reshaped = np.reshape(img, [1,47,47,1])
			images.append(img_reshaped)

	images = np.vstack(images)
	pred_classes = model.predict(images, batch_size=10, verbose=0)
	predicted_indices = np.argmax(pred_classes, axis=1)
	predicted_labels = [classes[i] for i in predicted_indices]
	prediction_certainty = [pred_classes[i][predicted_indices[i]] for i in range(len(images))]
	_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	if(black_perspective):
		predicted_labels.reverse()

	return predicted_labels

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

	info = engine.analyse(board, chess.engine.Limit(time=0.5))

	# Remember to close the engine after use
	engine.quit()

	print(f"{info['pv'][0]}", end="")
	if(info["score"].relative.score() != None):
		print(f" ({info['score'].white().score() / 100})")
	else:
		print(f"({info['score'].white()})")
	
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

def display_board(fen, black_perspective):
	pieces = {
		'P': '♙',
		'N': '♘',
		'B': '♗',
		'R': '♖',
		'Q': '♕',
		'K': '♔',
		'p': '♟',
		'n': '♞',
		'b': '♝',
		'r': '♜',
		'q': '♛',
		'k': '♚',
	}

	rows = fen.split()[0].split('/')
	board = []
	for row in rows:
		board_row = []
		for char in row:
			if char.isdigit():
				board_row.extend([''] * int(char))
			else:
				board_row.append(pieces[char])
		board.append(board_row)

	if black_perspective:
		board = reversed(board)

	print()

	# add column letters at the top
	if not black_perspective:
		print('    a  b  c  d  e  f  g  h')
	else:
		print('    h  g  f  e  d  c  b  a')
	for i, row in enumerate(board):
		# add row number on left side
		if not black_perspective:
			print(' ' + str(8 - i) + ' ', end='')
		else:
			print(' ' + str(i + 1) + ' ', end='')
		for j, piece in enumerate(reversed(row) if black_perspective else row):
			# determine background color for square
			if (i + j) % 2 == 0:
				bg_color = '\x1b[48;5;233m'  # gray
			else:
				bg_color = '\x1b[40m'  # black
			# add piece with background color
			if piece:
				print(bg_color + ' ' + piece + ' ' + '\x1b[0m', end='')
			else:
				print(bg_color + '   ' + '\x1b[0m', end='')
		# add row number on right side and move to next line
		if not black_perspective:
			print(' ' + str(8 - i))
		else:
			print(' ' + str(i + 1))
	# add column letters at the bottom
	if not black_perspective:
		print('    a  b  c  d  e  f  g  h')
	else:
		print('    h  g  f  e  d  c  b  a')

def init_window():
	# Create a transparent tkinter window
	window = tk.Tk()
	window.overrideredirect(True)
	window.wait_visibility(window)
	# window.attributes('-alpha', 0.7)
	window.lift()
	window.wm_attributes("-topmost", True)

	# Create a canvas on the window
	canvas = tk.Canvas(window, width=window.winfo_screenwidth(), height=window.winfo_screenheight(), 
		highlightthickness=0)

	# Function to handle mouse click events
	def on_click(event):
		pass  # Do nothing to disable clicking

	# Bind mouse click event to the canvas and prevent further processing
	canvas.bind("<Button-1>", on_click)
	canvas.bind("<Button-2>", on_click)
	canvas.bind("<Button-3>", on_click)

	canvas.configure(bg='green')
	canvas.pack()
	# Make the window click-through
	window.attributes("-transparentcolor", "green")

	return window, canvas

square_from = None
square_to = None
def draw_move(window, canvas, from_sq, to_sq, cell_size):
	global square_from, square_to

	if square_from:
		canvas.delete(square_from)  # Remove the previous circle
	if square_to:
		canvas.delete(square_to)  # Remove the previous circle

	square_x_from, square_y_from = from_sq[0], from_sq[1]  # Coordinates where you want to draw the circle
	square_x_to, square_y_to = to_sq[0], to_sq[1]  # Coordinates where you want to draw the circle

	# Define the side length of the square
	square_side = cell_size//2

	# Draw the square for the 'circle_from'
	square_from = canvas.create_rectangle(
		square_x_from - square_side, square_y_from - square_side,
		square_x_from + square_side, square_y_from + square_side,
		width=2, outline='red', fill='green'
	)

	canvas.coords(square_from,
		square_x_from - square_side, square_y_from - square_side,
		square_x_from + square_side, square_y_from + square_side
	)

	# Draw the square for the 'circle_to'
	square_to = canvas.create_rectangle(
		square_x_to - square_side, square_y_to - square_side,
		square_x_to + square_side, square_y_to + square_side,
		width=2, outline='aqua', fill='green'
	)

	canvas.coords(square_to,
		square_x_to - square_side, square_y_to - square_side,
		square_x_to + square_side, square_y_to + square_side
	)

	window.update()

	return window

def main():
	if(len(sys.argv) < 2):
		print("Example: python chess_cv.py b")
		exit()

	player_color = sys.argv[1]
	# black_perspective = sys.argv[2].lower() == 'true'
	black_perspective = False
	if(player_color == "b"):
		black_perspective = True

	window, canvas = init_window()

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
			chessboard, square_to_coords, cell_size = find_chessboard(img)
			vertical_lines = find_lines(chessboard)
			squares = find_squares(chessboard, vertical_lines)
			pieces = predict_board(squares, black_perspective)
			fen = calculate_fen(pieces, player_color)
			display_board(fen, black_perspective)

			print()
			print("link: " + url_normalize("https://lichess.org/analysis/fromPosition/" + fen))
			print()

			best_move = analyze_position(fen)

			# extract source and destination square coordinates
			row, column = notation_to_coordinates(best_move[0:2], black_perspective)
			from_sq = square_to_coords[row * 8 + column]
			row, column = notation_to_coordinates(best_move[2:4], black_perspective)
			to_sq = square_to_coords[row * 8 + column]

			draw_move(window, canvas, from_sq, to_sq, cell_size)
		
			# move_and_click(from_sq[0], from_sq[1])
			# move_and_click(to_sq[0], to_sq[1])

			# time.sleep(0.5)
		except KeyboardInterrupt:
			break
		except Exception as e:
			window.update()
			print(e)

main()