# chess_cv

A python script that finds the biggest contour on the screen, and if it's 
a chessboard it then calculates the best move in the position and the 
advantage using the locally downloaded stockfish and then plays it. 
It also creates the FEN(Forsyth-Edwards Notation) and a link to lichess 
online stockfish engine for further analysis with gui. Useful for analyzing
puzzles found in 2D images or even as a cheating tool on online chess games.

The keras model was trained for image classification using this dataset 
https://universe.roboflow.com/chess-project/2d-chessboard-and-chess-pieces
by splitting the chessboards that were labeled for yolo in squares with labels
using the split.py script. Some extra images were also added.

## Download Stockfish

Go to this url https://stockfishchess.org/ and downloaded the latest version of
stockfish. The engine must be in the same folder named stockfish.exe. 