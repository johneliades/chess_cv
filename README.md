# chess_cv

A python script that finds the onscreen chessboard and then calculates 
the best move in the position using the locally downloaded stockfish 
and then plays it. It also creates the FEN(Forsyth-Edwards Notation) 
and a link to the lichess online stockfish engine for further analysis 
with gui. Useful for analyzing puzzles found in 2D images or even as 
a cheating tool on online chess games. Can also be used in combination
with a trained chess AI in order to determine its elo ranking on popular 
chess platforms by having it play automatically online. 

<p align="center">
  <img src="https://github.com/johneliades/chess_cv/blob/main/lichess_preview.gif" alt="animated" />
</p>

<p align="center">
  <img src="https://github.com/johneliades/chess_cv/blob/main/puzzle_preview.gif" alt="animated" />
</p>


The keras model was trained for image classification using this dataset 
https://universe.roboflow.com/chess-project/2d-chessboard-and-chess-pieces
by splitting the chessboards that were labeled for yolo in squares using the 
split.py script. Some extra images were also added.

## Clone

Clone the repository locally by entering the following command:
```
git clone https://github.com/johneliades/chess_cv.git
```
Or by clicking on the green "Clone or download" button on top and then 
decompressing the zip.

Then install the missing libraries in a virtual environment:

```
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt && deactivate
```

## Download Stockfish

Go to this url https://stockfishchess.org/ and downloaded the latest version of
stockfish. The engine must be in the same folder named stockfish.exe. 

## Run

Then you can run:

```
.venv\Scripts\activate
python chess_cv.py b
```

for black, or

```
.venv\Scripts\activate
python chess_cv.py w
```

for white.
