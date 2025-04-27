# LinkedIn Queens Game Solver

This project provides an automated solution for the LinkedIn Queens puzzle game. It uses computer vision to analyze the game board and provides the solution that you can manually apply in the game.

## Features

- **Automated Screenshot Capture**: Captures the game board from your screen
- **Computer Vision Processing**: Detects the grid, cells, and color regions
- **Intelligent Solving Algorithm**: Uses constraint propagation and backtracking to find solutions
- **Visual Debugging**: Generates debug images showing detected grid and regions

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- scikit-learn
- Pillow (PIL)
- pyautogui
- matplotlib

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install opencv-python numpy scikit-learn pillow pyautogui matplotlib
```

## How to Use

1. Navigate to the LinkedIn Queens game in your browser
2. Run the solver:
```bash
python queens-screenshot.py
```
3. You will have 5 seconds to position your browser window
4. The program will:
   - Capture a screenshot of the game
   - Analyze the board layout and color regions
   - Solve the puzzle using constraint propagation and backtracking
   - Display the solution in the console
5. Manually place the queens in the game according to the solution shown in the console

## Current Limitations

- The grid size is currently hardcoded to 10x10. You need to modify the code to change the grid size.
- The solution is only displayed in the console and needs to be manually applied in the game.

## Future Work

1. **Dynamic Grid Size Detection**:
   - Automatically detect the grid size from the screenshot
   - Remove the hardcoded 10x10 limitation
   - Handle different board sizes dynamically

2. **Automated Gameplay**:
   - Implement automatic mouse clicks to place queens
   - Mark invalid cells directly in the browser
   - Add visual feedback during the solving process

## How It Works

1. **Image Processing**:
   - Captures a screenshot of the game
   - Detects the grid boundaries and cell positions
   - Identifies color regions using K-means clustering

2. **Game Solving**:
   - Uses constraint propagation to identify forced moves
   - Marks cells that cannot contain queens
   - Places queens in cells that must contain them
   - Uses backtracking for remaining cells if needed

3. **Output**:
   - Prints the solution to the console
   - Shows the board state with queens (Q) and invalid cells (X)
   - Provides visual feedback through debug images

## Debugging

The program generates several debug images:
- `detected_grid.jpg`: Shows the detected grid and cell boundaries
- `grid_lines_detected.jpg`: Shows detected grid lines
- `grid_lines_hough.jpg`: Shows grid lines detected using Hough transform

## Notes

- The solver assumes the game is visible on your screen
- Make sure the game window is not obscured by other windows
- The solver works best with a clear view of the game board
- The program includes error handling for various edge cases
- Currently, you need to manually place the queens in the game based on the console output
- The grid size is fixed at 10x10 and needs to be modified in the code for different sizes

## License

This project is open source and available under the MIT License.
