import numpy as np
import cv2
import pyautogui
import time
from PIL import Image, ImageGrab
from collections import defaultdict, Counter
import time
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


class ImageProcessor:
    """Process screenshots of the LinkedIn Queens game"""
    
    @staticmethod
    def capture_screenshot():
        """Capture a screenshot and return it as an OpenCV image"""
        # Give user time to navigate to the game
        print("You have 5 seconds to navigate to the LinkedIn Queens game...")
        time.sleep(5)
        
        screenshot = ImageGrab.grab()
        # Convert PIL image to OpenCV format (RGB to BGR)
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        return screenshot
    
    
    
    @staticmethod
    def detect_grid(image):
        """Detect the game grid in the screenshot"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply blur and edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Find the grid contour (usually the largest rectangular contour)
        grid_contour = None
        for contour in contours[:10]:  # Check top 10 largest contours
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # If contour has 4 corners, it's likely the grid
            if len(approx) == 4:
                grid_contour = approx
                break
        
        if grid_contour is None:
            print("Could not detect the game grid. Using default grid area.")
            h, w = image.shape[:2]
            # Default grid - center 60% of the screen
            grid_rect = (int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8))
            return grid_rect
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(grid_contour)
        return (x, y, x+w, y+h)
    
    def detect_grid_size_from_boundaries(image, grid_bounds):
        """Detect grid size by focusing on the black boundaries between cells"""
        import numpy as np
        import cv2
        from scipy import signal
        
        x1, y1, x2, y2 = grid_bounds
        grid_image = image[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)
        
        # Create a copy for visualization
        debug_image = grid_image.copy()
        
        # Apply threshold to isolate dark lines
        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Optionally dilate to make boundaries more prominent
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        # Get image dimensions
        height, width = binary.shape
        
        # Create horizontal and vertical projections
        h_projection = np.sum(binary, axis=1)
        v_projection = np.sum(binary, axis=0)
        
        # Save projections as images for debugging
        h_proj_img = np.zeros((height, 300), dtype=np.uint8)
        v_proj_img = np.zeros((300, width), dtype=np.uint8)
        
        # Scale projections to fit visualization
        h_scale = 300 / np.max(h_projection) if np.max(h_projection) > 0 else 1
        v_scale = 300 / np.max(v_projection) if np.max(v_projection) > 0 else 1
        
        for i in range(height):
            cv2.line(h_proj_img, (0, i), (int(h_projection[i] * h_scale), i), 255, 1)
        
        for i in range(width):
            cv2.line(v_proj_img, (i, 300-1), (i, 300-1-int(v_projection[i] * v_scale)), 255, 1)
        
        cv2.imwrite("h_projection.jpg", h_proj_img)
        cv2.imwrite("v_projection.jpg", v_proj_img)
        
        # Apply adaptive thresholding to projections to find peaks
        def find_boundaries_from_projection(projection, min_distance_percent=0.05):
            # Convert to percentage of image dimension
            min_distance = int(len(projection) * min_distance_percent)
            
            # Calculate dynamic threshold based on projection values
            threshold = np.mean(projection) * 1.2
            
            # Find peaks (boundary lines)
            peaks, _ = signal.find_peaks(projection, height=threshold, distance=min_distance)
            
            # If no peaks found, try a lower threshold
            if len(peaks) < 2:
                threshold = np.mean(projection) * 0.8
                peaks, _ = signal.find_peaks(projection, height=threshold, distance=min_distance)
            
            # Debug output showing detected peaks (boundaries)
            print(f"Detected {len(peaks)} peaks with threshold {threshold:.2f} and min_distance {min_distance}")
            
            return peaks
        
        # Find boundaries
        h_boundaries = find_boundaries_from_projection(h_projection)
        v_boundaries = find_boundaries_from_projection(v_projection)
        
        # If we still don't have enough boundaries, try another approach
        if len(h_boundaries) < 2 or len(v_boundaries) < 2:
            # Apply edge detection instead
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Apply Hough Line Transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(min(height, width)/3))
            
            if lines is not None:
                h_lines = []
                v_lines = []
                
                for rho, theta in lines[:, 0]:
                    # Separate horizontal and vertical lines
                    if abs(theta) < 0.3 or abs(theta - np.pi) < 0.3:  # Vertical lines
                        v_lines.append(abs(rho))
                    elif abs(theta - np.pi/2) < 0.3:  # Horizontal lines
                        h_lines.append(abs(rho))
                
                # Sort and remove duplicates
                h_lines = sorted(h_lines)
                v_lines = sorted(v_lines)
                
                def remove_close_lines(lines, threshold=10):
                    if not lines:
                        return []
                    filtered = [lines[0]]
                    for line in lines[1:]:
                        if line - filtered[-1] > threshold:
                            filtered.append(line)
                    return filtered
                
                h_boundaries = remove_close_lines(h_lines)
                v_boundaries = remove_close_lines(v_lines)
        
        # Draw boundaries on debug image
        for y in h_boundaries:
            cv2.line(debug_image, (0, y), (width, y), (0, 255, 0), 2)
        for x in v_boundaries:
            cv2.line(debug_image, (x, 0), (x, height), (0, 0, 255), 2)
        
        cv2.imwrite("detected_boundaries.jpg", debug_image)
        cv2.imwrite("binary_boundaries.jpg", binary)
        
        # Number of cells = number of boundaries - 1 (for a grid with n cells, there are n+1 lines)
        rows = len(h_boundaries) - 1
        cols = len(v_boundaries) - 1
        
        # Ensure we have at least 2x2 grid
        rows = max(2, rows)
        cols = max(2, cols)
        
        # Additional check: try to detect cell intersections
        intersections = []
        for y in h_boundaries:
            for x in v_boundaries:
                # Verify if this is really an intersection (black point)
                region = binary[max(0, y-5):min(height, y+5), max(0, x-5):min(width, x+5)]
                if np.mean(region) > 127:  # If the region is mostly white, it's an intersection
                    intersections.append((x, y))
                    cv2.circle(debug_image, (x, y), 5, (255, 0, 0), -1)
        
        cv2.imwrite("intersections.jpg", debug_image)
        
        # If we have enough intersections, the grid size is more reliable
        if len(intersections) >= 4:
            # Count unique x and y coordinates
            x_coords = sorted(list(set([pt[0] for pt in intersections])))
            y_coords = sorted(list(set([pt[1] for pt in intersections])))
            
            # Merge close coordinates
            def merge_close_coords(coords, threshold=10):
                if not coords:
                    return []
                merged = [coords[0]]
                for coord in coords[1:]:
                    if coord - merged[-1] > threshold:
                        merged.append(coord)
                return merged
            
            x_merged = merge_close_coords(x_coords)
            y_merged = merge_close_coords(y_coords)
            
            # Grid size based on intersections
            cols = len(x_merged) - 1
            rows = len(y_merged) - 1
            
            # Ensure we have at least 2x2 grid
            rows = max(2, rows)
            cols = max(2, cols)
        
        # Sanity check: grid should be roughly square and have at least 2x2 cells
        if (rows < 2 or cols < 2) or abs(rows - cols) > 3:
            print(f"Warning: Detected grid size {rows}x{cols} seems unusual, using default 9x9")
            print(f"Number of horizontal boundaries: {len(h_boundaries)}, vertical boundaries: {len(v_boundaries)}")
            return 9, 9
        
        print(f"Detected grid size: {rows}x{cols}")
        print(f"Found {len(h_boundaries)} horizontal boundaries and {len(v_boundaries)} vertical boundaries")
        return rows, cols

    def detect_grid_cells_with_boundaries(image, grid_bounds):
        """Detect individual cells within the grid using boundary detection"""
        # First detect the grid size
        num_rows, num_cols = ImageProcessor.detect_grid_size_from_boundaries(image, grid_bounds)
        
        x1, y1, x2, y2 = grid_bounds
        grid_width = x2 - x1
        grid_height = y2 - y1
        
        # Calculate cell dimensions
        cell_width = grid_width / num_cols
        cell_height = grid_height / num_rows
        
        # Create cell coordinates
        cells = []
        for row in range(num_rows):
            cell_row = []
            for col in range(num_cols):
                cell_x1 = int(x1 + col * cell_width)
                cell_y1 = int(y1 + row * cell_height)
                cell_x2 = int(x1 + (col + 1) * cell_width)
                cell_y2 = int(y1 + (row + 1) * cell_height)
                cell_row.append((cell_x1, cell_y1, cell_x2, cell_y2))
            cells.append(cell_row)
        
        return cells

    def process_screenshot_with_boundaries(image_path=None):
        """Process a screenshot with boundary-based grid detection"""
        image = ImageProcessor.capture_screenshot()
        
        # Detect grid
        grid_bounds = ImageProcessor.detect_grid(image)
        print(f"Detected grid at: {grid_bounds}")
        
        # Detect cells with boundary-based grid size detection
        cells = ImageProcessor.detect_grid_cells_with_boundaries(image, grid_bounds)
        
        # Get grid dimensions
        board_size = len(cells)
        cols = len(cells[0]) if cells else 0
        
        # Identify color regions
        color_regions = ImageProcessor.identify_color_regions(image, cells)
        
        # For now, always start with an empty board
        board_state = [[0 for _ in range(cols)] for _ in range(board_size)]
        
        # Debug: Draw grid on image
        debug_image = image.copy()
        for row in range(board_size):
            for col in range(len(cells[row])):
                x1, y1, x2, y2 = cells[row][col]
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put region number in center of cell
                cv2.putText(debug_image, str(color_regions[row][col]), 
                            (x1 + 20, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 0, 255), 2)
        
        # Save debug image
        cv2.imwrite("detected_grid_with_boundaries.jpg", debug_image)
        print(f"Detected grid size: {board_size}x{cols}")
        
        return board_size, color_regions, board_state


        
    @staticmethod
    def detect_grid_cells(image, grid_bounds):
        """Detect individual cells within the grid with boundary detection"""
        return ImageProcessor.detect_grid_cells_with_boundaries(image, grid_bounds)
    
    @staticmethod
    def identify_color_regions(image, cells):
        """Identify color regions based on cell colors with dynamic number of clusters"""
        board_size_row = len(cells)
        board_size_col = len(cells[0])
        board_size = max(board_size_row, board_size_col)
        colors = []
        
        # Extract dominant color from each cell
        for row in range(board_size_row):
            color_row = []
            for col in range(board_size_col):
                x1, y1, x2, y2 = cells[row][col]
                # Get center area of cell (to avoid borders)
                padding = 5
                cell_image = image[y1+padding:y2-padding, x1+padding:x2-padding]
                
                # Get average color
                avg_color = np.mean(cell_image, axis=(0, 1))
                color_row.append(avg_color)
            colors.append(color_row)
        
        # Convert to numpy array for clustering
        colors_array = np.array([color for row in colors for color in row])
        
        # Use KMeans to cluster colors (number of clusters is approximately the number of regions)
        # We dynamically determine the number of clusters based on grid size
        # A good starting point might be sqrt(board_size)
        num_clusters = board_size
        
        # You can also use Elbow method or other techniques to determine optimal clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(colors_array)
        
        # Reshape clusters back to grid
        color_regions = clusters.reshape(len(cells), len(cells[0]))
        
        return color_regions.tolist()
    
    @staticmethod
    def process_screenshot(image_path=None):
        """Process a screenshot with improved grid detection"""
        return ImageProcessor.process_screenshot_with_boundaries()

    @staticmethod
    def get_cell_coordinates(grid_bounds, row, col, board_size=None):
        """Convert grid cell indices to screen coordinates using dynamic board size"""
        x1, y1, x2, y2 = grid_bounds
        
        # If board_size is not provided, detect it
        if board_size is None:
            num_rows, num_cols = ImageProcessor.detect_grid_size(ImageProcessor.capture_screenshot(), grid_bounds)
            cell_width = (x2 - x1) / num_cols
            cell_height = (y2 - y1) / num_rows
        else:
            cell_width = (x2 - x1) / board_size
            cell_height = (y2 - y1) / board_size
        
        center_x = int(x1 + (col + 0.5) * cell_width)
        center_y = int(y1 + (row + 0.5) * cell_height)
        
        return center_x, center_y

class GameAutomation:
    """Base class for game automation"""
    
    def __init__(self, game):
        self.game = game
    
    def solve_game(self):
        """Solve the Queens game"""
        solved, board, queen_positions = self.game.solve()
        
        if solved:
            print("Solution found!")
            self.game.print_board()
            print("Queen positions:", queen_positions)
            return board, queen_positions
        else:
            print("No solution found.")
            return None, None


class ScreenshotBasedAutomation(GameAutomation):
    """Automate the game using screenshots and mouse clicks"""
    
    def __init__(self, game, grid_bounds=None):
        super().__init__(game)
        self.grid_bounds = grid_bounds or self._detect_grid()
    
    def _detect_grid(self):
        """Detect the game grid boundaries"""
        # In a real implementation, you'd use image processing to find the grid
        # For now, return placeholder coordinates
        top_left = (335, 290)
        bottom_right = (815, 770)
        return (top_left, bottom_right)
    
    def get_cell_coords(self, row, col):
        """Get screen coordinates for a cell"""
        (x1, y1), (x2, y2) = self.grid_bounds
        cell_width = (x2 - x1) / self.game.size
        cell_height = (y2 - y1) / self.game.size
        
        x = x1 + col * cell_width + cell_width / 2
        y = y1 + row * cell_height + cell_height / 2
        
        return int(x), int(y)
    
    def apply_solution(self, board):
        """Apply the solution by clicking on cells"""
        current_board = self.game.board
        
        for row in range(self.game.size):
            for col in range(self.game.size):
                # Only apply changes to empty cells
                if current_board[row][col] == 0:
                    if board[row][col] == 1:  # X mark
                        self._click_cell(row, col, single=True)
                    elif board[row][col] == 2:  # Queen
                        self._click_cell(row, col, single=False)
    
    def _click_cell(self, row, col, single=True):
        """Click on a cell (single for X, double for Queen)"""
        x, y = self.get_cell_coords(row, col)
        
        pyautogui.moveTo(x, y, duration=0.1)
        if single:
            pyautogui.click()
            print(f"Single clicked at ({row}, {col}) for X")
        else:
            pyautogui.doubleClick()
            print(f"Double clicked at ({row}, {col}) for Queen")
        
        time.sleep(0.2)  # Small delay between actions

class OptimizedQueensGame:
    def __init__(self, size, color_regions, initial_state=None):
        """
        Initialize the game board with optimized solving capabilities
        
        Args:
            size: The size of the board (NxN)
            color_regions: A 2D list describing color regions
            initial_state: Optional initial board state (0: empty, 1: X, 2: Queen)
        """
        self.size = size
        self.color_regions = np.array(color_regions)
        self.unique_regions = set(np.unique(self.color_regions))
        self.num_regions = len(self.unique_regions)
        
        # Ensure we have the right number of queens to place
        if self.num_regions != self.size:
            print(f"Warning: Number of regions ({self.num_regions}) doesn't match board size ({self.size})")
        
        if initial_state:
            self.board = np.array(initial_state)
        else:
            self.board = np.zeros((size, size), dtype=int)
        
        # Create mappings for rows, columns, and regions
        self.region_cells = defaultdict(list)
        for r in range(size):
            for c in range(size):
                region = self.color_regions[r, c]
                self.region_cells[region].append((r, c))
        
        # Track placed queens
        self.queen_positions = []
        for r in range(size):
            for c in range(size):
                if self.board[r, c] == 2:
                    self.queen_positions.append((r, c))
        
        # Track constraints
        self.rows_with_queen = set()
        self.cols_with_queen = set()
        self.regions_with_queen = set()
        
        for r, c in self.queen_positions:
            self.rows_with_queen.add(r)
            self.cols_with_queen.add(c)
            self.regions_with_queen.add(self.color_regions[r, c])
    
    def mark_invalid_cells(self):
        """Mark cells that cannot contain a queen with X"""
        changed = False
        
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 0 and not self.can_place_queen(r, c):
                    self.board[r, c] = 1  # Mark as X
                    changed = True
        
        return changed
    
    def can_place_queen(self, row, col):
        """Check if a queen can be placed at this position"""
        # Check if cell already has something
        if self.board[row, col] != 0:
            return False
        
        # Check if row, column or region already has a queen
        if row in self.rows_with_queen or col in self.cols_with_queen:
            return False
        
        region = self.color_regions[row, col]
        if region in self.regions_with_queen:
            return False
        
        # Check diagonals and adjacent cells
        for qr, qc in self.queen_positions:
            # # Check diagonals
            # if abs(row - qr) == abs(col - qc):
            #     return False
            
            # Check adjacent cells (including diagonals)
            if max(abs(row - qr), abs(col - qc)) <= 1:
                return False
        
        return True
    
    def find_forced_placements(self):
        """Find cells where a queen must be placed due to constraints"""
        placed_queens = []
        
        # Check rows with no queens yet
        for row in range(self.size):
            if row not in self.rows_with_queen:
                valid_positions = []
                for col in range(self.size):
                    if self.can_place_queen(row, col):
                        valid_positions.append((row, col))
                
                # If only one valid position in row, must place queen there
                if len(valid_positions) == 1:
                    r, c = valid_positions[0]
                    self.place_queen(r, c)
                    placed_queens.append((r, c))
        
        # Check columns with no queens yet
        for col in range(self.size):
            if col not in self.cols_with_queen:
                valid_positions = []
                for row in range(self.size):
                    if self.can_place_queen(row, col):
                        valid_positions.append((row, col))
                
                # If only one valid position in column, must place queen there
                if len(valid_positions) == 1:
                    r, c = valid_positions[0]
                    self.place_queen(r, c)
                    placed_queens.append((r, c))
        
        # Check regions with no queens yet
        for region in self.unique_regions:
            if region not in self.regions_with_queen:
                valid_positions = []
                for r, c in self.region_cells[region]:
                    if self.can_place_queen(r, c):
                        valid_positions.append((r, c))
                
                # If only one valid position in region, must place queen there
                if len(valid_positions) == 1:
                    r, c = valid_positions[0]
                    self.place_queen(r, c)
                    placed_queens.append((r, c))
        
        return placed_queens
    
    def place_queen(self, row, col):
        """Place a queen at the specified position and update constraints"""
        if not self.can_place_queen(row, col):
            return False
        
        self.board[row, col] = 2
        self.queen_positions.append((row, col))
        
        # Update constraints
        self.rows_with_queen.add(row)
        self.cols_with_queen.add(col)
        self.regions_with_queen.add(self.color_regions[row, col])
        
        # Mark invalid cells
        self.mark_invalid_cells()
        
        return True
    
    def find_singleton_constraints(self):
        """
        Find 'singleton' constraints - where a row/column/region has only one valid cell
        """
        changed = False
        
        # Structure to hold valid positions for each row, column, and region
        valid_in_row = defaultdict(list)
        valid_in_col = defaultdict(list)
        valid_in_region = defaultdict(list)
        
        # Find all valid positions
        for r in range(self.size):
            for c in range(self.size):
                if self.can_place_queen(r, c):
                    valid_in_row[r].append((r, c))
                    valid_in_col[c].append((r, c))
                    region = self.color_regions[r, c]
                    valid_in_region[region].append((r, c))
        
        # Check for singletons
        for r, positions in valid_in_row.items():
            if len(positions) == 1:
                row, col = positions[0]
                if self.place_queen(row, col):
                    changed = True
        
        for c, positions in valid_in_col.items():
            if len(positions) == 1:
                row, col = positions[0]
                if self.place_queen(row, col):
                    changed = True
        
        for region, positions in valid_in_region.items():
            if len(positions) == 1:
                row, col = positions[0]
                if self.place_queen(row, col):
                    changed = True
        
        return changed
    
    def analyze_constraints(self):
        """
        Analyze constraints to identify cells that must be X or must have a queen
        Returns True if any changes were made
        """
        changed = False
        
        # First, mark all cells that clearly can't have a queen
        if self.mark_invalid_cells():
            changed = True
        
        # Find forced placements based on row/column/region constraints
        if self.find_forced_placements():
            changed = True
        
        # Look for singleton constraints
        if self.find_singleton_constraints():
            changed = True
        
        return changed
    
    def solve(self):
        """
        Solve the Queens game using constraint propagation first, 
        then backtracking if needed
        """
        # Step 1: Apply constraint propagation until no more changes
        while self.analyze_constraints():
            pass
        
        # Check if we've already solved the puzzle
        if len(self.queen_positions) == self.size:
            return True, self.board.tolist(), self.queen_positions
        
        # Step 2: If constraint propagation didn't solve it, use backtracking
        return self.solve_with_backtracking()
    
    def solve_with_backtracking(self):
        """Use backtracking for the remaining cells"""
        # Find cells still available for queens
        available_cells = []
        for r in range(self.size):
            for c in range(self.size):
                if self.can_place_queen(r, c):
                    available_cells.append((r, c))
        
        # Sort by most constrained first (heuristic to improve efficiency)
        available_cells.sort(key=lambda pos: self.count_constraints(*pos))
        
        # Save current state for backtracking
        original_board = self.board.copy()
        original_queens = self.queen_positions.copy()
        original_rows = self.rows_with_queen.copy()
        original_cols = self.cols_with_queen.copy()
        original_regions = self.regions_with_queen.copy()
        
        # Try each available cell
        for r, c in available_cells:
            if self.place_queen(r, c):
                # Try to solve from this state
                solved, board, queens = self.solve()
                if solved:
                    return True, board, queens
                
                # Backtrack if not solved
                self.board = original_board.copy()
                self.queen_positions = original_queens.copy()
                self.rows_with_queen = original_rows.copy()
                self.cols_with_queen = original_cols.copy()
                self.regions_with_queen = original_regions.copy()
        
        return False, self.board.tolist(), self.queen_positions
    
    def count_constraints(self, row, col):
        """Count how many cells would be eliminated if a queen is placed here"""
        # This is a heuristic to choose which cell to try first in backtracking
        count = 0
        
        # Count cells in same row and column
        for i in range(self.size):
            if self.board[row, i] == 0 and i != col:
                count += 1
            if self.board[i, col] == 0 and i != row:
                count += 1
        
        # Count cells in same region
        region = self.color_regions[row, col]
        for r, c in self.region_cells[region]:
            if (r != row or c != col) and self.board[r, c] == 0:
                count += 1
        
        # Count diagonals and adjacent cells
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 0 and (r != row or c != col):
                    # Diagonal
                    if abs(r - row) == abs(c - col):
                        count += 1
                    # Adjacent (including diagonal)
                    elif max(abs(r - row), abs(c - col)) <= 1:
                        count += 1
        
        return count
    
    def print_board(self):
        """Print the current board state"""
        symbols = {0: '.', 1: 'X', 2: 'Q'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()


def main():
    
    # In a real implementation, you'd capture the screen like:
    # screenshot = ImageGrab.grab()
    # screenshot.save(screenshot_path)
    
    # Process the screenshot to get game state
    board_size, color_regions, board_state = ImageProcessor.process_screenshot()

    # For the optimized version
    start_time = time.time()
    # Create and solve the game
    game = OptimizedQueensGame(board_size, color_regions, board_state)
    print("Initial board:")
    game.print_board()
    
    print("Starting constraint analysis...")
    game.analyze_constraints()
    print("After constraint analysis:")
    game.print_board()
    
    print("Solving remaining positions...")
    solved, final_board, queen_positions = game.solve()
    
    if solved:
        print("Solution found!")
        print("Final board:")
        game.print_board()
        print("Queen positions:", queen_positions)
    else:
        print("No solution found.")
        
    end_time = time.time()
    print(f"Time taken for optimized version: {end_time - start_time:.2f} seconds")
        # To apply solution with mouse clicks:
        # screen_automation = ScreenshotBasedAutomation(game)
        # screen_automation.apply_solution(solution_board)


if __name__ == "__main__":
    main()