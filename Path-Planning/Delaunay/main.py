import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from track_generator import create_oval_track, generate_track_positions

def create_delaunay_triangulation(cone_positions, n):
    triangulations = []
    
    for i in range(0, len(cone_positions) - 2, n - 2):
        points_subset = cone_positions[i:i + n]
        if len(points_subset) >= 3:
            tri = Delaunay(points_subset)
            triangulations.append((points_subset, tri))
    
    return triangulations

def find_edges(triangulations, cone_positions):
    """
    Finds all edges across all triangulations and adds first/last edges.
    
    :param triangulations: List of triangulations.
    :param cone_positions: Complete list of cone positions.
    :return: Dictionary of edges and their occurrence count, and edge points.
    """
    edge_count = {}
    edge_points = {}
    
    # Add first and last edges
    first_edge = tuple(sorted([tuple(cone_positions[0]), tuple(cone_positions[1])]))
    last_edge = tuple(sorted([tuple(cone_positions[-2]), tuple(cone_positions[-1])]))
    
    edge_count[first_edge] = 1
    edge_count[last_edge] = 1
    edge_points[first_edge] = (cone_positions[0], cone_positions[1])
    edge_points[last_edge] = (cone_positions[-2], cone_positions[-1])
    
    for points_subset, tri in triangulations:
        for simplex in tri.simplices:
            for i in range(3):
                p1_idx, p2_idx = simplex[i], simplex[(i+1)%3]
                p1, p2 = points_subset[p1_idx], points_subset[p2_idx]
                
                edge_key = tuple(sorted([tuple(p1), tuple(p2)]))
                
                edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
                edge_points[edge_key] = (p1, p2)
    
    return edge_count, edge_points

def order_midpoints(midpoints):
    """
    Orders midpoints using a nearest-neighbor approach to improve path continuity.
    Ensures the path forms a loop by ending at the starting point.

    :param midpoints: List of midpoint coordinates.
    :return: Ordered list of midpoints forming a closed loop.
    """
    if len(midpoints) < 2:
        return midpoints
    
    ordered_midpoints = [midpoints[0]]
    remaining_points = list(midpoints[1:])
    
    while remaining_points:
        last_point = ordered_midpoints[-1]
        nearest_index = np.argmin([np.linalg.norm(last_point - point) for point in remaining_points])
        ordered_midpoints.append(remaining_points.pop(nearest_index))
    
    # Close the loop by adding the starting midpoint to the end of the list
    ordered_midpoints.append(ordered_midpoints[0])
    
    return np.array(ordered_midpoints)

def find_path_edges(triangulations, cone_positions):
    """
    Finds all edges for the path, including first and last edges, and orders them.

    :param triangulations: List of triangulations.
    :param cone_positions: Complete list of cone positions.
    :return: Ordered list of edge midpoints.
    """
    edge_count, edge_points = find_edges(triangulations, cone_positions)
    
    midpoints = []
    for edge_key, count in edge_count.items():
        if count > 1 or edge_key == tuple(sorted([tuple(cone_positions[0]), tuple(cone_positions[1])])) or \
            edge_key == tuple(sorted([tuple(cone_positions[-2]), tuple(cone_positions[-1])])):
            p1, p2 = edge_points[edge_key]
            midpoint = (p1 + p2) / 2
            midpoints.append(midpoint)
    
    ordered_midpoints = order_midpoints(np.array(midpoints))
    return ordered_midpoints

def interpolate_path(midpoints, num_points=10):
    if len(midpoints) < 2:
        return np.array([])
    
    path = []
    for i in range(len(midpoints) - 1):
        start = midpoints[i]
        end = midpoints[i + 1]
        
        for t in np.linspace(0, 1, num_points):
            point = start + t * (end - start)
            path.append(point)
    
    return np.array(path)

def plot_triangulation_and_path(cone_positions, triangulations, midpoints, path):
    plt.figure(figsize=(10, 10))
    
    # Plot triangulations
    for points_subset, tri in triangulations:
        plt.triplot(points_subset[:, 0], points_subset[:, 1], tri.simplices.copy(), color='blue')
    
    # Plot cones
    plt.plot(cone_positions[:, 0], cone_positions[:, 1], 'o', color='red', label='Cones')

    # Plot midpoints with a gradient color
    if len(midpoints) > 0:
        plt.plot(midpoints[:, 0], midpoints[:, 1], 'o-', color='green', label='Ordered Edge Midpoints (Closed Loop)')

    # Plot interpolated path
    if len(path) > 0:
        plt.plot(path[:, 0], path[:, 1], '-', color='orange', label='Interpolated Path')

    plt.scatter(midpoints[0, 0], midpoints[0, 1], color='purple', s=100, label='Path Start/End')
    plt.title("Delaunay Triangulation")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # cone_positions = np.array([
    #     [0, 0], [1, 2], [2, 1], [3, 3],
    #     [4, 1], [5, 2], [6, 0], [7, 2],
    #     [9, 1], [8, 3], [10, 2], [11, 3]
    # ])
    cone_positions = create_oval_track(radius=10, width=1, num_points=30)   
    #cone_positions = generate_track_positions()

    n = 4  # Number of points in each overlapping set
    triangulations = create_delaunay_triangulation(cone_positions, n)
    midpoints = find_path_edges(triangulations, cone_positions)
    interpolated_path = interpolate_path(midpoints)
    plot_triangulation_and_path(cone_positions, triangulations, midpoints, interpolated_path)