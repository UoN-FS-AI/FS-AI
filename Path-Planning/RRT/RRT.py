import matplotlib.pyplot as plt
import random
import math
import numpy as np
from shapely.geometry import LineString, Polygon, Point

# Define the Node class
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

# Define the RRT class
class RRT:
    def __init__(self, start, goal, obstacles, map_size, track_polygon, max_iter=1000, step_size=5):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.obstacles = obstacles  # Obstacles are cones on the track
        self.map_width, self.map_height = map_size
        self.track_polygon = track_polygon  # The track area polygon
        self.max_iter = max_iter
        self.step_size = step_size
        self.node_list = [self.start]

    def get_random_point(self):
        while True:
            x = random.uniform(0, self.map_width)
            y = random.uniform(0, self.map_height)
            point = Point(x, y)
            if self.track_polygon.contains(point):
                return (x, y)

    def get_nearest_node(self, random_point):
        distances = [
            (node.x - random_point[0])**2 + (node.y - random_point[1])**2
            for node in self.node_list
        ]
        nearest_index = distances.index(min(distances))
        return self.node_list[nearest_index]

    def is_collision_free(self, node1, node2):
        line = LineString([(node1.x, node1.y), (node2.x, node2.y)])
        # Check if the line is within the track
        if not self.track_polygon.contains(line):
            return False
        # Check collision with cones
        for (ox, oy, r) in self.obstacles:
            if circle_line_collision(ox, oy, r, node1.x, node1.y, node2.x, node2.y):
                return False
        return True

    def plan(self):
        for _ in range(self.max_iter):
            rand_point = self.get_random_point()
            nearest_node = self.get_nearest_node(rand_point)
            theta = math.atan2(rand_point[1] - nearest_node.y, rand_point[0] - nearest_node.x)
            new_x = nearest_node.x + self.step_size * math.cos(theta)
            new_y = nearest_node.y + self.step_size * math.sin(theta)
            new_node = Node(new_x, new_y)
            new_node.parent = nearest_node

            if not self.is_collision_free(nearest_node, new_node):
                continue

            self.node_list.append(new_node)

            if self.distance_to_goal(new_node.x, new_node.y) <= self.step_size:
                final_node = Node(self.goal.x, self.goal.y)
                final_node.parent = new_node
                if self.is_collision_free(new_node, final_node):
                    self.node_list.append(final_node)
                    return self.generate_final_course(len(self.node_list) - 1)
        return None  # Failed to find a path

    def distance_to_goal(self, x, y):
        return math.hypot(x - self.goal.x, y - self.goal.y)

    def generate_final_course(self, goal_index):
        path = []
        node = self.node_list[goal_index]
        while node.parent is not None:
            path.append((node.x, node.y))
            node = node.parent
        path.append((self.start.x, self.start.y))
        return path[::-1]

def circle_line_collision(cx, cy, radius, x1, y1, x2, y2):
    # Collision detection between a circle and a line segment
    line = LineString([(x1, y1), (x2, y2)])
    circle = Point(cx, cy).buffer(radius)
    return line.intersects(circle)

def generate_race_track():
    # Generate a simple race track using a sine wave as the center line
    x = np.linspace(0, 100, 500)
    y = 50 + 20 * np.sin(x / 10)
    center_line = list(zip(x, y))
    return center_line

def place_cones_and_get_track_polygon(center_line, cone_spacing=5, offset=10):
    obstacles = []
    left_edge = []
    right_edge = []
    for i in range(0, len(center_line), cone_spacing):
        cx, cy = center_line[i]

        # Determine the direction perpendicular to the track at this point
        if i + 1 < len(center_line):
            dx = center_line[i + 1][0] - cx
            dy = center_line[i + 1][1] - cy
        else:
            dx = cx - center_line[i - 1][0]
            dy = cy - center_line[i - 1][1]
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        nx = -dy / length  # Normal vector (perpendicular)
        ny = dx / length

        # Place cones on both sides of the center line
        left_cone = (cx + nx * offset, cy + ny * offset, 1)
        right_cone = (cx - nx * offset, cy - ny * offset, 1)
        obstacles.append(left_cone)
        obstacles.append(right_cone)

        # Store the left and right edges
        left_edge.append((cx + nx * offset, cy + ny * offset))
        right_edge.append((cx - nx * offset, cy - ny * offset))

    # Create the track polygon
    right_edge.reverse()
    track_boundary = left_edge + right_edge
    track_polygon = Polygon(track_boundary)

    return obstacles, track_polygon

def main():
    start = (5, 50)
    goal = (95, 50)
    map_size = (100, 100)

    # Generate race track and cones
    center_line = generate_race_track()
    obstacles, track_polygon = place_cones_and_get_track_polygon(center_line)

    rrt = RRT(start, goal, obstacles, map_size, track_polygon)
    path = rrt.plan()

    # Plotting
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Plot race track area
    x_poly, y_poly = track_polygon.exterior.xy
    plt.fill(x_poly, y_poly, color='lightgray', alpha=0.5, label='Track Boundary')

    # Plot race track center line
    x_center, y_center = zip(*center_line)
    plt.plot(x_center, y_center, '--', color='gray', label='Track Center Line')

    # Plot cones as black circles
    for (ox, oy, r) in obstacles:
        cone = plt.Circle((ox, oy), r, color='black')
        ax.add_patch(cone)

    # Plot start and goal points
    plt.plot(start[0], start[1], "or", markersize=8, label="Start")
    plt.plot(goal[0], goal[1], "ob", markersize=8, label="Goal")

    # Plot RRT nodes and paths
    for node in rrt.node_list:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")

    # Plot the final path
    if path is not None:
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r', linewidth=2, label="Planned Path")
        print("Path found!")
    else:
        print("No path found!")

    plt.xlim(0, map_size[0])
    plt.ylim(0, map_size[1])
    plt.title("RRT Path Planning on a Race Track with Cones and Track Boundaries")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()