import numpy as np

def create_oval_track(radius=5, width=1, num_points=30):
    angles = np.linspace(0, 2 * np.pi, num_points // 2)
    oval_points = []
    
    for angle in angles:
        # Place the cone on one side of the track
        inner_x = (radius - width) * np.cos(angle)
        inner_y = (radius * 0.5 - width) * np.sin(angle)
        oval_points.append([inner_x, inner_y])

        # Then place the cone on the opposite side of the track
        outer_x = (radius + width) * np.cos(angle)
        outer_y = (radius * 0.5 + width) * np.sin(angle)
        oval_points.append([outer_x, outer_y])

    return np.array(oval_points)

def generate_track_positions(num_points=24, width=100, height=60, noise_factor=0.05):
    
    if num_points % 2 != 0:
        raise ValueError("num_points must be even")

    t = np.linspace(0, 2 * np.pi, num_points // 2, endpoint=False)
    
    # Generate base oval shape
    x_left = -width / 2 * np.cos(t)
    y_left = height / 2 * np.sin(t)
    
    x_right = width / 2 * np.cos(t)
    y_right = height / 2 * np.sin(t)
    
    # Add some noise to make it more interesting
    noise = np.random.normal(0, noise_factor * min(width, height), num_points // 2)
    
    x_left += noise
    y_left += noise
    x_right += noise
    y_right += noise
    
    # Combine left and right sides, alternating points
    x = np.empty(num_points)
    y = np.empty(num_points)
    x[0::2] = x_left
    x[1::2] = x_right
    y[0::2] = y_left
    y[1::2] = y_right
    
    # Ensure the track is closed by adjusting the last points
    x[-2:] = [x_left[0], x_right[0]]
    y[-2:] = [y_left[0], y_right[0]]
    
    return np.column_stack([x, y])