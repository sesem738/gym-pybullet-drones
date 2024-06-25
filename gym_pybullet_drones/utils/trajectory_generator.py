import numpy as np

class Waypoints_Generator:
    def spiral(self):
        

        # Parameters for the spiral
        radius = 2.0  # Radius of the spiral
        height = 5.0   # Total height of the spiral
        num_revolutions = 3  # Number of full revolutions 
        num_points = 100  # Number of waypoints on the spiral

        # Generate angles for the spiral (from 0 to number of revolutions * 2pi)
        theta = np.linspace(0, num_revolutions * 2 * np.pi, num_points)

        # Calculate x, y, z coordinates
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.linspace(0, height, num_points)  # Z increases linearly with angle

        # Create a list of waypoints
        waypoints = []
        for i in range(num_points):
            waypoints.append([x[i], y[i], z[i]])

        return waypoints