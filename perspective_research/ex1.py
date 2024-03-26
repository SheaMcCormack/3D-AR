import numpy as np
import matplotlib.pyplot as plt

# Define functions to generate point clouds

def generate_line():
    # Generate points along the x-axis
    x = np.random.rand(100, 1)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    return np.hstack((x, y, z, np.ones_like(x))).T

def generate_circle():
    # Generate points for a circle in the XY plane
    theta = np.random.uniform(0, 2 * np.pi, size=(100, 1))
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.zeros_like(x) 
    return np.hstack((x, y, z, np.ones_like(x))).T

def generate_cube():
    # Generate 200 0or1 points then generate 100 rand(0,1) points. Shuffle
    temp = np.hstack((np.random.randint(0, 2, size=(100, 2)), np.random.rand(100, 1)))
    np.apply_along_axis(np.random.shuffle, axis=1, arr=temp)
    return np.hstack((temp, np.ones((100, 1)))).T    

def translate_and_rotate(p, t, theta):
    # Translation Matrix
    tx, ty, tz = t
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

    # Rotation matrix 
    theta_x, theta_y, theta_z = theta
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x), np.cos(theta_x), 0],
        [0, 0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]
    ])

    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    R = np.dot(np.dot(Rz, Ry), Rx)

    # Generate the new matrices
    #T = R*T # Translation matrix: H = RzRyRxT
    T = np.dot(R, T)
    #p = T*p # Projective transformation : X'=HX
    p = np.dot(T, p)

    return T, p

def orthographic_projection(points):
    projection_matrix = np.array([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,0,1]])
    
    return np.dot(projection_matrix,points)

def perspective_projection(points, f, z):
    projection_matrix = np.array([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,0, f/z]])
    
    return np.dot(projection_matrix,points)

def plot_shape(ax, points, title):
    ax.scatter(points[0, :], points[1, :], points[2, :], label=title)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_projection(ax, points, title):
    x, y = points[0, :], points[1, :]
    ax.scatter(x, y)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

# Generate point clouds
line = generate_line()
circle = generate_circle()
cube = generate_cube()

# Plotting each shape on a separate subplot
fig = plt.figure(figsize=(30, 10))

def plot_shapes():
    ax1 = fig.add_subplot(231, projection='3d')
    plot_shape(ax1, line, 'Line')

    ax2 = fig.add_subplot(232, projection='3d')
    plot_shape(ax2, circle, 'Circle')

    ax3 = fig.add_subplot(233, projection='3d')
    plot_shape(ax3, cube, 'Cube')
plot_shapes()
# Straight orthographic projection
projected_points = orthographic_projection(line)
ax4 = fig.add_subplot(234)
plot_projection(ax4, projected_points, 'Orthographic Projection')

projected_points = orthographic_projection(circle)
ax5 = fig.add_subplot(235)
plot_projection(ax5, projected_points, 'Orthographic Projection')

projected_points = orthographic_projection(cube)
ax6 = fig.add_subplot(236)
plot_projection(ax6, projected_points, 'Orthographic Projection')
plt.tight_layout()
plt.show()

# Reset the plot
plt.close(fig)
fig = plt.figure(figsize=(30, 10))
plot_shapes()

# Perspective projection without rotation
f = 1000000000
z = 1
projected_points = perspective_projection(line, f, z)
ax4 = fig.add_subplot(234)
plot_projection(ax4, projected_points, 'Perspective Without rotation')

projected_points = perspective_projection(circle, f, z)
ax5 = fig.add_subplot(235)
plot_projection(ax5, projected_points, 'Perspective Without rotation')

projected_points = perspective_projection(cube, f, z)
ax6 = fig.add_subplot(236)
plot_projection(ax6, projected_points, 'Perspective Without rotation')
plt.tight_layout()
plt.show()

# Perspective projection with one or more rotations

# Parameters for translation and rotation
t = np.array([0, 0, -5])  # Translation amounts
theta = np.radians(np.array([30, 45, 60]))  # Rotation angles in degrees


# Reset the plot
plt.close(fig)
fig = plt.figure(figsize=(30, 10))
plot_shapes()

# Assuming f and z have already been defined above
H, transformed_points = translate_and_rotate(line, t, theta)
projected_points = perspective_projection(transformed_points, f, z)
ax4 = fig.add_subplot(234)
plot_projection(ax4, projected_points, 'Perspective With rotation')

H, transformed_points = translate_and_rotate(circle, t, theta)
projected_points = perspective_projection(transformed_points, f, z)
ax5 = fig.add_subplot(235)
plot_projection(ax5, projected_points, 'Perspective With rotation')

H, transformed_points = translate_and_rotate(cube, t, theta)
projected_points = perspective_projection(transformed_points, f, z)
ax6 = fig.add_subplot(236)
plot_projection(ax6, projected_points, 'Perspective With rotation')
plt.tight_layout()
plt.show()
