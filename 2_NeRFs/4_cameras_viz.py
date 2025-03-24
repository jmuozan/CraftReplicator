import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

def load_transforms_json(file_path):
    """Load the transforms.json file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_camera_positions_and_orientations(data):
    """Extract camera positions and orientations from transforms.json"""
    cameras = []
    
    for frame in data['frames']:
        # Extract transform matrix
        transform_matrix = np.array(frame['transform_matrix'])
        
        # The position is in the 4th column (indices 0-2, row 3)
        position = transform_matrix[:3, 3]
        
        # The orientation vectors are in the first 3 columns
        # We'll use the negative z-axis as the camera's forward direction
        # and the y-axis as the camera's up direction
        forward = -transform_matrix[:3, 2]  # Negative z-axis
        up = transform_matrix[:3, 1]        # y-axis
        right = transform_matrix[:3, 0]     # x-axis
        
        # Scale the orientation vectors to make them visible
        scale = 0.3
        
        cameras.append({
            'position': position,
            'forward': forward * scale,
            'up': up * scale,
            'right': right * scale,
            'filename': frame.get('file_path', '')
        })
    
    return cameras

def create_camera_pyramid(position, forward, up, right, scale=0.1):
    """Create a pyramid to represent the camera"""
    # Calculate the camera frustum points
    apex = position
    # Base points of the pyramid (scaled to create a frustum)
    base_center = position + forward
    top_left = base_center + up * scale - right * scale
    top_right = base_center + up * scale + right * scale
    bottom_right = base_center - up * scale + right * scale
    bottom_left = base_center - up * scale - right * scale
    
    # Define the vertices of the frustum
    vertices = [apex, top_left, top_right, bottom_right, bottom_left]
    
    # Define the faces of the frustum
    faces = [
        [0, 1, 2],  # Top face
        [0, 2, 3],  # Right face
        [0, 3, 4],  # Bottom face
        [0, 4, 1],  # Left face
        [1, 2, 3, 4]  # Base face
    ]
    
    return vertices, faces

def plot_cameras(cameras, ax, plot_arrows=True):
    """Plot cameras as pyramids and arrows for orientation"""
    # Plot camera positions
    camera_positions = np.array([cam['position'] for cam in cameras])
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
              c='blue', marker='o', s=50, label='Camera Positions')
    
    # Plot camera orientation arrows
    if plot_arrows:
        for cam in cameras:
            pos = cam['position']
            ax.quiver(pos[0], pos[1], pos[2], 
                     cam['forward'][0], cam['forward'][1], cam['forward'][2], 
                     color='red', label='Forward' if cam is cameras[0] else "")
            ax.quiver(pos[0], pos[1], pos[2], 
                     cam['up'][0], cam['up'][1], cam['up'][2], 
                     color='green', label='Up' if cam is cameras[0] else "")
            ax.quiver(pos[0], pos[1], pos[2], 
                     cam['right'][0], cam['right'][1], cam['right'][2], 
                     color='blue', label='Right' if cam is cameras[0] else "")
    
    # Plot camera frustums
    for cam in cameras:
        vertices, faces = create_camera_pyramid(
            cam['position'], cam['forward'], cam['up'], cam['right']
        )
        
        # Create a Poly3DCollection
        polygon = Poly3DCollection(
            [[vertices[idx] for idx in face] for face in faces],
            alpha=0.3, facecolor='cyan', edgecolor='black'
        )
        ax.add_collection3d(polygon)
    
    # Add filename labels if available
    for cam in cameras:
        if 'filename' in cam:
            ax.text(
                cam['position'][0], cam['position'][1], cam['position'][2], 
                cam['filename'].split('/')[-1], 
                size=8, zorder=1, color='k'
            )

def visualize_cameras(data, save_to_file=None):
    """Create a 3D visualization of cameras"""
    cameras = get_camera_positions_and_orientations(data)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    plot_cameras(cameras, ax)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions and Orientations')
    
    # Add legend for the first occurrence of each type
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Set equal aspect ratio
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    
    max_range = max(x_range, y_range, z_range) / 2.0
    
    mid_x = (x_limits[1] + x_limits[0]) / 2.0
    mid_y = (y_limits[1] + y_limits[0]) / 2.0
    mid_z = (z_limits[1] + z_limits[0]) / 2.0
    
    ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
    ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
    ax.set_zlim3d([mid_z - max_range, mid_z + max_range])
    
    if save_to_file:
        plt.savefig(save_to_file, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax, cameras

def create_animation(data, output_file='camera_animation.mp4'):
    """Create a rotating animation of the camera visualization"""
    cameras = get_camera_positions_and_orientations(data)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    plot_cameras(cameras, ax, plot_arrows=False)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions and Orientations')
    
    # Set equal aspect ratio
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    
    max_range = max(x_range, y_range, z_range) / 2.0
    
    mid_x = (x_limits[1] + x_limits[0]) / 2.0
    mid_y = (y_limits[1] + y_limits[0]) / 2.0
    mid_z = (z_limits[1] + z_limits[0]) / 2.0
    
    ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
    ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
    ax.set_zlim3d([mid_z - max_range, mid_z + max_range])
    
    def rotate(angle):
        ax.view_init(elev=20, azim=angle)
        return fig,
    
    # Create a 360-degree rotation animation
    anim = animation.FuncAnimation(fig, rotate, frames=range(0, 360, 2), interval=50)
    
    # Save animation
    writer = animation.FFMpegWriter(fps=30, bitrate=5000)
    anim.save(output_file, writer=writer)
    
    plt.close()
    return output_file

if __name__ == "__main__":
    # Usage example:
    transform_file = "transforms.json"
    
    try:
        data = load_transforms_json(transform_file)
        
        # Static visualization
        visualize_cameras(data, save_to_file="camera_visualization.png")
        
        # Create animation (uncomment to use)
        # create_animation(data)
        
        print("Visualization complete!")
        
    except FileNotFoundError:
        print(f"Error: '{transform_file}' not found.")
        print("Please provide the correct path to your transforms.json file.")
    except json.JSONDecodeError:
        print(f"Error: '{transform_file}' is not a valid JSON file.")
    except Exception as e:
        print(f"Error: {str(e)}")