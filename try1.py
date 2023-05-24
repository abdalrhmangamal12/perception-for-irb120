import numpy as np
import cv2
import pyrealsense2 as rs

# Create a pipeline object
pipeline = rs.pipeline()

# Create a config object and configure the pipeline to stream both depth and color
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline and get the depth sensor's intrinsics (needed for deprojecting pixels to points)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Create a pointcloud object and a colorizer object
pc = rs.pointcloud()
colorizer = rs.colorizer()

# Process frames from the pipeline
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert the depth and color frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Create a pointcloud from the depth image and the intrinsics
        points = pc.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices())
        vertex_count = vertices.shape[0]

        # Create an array of RGB values for each point in the pointcloud
        colorizer.process(frames)
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        rgb = np.zeros((vertex_count, 3), dtype=np.uint8)
        for i in range(vertex_count):
            x, y, z = vertices[i]
            pixel = [int(x), int(y)]
            if x >= 0 and y >= 0 and pixel[0] < 640 and pixel[1] < 480:
                rgb[i] = color_image[pixel[1], pixel[0]]

        # Combine the vertices and RGB values into a single array
        print(vertices)
        points_with_color = np.concatenate((vertices, rgb), axis=1)

        # Display the pointcloud
        cv2.namedWindow('Pointcloud', cv2.WINDOW_NORMAL)
        cv2.imshow('Pointcloud', colorized_depth)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

finally:
    # Stop the pipeline and close all windows
    pipeline.stop()
    cv2.destroyAllWindows()

