import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import cv2

#print(o3d.t.io.RealSenseSensor.list_devices())

point_cloud_file = './point-cloud-file/'

pipeline = rs.pipeline()
config = rs.config()
# 使用realsense获得图片的格式，这里很重要
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
pipeline.start(config)
align = rs.align(rs.stream.color)

to_reset = True
# 可视化实例初始化
vis = o3d.visualization.Visualizer()
vis.create_window()
# 点云类初始化
pointcloud = o3d.geometry.PointCloud()
vis.add_geometry(pointcloud)
opt = vis.get_render_option()
#opt.background_color = np.asanyarray([0, 0, 0])
#opt.point_size = 1
opt.show_coordinate_frame = True

frames = pipeline.wait_for_frames()
frames = pipeline.wait_for_frames()
frames = pipeline.wait_for_frames()

# 点云流
while True:
    # 从real sense拿到RGB-D
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    profile = aligned_frames.get_profile()
    # 相机内参
    intrinsics = profile.as_video_stream_profile().get_intrinsics()
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(\
        intrinsics.width, intrinsics.height, intrinsics.fx, \
        intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    # 对深度图空间滤波
    spatial = rs.spatial_filter()
    #spatial.set_option(rs.option.filter_magnitude, 5)
    #spatial.set_option(rs.option.filter_smooth_alpha, 1)
    #spatial.set_option(rs.option.filter_smooth_delta, 50)
    filtered_depth = spatial.process(depth_frame)

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(filtered_depth.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((color_image, depth_colormap))
    cv2.namedWindow('rgb-d', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('rgb-d', images)

    # 从BGR转化到RGB
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    img_color = o3d.geometry.Image(color_image)
    img_depth = o3d.geometry.Image(depth_image)

    # 设置参数convert_rgb_to_intensity，默认为True
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, \
                                                              convert_rgb_to_intensity = False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        localtime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        pcdfile_path = point_cloud_file + localtime + ".pcd"
        rgb_path = point_cloud_file + localtime + ".jpg"
        depth_path = point_cloud_file + localtime + ".npy"
        print(pcdfile_path)
        o3d.io.write_point_cloud(pcdfile_path, pcd)
        cv2.imwrite(rgb_path, color_image)
        np.save(depth_path, depth_image)

    # 把点云的xyz和rgb转为numpy格式，传到pointcloud
    pcd_points = np.asarray(pcd.points).reshape((-1, 3))
    pcd_colors = np.asarray(pcd.colors).reshape((-1, 3))
    pointcloud.colors = o3d.utility.Vector3dVector(pcd_colors)
    pointcloud.points = o3d.utility.Vector3dVector(pcd_points)

    # update vis
    vis.update_geometry(pointcloud)
    if to_reset:
        vis.reset_view_point(True)
        to_reset = False
    vis.poll_events()
    vis.update_renderer()

    if key & 0xFF == ord('q') or key == 27:
        break

cv2.destroyWindow('rgb-d')
vis.clear_geometries()
vis.destroy_window()
pipeline.stop()
print('done')
