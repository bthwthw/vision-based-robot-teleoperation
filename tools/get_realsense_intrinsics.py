# tools/get_realsense_intrinsics.py
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print(f"fx={intr.fx}, fy={intr.fy}, ppx={intr.ppx}, ppy={intr.ppy}, width={intr.width}, height={intr.height}")

pipeline.stop()

# fx=616.1630249023438, fy=616.451416015625, ppx=314.67425537109375, ppy=247.34315490722656, width=640, height=480