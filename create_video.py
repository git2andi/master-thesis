import cv2
import os
import xml.etree.ElementTree as ET
from natsort import natsorted

# === CONFIGURATION ===
frames_folder = '001-001_frames'
annotations_folder = '001-001_annotations'
output_video_path = 'output/001-001_output_annotated.mp4'
fps = 25  # adjust based on metadata if available

# === COLLECT FRAME FILES ===
frame_files = [f for f in os.listdir(frames_folder) if f.endswith('.jpg')]
frame_files = natsorted(frame_files)


# === LOAD FIRST FRAME TO GET SIZE ===
first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
height, width, _ = first_frame.shape

# === VIDEO WRITER SETUP ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# === PROCESS FRAMES WITH ANNOTATIONS ===
for filename in frame_files:
    frame_path = os.path.join(frames_folder, filename)
    annotation_path = os.path.join(annotations_folder, filename.replace('.jpg', '.xml'))
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Warning: couldn't read frame {frame_path}")
        continue

    # Check if annotation exists
    if os.path.exists(annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = obj.find('name').text if obj.find('name') is not None else "polyp"
                cv2.putText(frame, label, (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    video_writer.write(frame)

video_writer.release()
print(f"Annotated video saved to {output_video_path}")
