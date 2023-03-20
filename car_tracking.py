import cv2
import sys
import matplotlib.pyplot as plt

video_input_file_name = "car_passing_by.mp4"


def draw_rectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)


def display_rectangle(frame, bbox):
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()
    draw_rectangle(frameCopy, bbox)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
    plt.imshow(frameCopy)
    plt.axis('off')


def draw_text(frame, txt, location, color = (50, 170, 50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

# Set up tracker
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'CSRT', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE']

# Change the index to change the tracker type
tracker_type = tracker_types[0]

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy_TrackerBoosting.create()
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMILcreate()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
elif tracker_type == 'CSRT':
    tracker = cv2.legacy_TrackerCSRT.create()
elif tracker_type == 'TLD':
    tracker = cv2.legacy_TrackerTLD.create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy_TrackerMedianFlow.create()
elif tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
else:
    tracker = cv2.legacy_TrackerMOSSE.create()

# Read video
video = cv2.VideoCapture(video_input_file_name)
ok, frame = video.read()

# Exit if video not opened
if not video.isOpened():
    print("Could not open video")
    sys.exit()
else:
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_output_file_name = 'car_passing_by_tracking_' + tracker_type + '.mp4'
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_out = cv2.VideoWriter(video_output_file_name, fourcc, 10, (width, height))

# Define a bounding box
bbox = (120, 500, 200, 150)             # bounding box defined according to the resolution
#bbox = cv2.selectROI(frame, False)     # bounding box to be drawn by the user

display_rectangle(frame, bbox)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()
    if not ok:
        break

        # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
    if ok:
        draw_rectangle(frame, bbox)
    else:
        draw_text(frame, "Tracking failure detected", (80, 140), (0, 0, 255))

    # Display Info
    draw_text(frame, tracker_type + " Tracker", (80, 60))
    draw_text(frame, "FPS : " + str(int(fps)), (80, 100))

    # Write frame to video
    video_out.write(frame)

video.release()
video_out.release()