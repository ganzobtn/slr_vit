import cv2
import os
import argparse

def list_video_dimensions(folder_path):
    dimensions = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mkv')):
                path = os.path.join(root, file)
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    dim = (width, height)
                    if dim in dimensions:
                        dimensions[dim] += 1
                    else:
                        dimensions[dim] = 1
                cap.release()

    for dim, count in dimensions.items():
        print(f"Dimension {dim[0]}x{dim[1]}: {count} video(s)")

def main():
    parser = argparse.ArgumentParser(description='List dimensions of videos in a folder.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing videos')
    args = parser.parse_args()
    list_video_dimensions(args.folder_path)

if __name__ == "__main__":
    main()
