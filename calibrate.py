import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os

from scipy.signal import argrelmin

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def pick_frames(video_file:str, target_dir:str, square_size:float, width:int=9, height:int=6, use_prev=True):
    """
    If prev=False, picks frames and stores them in target_dir. If prev=True, uses target_dir images then calibrates
    """

    cap = cv2.VideoCapture(video_file)
    print(f"frames in video: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"fps: {int(cap.get(cv2.CAP_PROP_FPS))}")
    print(f"(width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}, height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")

    ret, first_frame = cap.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5,5), cv2.BORDER_DEFAULT)

    gray = None

    imgPoints = []

    #############################
    # PICK IMAGES OR USE FOLDER #
    #############################
    if not use_prev:
        diffs = [np.inf]
        for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1))):
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5,5), cv2.BORDER_DEFAULT)

            diff = cv2.absdiff(prev_gray, gray)

            diffs.append(np.sum(diff))

            prev_gray = gray

        indices = argrelmin(np.asarray(diffs), order=10)[0]

        images = []

        for idx in tqdm(indices.ravel()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            cv2.imshow("frame", frame)
            key = cv2.waitKey(0) & 0xff
            if key == ord('w'):
                cv2.imwrite(os.path.join(target_dir, f"{idx}.png"), frame)
                images.append(frame)
            elif key == ord('q'):
                return
    else:
        images = list(filter(lambda img: not img is None, [cv2.imread(addr) for addr in tqdm([os.path.join(target_dir, path) for path in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, path))])]))

    cv2.destroyAllWindows()

    print("Images used:", len(images))


    #################################
    # FIND CHECKERBOARD & CALIBRATE #
    #################################

    objPoint = np.zeros((height*width, 3), np.float32)
    objPoint[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objPoint = objPoint * square_size

    for image in tqdm(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(image, (width, height))

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (8, 8), (-1, -1), criteria)
            imgPoints.append(corners2)
            cv2.drawChessboardCorners(image, (width,height), corners2, ret)

        cv2.imshow("image", image)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()

    objPoints = [objPoint for i in range(len(imgPoints))]
    print("Calibrating")
    ret, intrinsic, distortion, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)

    return intrinsic, distortion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create calibration matrices")

    parser.add_argument("--video_file", "-v", type=str, required=True)
    parser.add_argument("--save_folder", "-f", type=str, required=True)
    parser.add_argument("--square_size", "-s", type=float, required=True)
    parser.add_argument("--use_saved_photos", "-u", help="Use photos from", action="store_true", default=False)

    args = parser.parse_args()

    intrinsic, distortion = pick_frames(args.video_file, args.save_folder, args.square_size, use_prev=args.use_saved_photos)

    print("Intrinsic:\n", intrinsic)
    print("\n\nDistortion:\n",distortion)

    np.save(os.path.join(args.save_folder, "intrinsic.npy"), intrinsic)
    np.save(os.path.join(args.save_folder, "distortion.npy"), distortion)

