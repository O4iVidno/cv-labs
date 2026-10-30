import os
import time
from pathlib import Path
import numpy as np
import cv2

os.chdir(Path(__file__).resolve().parent.parent)


def detect_and_describe(img, detector):
    keypoints, descriptors = detector.detectAndCompute(img, None)
    return keypoints, descriptors


def save_keypoints(img, keypoints, path):
    cv2.imwrite(path, cv2.drawKeypoints(img, keypoints, outImage=None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT))


def passes_lowe(d1, d2, ratio):
    return d1 < ratio * d2


def filter_lowe(knn, ratio):
    matches_lowe = []
    for i in range(len(knn)):
        if passes_lowe(knn[i][0].distance, knn[i][1].distance, ratio):
            matches_lowe.append(knn[i][0])
    return matches_lowe


def locate_object(matches_lowe, keypoints_ref, keypoints_frame, w, h, corners_ref):
    min_count_inliers = 8
    min_area_ratio = 0.025

    # для гомографии нужно минимум 4 пары
    if len(matches_lowe) < 4:
        return None

    points_ref = []
    points_frame = []

    for m in matches_lowe:
        index_ref = m.queryIdx
        index_frame = m.trainIdx
        points_ref.append(keypoints_ref[index_ref].pt)
        points_frame.append(keypoints_frame[index_frame].pt)

    # преобразование
    points_ref = np.array(points_ref, np.float32).reshape(-1, 1, 2)
    points_frame = np.array(points_frame, np.float32).reshape(-1, 1, 2)

    homography, inliers = cv2.findHomography(points_ref, points_frame, cv2.RANSAC)

    # матрицу найти не удалось
    if homography is None:
        return None

    if int(np.sum(inliers)) >= min_count_inliers:
        corners_frame = np.array(cv2.perspectiveTransform(corners_ref, homography), dtype=np.int32).reshape(-1, 4, 2)
        # площадь изображения
        if (cv2.contourArea(corners_frame[0]) >= min_area_ratio * (w*h)):
            # рамка без перекрутов и вдавливаний
            if (cv2.isContourConvex(corners_frame[0])):
                return corners_frame

    # объект не виден, ни одна проверка не прошла
    return None


def draw_border(img, corners_frame, caption):
    # рисует прямо на img
    cv2.polylines(img, corners_frame, isClosed=True, color=(0,255,0), thickness=1, lineType=cv2.LINE_8)
    cv2.putText(img, str(caption), corners_frame[0][3] + (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    return img


def save_border(img, corners_frame, caption, name):
    draw_border(img, corners_frame, caption)
    return cv2.imwrite(str(name), img)


def longest_run(border_on_frame):
    # самая длинная серия кадров подряд, где рамка есть
    longest = 0
    current = 0
    for i in range(len(border_on_frame)):
        if border_on_frame[i] == 1:
            current += 1
        else:
            current = 0
        longest = max(longest, current)
    return longest


def make_reference(video_path, ref_path=None):
    # эталон из картинки, если путь передан
    if ref_path is not None:
        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if ref is None:
            raise Exception("reference not read: " + ref_path)
        return ref

    # иначе первый кадр видео
    video = cv2.VideoCapture(video_path)
    ok, first_frame = video.read()
    video.release()
    if not ok:
        raise Exception("video not read: " + video_path)
    return cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)


def track_video(video_path, path, method, ref=None, orb_features=800):
    k = 2
    ratio = 0.75

    if method == "SIFT":
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        detector = cv2.SIFT_create()
    elif method == "ORB":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        detector = cv2.ORB_create(nfeatures=orb_features)
    else: raise Exception("method must be SIFT or ORB")

    video = cv2.VideoCapture(video_path)
    if video.isOpened():
        # параметры берем из этого же видео
        fps = round(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # если эталон не передали, берем первый кадр этого же видео
        if ref is None:
            ref = make_reference(video_path)
        keypoints_ref, descriptors_ref = detect_and_describe(ref, detector)

        # углы эталона под его размер
        ref_h, ref_w = ref.shape
        corners_ref = np.array([(0, 0), (ref_w-1, 0), (ref_w-1, ref_h-1), (0, ref_h-1)], np.float32).reshape(-1, 1, 2)

        # 1 – рамка есть, 0 – нет
        border_on_frame = []
        # время поиска объекта на каждом кадре, в секундах
        times = []

        # путь может прийти строкой, unlink есть только у Path
        path = Path(path)
        path.unlink(missing_ok=True)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
        if writer.isOpened():
            for i in range(frame_count):
                ok, frame = video.read()
                if not ok:
                    break
                frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # засекаем время только на поиск объекта
                start = time.perf_counter()
                keypoints_frame, descriptors_frame = detect_and_describe(frame_grey, detector)

                # на размытом кадре точек может не быть, а для k=2 нужно хотя бы две
                if descriptors_frame is None or len(descriptors_frame) < 2:
                    corners_frame = None
                else:
                    knn = matcher.knnMatch(descriptors_ref, descriptors_frame, k=k)
                    matches_lowe = filter_lowe(knn, ratio)
                    corners_frame = locate_object(matches_lowe, keypoints_ref, keypoints_frame, width, height, corners_ref)

                times.append(time.perf_counter() - start)

                if corners_frame is not None:
                    draw_border(frame, corners_frame, "Mona Lisa")
                    border_on_frame.append(1)
                else:
                    border_on_frame.append(0)
                writer.write(frame)
        writer.release()
        video.release()

        # итоги прогона
        frames = len(border_on_frame)
        with_border = sum(border_on_frame)
        share_percent = round(100 * with_border / frames, 1)
        median_ms = round(float(np.median(times)) * 1000, 1)
        total_s = round(sum(times), 1)

        print(video_path, method)
        print('  кадров с рамкой:', with_border, 'из', frames, '(' + str(share_percent) + ' %)')
        print('  самая длинная серия подряд:', longest_run(border_on_frame), 'кадров')
        print('  время на кадр, медиана:', median_ms, 'мс, всего', total_s, 'с')


if __name__ == "__main__":
    track_video('data/mona-lisa.mp4', 'results/SIFT/video/tracked_SIFT.mp4', "SIFT")
    track_video('data/mona-lisa.mp4', 'results/ORB/video/tracked_ORB.mp4', "ORB")
    track_video('data/mona-lisa-blur.mp4', 'results/SIFT/video/tracked_blur_SIFT.mp4', "SIFT")
    track_video('data/mona-lisa-blur.mp4', 'results/ORB/video/tracked_blur_ORB.mp4', "ORB")
    track_video('data/mona-lisa-blur-extra-credit.mp4', 'results/SIFT/video/tracked_blur_extra_SIFT.mp4', "SIFT")
    track_video('data/mona-lisa-blur-extra-credit.mp4', 'results/ORB/video/tracked_blur_extra_ORB.mp4', "ORB")

    # собственное видео, эталон – отдельная картинка, у ORB 2000 точек
    ref_my = make_reference('data/My/my_video.mp4', 'results/my_reference.png')
    track_video('data/My/my_video.mp4', 'results/SIFT/video/my_video_SIFT.mp4', "SIFT", ref_my)
    track_video('data/My/my_video.mp4', 'results/ORB/video/my_video_ORB.mp4', "ORB", ref_my, 2000)


# numbered_path = Path('results/numbered.mp4')
# frame_100 = cv2.imread('results/frame_100.png', cv2.IMREAD_GRAYSCALE)
# frame_100_color = cv2.imread('results/frame_100.png')
# ok, frame = video.read()
# if ok:
#     cv2.imwrite('results/reference.png', frame)


# video.set(cv2.CAP_PROP_POS_FRAMES, 0)
# numbered_path.unlink(missing_ok=True)
# writer = cv2.VideoWriter(numbered_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
# for i in range(frame_count):
#     ok, frame = video.read()
#
#     if not ok:
#         break
#     if i == 100:
#         cv2.imwrite('results/frame_100.png', frame)
#     cv2.putText(frame, str(i), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
#     writer.write(frame)
# writer.release()
# video.release()
# разбор кадра 100 для блоков 2–4, картинки уже в results/
# detector_SIFT = cv2.SIFT_create()
# detector_ORB = cv2.ORB_create(nfeatures=2000)


# keypoints_ref_SIFT, descriptors_ref_SIFT = detect_and_describe(reference, detector_SIFT)
# keypoints_ref_ORB, descriptors_ref_ORB = detect_and_describe(reference, detector_ORB)
# keypoints_100_SIFT, descriptors_100_SIFT = detect_and_describe(frame_100, detector_SIFT)
# keypoints_100_ORB, descriptors_100_ORB = detect_and_describe(frame_100, detector_ORB)

# save_keypoints(reference, keypoints_ref_SIFT, 'results/SIFT/keypoints_ref_SIFT.png')
# save_keypoints(reference, keypoints_ref_ORB, 'results/ORB/keypoints_ref_ORB.png')
# save_keypoints(frame_100, keypoints_100_SIFT, 'results/SIFT/keypoints_100_SIFT.png')
# save_keypoints(frame_100, keypoints_100_ORB, 'results/ORB/keypoints_100_ORB.png')


# matcher_SIFT = cv2.BFMatcher(cv2.NORM_L2)
# matcher_ORB = cv2.BFMatcher(cv2.NORM_HAMMING)


# matches_all_SIFT = matcher_SIFT.match(descriptors_ref_SIFT, descriptors_100_SIFT)
# matches_all_ORB = matcher_ORB.match(descriptors_ref_ORB, descriptors_100_ORB)


# knn_SIFT = matcher_SIFT.knnMatch(descriptors_ref_SIFT, descriptors_100_SIFT, k=k)
# knn_ORB = matcher_ORB.knnMatch(descriptors_ref_ORB, descriptors_100_ORB, k=k)


# img_matches_all_SIFT = cv2.drawMatches(reference, keypoints_ref_SIFT, frame_100, keypoints_100_SIFT, matches_all_SIFT, outImg=None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# img_matches_all_ORB = cv2.drawMatches(reference, keypoints_ref_ORB, frame_100, keypoints_100_ORB, matches_all_ORB, outImg=None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# cv2.imwrite('results/SIFT/matches_all_100_SIFT.png', img_matches_all_SIFT)
# cv2.imwrite('results/ORB/matches_all_100_ORB.png', img_matches_all_ORB)


# matches_lowe_SIFT = filter_lowe(knn_SIFT, 0.75)
# matches_lowe_ORB = filter_lowe(knn_ORB, 0.75)

# img_matches_lowe_SIFT = cv2.drawMatches(reference, keypoints_ref_SIFT, frame_100, keypoints_100_SIFT, matches_lowe_SIFT, outImg=None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# img_matches_lowe_ORB = cv2.drawMatches(reference, keypoints_ref_ORB, frame_100, keypoints_100_ORB, matches_lowe_ORB, outImg=None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# cv2.imwrite('results/SIFT/matches_lowe_100_SIFT.png', img_matches_lowe_SIFT)
# cv2.imwrite('results/ORB/matches_lowe_100_ORB.png', img_matches_lowe_ORB)
# corners_100_SIFT, inliers_100_SIFT = locate_object(matches_lowe_SIFT, keypoints_ref_SIFT, keypoints_100_SIFT)
# corners_100_ORB, inliers_100_ORB = locate_object(matches_lowe_ORB, keypoints_ref_ORB, keypoints_100_ORB)


# if corners_100_SIFT is not None:
#     draw_write_border(frame_100_color.copy(), corners_100_SIFT, "Mona Lisa", 'results/SIFT/border_100_SIFT.png')
# if corners_100_ORB is not None:
#     draw_write_border(frame_100_color.copy(), corners_100_ORB, "Mona Lisa", 'results/ORB/border_100_ORB.png')