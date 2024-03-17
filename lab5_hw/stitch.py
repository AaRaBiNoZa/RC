import os
from collections import defaultdict
from typing import List, Tuple, Union, Dict, Collection

import cv2
import numpy as np

# gotten from lab 4
camera_matrix = np.array(
    [[1.28613634e+03, 0.00000000e+00, 6.92097797e+02],
     [0.00000000e+00, 1.30040995e+03, 4.25368745e+02],
     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00], ]
)

dist_coeffs = np.array(
    [[-0.30566098, -0.17641842, -0.00495141, -0.00106297, 0.72164286]]
)


def show(
        img_list: List[np.ndarray],
        window_name: str = ''
) -> None:
    """
    Displays each image from input list one by one.
    :param img_list:
    :param window_name:
    :return: None
    """
    s = np.concatenate(img_list, axis=1)

    # Using cv2.imshow() method
    # Displaying the image
    cv2.imshow(window_name, s)

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()


def load_pictures(
        dirname: str = 'data'
) -> Dict[str, np.ndarray]:
    images = {}

    for filename in os.listdir(dirname):
        print(filename)
        path = os.path.join(dirname, filename)

        img = cv2.imread(path)

        images[filename] = img

    return images


def undistort(
        images: Dict[str, np.ndarray],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        do_show: bool = False
) -> Dict[str, np.ndarray]:
    """
    Undistorts each image given a camera matrix and distortion coefficients.
    :param images:
    :param camera_matrix:
    :param dist_coeffs:
    :param do_show:
    :return: List of undistorted images
    """
    alpha = 0.

    any_image = list(images.values())[0]
    img_size = (any_image.shape[1], any_image.shape[0])

    assert (img_size == (1440, 960))

    rect_camera_matrix = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, img_size, alpha
    )[0]

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        np.eye(3),
        rect_camera_matrix,
        img_size,
        cv2.CV_32FC1
    )

    rect_images = {name: cv2.remap(img, map1, map2, cv2.INTER_LINEAR) for
                   name, img in images.items()}

    if do_show:
        for name in images.keys():
            show([images[name], rect_images[name]])

    return rect_images


def find_aruco_markers(
        img: np.ndarray
) -> Collection[np.ndarray]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    return detector.detectMarkers(img)


def find_aruco_pairs(
        img1: np.ndarray,
        img2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds corresponding aruco markers on the two images.
    :param img1:
    :param img2:
    :return: Tuple of (n,2) ndarrays with coordinates of corresponding points
    in both input images.
    """
    id_to_corner_dict = defaultdict(list)

    img1_corners, img1_ids, _ = find_aruco_markers(img1)
    img2_corners, img2_ids, _ = find_aruco_markers(img2)

    for corners, mark_id in zip(img1_corners, img1_ids):
        id_to_corner_dict[mark_id[0]].append(corners)
    for corners, mark_id in zip(img2_corners, img2_ids):
        id_to_corner_dict[mark_id[0]].append(corners)

    first_img_points = []
    second_img_points = []

    for corners_pair in id_to_corner_dict.values():
        # this means that the marker was in both images
        if len(corners_pair) == 2:
            first_img_points.append(corners_pair[0])
            second_img_points.append(corners_pair[1])

    first_img_points = np.concatenate(first_img_points, axis=0).reshape(-1, 2)
    second_img_points = np.concatenate(
        second_img_points, axis=0
    ).reshape(-1, 2)

    return first_img_points, second_img_points


def refine_pixel_locations(
        img: np.ndarray,
        points: np.ndarray
) -> np.ndarray:
    """
    :param img:
    :param points: ndarray of shape (n,2) with point xy coordinates
    :return: Refined coordinates of input points in a ndarray of shape (n,2)
    """
    points = points.reshape(-1, 1, 2)

    bl_w = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_points = cv2.cornerSubPix(
        bl_w,
        points,
        (11, 11),
        (-1, -1),
        criteria=(
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
    )

    return img_points.reshape(-1, 2)


def automatic_find_homography(
        source: np.ndarray,
        destination: np.ndarray,
        use_ransac: bool = True
) -> np.ndarray:
    """
    Given source and destination images computes a homography that transforms
    source into destination frame.
    :param source:
    :param destination:
    :param use_ransac: determines if all points should be taken into
    consideration or if RANSAC should be used
    :return: Homography matrix
    """
    source_points, destination_points = find_aruco_pairs(source, destination)

    source_points = refine_pixel_locations(source, source_points)
    destination_points = refine_pixel_locations(
        destination, destination_points
    )

    if use_ransac:
        homography = ransac(
            source_points, destination_points
        )
    else:
        homography = compute_homography(source_points, destination_points)

    return homography


def get_corners_centers(
        corners: np.ndarray
):
    """
    Computes centers for each corner of given ndarray of corners.
    :param corners: ndarray of shape (n,4,2) of corners coordinates in xy
    :return: ndarray of shape (n,2) of corners' centers in xy
    """
    return corners.mean(axis=1)


def determine_order(
        images: List[np.ndarray],
        do_show: bool = False
) -> np.ndarray:
    """
    Given a list of images determine their order left to right.
    It is enough that this function works on any subset of 3 images from the
    dataset, so we can just find
    at least 1 marker that appears in all 3 images (in these images there
    always exists such a marker).
    :param images:
    :param do_show: determines if should display the images with detected
    markers during the computation.
    :return: ndarray of image ids in correct order
    :raises: ValueError if a marker visible on all input images couldn't be
    found
    """
    aruco_id_to_x_coord = defaultdict(list)

    for img in images:
        corners, ids, _ = find_aruco_markers(img)
        corners = np.array(corners).squeeze()

        # corners_centers is a ndarray of shape (n,2) containing xy coords of
        # centers of corners
        corners_centers = get_corners_centers(corners)

        # order will be determined by x coordinates of the marker centers
        for idx, cor in zip(ids, corners_centers):
            aruco_id_to_x_coord[idx[0]].append(cor[0])

        if do_show:
            img_draw = img.copy()
            cv2.aruco.drawDetectedMarkers(img_draw, corners, ids)

            show([img_draw])

    # from left to right
    order = None
    for cors in aruco_id_to_x_coord.values():
        # a given marker was spotted in both images
        if len(cors) == len(images):
            order = np.argsort(cors)[::-1]
            return order

    if order is None:
        raise ValueError('No marker has appeared in all images')


def get_coords_of_plane(
        size: Tuple[int, int]
) -> np.ndarray:
    """
    Given 2d size returns (xy) coordinates in the image reference frame (origin
    in top left)
    :param size:
    :return: (n,2) ndarray of xy coordinates
    """
    xx, yy = np.meshgrid(
        np.arange(size[1]), np.arange(size[0])
    )

    coords = np.stack([xx, yy], axis=-1)

    return coords.reshape(-1, 2)


def to_homogenous(
        points: np.ndarray
) -> np.ndarray:
    """
    Given a ndarray of shape (n,2) describing coordinates in xy, transform it
    to (n,3) ndarray describing those points
    in homogenous coordinates.
    :param points:
    :return: (n,3) ndarray of vectors describing homogenous coordinates of
    points
    """
    one_stack = np.ones((points.shape[0], 1))
    return np.concatenate([points, one_stack], axis=-1)


def from_homogenous(
        points: np.ndarray
) -> np.ndarray:
    """
    Given (n,3) ndarray of vectors describing points in homogenous
    coordinates, transform them back into 2d.
    :param points:
    :return: (n,2) ndarray of point coordinates in xy
    """
    return points[..., :2] / points[..., 2].reshape(-1, 1)


def apply_homography(
        two_d_coords: np.ndarray,
        homography_matrix: np.ndarray
) -> np.ndarray:
    """
    Given (n,2) coordinates in xy, apply a homography and return the result.
    :param two_d_coords: (n,2) ndarray of 2d point coordinates
    :param homography_matrix:
    :return: (n,2) ndarray of points after transformation
    """
    vectors = to_homogenous(two_d_coords)
    vectors = vectors.reshape(-1, 3, 1)
    resulting_vectors = homography_matrix @ vectors
    return from_homogenous(resulting_vectors.reshape(-1, 3))


def compute_homography(
        source_pts: np.ndarray,
        dest_pts: np.ndarray
) -> np.ndarray:
    """
    Given source and destination point coordinates as (n,2) ndarrays,
    computes homography matrix.
    :param source_pts:
    :param dest_pts:
    :return: Homography matrix
    """
    A = np.zeros((source_pts.shape[0] * 2, 9), dtype=np.float64)

    for row, ((xs, ys), (xd, yd)) in enumerate(zip(source_pts, dest_pts)):
        A[2 * row] = np.array([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys, -xd])
        A[2 * row + 1] = np.array(
            [0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys, -yd]
        )

    _, _, V = np.linalg.svd(A)
    eingenvector = V[-1, :]

    return eingenvector.reshape(3, 3)


def test_homography(
        n=10,
        m=5
):
    """
    Tests if computing homography is correct.
    First picks a homography matrix randomly, then samples random points and
    transforms them.
    After that it uses compute_homography to try and recover the homography
    from points and checks if the two
    are the same (up to a scaling factor).
    :param n: number of repetitions
    :param m: number of points to be sampled (can't be lower than 4)
    :return: None
    """
    count = 0
    while count < n:
        rand_hom = np.random.rand(3, 3)

        if np.isclose(np.linalg.det(rand_hom), 0):
            continue

        rand_points = np.random.rand(m, 2)
        dest_points = apply_homography(rand_points, rand_hom)

        recovered_hom = compute_homography(rand_points, dest_points)
        scaling_factor = recovered_hom / rand_hom

        if not np.allclose(scaling_factor * rand_hom, recovered_hom):
            print(f'Test #{count} failed')
        else:
            print(f'Test #{count} passed')
        count += 1


def cumsum_reset(
        arr: np.ndarray,
        axis: int = 0
) -> np.ndarray:
    """
    Is exactly like np.cumsum, except when stumbling upon 0 in the array,
    resets the accumulator.
    :param arr:
    :param axis:
    :return: like np.cumsum(arr, axis=axis) but resets when seeing 0
    """
    no_reset = (arr == 1).cumsum(axis=axis)
    reset = (arr == 0)

    excess = np.maximum.accumulate(no_reset * reset, axis=axis)

    return no_reset - excess


def get_dist_transform_from_mask(
        mask: np.ndarray
) -> np.ndarray:
    """
    Computes distance transform (distance from the closest 0 for each pixel)
    based on binary mask of an image.
    Computes cumsum in both directions on both axes and then chooses
    elementwise minimum.
    It is needed, because the images can have holes inside (as a result
    of overlap).
    :param mask: binary mask of an image
    :return: a ndarray depicting distance from the background based on mask.
    Distance is a taxicab distance
    """
    left_right = cumsum_reset(mask, axis=-1)
    up_down = cumsum_reset(mask, axis=0)
    right_left = np.flip(
        cumsum_reset(np.flip(mask, axis=-1), axis=-1),
        axis=-1
    )
    down_up = np.flip(cumsum_reset(np.flip(mask, axis=0), axis=0), axis=0)

    return np.expand_dims(
        np.min([left_right, up_down, right_left, down_up], axis=0), axis=-1
    ).astype(np.int64)


def get_homography_size_and_origin(
        source: np.ndarray,
        destination: np.ndarray,
        proj_matrix: np.ndarray
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Given source and destination images with projection matrix transforming
    source to destination reference frame,
    compute the size and translation needed, so that the projection doesn't
    end up cropped.
    Translation is called origin, since it describes where the origin should
    be in the plane we end up projecting onto.
    :param source: source image
    :param destination: destination image
    :param proj_matrix:
    :return: size needed for the resulting image after applying homography
    and translation.
             Size is given in numpy order (yx), but translation is in xy
    """
    corners = np.array(
        [[0, 0], [0, source.shape[0]], [source.shape[1], 0],
         [source.shape[1], source.shape[0]], ]
    )

    corners = apply_homography(corners, proj_matrix)

    # compute output image size
    x_min = np.floor(min(0, corners[..., 0].min()))
    x_max = np.ceil(max(corners[..., 0].max(), destination.shape[1]))

    y_min = np.floor(min(0, corners[..., 1].min()))
    y_max = np.ceil(max(corners[..., 1].max(), destination.shape[0]))

    size = (int(y_max - y_min), int(x_max - x_min))
    origin = (-int(x_min), -int(y_min))

    return size, origin


def get_homography_size_and_origin_for_two(
        left: np.ndarray,
        destination: np.ndarray,
        right: np.ndarray,
        left_proj: np.ndarray,
        right_proj: np.ndarray
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Given left, right and destination images with projection matrices
    transforming left and right into destination reference frame,
    compute the size and translation needed, so that the projections don't
    end up cropped.
    Translation is called origin, since it describes where the origin should
    be in the plane we end up projecting onto.
    :param left: left source image
    :param destination: destination image
    :param right: right source image
    :param left_proj:
    :param right_proj:
    :return: size needed for the resulting image after applying homography
    and translation.
             Size is given in numpy order (yx), but translation is in xy
    """
    left_corners = np.array(
        [[0, 0], [0, left.shape[0]], [left.shape[1], 0],
         [left.shape[1], left.shape[0]], ]
    )

    left_corners = apply_homography(left_corners, left_proj)

    right_corners = np.array(
        [[0, 0], [0, right.shape[0]], [right.shape[1], 0],
         [right.shape[1], right.shape[0]], ]
    )

    right_corners = apply_homography(right_corners, right_proj)

    # compute output image size
    x_min = np.floor(
        min(0, left_corners[..., 0].min(), right_corners[..., 0].min())
    )
    x_max = np.ceil(
        max(
            left_corners[..., 0].max(),
            right_corners[..., 0].max(),
            destination.shape[1]
        )
    )

    y_min = np.floor(
        min(0, left_corners[..., 1].min(), right_corners[..., 1].min())
    )
    y_max = np.ceil(
        max(
            left_corners[..., 1].max(),
            right_corners[..., 1].max(),
            destination.shape[0]
        )
    )

    size = (int(y_max - y_min), int(x_max - x_min))
    origin = (-int(x_min), -int(y_min))

    return size, origin


def get_size_and_origin_from_two(
        s1: Tuple[int, int],
        s2: Tuple[int, int],
        o1: Tuple[int, int],
        o2: Tuple[int, int]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Given appropriate sizes and translations computed for two homographies,
    find a size and translation (origin) that will be good in terms of size
    for both transformations.
    Sizes given in yx,
    origins given in xy
    :param s1: computed size for the first projection
    :param s2: computed size for the second projection
    :param o1: computed origin for the first projection
    :param o2: computed origin for the second projection
    :return: combined size and origin
    """
    common_origin = max(o1[0], o2[0]), max(o1[1], o2[1])
    origin_translation_y = abs(o1[1] - o2[1])
    origin_translation_x = abs(o1[0] - o2[0])

    common_size_y = min(s1[0], s2[0]) + origin_translation_y
    common_size_x = min(s1[1], s2[1]) + origin_translation_x

    common_size = common_size_y, common_size_x

    return common_size, common_origin


def condition_for_err(
        distances: np.ndarray,
        best_val: Union[int, float],
        *args
) -> Tuple[bool, float]:
    """
    Only for use with RANSAC. (doing classess seemed like an overkill).
    Look in there for explanation.
    :param distances:
    :param best_val:
    :return: if given args pass this condition
    """
    this_val = np.linalg.norm(distances) / len(distances)

    return this_val < best_val, this_val


def condition_for_count(
        distances: np.ndarray,
        best_val: Union[int, float],
        t: int,
        *args
) -> Tuple[bool, float]:
    """
    Only for use with RANSAC. (doing classess seemed like an overkill).
    Look in there for explanation.
    :param distances:
    :param best_val:
    :param t:
    :return: if given args pass this condition
    """
    this_val = (distances < t).sum()

    return this_val > best_val, this_val


def ransac(
        source_points: np.ndarray,
        dest_points: np.ndarray,
        n: int = 8,
        k: int = 1000,
        t: float = 3.,
        optimze: str = 'err',
        d: int = 2,
        log: bool = False
) -> np.ndarray:
    """
    RANSAC implementation for finding homography between source_points and
    dest_points.
    Can be set to optimize either count of inliers or l2 norm of distances
    between predictions and inliers.
    When we sample random points, we make sure to try to sample every point
    from
    some marker. It makes ransac:
    a) more deterministic (fewer permutations)
    b) adjust the homography to markers - without it, things like a line on the
    wall or lines on the floor maybe matched better, but there were ghosts
    in the middle, because the toys (with markers) were poorly represented.
    Now the toys are much better visible, and the rest of the room has some
    errors but overall it's good.
    :param source_points: we assume they're in order of the markers
    :param dest_points: we assume they're in order of the markers
    :param n: number of points to be sampled in each iteration
    :param k: number of iterations
    :param t: threshold to count sth as an inlier
    :param optimze: can be either 'err' or 'count'
    :param d: determines the least amount of predictions within threshold
              the model has to have to be considered maybe good
    :param log: if should print diagnostic info
    :return: homography matrix
    """
    best_model = None
    best_fitness = None
    fitness_condition = None

    if optimze == 'err':
        fitness_condition = condition_for_err
        best_fitness = np.inf
    else:
        fitness_condition = condition_for_count
        best_fitness = 0

    data = np.concatenate([source_points, dest_points], axis=-1)
    for _ in range(k):
        # so we end up sampling all the points from a given marker when
        # possible
        np.random.shuffle(data.reshape(-1, 4, 4))
        maybe_inliers = data[:n, ...]

        sample_source_pts = maybe_inliers[..., :2]
        sample_dest_pts = maybe_inliers[..., 2:]

        maybe_model = compute_homography(sample_source_pts, sample_dest_pts)

        rest_of_data = data[n:]
        source_pts = rest_of_data[..., :2]
        dest_pts = rest_of_data[..., 2:]

        preds = apply_homography(source_pts, maybe_model)
        distances = np.linalg.norm(preds - dest_pts, axis=-1)

        inlier_mask = distances < t

        if inlier_mask.sum() >= d:
            inliers = np.concatenate(
                [rest_of_data[inlier_mask], maybe_inliers], axis=0
            )
            better_model = compute_homography(
                inliers[..., :2], inliers[..., 2:]
            )

            preds = apply_homography(inliers[..., :2], better_model)
            distances = np.linalg.norm(preds - inliers[..., 2:], axis=-1)

            is_better, this_fitness = fitness_condition(
                distances, best_fitness, t
            )
            if is_better:
                best_fitness = this_fitness
                best_model = better_model
    if best_model is None:
        raise ValueError('RANSAC ERROR: Change parameters')

    if log:
        print(best_fitness)
    return best_model


def project(
        image: np.ndarray,
        projection_matrix: np.ndarray,
        output_size=None,
        new_origin=(0, 0),
        dtype=np.uint8,
        stitches_pixel_dist=0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects an image onto a plane with a given output size and origin (if
    not given, constructed by default, but the
    resulting image may be cropped). Returns masks useful for stitching later
    on.
    One mask is just a binary mask with input image after transformation
    shape (distance transform can break if there
    are black values in the input image, so it will be computed with this
    mask).
    Second mask is a binary mask of an image cropped by stitches_pixel_dist
    from every side, usefull later on for
    blending.
    :param image:
    :param projection_matrix:
    :param output_size: size of an output image (in numpy yx format)
    :param new_origin: origin position in the new image (int xy format)
    :param dtype: type of the result
    :param stitches_pixel_dist: how big of a cut should be represented in one
    of the output masks
    :return: a projected image, a binary mask of the projected image,
    a binary mask of a cropped projected image
    """
    assert (len(image.shape) == 3)  # dealing with 2d images with some channels

    inverse_projection = np.linalg.inv(projection_matrix)

    if output_size is None:
        output_shape = image.shape
    else:
        output_shape = (*output_size, image.shape[-1])

    result_image = np.zeros(output_shape)
    binary_mask = np.zeros(output_shape[:2])
    cropped_binary_mask = binary_mask.copy()

    # compute coordinates in resulting image frame
    result_img_coords = get_coords_of_plane(result_image.shape[:2])

    # this is an array of coordinates in destination frame of reference (that
    # is translated pixel coords)
    destination_coords = (result_img_coords.copy() - np.array(new_origin))

    # find correspondence in source image
    source_pts = apply_homography(destination_coords, inverse_projection)
    source_pts = np.rint(source_pts).astype(np.int64)  # nearest neighbour

    in_range_mask = (
            (source_pts[..., 0] >= 0) &
            (source_pts[..., 0] < image.shape[1]) &
            (source_pts[..., 1] >= 0) &
            (source_pts[..., 1] < image.shape[0])
    )

    cropped_in_range_mask = (
            (source_pts[..., 0] >= stitches_pixel_dist) &
            (source_pts[..., 0] < image.shape[1] - stitches_pixel_dist) &
            (source_pts[..., 1] >= stitches_pixel_dist) &
            (source_pts[..., 1] < image.shape[0] - stitches_pixel_dist)
    )

    valid_res_points = result_img_coords[in_range_mask]
    valid_source_points = source_pts[in_range_mask]

    result_image[
        valid_res_points[..., 1], valid_res_points[..., 0]
    ] = image[
        valid_source_points[..., 1], valid_source_points[..., 0]
    ]
    binary_mask[valid_res_points[..., 1], valid_res_points[..., 0]] = 1

    cropped_mask_points = result_img_coords[cropped_in_range_mask]
    cropped_binary_mask[
        cropped_mask_points[..., 1], cropped_mask_points[..., 0]
    ] = 1

    return result_image.astype(dtype), binary_mask, cropped_binary_mask


def translate_and_resize(
        img: np.ndarray,
        output_size: Tuple[int, int],
        new_origin: Tuple[int, int] = (0, 0),
        stitches_pixel_dist: int = 0,
        dtype=np.uint8
):
    """
    Given an image, move it onto a "plane" with different size and origin.
    Can be thought as simplified version of project(), since the homography
    is just a translation.
    :param img:
    :param output_size: size of an output image (2d in numpy order)
    :param new_origin: origin position in the new image
    :param stitches_pixel_dist: how big of a cut should be represented in one
    of the output masks
    :param dtype: type of the result
    :return: a translated image, a binary mask of the translated image,
    a binary mask of a cropped translated image
    """
    assert (len(img.shape) == 3)

    dest_plane = np.zeros((*output_size, img.shape[2]), dtype=dtype)

    dest_plane[
    new_origin[1]:new_origin[1] + img.shape[0],
    new_origin[0]:new_origin[0] + img.shape[1]
    ] = img.astype(dtype)

    binary_mask = np.zeros(dest_plane.shape[:2])
    binary_mask[
    new_origin[1]:new_origin[1] + img.shape[0],
    new_origin[0]:new_origin[0] + img.shape[1]
    ] = 1

    cropped_binary_mask = np.zeros(dest_plane.shape[:2])
    cropped_binary_mask[
    new_origin[1] + stitches_pixel_dist:new_origin[1] + img.shape[
        0
    ] - stitches_pixel_dist,
    new_origin[0] + stitches_pixel_dist:new_origin[0] + img.shape[
        1
    ] - stitches_pixel_dist
    ] = 1

    return dest_plane, binary_mask, cropped_binary_mask


def warp(
        source: np.ndarray,
        destination: np.ndarray,
        blending_pixels_number: int = 0,
        over: bool = True,
        proj_matrix: np.ndarray = None, ):
    """
    Given a source and destination, project source into
    destination reference frame and adjust the size of the resulting image
    so nothing ends up cropped.
    While stitching uses weighted average to decrease seams.
    Weights are not computed based on the whole images but rather some cropped
    versions. Size of the crop is a parameter.
    :param source:
    :param destination:
    :param blending_pixels_number: size of pixel to crop from edges for weights
    :param over: whether source image should be on top or below the destination
    :param proj_matrix:
    :return:
    """
    if proj_matrix is None:
        proj_matrix = automatic_find_homography(source, destination)

    result_size, origin_in_output = get_homography_size_and_origin(
        source,
        destination,
        proj_matrix
    )

    result_image = np.zeros((*result_size, source.shape[2]))
    source_proj, source_proj_binary_mask, cropped_source_proj_mask = project(
        source,
        proj_matrix,
        result_image.shape[:2],
        origin_in_output,
        stitches_pixel_dist=blending_pixels_number
    )

    # place destination in result image
    dest_proj, dest_proj_mask, cropped_dest_proj_mask = translate_and_resize(
        destination,
        result_size,
        origin_in_output,
        blending_pixels_number
    )

    if over:
        dest_proj_mask *= (1 - cropped_source_proj_mask)
    else:
        source_proj_binary_mask *= (1 - cropped_dest_proj_mask)

    dest_proj_weights = get_dist_transform_from_mask(dest_proj_mask)
    source_proj_weights = get_dist_transform_from_mask(source_proj_binary_mask)

    result_image = (
            (
                    (
                            source_proj * source_proj_weights
                    ) + (
                            dest_proj * dest_proj_weights)
            ) / (
                    source_proj_weights + dest_proj_weights + 1e-8
            )
    ).astype(
        np.uint8
    )

    return result_image


def stitch(
        left_img,
        destination_img,
        right_img,
        blending_pixels_number=50
):
    """
    Stitches 3 images together as in the lecture, order from top is left, mid,
    right.
    :param left_img:
    :param destination_img:
    :param right_img:
    :param blending_pixels_number: determines how much cropped should the mask
    be for weight computation
    :return: an image that is a combination of left, destination and right
    """
    left_to_dest_proj = automatic_find_homography(left_img, destination_img)
    right_to_dest_proj = automatic_find_homography(right_img, destination_img)

    result_size, origin_in_output = get_homography_size_and_origin_for_two(
        left_img,
        destination_img,
        right_img,
        left_to_dest_proj,
        right_to_dest_proj
    )

    result_image = np.zeros((*result_size, 3))

    left_proj, left_proj_mask, cropped_left_proj_mask = project(
        left_img,
        left_to_dest_proj,
        result_image.shape[:2],
        origin_in_output,
        stitches_pixel_dist=blending_pixels_number
    )

    right_proj, right_proj_mask, cropped_right_proj_mask = project(
        right_img,
        right_to_dest_proj,
        result_image.shape[:2],
        origin_in_output,
        stitches_pixel_dist=blending_pixels_number
    )

    # place destination in result image
    dest_proj, dest_proj_mask, cropped_dest_proj_mask = translate_and_resize(
        destination_img,
        result_size,
        origin_in_output,
        blending_pixels_number
    )

    # ensures that left covers middle which covers right
    dest_proj_mask *= (1 - cropped_left_proj_mask)
    right_proj_mask *= (1 - cropped_dest_proj_mask)

    dest_proj_weights = get_dist_transform_from_mask(dest_proj_mask).astype(
        np.float64
    )
    left_proj_weights = get_dist_transform_from_mask(left_proj_mask).astype(
        np.float64
    )
    right_proj_weights = get_dist_transform_from_mask(right_proj_mask).astype(
        np.float64
    )

    result_image = ((
                            (
                                    dest_proj * dest_proj_weights
                            ) + (
                                    left_proj * left_proj_weights
                            ) + (
                                    right_proj * right_proj_weights
                            )
                    ) / (
                            dest_proj_weights + left_proj_weights +
                            right_proj_weights + 1e-8
                    )
                    ).astype(
        np.uint8
    )

    return result_image


def task1(
        do_show: bool = False
):
    images = load_pictures()
    return undistort(
        images, camera_matrix, dist_coeffs, do_show=do_show
    )


def task2(
        image: np.ndarray,
        proj_matrix: np.ndarray,
):
    new_origin = (0, 0)

    projection, _, _ = project(image, proj_matrix)

    show([image, projection], 'task2')


def task3():
    test_homography()


def task4():
    dest_pts = [  # hw12
        np.array([769., 748.]), np.array([874., 323.]),  #
        np.array([965., 253.]),  #
        np.array([1111., 625.]),  #
        np.array([1209., 504.]),  #
        np.array([1303., 346.]),  #

    ]
    source_pts = [  # hw13
        np.array([373., 847.]), np.array([538., 374.]),  #
        np.array([640., 295.]),  #
        np.array([786., 685.]),  #
        np.array([909., 548.]),  #
        np.array([1012., 376.]),  #
    ]

    homography_matrix = compute_homography(
        np.array(source_pts), np.array(dest_pts)
    )

    return homography_matrix


def task5(
        hw13_img,
        hw12_img
):
    proj_matrix = task4()

    result = warp(hw13_img, hw12_img, 50, False, proj_matrix)

    cv2.imwrite('./task_5_stitched.jpg', result)


def task6(
        hw13_img,
        hw12_img
):
    # projection matrix
    proj_matrix = automatic_find_homography(hw13_img, hw12_img, False)
    result = warp(hw13_img, hw12_img, 50, False, proj_matrix)

    cv2.imwrite('./task_6_stitched.jpg', result)


def task7(
        images
):
    images_in_name_order = np.array(
        [images['hw11.jpg'], images['hw12.jpg'], images['hw13.jpg'],
         images['hw14.jpg']]
    )

    first_subset = images_in_name_order[[0, 1, 2]]
    first_panorama_order = determine_order(first_subset)
    first_left, first_mid, first_right = first_subset[first_panorama_order]
    first_panorama = stitch(first_left, first_mid, first_right, 50)

    second_subset = images_in_name_order[[0, 3, 2]]
    second_panorama_order = determine_order(second_subset)
    second_left, second_mid, second_right = second_subset[
        second_panorama_order]
    second_panorama = stitch(second_left, second_mid, second_right, 25)

    cv2.imwrite('./task_7_1.jpg', first_panorama)
    cv2.imwrite('./task_7_2.jpg', second_panorama)


if __name__ == '__main__':
    images = task1()

    # task here bcs to do task3 i need some homography
    homography_for_task_2 = task4()
    task2(images['hw13.jpg'], homography_for_task_2)

    task3()

    task5(images['hw13.jpg'], images['hw12.jpg'])
    task6(images['hw13.jpg'], images['hw12.jpg'])
    task7(images)
