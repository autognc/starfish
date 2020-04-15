from mathutils import Vector


def project_keypoints_onto_image(keypoints, scene, obj, camera):
    """Converts 3D keypoints of an object into their corresponding 2D coordinates on the image.

    This function takes a list of keypoints represented as 3D coordinates in object space, and then projects them
    onto the camera to get their corresponding 2D coordinates on the image. It uses the current location and
    orientation of the input Blender objects. Typical usage would be to call this function after
    `Frame.setup <starfish.Frame.setup>` and then store the 2D locations as metadata for that frame::

        frame.setup(scene, obj, camera, sun)
        frame.keypoints = project_keypoints_onto_image(keypoints, scene, obj, camera)
        with open('meta...', 'w') as f:
            f.write(frame.dumps())

    :param keypoints: a list of 3D coordinates corresponding to the locations of the keypoints in the object space, e.g.
        the output of `generate_keypoints <starfish.annotation.generate_keypoints>`
    :param scene: (BlendDataObject): the scene to use for aspect ratio calculations. Note that this should be the
        scene that you intend to perform the final render in, not necessarily the one that your objects exist in. If
        you render in a scene that has an output resolution with a different aspect ratio than the output
        resolution of this scene, then the results may be incorrect.
    :param obj: (BlendDataObject): the object to use
    :param camera: (BlendDataObject): the camera to use

    :return: a list of (y, x) coordinates in the same order as ``keypoints`` where (0, 0) is the top left corner of
        the image and (1, 1) is the bottom right
    """
    from bpy_extras.object_utils import world_to_camera_view
    results = []
    for keypoint in keypoints:
        camera_coord = world_to_camera_view(scene, camera, obj.matrix_world @ Vector(keypoint))
        results.append((1 - camera_coord.y, camera_coord.x))
    return results
