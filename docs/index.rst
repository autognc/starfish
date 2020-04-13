====================================
Documentation for Starfish
====================================
Starfish is a Python library for automatically creating synthetic, labeled image data using Blender.

This library it extends Blender's powerful Python scripting ability, providing utilities that make it easy to generate
long sequences of synthetic images with intuitive parameters. It was designed for people who, like me, know Python
better than they know Blender.

These sequences can be smoothly interpolated from waypoint to waypoint, much like a traditional keyframe-based
animation. They can also be exhaustive or random, containing images with various object poses, backgrounds, and lighting
conditions. The intended use for these sequences is the generation of training and evaluation data for machine learning
tasks where annotated real data may be difficult to obtain.

.. contents:: Contents
    :local:
    :depth: 2

.. toctree::
    :hidden:

    Home <self>
    api/index

Installation
------------------------------------
#. Identify the location of your Blender scripts directory. This can be done by opening Blender, clicking on the
   'Scripting' tab, and entering ``bpy.utils.script_path_user()`` in the Python console at the bottom. Generally, on
   Linux, the default location is ``~/.config/blender/[VERSION]/scripts``. From now on, this path will be referred to as
   ``[SCRIPTS_DIR]``.
#. Create the addon modules directory, if it does not exist already: ``mkdir -p [SCRIPTS_DIR]/addons/modules``
#. Install the library to Blender: ``pip install https://github.com/autognc/starfish --no-deps --target
   [SCRIPTS_DIR]/addons/modules``. Starfish does not require any additional packages besides what is already bundled
   with Blender, which is why ``--no-deps`` can be used.

Starfish can also be pip-installed normally without Blender for testing purposes or for independent usage of the
`postprocessing` module.

Quickstart
------------------------------------

Recommended Reading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To use Starfish, you're probably going to have to interact with Blender using the `Python API
<https://docs.blender.org/api/current/>`_. This library also makes heavy use of `mathutils
<https://docs.blender.org/api/current/mathutils.html>`_, which is an independent math library that comes bundled with
Blender.

Running Scripts in Blender
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The easiest way to experiment with the library is by opening Blender, navigating to the Scripting tab, and hitting the
plus button to create a new script. You can then import starfish, write some code, and hit ``Alt+P`` to see what it does.

Once you're ready to execute a more long-running script, you can write it outside Blender and then execute it using
``blender file.blend --background --python script.py`` (or ``blender file.blend -b -P script.py`` for short).

Generating Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Frames
"""""""""""""""""""
At the core of Starfish is the `Frame <starfish.Frame>` class, which represents a single image of a single object. A frame is defined by 6 parameters::

    frame = starfish.Frame(
        position=(0, 0, 0),
        distance=10,
        pose=mathutils.Euler([0, math.pi, 0]),
        lighting=mathutils.Euler([0, 0, 0]),
        offset=(0.3, 0.7),
        background=mathutils.Euler([math.pi / 2, 0, 0])
    )

See the `starfish.Frame` documentation for more details about what each parameter means. Once you have a frame object, you use it to
'set up' your scene::
    
    frame.setup(
        bpy.data.scenes['MyScene'],
        bpy.data.objects['MyObject'],
        bpy.data.objects['Main_Camera'],
        bpy.data.objects['The_Sun']
    )

This moves all the objects so that the image that the camera sees matches up with the parameters in the frame object. At
this point, you can render the frame using `bpy.ops.render.render`. You can also dump metadata about a frame into JSON
format using the `Frame.dumps <starfish.Frame.dumps>` method.

Sequences
"""""""""""""""""""
Of course, Starfish wouldn't be very useful without the ability to create multiple frames at once. The `Sequence
<starfish.Sequence>` class is essentially just a list of frames, but with several classmethod constructors for
generating these sequences of frames in different ways. For example, `Sequence.interpolated
<starfish.Sequence.interpolated>` generates 'animated' sequences that smoothly interpolate between keyframes, and
`Sequence.exhaustive <starfish.Sequence.exhaustive>` generates long sequences that contain every possible combination of
the parameters given.

Once you've created a sequences, you can iterate through its frames like so::

    seq = starfish.Sequence...

    for frame in seq:
        frame.setup(...)
        bpy.ops.render.render()

The `Sequence.bake <starfish.Sequence.bake>` method also provides an easy way to 'preview' sequences that you're working
on in Blender. See `Sequence <starfish.Sequence>` for more detail.

Postprocessing
"""""""""""""""""""
Starfish also contains a (currently small) `postprocessing module <starfish.postprocessing>` for common image
postprocessing operations.

One common type of annotation generated for computer vision task is some sort of segmentation mask (e.g. using the `ID
Mask Node <https://docs.blender.org/manual/en/latest/compositing/types/converter/id_mask.html>`) where having perfectly
uniform colors is important. Unfortunately, I've often encountered an issue in Blender where the output colors differ
slightly: for example, instead of the background being solid ``rgb(0, 0, 0)`` black, it will actually be a random mix of
``rgb(0, 0, 1)``, ``rgb(1, 1, 0)``, etc. The `normalize_mask_colors <starfish.postprocessing.normalize_mask_colors>`
function can be used to clean up such images.

Once a mask has been cleaned up, `get_bounding_boxes_from_mask <starfish.postprocessing.get_bounding_boxes_from_mask>`
and `get_centroids_from_mask <starfish.postprocessing.get_centroids_from_mask>` can be used to get the bounding boxes
and centroids of segmented areas, respectively.

Example Script
^^^^^^^^^^^^^^^^^^^^^^
All together, here is what an image generation script might look like:

.. literalinclude:: ../example.py
