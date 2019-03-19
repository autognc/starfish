# synthetic-spacecraft-imagery
Python library for automatically creating synthetic training data using Blender.

## Quickstart (for now)
1. Open Blender and create your scene. Make sure you have, at minimum, an object, a camera, and a sun lamp.
2. Go to the "Scripting" tab and copy the entirety of `ssi/main.py` into the built-in text editor.
3. Scroll down to the bottom and modify as necessary below the `# test code` comment.
4. Press `Alt+P` to run the code.

## Notes
* This code has only been tested with Blender 2.8 beta.
* There is a known issue with the `offset` parameter, in that it is sometimes slightly wrong (usually on the y-axis). This appears to be due to a bug in Blender's `Camera.angle_x` and `Camera.angle_y` attributes.
