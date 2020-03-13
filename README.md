# starfish
Python library for automatically creating synthetic training data using Blender.

## Installation
1. Clone the repository: `git clone https://github.com/autognc/starfish`
2. Identify the location of your Blender scripts directory. This can be done by opening Blender, clicking on the "Scripting" tab, and entering `bpy.utils.script_path_user()` in the Python console at the bottom. Generally, on Linux, the default location is `~/.config/blender/[VERSION]/scripts`. From now on, this path will be referred to as `[SCRIPTS_DIR]`.
3. Create the addon modules directory, if it does not exist already: `mkdir -p [SCRIPTS_DIR]/addons/modules`
4. Enter the repository folder: `cd starfish`
5. Install the library to blender: `pip install . --target [SCRIPTS_DIR]/addons/modules`

## Quickstart
1. Open Blender and create your scene. Make sure you have, at minimum, an object, a camera, and a sun lamp.
2. Click on the "Scripting" tab.
3. Enter your script into the text panel on the upper left, and press `Alt+P` to run. See `example.py` for example usage.

## Notes
* This code has only been tested with Blender 2.8 beta.
