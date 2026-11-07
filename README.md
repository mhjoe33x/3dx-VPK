# Re3DX+ VPK Tool

A Python desktop tool for viewing and editing the vinyl/sticker designs stored in **Re3DX+ car `.vpk` files**. It can replace individual stickers, recolour supported single-colour designs, preview changes on a car using its `.wpk` pack, and export UV templates for creating accurately aligned artwork.

> **Keep backups.** Work on copies of your game files whenever possible. Saving over an existing `.vpk` creates a `.bak` backup, but you should still keep your own untouched copy.

## Features

- Browse the sticker entries in a car's `.vpk` file using a grid based on the in-game sticker menu.
- Replace sticker artwork with image files. Transparent PNG or TGA artwork is recommended; images are prepared at 512 × 512 pixels.
- Recolour supported one-colour stickers (such as line, fire, and tribal designs) without having to edit the artwork externally.
- Undo sticker edits and save the edited car file, or use **Save as...** to write a separate copy.
- Export all sticker images to PNG, edit them in an image editor, and import the changed files back.
- Preview a selected sticker on the car using its `.wpk` car pack.
- View the car from several angles, choose optional body parts, adjust lighting and camera distance, and rotate the car by hand.
- Overlay the car's real UV panel borders on the sticker or on the car to check artwork placement.
- Export UV templates for one car or for a folder of car packs, including available aero kits.
- Fit roughly aligned artwork to the car's UV layout using draggable control pins.
- Export car textures and use command-line tools for inspection, replacement, packing, and UV checks.

## Requirements

- **Python 3.9 or newer**
- **Tkinter** (included with the standard Python installer for Windows; on some Linux distributions it must be installed separately)
- Python packages listed in [`requirements.txt`](requirements.txt):
  - NumPy
  - Pillow
- **Your own Re3DX+ game data files** (see below)

### You must provide your own game files

This repository contains the editor software only. **It does not include, host, or provide downloads for Re3DX+ game files or copyrighted game assets.** You must supply the game data files needed for the features you use, obtained from a copy or source you are authorized to access.

- A car's **`.vpk` file** is needed to open and edit that car's sticker/vinyl data.
- The matching car **`.wpk` file** is needed for the live 3D car preview and car UV-template tools.
- For some aero `.wpk` packs, the stock car `.wpk` must also be available in the same folder. See [Live car preview and UV workflow](#live-car-preview-and-uv-workflow).

The tool cannot supply missing game files. Keep your original files backed up and do not upload or redistribute game files in this repository. When asking for support, share error messages, logs, or screenshots rather than copyrighted game data.

## Installation and launch

### Windows

1. Install Python 3.9+ from [python.org](https://www.python.org/downloads/).
2. During installation, enable **Add Python to PATH**. Make sure the installation includes **Tcl/Tk and IDLE** so Tkinter is available.
3. Extract this repository to a folder.
4. Double-click `run_vpk_tool.bat`.

The Windows launcher checks for NumPy and Pillow and installs the requirements when they are missing. You can also open PowerShell in the project folder and run:

```powershell
py -3 -m pip install -r requirements.txt
py -3 vpk_tool.py
```

To open a particular car file directly:

```powershell
py -3 vpk_tool.py "C:\path\to\car.vpk"
```

You can also drag a `.vpk` file onto `run_vpk_tool.bat`.

### macOS and Linux

Install Python 3.9+, Tkinter, and the packages in `requirements.txt`. From a terminal in the project folder, run:

```bash
python3 -m pip install -r requirements.txt
python3 vpk_tool.py
```

Or use the launcher:

```bash
chmod +x run_vpk_tool.sh
./run_vpk_tool.sh
```

To open a car file directly:

```bash
python3 vpk_tool.py "/path/to/car.vpk"
```

## Basic sticker-editing workflow

1. **Open a car file.** Open the target car's `.vpk` from the game's vinyl folder.
2. **Select a sticker.** Click the sticker entry in the grid. The number identifies its place in the game's sticker menu.
3. **Replace or recolour it.** Choose **Replace image** to load your artwork, or use the colour control for supported single-colour designs. Use the UV overlay or live car preview to check alignment.
4. **Save your work.** Use **Save to game** to save to the opened file, or **Save as...** to create a separate file. Keep an untouched backup of the original.

### Sticker groups

| Sticker numbers | Group |
|---|---|
| 1–10 | Line 1 |
| 11–20 | Line 2 |
| 21–30 | Fire 1 |
| 31–40 | Fire 2 |
| 41 | Namco |
| 42 | Pac-Man |
| 43–47 | Works colours |
| 48–57 | Line 3 |
| 58–67 | Fire 3 |
| 68–77 | Tribal |

Some designs have separate images for the normal body and aero bodies. Some Works-colour designs also have paint-specific or carbon-bonnet variants. The editor shows the relevant body or version selector when the opened file contains those variants.

## Live car preview and UV workflow

To preview your work, open a car `.vpk` and use **View → Show on the car...** (or the preview button in the main window). Select the matching car `.wpk` if the tool does not find it automatically.

The preview includes fixed camera views and a free-rotation view, optional body-part selections, lighting modes, and UV overlays. The UV template tools can export the car's panel layout as a PNG; paint on a separate layer, hide the template layer before exporting your artwork, and use **Replace image** to apply it.

For aero `.wpk` packs, keep the stock car `.wpk` in the same folder when the aero pack borrows textures from it. For example, keep `maz_FC3S.wpk` beside `maz_FC3S_aeroA.wpk`.

## Command-line usage

Run commands from the project directory. Replace `python` with `py -3` on Windows if needed.

```bash
# Open the graphical editor
python vpk_tool.py

# Open a car file in the editor
python vpk_tool.py CAR.vpk

# Show information about a VPK file
python vpk_tool.py info CAR.vpk

# Replace one sticker (writes to CAR.vpk unless OUT.vpk is supplied)
python vpk_tool.py replace CAR.vpk 21 my_sticker.png [OUT.vpk]

# Export a car's UV template (default size: 2048 pixels)
python vpk_tool.py uv CAR.wpk OUT.png [SIZE]

# Export UV templates for all car packs in a directory (includes subdirectories)
python vpk_tool.py uvall CARS_DIR OUT_DIR [SIZE]

# Draw a car's UV borders over a sticker image to check placement
python vpk_tool.py uvcheck CAR.wpk STICKER.png CHECK.png

# Unpack a VPK to PNG files and project.json
python vpk_tool.py unpack CAR.vpk DIR

# Rebuild a VPK from an unpacked project.json
python vpk_tool.py pack PROJECT.json OUT.vpk
```

The optional `[OUT.vpk]` and `[SIZE]` arguments are alternatives shown in the command syntax; do not type the square brackets themselves. The `uv` and `uvcheck` commands can also accept an optional texture name as a final argument; run `python vpk_tool.py` to print the built-in command help.

## Build a Windows executable (optional)

The repository includes `build_exe.bat`. On Windows, run it to install PyInstaller (if needed) and build a single-file executable:

```text
dist\vpk_tool.exe
```

The executable is built for Windows. Keep the Python scripts and supporting files in the repository as the editable source version.

## Repository contents

```text
.
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
├── run_vpk_tool.bat
├── run_vpk_tool.sh
├── build_exe.bat
├── vpk_tool.py
├── wpk_view.py
├── uv_fit.py
└── .gitignore
```

## Important notes

- The tool edits game data. Always keep an untouched copy of the original `.vpk` and test edited files on copies first.
- Use artwork made for the target car's UV layout. A sticker image made for a different car may not line up correctly.
- The live preview is a software-rendered reference; it does not reproduce all in-game lighting, reflections, or effects.
- This repository does not include game files, car packs, textures, or other copyrighted game assets. Obtain and use those files from a source you are authorized to use.
- Compatibility of edited files may depend on the game build and the target car pack. Verify changes in your own test environment before replacing files in a live installation.

## Development and attribution

This project was developed with AI-assisted programming. AI assistance was used to help design, write, and refine the tool; it does not change the licensing or ownership of third-party material used in the project.

The VPK format and sticker data layout were researched using Re3DX+ project files. In particular, the one-colour sticker detection and plane-encoding logic in `vpk_tool.py`'s `encode()` function is adapted from `tools/stage/pack_vinyl.py` in the Re3DX+ project. The relevant author attribution and MIT license notice are included in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Disclaimer

This is an independent, unofficial community tool. It is not affiliated with, endorsed by, or sponsored by Bandai Namco Entertainment or the developers/publishers of Re3DX+.

Re3DX+, WANGAN MIDNIGHT MAXIMUM TUNE 3DX+, and any associated names, trademarks, game code, artwork, vehicles, music, sounds, and other game content belong to their respective rights holders. This project does not claim ownership of those materials. The tool's license does not grant permission to use or redistribute game content.

**You must provide your own game files.** This repository does not include or provide downloads for `.vpk`, `.wpk`, or other game files or assets. Only use files you are authorized to access, and do not upload game files or extracted assets to this repository, issue tracker, or pull requests.

## License

The original code contributed specifically to this VPK Tool project is licensed under the MIT License; see [`LICENSE`](LICENSE). Portions adapted from other projects retain their original copyright and license terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the Re3DX+ attribution and MIT notice.

Before publishing, replace the `[YOUR NAME OR GITHUB USERNAME]` placeholder in `LICENSE` with the name or GitHub username you want to use as the copyright holder for your original contributions.
