#!/usr/bin/env python3
import os, sys, argparse

# Define parser to get path to images and options
parser = argparse.ArgumentParser(description="Script to rename images in a directory.")

parser.add_argument(
    "-p", "--path_folder",
    type=str,
    required=True,
    help="Path to the folder containing the images to rename.",
)

parser.add_argument('-e', '--extension',
                    type=str, 
                    default=".png", 
                    help="Extension of the images to rename. Default is .png")

parser.add_argument('-t', '--type_renaming', 
                    type=str, 
                    default="progressive_id", 
                    help="Type of renaming to perform. Options: 'progressive_id'.")

parser.add_argument('-s', '--start',
                    type=int, 
                    default=0, 
                    help="Starting number for renaming. Default is 0.")

parser.add_argument("--step", type=int, default=1, help="Step for renaming.")

parser.add_argument(
    "-o", "--output_folder",
    type=str,
    default=None,
    help="Path to the folder where the renamed images will be saved. If not specified, images will be renamed in the original folder.",
)

# Add help list
parser.add_argument('-h', '--help_list', action='store_true', help="Display a list of available options.")

# Parse the arguments
args = parser.parse_args()

def main():

    if args.help_list:
        print("Available options:")
        print("-p, --path_folder: Path to the folder containing the images to rename.")
        print("-e, --extension: Extension of the images to rename. Default is .png")
        print("-t, --type_renaming: Type of renaming to perform. Options: 'progressive_id'.")
        print("-s, --start: Starting number for renaming. Default is 0.")
        print("--step: Step for renaming.")
        print("-o, --output_folder: Path to the folder where the renamed images will be saved. If not specified, images will be renamed in the original folder.")
        sys.exit(0)

    # Get folder path
    folder_path = args.path_folder
    start_id = args.start
    step_id = args.step
    type_renaming = args.type_renaming

    if not os.path.exists(folder_path):
        print(f"Error: folder '{folder_path}' does not exist.")
        sys.exit(1)

    # Get a list of the image files in the folder, sorted by name
    image_extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # If extension is specified, use that one
    if args.extension:
        image_extensions: tuple = (args.extension,)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

    # Get extension of file
    _, ext_ = os.path.splitext(files[0])

    if args.output_folder is None:
        output_folder_path = folder_path
    else:
        output_folder_path = args.output_folder 

    # Iterate over the files and rename them with a progressive number starting from start_id
    for i, filename in enumerate(files):

        if os.path.splitext(filename)[1] != ext_:
            continue

        print(f"Renaming {filename} to {i * step_id + start_id:06}.{ext_}")
        
        new_name = f"{i * step_id + start_id:06}.{ext_}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(output_folder_path, new_name)

        os.rename(old_path, new_path)

    print("Renaming operation completed.")

if __name__ == "__main__":
    main()

