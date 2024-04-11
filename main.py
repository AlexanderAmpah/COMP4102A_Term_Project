import os
import argparse
import letter_detection as detector
import input as proc
import mst
import letters as l


def main():

    def is_valid_file(parser, filename):
        if not os.path.isfile(filename):
            parser.error(f'The file: "{filename}" does not exist!')

        else:
            return filename
        
        
    def is_valid_save(parser, model, load=True):
        if not os.path.isfile(model) and load:
            parser.error(f'The file: "{model}" does not exist!')

        elif len(model) < 11 or model[-11:] != '.weights.h5':
            parser.error(f'The file: "{model}" is not the proper extension! Must end with ".weights.h5".')

        else:
            return model
        

    parser = argparse.ArgumentParser(
        prog='handToText',
        description="Reads handwritten text from an image file. Handwritten text must be well lit and printed. Adequate spacing must be between the letters such that the letters are neither connected nor is one letter 'over' another.",
        usage='%(prog)s [filename] [--model] [model params file]', 
        epilog="By Ainan Kashif, Oko Ampah, and Emmanuel Richard. For COMP 4102A in Winter semester 2024."
    )

    parser.add_argument(
        'path', 
        help="File path to handwritten image. Handwritten text must be well lit and printed. Adequate spacing must be between the letters such that the letters are neither connected nor is one letter 'over' another.", 
        metavar='image', 
        type=lambda f: is_valid_file(parser, f)
    )

    parser.add_argument(
        '--train', 
        action='store_true', 
        help='Trains model.'
    )

    parser.add_argument(
        '--test', 
        action='store_true', 
        help='Tests model.'
    )

    parser.add_argument(
        '-l', 
        '--load', 
        dest='loadfile', 
        help='Model parameters loadfile. Must be ".weights.h5" extension.',
        type=lambda f: is_valid_save(parser, f, load=True) 
    )

    parser.add_argument(
        '-s', 
        '--save', 
        dest='savefile', 
        help='Model parameters savefile. Must be ".weights.h5" extension.', 
        type=lambda f: is_valid_save(parser, f, load=False)
    )

    # use image first blur, then get boxes, then compute mst, then merge i and j, then extract letters
    # To preserve aspect ratio, resize so height or width is 28 pixels in length
    # copy image to a 28 x 28 array prefilled with background colour values

    args = parser.parse_args()
    path = args.path

    savefile = args.savefile
    loadfile = args.loadfile

    img = proc.loadImg(path)
    blur = proc.blur(img)
    _, boxes, _ = proc.box_letters(blur)
    boxes = proc.filter_boxes(boxes)

    centers = mst.calculate_centers(boxes)
    dist, _ = mst.distance_matrix(centers)
    tree = mst.min_spanning_tree(dist, centers)

    boxes, tree = l.group_ij(boxes, tree)
    letters = l.extract_letters(blur, boxes)

    # Resize for neural network
    # Keep aspect ratio while making image 28 x 28

    model = detector.build_model()

    if args.train:
        train, test, val = detector.load_dataset()  # X and y pairs
        history = detector.train_model(model, train, val)

        if not savefile is None:
            detector.save_model(model, savefile)

        # TODO: Plot train accuracy and loss over epochs using history object

    elif not loadfile is None:
        detector.load_model(model, loadfile)

    else:
        parser.error('Error! Loadfile or train option not specified.')

    if args.test:
        train, test, val = detector.load_dataset()
        loss, acc = detector.test_model(model, test)

        # TODO: Plot test accuracy and loss over epochs


    for image in letters:
        letter, index = detector.classify(model, image)

        proc_image = detector.preprocess_image(image)

        proc.plotImg(image, title=letter)
        proc.plotImg(proc_image, title=f'How the model sees {letter}')

        print(index, letter)


if __name__ == "__main__":
    main()