import os
import argparse

def create_txt(xml_dir):
    for i in os.listdir(xml_dir):
        a, b = os.path.splitext(i)
        print(a)
        if not a == '.ipynb_checkpoints':
            with open(xml_dir.split('xmls')[0] + 'trainval.txt', 'a') as f:
                f.write(a + ' 0\n')


if __name__ == "__main__":
    # Path of the images
    parser = argparse.ArgumentParser()
    parser.add_argument("xml_dir")
    args = parser.parse_args()
    create_txt(args.xml_dir)