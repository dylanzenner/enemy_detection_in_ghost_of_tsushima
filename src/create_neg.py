import os


def generate_negative_description_file():
    """
    Creates the required negative description file to be used for the cascade classifier
    :return: None
    """
    with open("neg.txt", "w") as f:
        for filename in os.listdir("negative_images"):
            f.write("negative_images/" + filename + "\n")
            
 if __name__ == "__main__":
    generate_negative_description_file()
