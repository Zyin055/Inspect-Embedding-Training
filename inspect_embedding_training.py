import os
import csv
import sys
import torch
import getopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor

#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#######################################################################################################################
#                                                      CONFIG                                                         #
#######################################################################################################################
SAVE_LOSS_GRAPH_IMG: bool = True              # Create a .jpg of the Loss graph
SAVE_VECTOR_GRAPH_IMG: bool = True            # Create a .jpg of the Vector graph

SHOW_PLOTS_AFTER_GENERATION: bool = False      # Show the Loss and Vector graphs after running this script

GRAPH_IMAGE_SIZE: tuple[int, int] = (19, 9)   # (X,Y) tuple in inches (multiply by 100 for (X,Y) pixel size for the output graphs)
GRAPH_SHOW_TITLE: bool = True                 # Adds the embed name at the top of the graphs

VECTOR_GRAPH_SHOW_LEARNING_RATE: bool = True  # Adds the learning rate labels and vertical lines on the vector graph
VECTOR_GRAPH_LIMIT_NUM_VECTORS: int = 100     # Limits to this number of vectors drawn on the vector graph to this many lines, set to 0 or None to draw all the vectors. Normally there are 768 vectors per token.
#######################################################################################################################
#                                                    END CONFIG                                                       #
#######################################################################################################################


BASEDIR: str = os.path.realpath(os.path.dirname(__file__))   # the path where this .py file is located, ex "C:\Stable Diffusion\textual_inversion\2022-12-30\EmbedFolderName"
output_dir: str = BASEDIR    # where the output graph images are saved
working_dir: str = BASEDIR   # where we look for embeddings


def parse_args(argv) -> None:
    try:
        opts, args = getopt.getopt(argv[1:], "h", ["help", "dir=", "out="])
    except getopt.GetoptError as e:
        sys.exit(e)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Place this Python file in the textual inversion folder in the specific embedding folder you want to analyze (next to textual_inversion_loss.csv). Optionally, you can use the --dir \"/path/to/folder\" launch argument to specify the folder to use.")
            print("launch args:")
            print("--help -h")
            print("    This help message.")
            print("--dir")
            print("    The \"/path/to/embedding/folder\" to use instead of the local path where this script is at. This directory should have the textual_inversion_loss.csv file in it.")
            print("--out")
            print("    The \"/path/to/an/output/folder\" to use instead of the local path for outputting images.")
            sys.exit(0)
        # elif opt in ("-d", "--dir"):
        elif opt == "--dir":
            global working_dir
            working_dir = arg
            print(f"Directory set to: {working_dir}")
        elif opt == "--out":
            global output_dir
            output_dir = arg
            print(f"Output directory set to: {output_dir}")



def load_textual_inversion_loss_data_from_file(file: str) -> dict[int, dict[str, str]]:
    if os.path.isfile(file):
        with open(file) as metadata_file:
            return {int(rec["step"]): rec for rec in csv.DictReader(metadata_file)}
    else:
        print(f"[ERROR] Could not find file: {file}")
        print("[ERROR] This error could happen if this script is set to use the wrong directory, or if not enough training steps have passed for the file to be created yet. In Automatic1111 Web UI, try lowering the value for the setting \"Save an csv containing the loss to log directory every N steps, 0 to disable\".")
        sys.exit(os.EX_NOTFOUND)



def analyze_embedding_files(embedding_dir: str) -> (dict[int, Tensor], str, int, int):
    DIMS_PER_VECTOR = 768
    embed_name = None        # "EmbedName"
    number_of_embedding_files = 0
    highest_step = -1
    vector_data = {}
    try:
        for embed_file_name in os.listdir(embedding_dir):  # "EmbedName-500.pt"
            if not embed_file_name.endswith(".pt"):
                continue
            embed_path = os.path.join(embedding_dir, embed_file_name)
            embed = torch.load(embed_path, map_location="cpu")
            v = embed["string_to_param"]["*"]
            step = embed["step"] + 1  # starts counting at 0, so add 1
            vector_data[step] = torch.flatten(v).tolist()
            number_of_embedding_files += 1

            # print(f"step:{step}")

            if step > highest_step:
                highest_step = step  # + 1
                embed_name = embed_file_name  # save the file name with the highest step count
                #vectors_per_token = int(len(vector_data[step]) / DIMS_PER_VECTOR)

            # print(f"Loaded data from embedding: {embed_file_name}")

    except FileNotFoundError as e:
        print("[ERROR] Make sure to place this Python file in the textual inversion folder in the specific embedding folder you want to analyze (next to textual_inversion_loss.csv). Optionally, you can use the --dir \"/path/to/embedding/folder\" launch argument to specify the folder to use.")
        sys.exit(e)

    vectors_per_token = int(len(vector_data[highest_step]) / DIMS_PER_VECTOR)
    embed_name = embed_name.replace(".pt", "")[:-(len(str(highest_step)) + 1)]  # "EmbedName", trim "-XX.pt" off the end
    print(f"This embedding has {vectors_per_token} vectors per token.")
    print(f"Loaded {number_of_embedding_files} embedding files up to training step {highest_step}.")

    return vector_data, embed_name, highest_step, number_of_embedding_files



def create_loss_plot(title: str, data: dict, save_img: bool, output_file_name: str) -> None:
    plt.figure(figsize=GRAPH_IMAGE_SIZE)
    loss = pd.DataFrame(data).T.sort_index()["loss"].astype(float).loc[1:]
    #loss = pd.DataFrame(data).T.sort_index()["loss"].astype(float).iloc[1:]    # don't know what the difference is between loc and iloc, both work.
    # print("loss:")
    # print(loss)

    # x: step, y: loss
    plt.scatter(loss.index, loss, color=(0, 0, 0, 0.2))  # point values
    plt.plot(loss.rolling(window=5).mean(), color=(0, 0, 0, 0.4))  # best fit line over 5 steps
    plt.plot(loss.rolling(window=50).mean(), color=(0, 0, 0, 0.7))  # best fit line over 50 steps

    z1 = np.polyfit(loss.index, loss, 1)
    p1 = np.poly1d(z1)(loss.index)
    z2 = np.polyfit(loss.index, loss, 5)
    p2 = np.poly1d(z2)(loss.index)
    p3 = np.poly1d([0, z1[1]])(loss.index)
    
    plt.plot(p3, ":", color=(0, 0, 0, 0.5))
    plt.plot(p1, color=(1, 0, 1, 0.5))
    plt.plot(p2, color=(0, 1, 1, 0.5))

    min_y_value, max_y_value = plt.gca().get_ylim()
    min_x_value, max_x_value = plt.gca().get_xlim()

    if title is not None:
        x = (min_x_value + max_x_value) / 2
        y = (max_y_value - min_y_value) * 1.10 + min_y_value
        plt.text(x, y, title, fontsize="20", ha="center", va="center")  # Title at the top

    if save_img:
        plt.savefig(output_dir + "/" + output_file_name)
        print(f"Created: {output_file_name}")


def create_vector_plot(title: str, data: dict[int, dict[int, Tensor]], learn_rate_changes: dict[int, (int, float)], highest_step: int, show_learning_rate: bool, save_img: bool, output_file_name: str) -> None:
    global VECTOR_GRAPH_LIMIT_NUM_VECTORS

    vectors_shown_text = None
    if 0 < VECTOR_GRAPH_LIMIT_NUM_VECTORS < len(data[highest_step]):
        vectors_shown_text = f"{VECTOR_GRAPH_LIMIT_NUM_VECTORS} of {len(data[highest_step])} vectors shown"
        for step in data:
            data[step] = data[step][:VECTOR_GRAPH_LIMIT_NUM_VECTORS]

    plt.figure(figsize=GRAPH_IMAGE_SIZE)
    plt.plot(pd.DataFrame(data).T.sort_index())
    min_y_value, max_y_value = plt.gca().get_ylim()
    min_x_value, max_x_value = plt.gca().get_xlim()

    # plot the Learn rate: XX labels and lines
    for i in learn_rate_changes:
        #print(f"i={i}, len(learn_rate_changes)={len(learn_rate_changes)}")
        step, learn_rate = learn_rate_changes[i]
        nextStep = highest_step
        if i < len(learn_rate_changes) - 1:
            nextStep = learn_rate_changes[i+1][0]

        #print(f"step={step}, nextStep={nextStep}")
        if show_learning_rate:
            x = (step + nextStep) / 2
            y = (max_y_value - min_y_value) * 1.03 + min_y_value
            # if i % 2 == 1:
            #     y += (max_y_value - min_y_value) * 0.06
            #print(f"Learn rate: {learn_rate} at ({x},{y}), step={step}, nextStep={nextStep}")
            plt.text(x, y, f"Learn rate:\n{learn_rate}", fontsize="12", ha="center", va="center")    #Learn rate labels

            if len(learn_rate_changes) > 1: #create vertical line only if there is a learning rate change
                plt.axvline(step, color=(0, 0, 0, 0.4), linestyle="dotted")

    if vectors_shown_text is not None:
        x = (min_x_value + max_x_value) / 2
        y = min_y_value - (max_y_value - min_y_value) * 0.10
        plt.text(x, y, vectors_shown_text, fontsize="10", ha="center", va="center")  # Title at the top

    if title is not None:
        x = (min_x_value + max_x_value) / 2
        y = (max_y_value - min_y_value) * 1.10 + min_y_value
        plt.text(x, y, title, fontsize="20", ha="center", va="center")  # Title at the top

    if save_img:
        plt.savefig(output_dir + "/" + output_file_name)
        print(f"Created: {output_file_name}")



def main():

    parse_args(sys.argv)

    if not SAVE_LOSS_GRAPH_IMG and not SAVE_VECTOR_GRAPH_IMG and not SHOW_PLOTS_AFTER_GENERATION:
        print("[ERROR] Not set to create or show any plots. Change the global variables so that this script can do something.")
        sys.exit(os.EX_CONFIG)


    embeddings_dir = os.path.join(working_dir, "embeddings")
    vector_data, embed_name, highest_step, number_of_embedding_files = analyze_embedding_files(embeddings_dir)
    #print(f"vector_data:")
    #print(vector_data)  # dict{step:[768xDimensions of floats]}
    #print(pd.DataFrame(vector_data).T.sort_index())  # massive amounts of numbers, [step][768 x Dimensions of floats]


    print("Generating graphs...")


    loss_csv_file = os.path.join(working_dir, "textual_inversion_loss.csv")
    textual_inversion_loss_data = load_textual_inversion_loss_data_from_file(loss_csv_file)
    #print(f"textual_inversion_loss_data:")
    #print(textual_inversion_loss_data)
    #print(pd.DataFrame(textual_inversion_loss_data).T.sort_index())   # the same format as the textual_inversion_loss.csv file

    learn_rate_changes = {} # dict[index: int, (step: int, learn rate: float)]
    last_learn_rate = -1
    new_i = 0
    for step in textual_inversion_loss_data:
        new_learn_rate = textual_inversion_loss_data[step]["learn_rate"]
        #print(f"step {step} -> {new_learn_rate} learn rate")
        if last_learn_rate != new_learn_rate:
            learn_rate_changes[new_i] = (step, float(new_learn_rate))
            last_learn_rate = new_learn_rate

            print(f"Learning rate at step {step}: {learn_rate_changes[new_i][1]}")
            new_i += 1

    if len(learn_rate_changes) == 1:
        learn_rate_changes[0] = (0, learn_rate_changes[0][1])    # only 1 learning rate change, set starting step to 0. makes the Learn rate label centered

    #print(f"learn_rate_changes: {learn_rate_changes}")

    if SAVE_LOSS_GRAPH_IMG or SHOW_PLOTS_AFTER_GENERATION:  # loss plot
        output_file_name = f"{embed_name}-{highest_step}-loss.jpg"
        create_loss_plot(title=embed_name if GRAPH_SHOW_TITLE else None,
                         data=textual_inversion_loss_data,
                         save_img=SAVE_LOSS_GRAPH_IMG,
                         output_file_name=output_file_name)
    else:
        print("Skipping making a loss plot.")


    if number_of_embedding_files == 0:
        print(f"[ERROR] Could not find any embedding .pt files in {embeddings_dir}")
        sys.exit(os.EX_NOTFOUND)
    elif number_of_embedding_files == 1:
        print("[WARNING] Only 1 embedding file found, the vector plot won't show any useful data.")


    if SAVE_VECTOR_GRAPH_IMG or SHOW_PLOTS_AFTER_GENERATION:  # vector plot
        if 0 < VECTOR_GRAPH_LIMIT_NUM_VECTORS < len(vector_data[highest_step]):
            output_file_name = f"{embed_name}-{highest_step}-vector-({VECTOR_GRAPH_LIMIT_NUM_VECTORS}-vector-limit).jpg"
        else:
            output_file_name = f"{embed_name}-{highest_step}-vector.jpg"
        create_vector_plot(title=embed_name if GRAPH_SHOW_TITLE else None,
                           data=vector_data,
                           learn_rate_changes=learn_rate_changes,
                           highest_step=highest_step,
                           show_learning_rate=VECTOR_GRAPH_SHOW_LEARNING_RATE,
                           save_img=SAVE_VECTOR_GRAPH_IMG,
                           output_file_name=output_file_name,
                           )
    else:
        print("Skipping making a vector plot.")

    if SHOW_PLOTS_AFTER_GENERATION:
        plt.show()  # shows both plots

    print("Done!")


if __name__ == "__main__":
    main()
