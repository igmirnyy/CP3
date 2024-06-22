from matplotlib import pyplot as plt
import numpy as np
import taichi
import argparse

from PIL import Image
from enums import SimulationModeEnum, ColorsEnum
from simulation import light, image, simulation_step_accumulate, simulation_step
import config as cfg


argparser = argparse.ArgumentParser()
argparser.add_argument("-m", "--mode", choices=list(SimulationModeEnum))
argparser.add_argument("-i", "--iter", type=int)
argparser.add_argument("-acc", "--accumulate", type=float)
argparser.add_argument("-c", "--color", choices=list(ColorsEnum))
argparser.add_argument("-f", "--filename")
args = argparser.parse_args()



mode = SimulationModeEnum.animation if args.mode is None else args.mode
cfg.acc = cfg.acc if args.accumulate is None else args.accumulate
n_iter = cfg.n_iter if args.iter is None else args.iter
color = ColorsEnum.all if args.color is None else args.color
filename = "out" if args.filename is None else args.filename
match mode:
    case SimulationModeEnum.animation:
        gui = taichi.GUI("Light Refraction", res=(cfg.SCREEN_WIDTH, cfg.SCREEN_HEIGHT), fast_gui=True)
        while gui.running:
            if gui.get_event(taichi.GUI.PRESS):
                if gui.event.key == taichi.GUI.ESCAPE:
                    break
            if cfg.acc:
                simulation_step_accumulate()
                gui.set_image(image)
            else:
                simulation_step()
                gui.set_image(light.present)
            gui.show()
        gui.close()
    case SimulationModeEnum.picture:
        for _ in range(n_iter):
            simulation_step_accumulate()
        img = image.to_numpy()
        
        screen = img[8 * cfg.SCREEN_WIDTH//9, :, :]
        fig, ax = plt.subplots(4, figsize=(6, 15))
        for i, (ax_i, color) in enumerate(zip(ax, ["red", "green", "blue"])):
            color_image = screen[:, i]
            ax_i.plot(color_image, color=color)
            ax_i.set_title(color.title())
            ax_i.set_xlabel("y")
            ax_i.set_ylabel("brightness")
        for i, color in enumerate(["red", "green", "blue"]):
            color_image = screen[:, i]
            ax[-1].plot(color_image, color=color)
        ax[-1].set_title("All")
        ax[-1].set_xlabel("y")
        ax[-1].set_ylabel("brightness")
        fig.suptitle("Dispersion picture at the screen")
        fig.tight_layout()
        taichi.tools.imwrite(image, f"./output/{filename}_image.png")
        plt.savefig(f"./output/{filename}.png", dpi=300)
        
    case SimulationModeEnum.video:
        video_writer = taichi.tools.VideoManager(output_dir="./output", video_filename=f"{filename}.mp4", framerate=60, automatic_build=False)
        
        for i in range(n_iter):
            if color == ColorsEnum.all:
                simulation_step_accumulate()
                img = image.to_numpy()
            else:
                simulation_step()
                img = light.present.to_numpy()
                match color:
                    case ColorsEnum.red:
                        img[:, :, 1] = 0
                        img[:, :, 2] = 0
                        img_max = img.max()
                        img = np.clip(img / img_max, 0.0, 1.0)
                    case ColorsEnum.green:
                        img[:, :, 0] = 0
                        img[:, :, 2] = 0
                        img_max = img.max()
                        img = np.clip(img / img_max, 0.0, 1.0)
                    case ColorsEnum.blue:
                        img[:, :, 0] = 0
                        img[:, :, 1] = 0
                        img_max = img.max()
                        img = np.clip(img / img_max, 0.0, 1.0)
            video_writer.write_frame(img)
        
        
        video_writer.make_video(gif=False, mp4=True)