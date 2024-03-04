import sys
import timeit
import PIL
import imageio
import numpy as np
from funcs import init_boids, step
import config as cfg
import datetime
import arcade



class Boids(arcade.Window):
    """
    Main application class.
    """

    def __init__(self):

        # Call the parent class and set up the window
        super().__init__(cfg.SCREEN_WIDTH, cfg.SCREEN_HEIGHT, cfg.SCREEN_TITLE)

        arcade.set_background_color(arcade.csscolor.CORNFLOWER_BLUE)

    def setup(self):
        """Set up the game here. Call this function to restart the game."""
        self.boids = np.zeros((cfg.N, 6), dtype=np.float64)
        self.n_neighbors = 10

        init_boids(self.boids, cfg.ASPECT_RATIO, cfg.VELOCITY_RANGE)
    

        self.frame_count = 0
        self.background = arcade.load_texture("Boids/assets/brickwall.jpg")
        self.bat =  arcade.load_texture("Boids/assets/bat.png")
        self.active_bat = arcade.load_texture("Boids/assets/active_bat.png")
        self.neighbor_bat = arcade.load_texture("Boids/assets/neighbor_bat.png")
        self.bats = arcade.SpriteList()
        self.visible_boids = cfg.N
        self.coefs = np.array(list(cfg.COEFFITIENTS.values()), dtype=np.float64)
        self.bats.append(arcade.Sprite(texture=self.active_bat, center_x= self.boids[0][0] * cfg.SCREEN_HEIGHT, center_y= self.boids[0][0] * cfg.SCREEN_HEIGHT, scale=cfg.BOID_SCALE))
        for boid in self.boids[1:]:
               self.bats.append(arcade.Sprite(texture=self.bat, center_x= boid[0] * cfg.SCREEN_HEIGHT, center_y= boid[0] * cfg.SCREEN_HEIGHT, scale=cfg.BOID_SCALE))
        self.processing_time = 0

        if cfg.VIDEO:
            self.frames = 0
            self.writer = imageio.get_writer(f"Boids/simulation,{cfg.N=}.mp4", fps = 60)
        # Time for on_draw
        self.draw_time = 0

        # Variables used to calculate frames per second
        self.frame_count = 0
        self.fps_start_timer = None
        self.fps = None
    
    def on_draw(self):
        """Render the screen."""
        arcade.start_render()
        self.clear()
        start_time = timeit.default_timer()
 
        fps_calculation_freq = 60  
        if self.frame_count % fps_calculation_freq == 0:
            if self.fps_start_timer is not None:             # Calculate FPS
                total_time = timeit.default_timer() - self.fps_start_timer
                self.fps = fps_calculation_freq / total_time
            self.fps_start_timer = timeit.default_timer()
        self.frame_count += 1
        # Draw the background texture
        arcade.draw_lrwh_rectangle_textured(0, 0,
                                            cfg.SCREEN_WIDTH, cfg.SCREEN_HEIGHT,
                                            self.background)
        arcade.draw_circle_filled(self.bats[0].center_x, self.bats[0].center_y , cfg.PERCEPTION * cfg.SCREEN_HEIGHT, (255, 255, 191, 30 ))
        self.bats.draw()
        # Code to draw the screen goes here
        arcade.draw_lrtb_rectangle_filled(0, 200, cfg.SCREEN_HEIGHT, cfg.SCREEN_HEIGHT - 170, arcade.csscolor.ANTIQUE_WHITE)
      # Display timings
        output = f"Processing time: {self.processing_time:.3f}"
        arcade.draw_text(output, 20, cfg.SCREEN_HEIGHT - 15, arcade.color.BLACK, 12)
        output = f"Drawing time: {self.draw_time:.3f}"
        arcade.draw_text(output, 20, cfg.SCREEN_HEIGHT - 30, arcade.color.BLACK, 12)
        if self.fps is not None:
            output = f"FPS: {self.fps:.0f}"
            arcade.draw_text(output, 20, cfg.SCREEN_HEIGHT - 45, arcade.color.BLACK, 12)
        # Stop the draw timer, and calculate total on_draw time.
        arcade.draw_text(f"N: {cfg.N}", 20, cfg.SCREEN_HEIGHT - 60, arcade.color.BLACK, 12)
        arcade.draw_text(f"Visible boids: {self.visible_boids}", 20, cfg.SCREEN_HEIGHT - 75, arcade.color.BLACK, 12)
        arcade.draw_text(f"N_neighbors: {cfg.N_NEIGHBORS}", 20, cfg.SCREEN_HEIGHT - 90, arcade.color.BLACK, 12)
        for i, (param, value) in enumerate(cfg.COEFFITIENTS.items()):
            arcade.draw_text(f"{param}: {value}", 20, cfg.SCREEN_HEIGHT - 90 - 15 * (i+1), arcade.color.BLACK, 12)
        self.draw_time = timeit.default_timer() - start_time
        if cfg.VIDEO:
            image = arcade.draw_commands.get_image()
            self.writer.append_data(np.array(image))
            self.frames += 1
            if self.frames >= cfg.FRAMES:
                sys.exit()

    def on_update(self, delta_time):
        """ Movement and game logic """

        # Start timing how long this takes
        start_time = timeit.default_timer()

        # Call update on all sprites (The sprites don't do much in this
        # example though.)
        mask = step(self.boids, cfg.ASPECT_RATIO, cfg.PERCEPTION, self.coefs,cfg.N_NEIGHBORS, cfg.VELOCITY_RANGE, cfg.ACCELERATION_RANGE, cfg.dt)
        self.visible_boids = 0
        for i, boid, bat in zip(range(cfg.N), self.boids, self.bats):
            x = boid[0]* cfg.SCREEN_HEIGHT
            y = boid[1]* cfg.SCREEN_HEIGHT
            bat.center_x = x
            bat.center_y = y
            if 0 <= x <= cfg.SCREEN_WIDTH and 0 <= y <= cfg.SCREEN_HEIGHT:
                self.visible_boids += 1
            bat.angle = - np.arctan2(boid[2], boid[3]) /np.pi * 180
            if i!=0:
                if mask[0][i]:
                    bat.texture = self.neighbor_bat
                else:
                    bat.texture = self.bat
        # Stop the draw timer, and calculate total on_draw time.
        self.processing_time = timeit.default_timer() - start_time

def main():
    """Main function"""
    window = Boids()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()