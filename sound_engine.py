import os
import pygame

class SoundEngine:

    def __init__(self, folder):

        pygame.mixer.init()

        self.sounds = {}

        for file in os.listdir(folder):

            if file.endswith(".wav"):

                name = file.replace(".wav", "")

                self.sounds[name] = pygame.mixer.Sound(
                    os.path.join(folder, file)
                )

    def play(self, name, volume=1.0):

        if name in self.sounds:

            s = self.sounds[name]

            s.set_volume(volume)

            s.play()

    def quit(self):
        pygame.mixer.quit()