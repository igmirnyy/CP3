from enum import StrEnum


class SimulationModeEnum(StrEnum):
    picture = "picture"
    video = "video"
    animation = "animation"
    
    def __str__(self):
        return self.value

class ColorsEnum(StrEnum):
    all = "all"
    red = "red"
    green = "green"
    blue = "blue"
    
    def __str__(self):
        return self.value