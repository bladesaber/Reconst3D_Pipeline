import open3d.visualization.gui as gui

class InfoCheckbox(gui.Checkbox):
    name = ""

    def __init__(self, name):
        super(InfoCheckbox, self).__init__(name)
        self.name = name
