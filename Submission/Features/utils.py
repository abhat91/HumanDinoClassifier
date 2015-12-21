import numpy as np
import os

#Utilities used for the project may be stored here
class Util:
    #Get the path to the current folder
    @staticmethod
    def getCurrentFolder():
        """Get the path to the current folder"""
        folderPath=os.getcwd()
        return folderPath

