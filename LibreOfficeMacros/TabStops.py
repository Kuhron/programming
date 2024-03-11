# trying to automate tab stop alignment for linguistic interlinear glosses
# should work on a given example and gloss line which already have tab-separated components and are each only one line even if it overflows
# ideally this can split it into Example2 and Gloss for more lines if it overflows

# another idea once I know how to use this better: auto small-caps given a list of strings that should be small-capped, it should run on the given line (or can run along with the alignment routine and do the gloss line then), can have a user-specific file of abbreviations that should be small-capped, and use regex or similar to make sure it's not a substring of a word (e.g. don't want 'imm' small-capped in the word 'swimming')

# documentation for the PyUNO API that this uses to communicate with LibreOffice:
# - http://www.openoffice.org/udk/python/python-bridge.html
# - https://wiki.openoffice.org/wiki/Python
# - e.g. https://www.openoffice.org/api/docs/common/ref/com/sun/star/frame/XController.html


import os
from com.sun.star.style import TabStop
# from com.sun.star.style import TabAlign  # doesn't exist for some reason, I'll just do it myself


LOG_FP = "/home/kuhron/programming/LibreOfficeMacros/log.txt"
def print(x):
    if not os.path.exists(LOG_FP):
        open(LOG_FP, "w").close()
    with open(LOG_FP, "a") as f:
        f.write(str(x) + "\n")


class TabAlign:
    # https://www.openoffice.org/api/docs/common/ref/com/sun/star/style/TabAlign.html
    LEFT = 0
    CENTER = 1
    RIGHT = 2
    DECIMAL = 3
    DEFAULT = 4


# https://forum.openoffice.org/en/forum/viewtopic.php?t=106673
def tabs(*args):
    tbs = []
    for position, alignment in args:
        tb = TabStop()
        tb.Position = position
        tb.Alignment = alignment
        tbs.append(tb)
    tbs = tuple(tbs)
    return tbs


def get_cursor():
    xModel = XSCRIPTCONTEXT.getDocument()
    controller = xModel.getCurrentController()
    cursor = controller.getViewCursor()
    return cursor


def ClearTabStops():
    cursor = get_cursor()
    cursor.ParaTabStops = [] # this to clear tabs from paragraph


def MoveTabStops():
    # want to set properties of the current paragraph, NOT the whole style
    cursor = get_cursor()
    print(cursor.getText())

    # example of setting them:
    # cursor.ParaTabStops = tabs((3000, TabAlign.LEFT), (5000, TabAlign.CENTER), (10000, TabAlign.RIGHT))

    # TODO how to get width of a set of characters so we can set the tab stop to some point after that?
    # XFont: https://www.openoffice.org/api/docs/common/ref/com/sun/star/awt/XFont.html#getStringWidth
    # getStringWidth()

    print("finished setting tab stops")


# stuff to translate or use as guide, from https://forum.openoffice.org/en/forum/viewtopic.php?t=20217
# Dim Stops(1) as New com.sun.star.style.TabStop
# Doc = ThisComponent
# StyleFams = Doc.StyleFamilies
# ParaStyles = StyleFams.getByName("ParagraphStyles")
# DefaultStyle = ParaStyles.getByName("Standard")
# Stops(0).Position = 1270
# Stops(0).Alignment = com.sun.star.style.TabAlign.RIGHT
# Stops(1).Position = 2540
# DefaultStyle.ParaTabStops = Stops

g_exportedScripts = MoveTabStops,

print("\n---- begin running script ----\n")

