# trying to automate tab stop alignment for linguistic interlinear glosses
# should work on a given example and gloss line which already have tab-separated components and are each only one line even if it overflows
# ideally this can split it into Example2 and Gloss for more lines if it overflows

# documentation for the PyUNO API that this uses to communicate with LibreOffice:
# - http://www.openoffice.org/udk/python/python-bridge.html
# - https://wiki.openoffice.org/wiki/Python


import os


LOG_FP = "/home/kuhron/programming/LibreOfficeMacros/log.txt"
def print(x):
    if not os.path.exists(LOG_FP):
        open(LOG_FP, "w").close()
    with open(LOG_FP, "a") as f:
        f.write(str(x) + "\n")


def MoveTabStops():
    xModel = XSCRIPTCONTEXT.getDocument()
    xSelectionSupplier = xModel.getCurrentController()
    xIndexAccess = xSelectionSupplier.getSelection()
    print(xIndexAccess)

    # TODO how to get width of a set of characters so we can set the tab stop to some point after that?
    # XFont: https://www.openoffice.org/api/docs/common/ref/com/sun/star/awt/XFont.html#getStringWidth
    # getStringWidth()


g_exportedScripts = MoveTabStops,

print("\n---- begin running script ----\n")

