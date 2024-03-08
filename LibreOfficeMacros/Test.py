# HelloWorld python script for the scripting framework

#
# This file is part of the LibreOffice project.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# This file incorporates work covered by the following license notice:
#
#   Licensed to the Apache Software Foundation (ASF) under one or more
#   contributor license agreements. See the NOTICE file distributed
#   with this work for additional information regarding copyright
#   ownership. The ASF licenses this file to you under the Apache
#   License, Version 2.0 (the "License"); you may not use this file
#   except in compliance with the License. You may obtain a copy of
#   the License at http://www.apache.org/licenses/LICENSE-2.0 .
#

import time


def TestUserPythonMacro():
    """Prints a string into the current document.
    """

    # Get the doc from the scripting context which is made available to all
    # scripts.
    desktop = XSCRIPTCONTEXT.getDesktop()
    model = desktop.getCurrentComponent()

    # Check whether there's already an opened document.
    # Otherwise, create a new one
    if not hasattr(model, "Text"):
        model = desktop.loadComponentFromURL(
            "private:factory/swriter", "_blank", 0, ()
        )

    # get the XText interface
    text = model.Text

    # create an XTextRange at the end of the document
    tRange = text.End

    # and set the string
    tRange.String = f"asdf {time.time()} (in Python)"

    return None


# Capitalize.py copied from https://wiki.openoffice.org/wiki/PyUNO_samples so I can learn how this works

# helper function
def getNewString(theString):
    if not theString or len(theString) ==0:
        return ""
    # should we tokenize on "."?
    if theString[0].isupper() and len(theString)>=2 and theString[1].isupper():
    # first two chars are UC => first UC, rest LC
        newString=theString.capitalize()
    elif theString[0].isupper():
    # first char UC => all to LC
        newString=theString.lower()
    else: # all to UC.
        newString=theString.upper()
    return newString


def CapitalizePython():
    """Change the case of a selection, or current word from uppercase, to first char uppercase, to all lowercase to uppercase..."""

    # The context variable is of type XScriptContext and is available to
    # all BeanShell scripts executed by the Script Framework
    xModel = XSCRIPTCONTEXT.getDocument()

    #the writer controller impl supports the css.view.XSelectionSupplier interface
    xSelectionSupplier = xModel.getCurrentController()

    #see section 7.5.1 of developers' guide
    xIndexAccess = xSelectionSupplier.getSelection()
    count = xIndexAccess.getCount()
    for i in range(count):
        xTextRange = xIndexAccess.getByIndex(i)
        #print "string: " + xTextRange.getString()
        theString = xTextRange.getString()
        if len(theString) == 0:
            # sadly we can have a selection where nothing is selected
            # in this case we get the XWordCursor and make a selection!
            xText = xTextRange.getText()
            xWordCursor = xText.createTextCursorByRange(xTextRange)
            if not xWordCursor.isStartOfWord():
                xWordCursor.gotoStartOfWord(False)
            xWordCursor.gotoNextWord(True)
            theString = xWordCursor.getString()
            newString = getNewString(theString)
            if newString:
                xWordCursor.setString(newString)
                xSelectionSupplier.select(xWordCursor)
        else:
            newString = getNewString(theString)
            if newString:
                xTextRange.setString(newString)
                xSelectionSupplier.select(xTextRange)


# vim: set shiftwidth=4 softtabstop=4 expandtab:

# lists the scripts, that shall be visible inside OOo. Can be omitted, if
# all functions shall be visible, however here getNewString shall be suppressed
g_exportedScripts = CapitalizePython, TestUserPythonMacro
