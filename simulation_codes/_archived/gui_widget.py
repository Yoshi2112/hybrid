# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:56:00 2019

@author: Yoshi
"""

'''
Goal: Need some way to input values using a GUI - textbox + assign to variable
'''
from tkinter import Tk, Label, Button, Entry, IntVar, END, W, E

class SimInput:
    def __init__(self, master):
        self.master = master
        master.title("Calculator")
        master.geometry('680x420')
       

        vcmd = master.register(self.validate) # we have to wrap the command
        self.entry = Entry(master, validate="key", validatecommand=(vcmd, '%P'))

        self.add_button = Button(master, text="+", command=lambda: self.update("add"))
        self.subtract_button = Button(master, text="-", command=lambda: self.update("subtract"))
        self.reset_button = Button(master, text="Reset", command=lambda: self.update("reset"))

        # LAYOUT
        self.label = Label(master, text="Total:")
        self.label.grid(row=0, column=0, sticky=W)

        self.entry.grid(row=1, column=0, columnspan=3, sticky=W+E)

        self.add_button.grid(row=2, column=0)
        self.subtract_button.grid(row=2, column=1)
        self.reset_button.grid(row=2, column=2, sticky=W+E)

    def validate(self, new_text):
        '''
        Stops crap being put in the textbox
        '''
        if not new_text: # the field is being cleared
            self.entered_number = 0
            return True
        try:
            self.entered_number = int(new_text)
            return True
        except ValueError:
            return False

    def update(self, method):
        if method == "add":
            self.total += self.entered_number
        elif method == "subtract":
            self.total -= self.entered_number
        else: # reset
            self.total = 0

        self.entry.delete(0, END)

root   = Tk()
my_gui = SimInput(root)
root.mainloop()
        