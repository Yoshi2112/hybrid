'''
Goal: Need some way to input values using a GUI - textbox + assign to variable
'''

from tkinter import Tk, Label, Button, Entry, IntVar, END, W, E

class Calculator:

    def __init__(self, master):
        self.master = master
        master.title("Calculator")
        master.geometry('680x420')

        self.total = 0
        self.entered_number = 0

        self.total_label_text = IntVar()
        self.total_label_text.set(self.total)
        self.total_label = Label(master, textvariable=self.total_label_text)

        self.label = Label(master, text="Total:")

        vcmd = master.register(self.validate) # we have to wrap the command
        self.entry = Entry(master, validate="key", validatecommand=(vcmd, '%P'))

        self.add_button = Button(master, text="+", command=lambda: self.update("add"))
        self.subtract_button = Button(master, text="-", command=lambda: self.update("subtract"))
        self.reset_button = Button(master, text="Reset", command=lambda: self.update("reset"))

        # LAYOUT

        self.label.grid(row=0, column=0, sticky=W)
        self.total_label.grid(row=0, column=1, columnspan=2, sticky=E)

        self.entry.grid(row=1, column=0, columnspan=3, sticky=W+E)

        self.add_button.grid(row=2, column=0)
        self.subtract_button.grid(row=2, column=1)
        self.reset_button.grid(row=2, column=2, sticky=W+E)

    def validate(self, new_text):
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

        self.total_label_text.set(self.total)
        self.entry.delete(0, END)

root = Tk()
my_gui = Calculator(root)
root.mainloop()
# =============================================================================
# import tkinter as tk
# 
# class Dashboard():
#     def __init__(self, root):
#         self.root=root
#         root.title("Dashboard View")
# 
#         # the width, height and colors are temporary,
#         # until we have more of the GUI working.
#         buttonPanel = tk.Frame(root, background="green", width=200, height=200)
#         canvasPanel = tk.Frame(root, background="pink", width=500, height=500)
# 
#         # because these two panels are side-by-side, pack is the
#         # best choice:
#         buttonPanel.pack(side="left", fill="y")
#         canvasPanel.pack(side="right", fill="both", expand=True)
# 
#         # fill in these two areas:
#         self._create_buttons(buttonPanel)
#         self._create_canvas(canvasPanel)
# 
#     def _create_buttons(self, parent):
# 
#         b1=tk.Button(parent,text="Status", command=lambda: self._contents("#DC322F", "dashboard_content.txt"))
#         b2=tk.Button(parent,text="Processes", command=lambda: self._contents("#859900", "process.log"))
#         b3=tk.Button(parent,text="Links", command=lambda: self._contents("#B58900", "links.log"))
#         b4=tk.Button(parent,text="Traffic", command=lambda: self._contents("#268BD2", "version.log"))
#         b5=tk.Button(parent,text="App Version", command=lambda: self._contents("#D33682", "version.log"))
#         b6=tk.Button(parent,text="Archive/Purge", command=lambda: self._contents("#93A1A1", "cleanup.log"))
# 
#         b1.grid(row = 0,column = 0, sticky = "we")
#         b2.grid(row = 0,column = 1, sticky = "we")
#         b3.grid(row = 1,column = 0, sticky = "we")
#         b4.grid(row = 1,column = 1, sticky = "we")
#         b5.grid(row = 2,column = 0, sticky = "we")
#         b6.grid(row = 2,column = 1, sticky = "we")
# 
#     def _create_canvas(self, parent):
#         self.canvas=tk.Canvas(parent, width=1000, height=700,background='#002B36')
#         vsb = tk.Scrollbar(parent, command=self.canvas.yview, orient="vertical")
#         hsb = tk.Scrollbar(parent, command=self.canvas.xview, orient="horizontal")
#         self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
#         
#         vsb.grid(row=0, column=0, sticky="ns")
#         hsb.grid(row=1, column=0, sticky="ew")
#         self.canvas.grid(row=0, column=1, sticky="nsew")
#         
#         parent.grid_rowconfigure(0, weight=1)
#         parent.grid_columnconfigure(1, weight=1)
# 
# if __name__== '__main__':
#     root=tk.Tk()
#     board=Dashboard(root)
#     root.mainloop()
# =============================================================================

# =============================================================================
# import tkinter as tk
# 
# fields = ('Annual Rate', 'Number of Payments', 'Loan Principle', 'Monthly Payment', 'Remaining Loan')
# 
# def monthly_payment(entries):
#     # period rate:
#     r = (float(entries['Annual Rate'].get()) / 100) / 12
#     print("r", r)
#     # principal loan:
#     loan = float(entries['Loan Principle'].get())
#     n =  float(entries['Number of Payments'].get())
#     remaining_loan = float(entries['Remaining Loan'].get())
#     q = (1 + r)** n
#     monthly = r * ( (q * loan - remaining_loan) / ( q - 1 ))
#     monthly = ("%8.2f" % monthly).strip()
#     entries['Monthly Payment'].delete(0, tk.END)
#     entries['Monthly Payment'].insert(0, monthly )
#     print("Monthly Payment: %f" % float(monthly))
# 
# def final_balance(entries):
#     # period rate:
#     r = (float(entries['Annual Rate'].get()) / 100) / 12
#     print("r", r)
#     # principal loan:
#     loan = float(entries['Loan Principle'].get())
#     n =  float(entries['Number of Payments'].get()) 
#     monthly = float(entries['Monthly Payment'].get())
#     q = (1 + r) ** n
#     remaining = q * loan  - ( (q - 1) / r) * monthly
#     remaining = ("%8.2f" % remaining).strip()
#     entries['Remaining Loan'].delete(0, tk.END)
#     entries['Remaining Loan'].insert(0, remaining )
#     print("Remaining Loan: %f" % float(remaining))
# 
# def makeform(root, fields):
#     entries = {}
#     for field in fields:
#         print(field)
#         row = tk.Frame(root)
#         lab = tk.Label(row, width=22, text=field+": ", anchor='w')
#         ent = tk.Entry(row)
#         ent.insert(0, "0")
#         row.pack(side=tk.TOP, 
#                  fill=tk.X, 
#                  padx=5, 
#                  pady=5)
#         lab.pack(side=tk.LEFT)
#         ent.pack(side=tk.RIGHT, 
#                  expand=tk.YES, 
#                  fill=tk.X)
#         entries[field] = ent
#     return entries
# 
# if __name__ == '__main__':
#     root = tk.Tk()
#     ents = makeform(root, fields)
#     b1   = tk.Button(root, text='Final Balance',
#            command=(lambda e=ents: final_balance(e)))
#     b1.pack(side=tk.LEFT, padx=5, pady=5)
#     b2   = tk.Button(root, text='Monthly Payment',
#            command=(lambda e=ents: monthly_payment(e)))
#     b2.pack(side=tk.LEFT, padx=5, pady=5)
#     b3   = tk.Button(root, text='Quit', command=root.quit)
#     b3.pack(side=tk.LEFT, padx=5, pady=5)
#     root.mainloop()
# =============================================================================
