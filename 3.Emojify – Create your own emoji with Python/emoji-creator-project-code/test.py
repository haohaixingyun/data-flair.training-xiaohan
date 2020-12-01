import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import datetime
import numpy as np
import cv2
global cap1
def show_vid():
    print(datetime.datetime.now())
    cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1, (600, 500))

    if flag1 is None:
        print("Major error!")
    elif flag1:

        # global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        print(datetime.datetime.now())
        lmain.after(10, show_vid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()


if __name__ == '__main__':
    root = tk.Tk()
    img = ImageTk.PhotoImage(Image.open("logo.png"))
    heading = Label(root, image=img, bg='black')

    heading.pack()
    heading2 = Label(root, text="Photo to Emoji", pady=20, font=('arial', 45, 'bold'), bg='black', fg='#CDCDCD')

    heading2.pack()
    lmain = tk.Label(master=root, padx=50, bd=10)  # Label(父对象, padx=水平边距, pady=垂直边距) padx pady: 单位 像素
    lmain2 = tk.Label(master=root, bd=10)

    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900, y=350)

    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'
    exitbutton = Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)
    show_vid()

    root.mainloop()