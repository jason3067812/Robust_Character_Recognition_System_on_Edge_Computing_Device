# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:05:23 2020

@author: user
"""

import tkinter as tk
from tkinter import *
from pypylon import pylon
import cv2, os, sys, shutil

import numpy as np
import time, datetime
from PIL import Image, ImageTk
import pickle
import csv
import socket, pickle, struct


# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

def connect_server():
    # serverHost = 'DESKTOP-UR7UQ1L'
    serverHost = socket.gethostname()
    port = 54321
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket Created')
        server_ip = socket.gethostbyname(serverHost)
        print('IP Address of ' + serverHost + ' is ' + server_ip)
        s.connect((server_ip, port))
        print('Socket Connected to ' + serverHost + ' on ip ' + server_ip)
        message = 'agxid'
        encodeMessage = message.encode()
        s.sendall(encodeMessage)
        print('Message send successfully')
        reply = s.recv(4096)
        get_message = reply.decode()
        return get_message
        s.close()
    except  socket.error:
        print('Failed!')


update_loop = True
def win_closing():
    global update_loop
    update_loop = False
    camera.StopGrabbing()
    # cam1.release()
    root.destroy()

root = tk.Tk()
root.title('ID識別_v1.0.0_update_20201105')
root.resizable(0, 0)

root.protocol('WM_DELETE_WINDOW', win_closing)

# =============================================================================
# 上半部
# =============================================================================

cameraFrame = tk.LabelFrame(root, text='即時畫面', fg='blue', font=('標楷體', 16))
cameraFrame.grid(column=0, row=0, padx=2, pady=2)
cameraFrame_up = tk.Frame(cameraFrame)
cameraFrame_up.grid(column=0, row=0)
screen1 = tk.Label(cameraFrame_up)
screen1.grid(column=0, row=0, padx=2, pady=2)

#initScreen = np.zeros([int(2048/4), int(2448/4)], dtype=int) # H*W #
initScreen = np.zeros([int(2048/4), int(2448/4)]) #linux無法採用dtype=int
initScreen = Image.fromarray(initScreen)
initScreen_tk = ImageTk.PhotoImage(image=initScreen)
screen1.imgtk = initScreen_tk
screen1.config(image=initScreen_tk)

cameraFrame_down = tk.Frame(cameraFrame)
cameraFrame_down.grid(column=0, row=1)
camNum1 = tk.Label(cameraFrame_down, text='攝影機狀態:', font=('標楷體', 13, 'bold'), fg='gray').grid(column=0, row=0, padx=2, pady=2)
camStatus1 = tk.Label(cameraFrame_down, text='初始化', font=('標楷體', 13), fg='white', bg='orange')
camStatus1.grid(column=1, row=0, padx=2, pady=2)

show_time = tk.Label(cameraFrame, text='日期:0000-00-00 00:00:00', font=('標楷體', 13, 'bold'), fg='Coral')
show_time.grid(column=0, row=2, padx=2, pady=2)

# =============================================================================
# 下半部
# =============================================================================
infoFrame = tk.LabelFrame(root, text='系統資訊', fg='blue', font=('標楷體', 16))
infoFrame.grid(column=0, row=1, padx=2, pady=2)
left_frame = tk.Frame(infoFrame)
left_frame.grid(column=0, row=0)

currentID = tk.Label(left_frame, text='程控提供序號:', font=('標楷體', 40, 'bold'), fg='gray').grid(column=0, row=0, padx=2, pady=2)
L1_str = tk.StringVar()
currentID_box = tk.Entry(left_frame, textvariable=L1_str, bd=5, width=15, font=('標楷體', 40, 'bold'))
currentID_box.grid(column=1, row=0, padx=2, pady=2)

sysID = tk.Label(left_frame, text='系統辨識結果:', font=('標楷體', 40, 'bold'), fg='gray').grid(column=0, row=1, padx=2, pady=2)
sysID_box_str = tk.StringVar()
sysID_box = tk.Entry(left_frame, textvariable=sysID_box_str, bd=5, width=15, font=('標楷體', 40, 'bold'))
sysID_box.grid(column=1, row=1, padx=2, pady=2)

# 現場ID比對
# =============================================================================
med_frame = tk.Frame(infoFrame)
med_frame.grid(column=1, row=0, padx=2, pady=2)
compare_result = tk.Label(med_frame, text='比對結果:', font=('標楷體', 40, 'bold'), fg='gray').grid(column=0, row=0, padx=2, pady=2)
check_result = tk.Label(med_frame, text='?', font=('標楷體', 50, 'bold'), fg='red', bg='yellow')
check_result.grid(column=0, row=1, padx=2, pady=2)

# 建立一個list裝csv的答案
ans_lst = []
with open('./paper_ans.csv', 'r', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        ans_lst.append(row)

# ========填入csv(答案)======== # 目前是手動輸入
csv_row = 0
def input_ans():
    global csv_row
    csv_row += 1
    L1_str.set(ans_lst[csv_row][1])
    #print(csv_row)
# ============================= #
# 重寫一個 new_csv, 每當更新重寫一次
def call_new_csv():
    global ans_lst
    with open('./new_ans.csv', 'w', newline='') as newcsv:
        writer = csv.writer(newcsv)
        for i in ans_lst:
            writer.writerow(i)

Button(root, text='程控序號輸入下一張', font=('標楷體', 30, 'bold'), command=input_ans).grid(column=0, row=3, padx=2, pady=2)

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
serverHost = socket.gethostname()
port = 54321
  
os.makedirs('./images/', exist_ok=True)
# ==================================================================== #

while update_loop == True:
    root.update()
    show_time.configure(text=('日期:' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    if camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():

            image = converter.Convert(grabResult)
            img = image.GetArray()

            # result, frame = cv2.imencode('.jpg', img, encode_param)
            # data = pickle.dumps(frame, 0)
            # size = len(data)
            imgname = '{}.jpg'.format(time.strftime("%Y%m%d%H%M%S", time.localtime()))
            cv2.imwrite('./images/'+imgname, img)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                print('Socket Created')
                server_ip = socket.gethostbyname(serverHost)
                print('IP Address of ' + serverHost + ' is ' + server_ip)
                s.connect((server_ip, port))
                print('Socket Connected to ' + serverHost + ' on ip ' + server_ip)
                encodeMessage = imgname.encode()
                s.sendall(encodeMessage)
                print('Message send successfully')
                reply = s.recv(4096)
                get_message = reply.decode()
                sysID_box_str.set(get_message)
                s.close()
            except socket.error:
                print('Failed!')
                
            img = cv2.resize(img, (int(2448/4), int(2048/4)), interpolation=cv2.INTER_CUBIC)
            out = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            tmpImg = Image.fromarray(out)
            show = ImageTk.PhotoImage(image=tmpImg)
            screen1.imgtk = show
            screen1.config(image=show)
            camStatus1.configure(text=('執行中'), bg='green')
            time.sleep(0.5)
        else:
            camStatus1.configure(text=('執行錯誤'), bg='red')
            screen1.imgtk = initScreen_tk
            screen1.config(image=initScreen_tk)

        grabResult.Release()
            



  
           
        
        
