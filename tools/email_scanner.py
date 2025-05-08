# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:33:24 2019

@author: huang
"""

import email
import imaplib
from bs4 import beautifulSoup #handle the html files

import os 

username = 'carrickhuang81@gmail.com'
password = 'handsome81'

mail = imaplib.IMAP4_SSL("imap.gmail.com")
mail.login(username, password)

mail.select("inbox")

#creat new  folder 
#mail.create("Business")

#list folder
mail.list()

result, data = mail.uid('search', None, "All")

inbox_item_list = data[0].split()

#get the most recent email id
#most_recent = inbox_item_list[-1]

for item in inbox_item_list:
    result2, email_data = mail.uid('fetch', item, '(RFC822)')
    #if work for python 2, do not need to decode it
    raw_email = email_data[0][1].decode("utf-8")
    email_message = email.message_from_string(raw_email)
    to_ = email_message['To']
    from_ = email_message['From']
    subject_ = email_message['Subject']
    date_ = email_message['date']
    counter = 1
    for part in email_message.walk():
        if part.get_content_maintype() == 'multipart':
            continue 
        filename = part.get_filename()
        content_type = part.get_content_type()
        if not filename:
            ext = mimetypes.guess_extension(content_type)
            if not ext:
                ext = '.bin'
            if 'text' in content_type:
                ext = '.txt'
            elif 'html' in content_type:
                ext = 'html'
            filename = 'msg-part-%08d%s' (counter, ext)
        counter += 1
        
    #save file
    save_path = os.path.join(os.getcwd(), "emails", date_, subject_,)
    if not os.path.exisits(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, filename), 'wb') as fp:
        fp.write(part.get_payload(decode=True))
        
    #print(subject_)
    #print(content_type)
    #if "plain" in content_type:
    #    print(part.get_payload())
    #elif "html" in content_type:
        #how do we handle the html content
    #    html_ = part.get_payload()
    #    soup = BeautifulSoup(html_, "html.parser")
    #    text = soup.get_text()
    #    print(text)
    #else:
    #    print(content_type)
    