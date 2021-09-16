#!/usr/bin/env python
# --- --- --- 
# rcsi_utils
# --- --- --- 
"""
    Copyright (C) 2012 Bo Zhu http://about.bozhu.me
    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""
# rc4 encryption
# modified from source: https://github.com/bozhu/RC4-Python/blob/master/rc4.py
# --- --- --- 
import sys
# --- --- --- 
def RCsi_CRYPT(key, data):
    S = list(range(256))
    j = 0
    for i in list(range(256)):
        j = (j + S[i] + ord(key[i % len(key)])) % 256
        S[i], S[j] = S[j], S[i]
    j = 0
    y = 0
    out = []
    for char in data:
        j = (j + 1) % 256
        y = (y + S[j]) % 256
        S[j], S[y] = S[y], S[j]
        if sys.version_info.major == 2: #python 2
            out.append(unichr(ord(char) ^ S[(S[j] + S[y]) % 256]))
        else: #python 3
            out.append(chr(ord(char) ^ S[(S[j] + S[y]) % 256]))
    return ''.join(out)
# --- --- --- 
if __name__ == '__main__':
    # ---
    key = input("ENTER KEY:")
    
    while True:
        plain = input("ENTER TEXT:")
        # --- encription
        encrypted = RCsi_CRYPT(key, plain)
        print('encrypted: <{}> : <{}>'.format(type(encrypted),repr(encrypted)))
        # --- decryption
        decrypted = RCsi_CRYPT(key, encrypted)
        print('decrypted: <{}> : <{}>'.format(type(decrypted),repr(decrypted)))