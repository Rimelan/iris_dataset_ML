#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:16:43 2026

@author: a.blbn
"""
import load_data

if __name__ == '__main__':
    DIris,LIris = load_data.load_data("iris.csv")
    D = DIris[:, LIris != 0 ]
    L = LIris[LIris != 0]
    