#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:31:14 2017

@author: sonic
"""

# import modules
import pandas as pd

xls_file = pd.ExcelFile('Domains-and-glossary.xlsx')

print(xls_file.sheet_names)


df = xls_file.parse('Glossary (by domain)')
print (df.head())

