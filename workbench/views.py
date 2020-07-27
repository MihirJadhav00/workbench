from django.shortcuts import render
from .Bench_Tests import CSVPlotter,callerFunction
import os

def HomePage(requests):
    if requests.method == "POST":
        class_name = requests.POST.get("slct1")
        file_name = requests.POST.get("slct2")
        print(class_name,file_name)
        CSVPlotter("C:/Users/Mihir Jadhav/Downloads/web_bench/workbench/workbench/data/",class_name,file_name)
    
    return render(requests,"index.html")

def ConfirmPage(requests):
    if requests.method == "POST":
        bench_msg = requests.POST.get("commit_msg")
        callerFunction(bench_msg)
    return render(requests,"warning.html")