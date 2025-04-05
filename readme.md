# 拼音输入法 Pinyin Input with HMM Model
## 简介 Introduction
拼音输入法是一个基于隐马尔可夫模型(HMM)实现的从拼音转换成汉字的统计方法的人工智能。项目中实现了一元组和二元组概率的计算，及应用Laplace Smoothing的原理于转移概率的计算中。
Pinyin input is a Hidden Markov Model (HMM)-based Artificial Intelligence technique widely used in Natural Language Processing (NLP). This project successfully implements the calculation of unigram and bigram transition probabilities, incorporating Laplace smoothing to enhance accuracy.

## 运行方式 How to run the project
一元与二元词频表可通过运行以下指令生成，
python src/process_data.py
运行的结果将分别存储到processed_data目录下的1_word_prob.txt和2_word_prob.txt

生成词频表后，可通过以下指令执行程序：
(Linux指令)
python main.py <data/input.txt >data/output.txt
(Powershell指令)
Get-Content data/input.txt | python main.py | Out-File -Encoding utf8 data/output.txt

**注：由于输出方式不同，可能对准确率结果有微小的差异。实验报告中的准确率是直接运行原始代码所获得的，但为了支持命令行形式使用重定向运行程序，文件的写入和读取过程中换行符、缓冲区刷新或编码等细节不完全一致可能会造成结果存在微小的差异。