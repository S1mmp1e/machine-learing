import sys
import numpy as np
import pandas as pd

def process_data(input_path,output_path):
        data = pd.read_csv(input_path, sep='\t',header=0)#导入数据集,除去行头
        #print(data.shape)#输出一下数据集规模
        count_1 = 0
        count_2 = 0
        temp=data.iloc[1, -1]#取第一行的目标值作为一个可能性输出
        for row in data.iloc[:, -1]:
            if temp in row:
                count_1 += row.count(temp)#计算该输出的数量
        total_rows = data.shape[0]
        count_2=total_rows-count_1#计算不是该输出的数量
        #计算错误率
        if count_1 < count_2:
            error = float(count_1)/total_rows
        else:
            error = float(count_2)/total_rows
        prob_1 = float(count_1)/total_rows
        prob_2 = float(count_2)/total_rows
        entropy = 0.0
        entropy -= prob_1 * np.log2(prob_1)
        entropy -= prob_2 * np.log2(prob_2)
        # 将最后的数据写入输出文件
        with open(output_path, 'w') as f:
            f.write('entropy: '+str(entropy)+'\n')
            f.write('error: '+str(error)+'\n')

if __name__ == "__main__":
    # 确保提供了足够的命令行参数
    if len(sys.argv) != 3:
        print("请输入: python inspection.py input_file output_file")
        sys.exit(1)
    # 获取命令行参数
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    # 处理数据
    process_data(input_file, output_file)
    print("成功写入!")

##命令行脚本格式：inspection.py small_train.tsv 1.txt