import sys
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score

def process_data(train_input,test_input,max_depth1,train_out,test_out,metrics_out):

        data = pd.read_csv(train_input, sep='\t')#导入数据集
        #print(data.shape)#输出一下数据集规模

        #数据预处理，将特征值和目标值用数字进行替换
        outlabel = []
        inlabel = []
        #采用暴力比较的形式得到两个不一样的值作为intemp1和intemp2，作为特征值的字符串
        #采用暴力比较的形式得到两个不一样的值作为outtemp1和outtemp2，作为目标值的字符串
        intemp = data.iloc[1, 0]
        outtemp = data.iloc[1, -1]
        intemp2 = data.iloc[2, 0]
        outtemp2 = data.iloc[2, -1]
        i = 2
        while intemp == intemp2:
            i += 1
            intemp2 = data.iloc[i, 0]
        i = 2
        while outtemp == outtemp2:
            i += 1
            outtemp2 = data.iloc[i, -1]
        outlabel.append(outtemp)
        inlabel.append(intemp)
        outlabel.append(outtemp2)
        inlabel.append(intemp2)
        #对于特征值和目标值字符串进行0和1的替代
        if inlabel[0] != outlabel[0] and inlabel[0] != outlabel[1]:
            data = data.replace(inlabel[0], 0)
            data = data.replace(inlabel[1], 1)
            data = data.replace(outlabel[0], 0)
            data = data.replace(outlabel[1], 1)
        else:
            data = data.replace(outlabel[0], 0)
            data = data.replace(outlabel[1], 1)
        #print(data)#输入一下替换完后的样本
        X = data.iloc[:, :-1] # 特征列
        Y = data.iloc[:, -1]   # 目标列

        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth1)
        clf = clf.fit(X, Y)#生成决策树
        feature_names = X.columns
        text_representation = tree.export_text(clf,feature_names=feature_names)
        print(text_representation)#决策树可视化
        print("其中特征值中的0代表着"+str(inlabel[0])+'\n')
        print("其中特征值中的1代表着" + str(inlabel[1]) + '\n')
        print("其中目标值中的0代表着" + str(outlabel[0]) + '\n')
        print("其中目标值中的1代表着" + str(outlabel[1]) + '\n')

        #得到在训练集上的预测准确率
        predictions1 = clf.predict(X)
        accuracy1 = accuracy_score(Y, predictions1)
        error_store1 = 1 - accuracy1

        # 写入训练集上决策树的预测结果
        with open(train_out, 'w') as f:
            # 遍历预测结果，将每个预测的标签写入到文件中
            for label in predictions1:
                # 写入标签，注意根据你的数据决定是否需要转换格式
                if label == 0:
                    f.write(str(outlabel[0]) + '\n')
                else:
                    f.write(str(outlabel[1]) + '\n')

        #对测试数据进行预处理
        test_data = pd.read_csv(test_input, sep='\t')
        # test_data = test_data.drop(index=0, axis=0)
        test_data = test_data.replace(inlabel[0], 0)
        test_data = test_data.replace(inlabel[1], 1)
        test_data = test_data.replace(outlabel[0], 0)
        test_data = test_data.replace(outlabel[1], 1)
        Xtest=test_data.iloc[:, :-1]
        Ytest=test_data.iloc[:, -1]

        #得到预测准确率
        predictions = clf.predict(Xtest)
        accuracy = accuracy_score(Ytest, predictions)
        error_store = 1 - accuracy
        print("Accuracy:", accuracy)#在测试集上对决策树进行测试

        # 写入测试集上决策树的预测结果
        with open(test_out, 'w') as f:
            # 遍历预测结果，将每个预测的标签写入到文件中
            for label in predictions:
                # 写入标签，注意根据你的数据决定是否需要转换格式
                if label==0:
                    f.write(str(outlabel[0]) + '\n')
                else:
                    f.write(str(outlabel[1]) + '\n')

        # 将决策树错误率写入输出文件metrics_out
        with open(metrics_out, 'w') as f:
            #f.write(text_representation)  # 决策树可视化
            f.write('error(train):' + str(error_store1) + '\n')
            f.write('error(test):'+str(error_store))

if __name__ == "__main__":
    # 确保提供了足够的命令行参数
    if len(sys.argv) != 7:
        print("请输入正确的参数的形式")
        sys.exit(1)
    # 获取命令行参数
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    # 处理数据
    process_data(input_file, output_file, max_depth, train_out, test_out, metrics_out)
    print("成功写入!")

##脚本命令decision.py small_train.tsv small_test.tsv 3 train_out.label test_out.label 1.txt